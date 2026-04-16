"""OhioThinStrategy — Freqtrade IStrategy with full OHIO pipeline wiring.

Pipeline: OHLCV -> Feature -> Normalize -> Factor(7-axis) -> Stabilize
          -> Meta -> Fitness -> Policy -> Execute

Completed in FT-017: all pipeline modules are wired.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Order, Trade
from freqtrade.strategy.interface import IStrategy

from freqtrade.ohio.adapters.freqtrade.cross_asset_provider import (
    compute_breadth_dispersion,
    compute_correlation_stress,
    compute_peer_returns,
    fetch_peer_closes,
)
from freqtrade.ohio.adapters.freqtrade.exit_adapter import ExitParams
from freqtrade.strategy.parameters import DecimalParameter, IntParameter
from freqtrade.ohio.adapters.freqtrade.metadata_handler import save_entry_metadata
from freqtrade.ohio.adapters.freqtrade.position_adapter import (
    compute_leverage,
    compute_stake,
)
from freqtrade.ohio.adapters.freqtrade.risk_gate_adapter import RiskGateAdapter
from freqtrade.ohio.core.domain.models import (
    DataMode,
    ExecutionPolicy,
    StateMeta,
    StrategyFitness,
    StrategyMode,
)
from freqtrade.ohio.core.market_state.feature_builder import FeatureBuilder
from freqtrade.ohio.core.market_state.factors import FactorCalculator
from freqtrade.ohio.core.market_state.meta_calculator import MetaCalculator
from freqtrade.ohio.core.market_state.normalizer import Normalizer
from freqtrade.ohio.core.market_state.stabilizer import StateStabilizer
from freqtrade.ohio.core.portfolio_risk.drawdown_controller import DrawdownController
from freqtrade.ohio.core.portfolio_risk.kill_switch import KillSwitch
from freqtrade.ohio.core.strategy_router.fitness_estimator import FitnessEstimator
from freqtrade.ohio.core.strategy_router.policy_generator import PolicyGenerator
from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles


logger = logging.getLogger(__name__)


class OhioThinStrategy(IStrategy):
    """OHIO Market State Engine strategy — Freqtrade adapter.

    Pipeline: OHLCV -> Feature -> Normalize -> Factor(7-axis) -> Stabilize
              -> Meta -> Fitness -> Policy -> Execute
    """

    INTERFACE_VERSION = 3

    # --- Required IStrategy attributes ---
    timeframe = "1h"
    startup_candle_count = 4320  # 180 days warmup

    # Stoploss: hard ceiling — custom_stoploss narrows per mode
    stoploss = -0.341

    # ROI targets for 1H crypto — take profits quickly
    minimal_roi = {"0": 0.05, "24": 0.03, "72": 0.02, "168": 0.01, "336": 0}

    # Enable features we need
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    process_only_new_candles = True
    position_adjustment_enable = True
    # V2: long/short via trend direction (ohio_stable_trend).
    can_short = True

    # ------------------------------------------------------------------
    # Hyperopt parameters
    # ------------------------------------------------------------------

    # Hedge algorithm — hyperopt-optimized values
    hedge_eta = DecimalParameter(0.01, 0.50, default=0.16, decimals=2, space="buy", optimize=True)
    hedge_temperature = DecimalParameter(0.5, 5.0, default=0.5, decimals=1, space="buy", optimize=True)

    # Policy — hyperopt-optimized
    disabled_threshold = DecimalParameter(0.20, 0.60, default=0.35, decimals=2, space="buy", optimize=True)
    min_trend_confidence = DecimalParameter(0.05, 0.25, default=0.05, decimals=2, space="buy", optimize=False)  # noqa: E501

    # Regime maturity cooldown — Stage 1 optimized, LOCKED
    regime_cooldown_bars = IntParameter(0, 10, default=0, space="buy", optimize=False)

    # --- Mode-specific entry parameters (buy space) ---
    # >>> ALL STAGES COMPLETE — Final optimized values <<<

    # Trend Following — Stage 2a LOCKED (Sharpe -0.46, 300 epochs)
    tf_adx_threshold = DecimalParameter(15.0, 40.0, default=36.4, decimals=1, space="buy", optimize=False)
    tf_hurst_threshold = DecimalParameter(0.45, 0.70, default=0.70, decimals=2, space="buy", optimize=False)
    tf_kama_slope_threshold = DecimalParameter(0.0001, 0.005, default=0.005, decimals=4, space="buy", optimize=False)
    tf_vol_scale_target = DecimalParameter(0.5, 2.0, default=0.6, decimals=1, space="buy", optimize=False)

    # Mean Reversion — Stage 2b LOCKED (Sharpe -0.22, 300 epochs)
    mr_zscore_entry = DecimalParameter(1.5, 3.0, default=2.7, decimals=1, space="buy", optimize=False)
    mr_rsi_oversold = IntParameter(20, 40, default=29, space="buy", optimize=False)
    mr_rsi_overbought = IntParameter(60, 80, default=60, space="buy", optimize=False)
    mr_hurst_max = DecimalParameter(0.35, 0.55, default=0.46, decimals=2, space="buy", optimize=False)

    # Breakout — Stage 3a LOCKED (Sharpe -0.22, 200 epochs)
    bo_squeeze_min_bars = IntParameter(3, 10, default=5, space="buy", optimize=False)
    bo_donchian_window = IntParameter(10, 30, default=13, space="buy", optimize=False)

    # Defensive — Stage 3b LOCKED (Sharpe +0.38, 200 epochs)
    def_adx_max = DecimalParameter(15.0, 30.0, default=17.9, decimals=1, space="buy", optimize=False)
    def_vol_scale = DecimalParameter(0.05, 0.30, default=0.18, decimals=2, space="buy", optimize=False)
    def_min_trend_abs = DecimalParameter(0.01, 0.10, default=0.06, decimals=2, space="buy", optimize=False)

    # Exit timing — Stage 4a LOCKED (Sharpe +1.43, 400 epochs)
    time_exit_bars = IntParameter(6, 48, default=31, space="sell", optimize=False)
    time_exit_fitness = DecimalParameter(0.20, 0.55, default=0.47, decimals=2, space="sell", optimize=False)

    # Exit — regime (hyperopt-optimized)
    regime_exit_risk = DecimalParameter(0.50, 0.99, default=0.70, decimals=2, space="sell", optimize=True)

    # Exit — profit preserve (hyperopt-optimized)
    profit_preserve_profit = DecimalParameter(0.01, 0.08, default=0.04, decimals=2, space="sell", optimize=True)
    profit_preserve_fitness = DecimalParameter(0.30, 0.60, default=0.58, decimals=2, space="sell", optimize=False)

    # Stoploss (hyperopt-optimized)
    trailing_profit_threshold = DecimalParameter(0.005, 0.05, default=0.018, decimals=3, space="sell", optimize=True)
    trailing_profit_ratio = DecimalParameter(0.30, 0.70, default=0.63, decimals=2, space="sell", optimize=True)
    transition_tighten_factor = DecimalParameter(0.50, 0.90, default=0.50, decimals=2, space="sell", optimize=False)
    sl_hard_floor = DecimalParameter(-0.30, -0.10, default=-0.19, decimals=2, space="sell", optimize=False)

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------
    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # Pipeline stages — instantiated once
        self._feature_builder = FeatureBuilder()
        self._normalizer = Normalizer(window=4320)
        self._factor_calculator = FactorCalculator()
        self._stabilizer = StateStabilizer()
        self._meta_calculator = MetaCalculator()
        self._fitness_estimator = FitnessEstimator(
            hedge_eta=self.hedge_eta.value,
            hedge_temperature=self.hedge_temperature.value,
        )
        self._policy_generator = PolicyGenerator(
            disabled_threshold=self.disabled_threshold.value,
        )

        # Risk / portfolio
        self._dd_controller = DrawdownController()
        self._kill_switch = KillSwitch()
        self._risk_gate = RiskGateAdapter(self._dd_controller, self._kill_switch)

        # Strategy profiles
        self._profiles = load_default_profiles()

    # ------------------------------------------------------------------
    # informative_pairs — Cross-asset peer basket
    # ------------------------------------------------------------------
    def informative_pairs(self) -> list[tuple[str, str]]:
        """Declare peer basket pairs so Freqtrade pre-fetches their OHLCV data.

        Uses the futures contract format (pair:settle) to match the trading mode.
        Without this, dp.get_pair_dataframe() returns empty for non-whitelist peers,
        leaving correlation_stress and breadth_dispersion as NaN in live/dryrun.
        """
        from freqtrade.ohio.adapters.freqtrade.cross_asset_provider import DEFAULT_PEERS
        # Convert spot format to futures format for the exchange settle currency
        settle = self.config.get("stake_currency", "USDT")
        pairs = [
            (f"{pair}:{settle}", self.timeframe)
            for pair in DEFAULT_PEERS
        ]
        # 4H timeframe for each whitelist pair — multi-TF regime confirmation
        whitelist = self.dp.current_whitelist() if self.dp else []
        for pair in whitelist:
            pairs.append((pair, "4h"))
            pairs.append((pair, "1d"))  # NEW: daily regime filter
        return pairs

    # ------------------------------------------------------------------
    # populate_indicators — Full 7-Phase Pipeline
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Phases 2-7: Full OHIO pipeline."""
        logger.info(
            "ohio.populate_indicators | pair=%s | rows=%d",
            metadata["pair"],
            len(dataframe),
        )

        # Phase 2: Feature extraction
        dataframe = self._feature_builder.compute(dataframe)

        # Phase 2b: Cross-asset features (correlation_stress, breadth_dispersion)
        try:
            settle = self.config.get("stake_currency", "USDT")
            peer_closes = fetch_peer_closes(self.dp, metadata["pair"], settle=settle)
            if len(peer_closes) >= 2:
                peer_returns = compute_peer_returns(peer_closes)
                corr_series = compute_correlation_stress(peer_returns)
                breadth_series = compute_breadth_dispersion(peer_returns)
                # Align by position — peer and strategy df share candle timeline
                if len(corr_series) >= len(dataframe):
                    dataframe["ohio_feat_correlation_stress"] = corr_series.values[:len(dataframe)]
                    dataframe["ohio_feat_breadth_dispersion"] = (
                        breadth_series.values[:len(dataframe)]
                    )
                else:
                    pad = len(dataframe) - len(corr_series)
                    dataframe["ohio_feat_correlation_stress"] = np.concatenate(
                        [np.full(pad, np.nan), corr_series.values]
                    )
                    dataframe["ohio_feat_breadth_dispersion"] = np.concatenate(
                        [np.full(pad, np.nan), breadth_series.values]
                    )
                logger.info(
                    "ohio.cross_asset | pair=%s | peers=%d",
                    metadata["pair"],
                    len(peer_closes),
                )
            else:
                logger.warning(
                    "ohio.cross_asset | pair=%s | insufficient peers (%d), using NaN",
                    metadata["pair"],
                    len(peer_closes),
                )
        except Exception:
            logger.warning(
                "ohio.cross_asset | pair=%s | failed, using NaN fallback",
                metadata["pair"],
                exc_info=True,
            )

        # Phase 3: Normalization + Factor calculation
        dataframe = self._normalizer.normalize(dataframe)
        dataframe = self._factor_calculator.compute(dataframe)

        # Phase 4: State stabilization
        dataframe = self._stabilizer.stabilize(dataframe)

        # Phase 5: Meta + Fitness + Policy
        dataframe = self._meta_calculator.compute(dataframe)
        dataframe = self._fitness_estimator.compute_dataframe(dataframe)
        dataframe = self._policy_generator.generate_dataframe(dataframe)

        # Phase 2c: 4H multi-timeframe features for regime confirmation
        try:
            df_4h = self.dp.get_pair_dataframe(metadata["pair"], "4h")
            if df_4h is not None and len(df_4h) > 20:
                # 4H ADX — Wilder's ADX(14) inline (same logic as FeatureBuilder)
                period = 14
                alpha = 1.0 / period
                high_4h = df_4h["high"]
                low_4h = df_4h["low"]
                close_4h = df_4h["close"]
                hl = high_4h - low_4h
                hpc = (high_4h - close_4h.shift(1)).abs()
                lpc = (low_4h - close_4h.shift(1)).abs()
                tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
                up_move = high_4h.diff()
                down_move = -low_4h.diff()
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
                plus_dm_s = pd.Series(plus_dm, index=df_4h.index, dtype=float)
                minus_dm_s = pd.Series(minus_dm, index=df_4h.index, dtype=float)
                tr_smooth = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
                pdm_smooth = plus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
                mdm_smooth = minus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
                plus_di = 100 * pdm_smooth / tr_smooth.replace(0, np.nan)
                minus_di = 100 * mdm_smooth / tr_smooth.replace(0, np.nan)
                di_sum = plus_di + minus_di
                dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
                adx_4h = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

                # 4H SMA slope (SMA20 direction)
                sma_4h = close_4h.rolling(20).mean()
                sma_slope_4h = (sma_4h - sma_4h.shift(4)) / (sma_4h.shift(4) + 1e-10)

                # Merge by aligning timestamps — use merge_asof for proper time alignment
                df_4h_features = df_4h[["date"]].copy()
                df_4h_features["ohio_4h_adx"] = adx_4h.values
                df_4h_features["ohio_4h_sma_slope"] = sma_slope_4h.values

                # Merge into 1H dataframe by nearest date (4H candle applies to all 1H candles within it)
                if "date" in dataframe.columns:
                    dataframe = pd.merge_asof(
                        dataframe.sort_values("date"),
                        df_4h_features.sort_values("date"),
                        on="date",
                        direction="backward",
                    )
                else:
                    logger.warning("ohio.4h_features | no date column, skipping 4H merge")
            else:
                logger.warning("ohio.4h_features | insufficient 4H data for %s", metadata["pair"])
        except Exception:
            logger.warning("ohio.4h_features | failed for %s", metadata["pair"], exc_info=True)

        # Phase 2d: 1D daily regime filter
        try:
            df_1d = self.dp.get_pair_dataframe(metadata["pair"], "1d")
            if df_1d is not None and len(df_1d) > 50:
                close_1d = df_1d["close"]
                ema_fast_1d = close_1d.ewm(span=10, adjust=False).mean()
                ema_slow_1d = close_1d.ewm(span=20, adjust=False).mean()
                sma_50_1d = close_1d.rolling(50).mean()

                # Bull: 10 EMA > 20 EMA AND close > 50 SMA
                # Bear: 10 EMA < 20 EMA AND close < 50 SMA
                # Choppy: mixed signals
                regime_1d = pd.Series("choppy", index=df_1d.index, dtype=object)
                bull_cond = (ema_fast_1d > ema_slow_1d) & (close_1d > sma_50_1d)
                bear_cond = (ema_fast_1d < ema_slow_1d) & (close_1d < sma_50_1d)
                regime_1d.loc[bull_cond] = "bull"
                regime_1d.loc[bear_cond] = "bear"

                df_1d_feat = df_1d[["date"]].copy()
                df_1d_feat["ohio_1d_regime"] = regime_1d.values

                if "date" in dataframe.columns:
                    dataframe = pd.merge_asof(
                        dataframe.sort_values("date"),
                        df_1d_feat.sort_values("date"),
                        on="date",
                        direction="backward",
                    )
            else:
                logger.warning("ohio.1d_regime | insufficient 1D data for %s", metadata["pair"])
        except Exception:
            logger.warning("ohio.1d_regime | failed for %s", metadata["pair"], exc_info=True)

        return dataframe

    # ------------------------------------------------------------------
    # populate_entry_trend — Independent OR-combined Entry Signals
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals via independent OR-combined strategies.

        Architecture: Multiple independent signals fire independently.
        Ohio StateVector is used for risk sizing, NOT entry gating.
        """
        logger.info("ohio.populate_entry_trend | pair=%s", metadata["pair"])
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        # Entry strategies — MACD div removed (no edge), breakout removed (false breakouts)
        from freqtrade.ohio.adapters.freqtrade.entries.ema_cross import EMACrossEntry
        from freqtrade.ohio.adapters.freqtrade.entries.bb_dip import BBDipEntry

        entry_params = self._entry_params_dict()

        # EMA cross: both long and short (net positive in testing)
        ema = EMACrossEntry()
        df_ema = ema.compute_entries(dataframe.copy(), entry_params)
        for side, col, suffix in [("long", "enter_long", ""), ("short", "enter_short", "_short")]:
            mask = (df_ema[col] == 1) & (dataframe[col] == 0)
            dataframe.loc[mask, col] = 1
            dataframe.loc[mask, "enter_tag"] = f"ohio_ema_cross{suffix}"

        # BB dip: LONG ONLY (4H trend filter in bb_dip.py prevents buying in bear)
        bb = BBDipEntry()
        df_bb = bb.compute_entries(dataframe.copy(), entry_params)
        long_mask = (df_bb["enter_long"] == 1) & (dataframe["enter_long"] == 0)
        dataframe.loc[long_mask, "enter_long"] = 1
        dataframe.loc[long_mask, "enter_tag"] = "ohio_bb_dip"

        # Suppress longs when ohio_stable_trend is bearish (tightened from -0.20 to -0.10)
        bearish = dataframe.get("ohio_stable_trend", pd.Series(0.0, index=dataframe.index)) < -0.10
        suppress_long = bearish & (dataframe["enter_long"] == 1)
        dataframe.loc[suppress_long, "enter_long"] = 0
        dataframe.loc[suppress_long, "enter_tag"] = ""

        # Suppress shorts when ohio_stable_trend is bullish (tightened from 0.20 to 0.10)
        bullish = dataframe.get("ohio_stable_trend", pd.Series(0.0, index=dataframe.index)) > 0.10
        suppress_short = bullish & (dataframe["enter_short"] == 1)
        dataframe.loc[suppress_short, "enter_short"] = 0
        dataframe.loc[suppress_short, "enter_tag"] = ""

        # Meta signal gates — block entries during regime instability
        transition_risk = dataframe.get(
            "ohio_meta_transition_risk",
            pd.Series(0.0, index=dataframe.index),
        )
        confidence = dataframe.get(
            "ohio_meta_confidence",
            pd.Series(1.0, index=dataframe.index),
        )

        # Block all entries when regime is transitioning OR state is low quality
        meta_block = (transition_risk > 0.80) | (confidence < 0.30)
        block_mask = meta_block & (
            (dataframe["enter_long"] == 1) | (dataframe["enter_short"] == 1)
        )
        dataframe.loc[block_mask, "enter_long"] = 0
        dataframe.loc[block_mask, "enter_short"] = 0
        dataframe.loc[block_mask, "enter_tag"] = ""

        # Correlation stress gate — skip when whole market moves together
        corr_stress = dataframe.get(
            "ohio_factor_correlation",
            pd.Series(0.5, index=dataframe.index),
        )
        high_corr = corr_stress > 0.85
        corr_block = high_corr & (
            (dataframe["enter_long"] == 1) | (dataframe["enter_short"] == 1)
        )
        dataframe.loc[corr_block, "enter_long"] = 0
        dataframe.loc[corr_block, "enter_short"] = 0
        dataframe.loc[corr_block, "enter_tag"] = ""

        # 1D regime filter — asymmetric directional bias + no-trade in chop
        regime_1d = dataframe.get("ohio_1d_regime", pd.Series("choppy", index=dataframe.index))

        # In bull regime: block shorts (don't fight uptrend)
        bull_rows = regime_1d == "bull"
        block_shorts = bull_rows & (dataframe["enter_short"] == 1)
        dataframe.loc[block_shorts, "enter_short"] = 0
        dataframe.loc[block_shorts, "enter_tag"] = dataframe.loc[block_shorts, "enter_tag"].where(
            dataframe.loc[block_shorts, "enter_long"] == 1, ""
        )

        # In bear regime: block longs (don't fight downtrend)
        bear_rows = regime_1d == "bear"
        block_longs = bear_rows & (dataframe["enter_long"] == 1)
        dataframe.loc[block_longs, "enter_long"] = 0
        dataframe.loc[block_longs, "enter_tag"] = dataframe.loc[block_longs, "enter_tag"].where(
            dataframe.loc[block_longs, "enter_short"] == 1, ""
        )

        # In choppy regime: block ALL entries (no-trade zone)
        # Preserves capital during indecisive markets
        choppy_rows = regime_1d == "choppy"
        choppy_block = choppy_rows & (
            (dataframe["enter_long"] == 1) | (dataframe["enter_short"] == 1)
        )
        dataframe.loc[choppy_block, "enter_long"] = 0
        dataframe.loc[choppy_block, "enter_short"] = 0
        dataframe.loc[choppy_block, "enter_tag"] = ""

        return dataframe

    # ------------------------------------------------------------------
    # populate_exit_trend — Minimal (exits via custom_exit)
    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate exit signals (delegated to custom_exit)."""
        logger.info("ohio.populate_exit_trend | pair=%s", metadata["pair"])
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ------------------------------------------------------------------
    # custom_stake_amount — Kelly Sizing
    # ------------------------------------------------------------------
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Kelly-based position sizing via OHIO policy."""
        logger.info(
            "ohio.custom_stake_amount | pair=%s | entry_tag=%s | proposed=%.4f",
            pair,
            entry_tag,
            proposed_stake,
        )

        if min_stake is None:
            return proposed_stake

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return proposed_stake

        last = dataframe.iloc[-1]
        policy = self._build_policy_from_row(last)
        fitness = self._get_best_fitness(last)
        dd_scale = self._dd_controller.size_scale

        # Cross-asset correlation stress and per-asset volatility for sizing
        corr_stress = float(last.get("ohio_factor_correlation", 0.5))
        asset_vol = float(last.get("ohio_feat_atr_ratio_14", 0.0))

        stake = compute_stake(
            policy=policy,
            fitness_score=fitness,
            base_stake=proposed_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            dd_scale=dd_scale,
            correlation_stress=corr_stress,
            asset_volatility=asset_vol,
        )
        return stake

    # ------------------------------------------------------------------
    # leverage — Mode-specific
    # ------------------------------------------------------------------
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """1x leverage — leverage keeps hurting R:R until we redesign stops."""
        return 1.0

    # ------------------------------------------------------------------
    # confirm_trade_entry — Risk Gate
    # ------------------------------------------------------------------
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """RiskEngine gate — block entries that fail risk rules."""
        logger.info(
            "ohio.confirm_trade_entry | pair=%s | entry_tag=%s | side=%s",
            pair,
            entry_tag,
            side,
        )

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return False

        last = dataframe.iloc[-1]
        policy = self._build_policy_from_row(last)
        meta = self._build_meta_from_row(last)

        open_trades = len(Trade.get_trades_proxy(is_open=True))
        daily_loss_pct = self._compute_daily_loss_pct()

        allowed, reasons = self._risk_gate.confirm_entry(
            pair=pair,
            side=side,
            policy=policy,
            meta=meta,
            open_trade_count=open_trades,
            daily_loss_pct=daily_loss_pct,
        )

        if not allowed:
            logger.info("ohio.entry_blocked | pair=%s | reasons=%s", pair, reasons)

        return allowed

    # ------------------------------------------------------------------
    # order_filled — Metadata Storage
    # ------------------------------------------------------------------
    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs,
    ) -> None:
        """Save OHIO metadata on entry fill."""
        logger.info(
            "ohio.order_filled | pair=%s | trade_id=%s | side=%s",
            pair,
            trade.id,
            order.ft_order_side,
        )

        # Only save on entry fills
        if order.ft_order_side == trade.entry_side:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and len(dataframe) > 0:
                last = dataframe.iloc[-1]
                fitness = self._build_fitness_from_row(last)
                meta_snapshot = self._build_meta_from_row(last)
                save_entry_metadata(trade, trade.enter_tag, fitness, meta_snapshot)

                # Store entry ATR for partial-exit calculations
                entry_atr = float(last.get("ohio_feat_atr_ratio_14", 0.02))
                trade.set_custom_data("entry_atr", entry_atr)

    # ------------------------------------------------------------------
    # custom_stoploss — ATR Trailing Stoploss
    # ------------------------------------------------------------------
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """Structure-based stoploss with 3 phases:

        Phase 1 (bars 0-12): swing low/high - 0.5*ATR buffer (real invalidation)
        Phase 2 (+1R): breakeven + 0.2% lock
        Phase 3 (+2R): Chandelier trailing at 2.5x ATR
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return -0.04  # safe fallback

        last = dataframe.iloc[-1]
        atr_ratio = float(last.get("ohio_feat_atr_ratio_14", 0.02))
        entry_rate = trade.open_rate
        is_long = not trade.is_short

        # Determine entry type — MR (BB dip) needs wider stops vs TF (EMA cross)
        entry_tag = trade.enter_tag or ""
        is_mr = "bb_dip" in entry_tag
        is_tf = "ema_cross" in entry_tag or "breakout" in entry_tag

        # 1R definition: ~2x ATR magnitude
        one_r = 2.0 * atr_ratio

        bars = self._bars_since_entry(trade, current_time)

        # Phase 3: Chandelier trailing at +2R profit (applies to all types)
        if current_profit >= 2.0 * one_r:
            return -(2.5 * atr_ratio)

        # Phase 2: Breakeven+0.2% at +1R profit
        if current_profit >= one_r:
            lock = current_profit - one_r - 0.002
            return -max(0.0, lock) if lock > 0 else -0.005

        # Phase 1: Entry-type specific initial stop
        if is_mr:
            # MR (BB dip): wider stop — MR needs room for mean reversion to develop
            # Use 3.5x ATR, floor at -5%
            return max(-(atr_ratio * 3.5), -0.05)

        if is_tf:
            # TF (EMA cross/breakout): structure-based stop using recent swing
            # Keep structure stop active throughout trade (not just 12 bars)
            # Lookback grows with time to follow the trend
            lookback_n = min(len(dataframe), max(bars + 10, 20))
            lookback = dataframe.tail(lookback_n)
            if is_long:
                swing_low = float(lookback["low"].min())
                stop_price = swing_low - (0.5 * atr_ratio * entry_rate)
                stop_ratio = (stop_price / entry_rate) - 1.0
            else:
                swing_high = float(lookback["high"].max())
                stop_price = swing_high + (0.5 * atr_ratio * entry_rate)
                stop_ratio = 1.0 - (stop_price / entry_rate)
            # Widen clamp for long-held trades (let trends breathe)
            max_width = -0.06 if bars > 24 else -0.04
            return max(max_width, min(-0.015, stop_ratio))

        # Default for unrecognized or stalled trades
        return -(3.0 * atr_ratio)

    # ------------------------------------------------------------------
    # custom_exit — Simplified Exit
    # ------------------------------------------------------------------
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """Simplified exit — let ATR trailing and ROI handle most exits."""
        # Kill switch
        if self._kill_switch.active:
            return "ohio_kill_switch"

        # 1D regime-aware ROI (simulates different roi tables per regime)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is not None and len(dataframe) > 0:
            last = dataframe.iloc[-1]
            regime_1d = str(last.get("ohio_1d_regime", "choppy"))
            bars = self._bars_since_entry(trade, current_time)

            is_long = not trade.is_short

            # In bull regime, longs have HIGHER ROI target (let winners run)
            # In bear regime, shorts have HIGHER ROI target
            aligned_with_trend = (regime_1d == "bull" and is_long) or (
                regime_1d == "bear" and not is_long
            )

            if aligned_with_trend:
                # Aligned: hold for bigger wins
                if bars < 24 and current_profit >= 0.10:
                    return "ohio_aligned_roi_10pct"
                if bars < 72 and current_profit >= 0.06:
                    return "ohio_aligned_roi_6pct"
                if bars < 168 and current_profit >= 0.03:
                    return "ohio_aligned_roi_3pct"

        # Stale trade: losing > 2% for > 72 bars
        bars = self._bars_since_entry(trade, current_time)
        if bars > 72 and current_profit < -0.02:
            return "ohio_stale_exit"

        return None

    # ------------------------------------------------------------------
    # adjust_trade_position — Partial Exits (Scaling Out)
    # ------------------------------------------------------------------
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None:
        """Partial exits: 25% at 1R, 50% (of original) at 2R, 25% runner.

        Uses entry_atr saved in order_filled for 1R calculation.
        """
        entry_atr = trade.get_custom_data("entry_atr", default=0.02)
        if not isinstance(entry_atr, (int, float)):
            entry_atr = 0.02
        one_r = 2.0 * float(entry_atr)

        exits_taken = trade.get_custom_data("partial_exits_taken", default=0)
        if not isinstance(exits_taken, int):
            exits_taken = 0

        # Only scale down, never add
        if current_profit <= 0:
            return None

        # Get current total stake from filled entry orders
        filled_entries = trade.select_filled_orders(trade.entry_side)
        if not filled_entries:
            return None
        current_stake = sum(float(o.stake_amount or 0) for o in filled_entries)
        if current_stake <= 0:
            return None

        # First partial at 1R: exit 25% of current
        if exits_taken == 0 and current_profit >= one_r:
            trade.set_custom_data("partial_exits_taken", 1)
            return -(current_stake * 0.25)

        # Second partial at 2R: exit 50% of ORIGINAL (= 2/3 of remaining 75%)
        if exits_taken == 1 and current_profit >= 2.0 * one_r:
            trade.set_custom_data("partial_exits_taken", 2)
            # original_stake = current_stake / 0.75 (since we already exited 25%)
            original_stake = current_stake / 0.75
            return -(original_stake * 0.50)

        return None

    # ==================================================================
    # Helper methods
    # ==================================================================

    def _entry_params_dict(self) -> dict[str, float]:
        """Build entry params dict from current hyperopt parameter values."""
        return {
            # Trend Following
            "tf_adx_threshold": self.tf_adx_threshold.value,
            "tf_hurst_threshold": self.tf_hurst_threshold.value,
            "tf_kama_slope_threshold": self.tf_kama_slope_threshold.value,
            "tf_vol_scale_target": self.tf_vol_scale_target.value,
            # Mean Reversion
            "mr_zscore_entry": self.mr_zscore_entry.value,
            "mr_rsi_oversold": float(self.mr_rsi_oversold.value),
            "mr_rsi_overbought": float(self.mr_rsi_overbought.value),
            "mr_hurst_max": self.mr_hurst_max.value,
            # Breakout
            "bo_squeeze_min_bars": float(self.bo_squeeze_min_bars.value),
            "bo_donchian_window": float(self.bo_donchian_window.value),
            # Defensive
            "def_adx_max": self.def_adx_max.value,
            "def_vol_scale": self.def_vol_scale.value,
            "def_min_trend_abs": self.def_min_trend_abs.value,
        }

    def _exit_params(self) -> ExitParams:
        """Build ExitParams from current hyperopt parameter values."""
        return ExitParams(
            hard_floor=self.sl_hard_floor.value,
            transition_tighten_factor=self.transition_tighten_factor.value,
            trailing_profit_threshold=self.trailing_profit_threshold.value,
            trailing_profit_ratio=self.trailing_profit_ratio.value,
            time_exit_bars=self.time_exit_bars.value,
            time_exit_fitness=self.time_exit_fitness.value,
            regime_exit_risk=self.regime_exit_risk.value,
            profit_preserve_profit=self.profit_preserve_profit.value,
            profit_preserve_fitness=self.profit_preserve_fitness.value,
        )

    def _build_policy_from_row(self, row) -> ExecutionPolicy:
        """Build ExecutionPolicy from DataFrame row's ohio_policy_* columns."""
        try:
            strategy_mode = StrategyMode(str(row.get("ohio_active_mode", "defensive")))
        except ValueError:
            strategy_mode = StrategyMode.DEFENSIVE
        return ExecutionPolicy(
            strategy_mode=strategy_mode,
            enabled=bool(row.get("ohio_policy_enabled", False)),
            size_multiplier=float(row.get("ohio_policy_size_multiplier", 1.0)),
            entry_threshold_adj=float(row.get("ohio_policy_entry_threshold_adj", 0.0)),
            max_positions=int(row.get("ohio_policy_max_positions", 1)),
            stoploss_width_adj=float(row.get("ohio_policy_stoploss_width_adj", 0.0)),
        )

    def _build_meta_from_row(self, row) -> StateMeta:
        """Build StateMeta from DataFrame row's ohio_meta_* columns."""
        try:
            data_mode = DataMode(str(row.get("ohio_meta_data_mode", "full")))
        except ValueError:
            data_mode = DataMode.FULL
        return StateMeta(
            transition_risk=float(row.get("ohio_meta_transition_risk", 0.0)),
            confidence=float(row.get("ohio_meta_confidence", 0.0)),
            stability=float(row.get("ohio_meta_stability", 0.0)),
            data_mode=data_mode,
        )

    def _build_fitness_from_row(self, row) -> StrategyFitness:
        """Build StrategyFitness from DataFrame row's ohio_fitness_* columns."""
        return StrategyFitness(
            trend_following=float(row.get("ohio_fitness_trend_following", 0.0)),
            mean_reversion=float(row.get("ohio_fitness_mean_reversion", 0.0)),
            breakout=float(row.get("ohio_fitness_breakout", 0.0)),
            defensive=float(row.get("ohio_fitness_defensive", 0.0)),
        )

    def _get_best_fitness(self, row) -> float:
        """Get the best fitness score from the active mode."""
        active_mode = str(row.get("ohio_active_mode", "defensive"))
        col = f"ohio_fitness_{active_mode}"
        return float(row.get(col, 0.5))

    def _get_fitness_for_pair(self, pair: str) -> float:
        """Get current fitness score for a pair."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return 0.5
        return self._get_best_fitness(dataframe.iloc[-1])

    def _get_profile_for_mode(self, mode_str: str | None):
        """Get StrategyProfile for a mode string."""
        if mode_str is None:
            return None
        try:
            return self._profiles.get(StrategyMode(mode_str))
        except ValueError:
            return None

    @staticmethod
    def _extract_mode(entry_tag: str | None) -> str | None:
        """Extract mode name from entry_tag like 'ohio_trend_following' -> 'trend_following'."""
        if entry_tag and entry_tag.startswith("ohio_"):
            mode = entry_tag[5:]
            if mode.endswith("_short"):
                mode = mode[:-6]
            return mode
        return entry_tag

    @staticmethod
    def _bars_since_entry(trade, current_time: datetime) -> int:
        """Calculate bars since trade entry (timezone-safe)."""
        if trade.open_date_utc is None:
            return 0
        ct = current_time if current_time.tzinfo else current_time.replace(tzinfo=timezone.utc)
        od = trade.open_date_utc if trade.open_date_utc.tzinfo else trade.open_date_utc.replace(tzinfo=timezone.utc)
        delta = ct - od
        return max(0, int(delta.total_seconds() / 3600))

    @staticmethod
    def _compute_daily_loss_pct() -> float:
        """Compute today's realised loss as a fraction of starting balance."""
        today_start = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        closed_today = Trade.get_trades_proxy(
            is_open=False,
        )
        daily_pnl = sum(
            float(t.close_profit or 0.0)
            for t in closed_today
            if t.close_date and t.close_date >= today_start
        )
        return abs(min(0.0, daily_pnl))

    def _build_fitness_dict(self, row) -> dict[str, float]:
        """Build fitness dict from row for metadata storage."""
        return {
            mode.value: float(row.get(f"ohio_fitness_{mode.value}", 0.0))
            for mode in StrategyMode
        }
