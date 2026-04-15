"""OhioThinStrategy — Freqtrade IStrategy with full OHIO pipeline wiring.

Pipeline: OHLCV -> Feature -> Normalize -> Factor(7-axis) -> Stabilize
          -> Meta -> Fitness -> Policy -> Execute

Completed in FT-017: all pipeline modules are wired.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from pandas import DataFrame

from freqtrade.persistence import Order, Trade
from freqtrade.strategy.interface import IStrategy

from freqtrade.ohio.adapters.freqtrade.cross_asset_provider import (
    compute_breadth_dispersion,
    compute_correlation_stress,
    compute_peer_returns,
    fetch_peer_closes,
)
from freqtrade.ohio.adapters.freqtrade.exit_adapter import (
    ExitParams,
    compute_exit,
    compute_stoploss,
)
from freqtrade.strategy.parameters import DecimalParameter, IntParameter
from freqtrade.ohio.adapters.freqtrade.metadata_handler import (
    get_trade_mode,
    save_entry_metadata,
)
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
    stoploss = -0.20

    # Minimal ROI disabled — exits handled by custom_exit
    minimal_roi = {"0": 100}  # effectively disabled

    # Enable features we need
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    process_only_new_candles = True
    # V2: long/short via trend direction (ohio_stable_trend).
    can_short = True

    # ------------------------------------------------------------------
    # Hyperopt parameters
    # ------------------------------------------------------------------

    # Hedge algorithm
    hedge_eta = DecimalParameter(0.01, 0.50, default=0.10, decimals=2, space="buy", optimize=True)
    hedge_temperature = DecimalParameter(0.5, 5.0, default=2.0, decimals=1, space="buy", optimize=True)

    # Policy
    disabled_threshold = DecimalParameter(0.20, 0.60, default=0.40, decimals=2, space="buy", optimize=True)

    # Exit — timing
    time_exit_bars = IntParameter(6, 48, default=12, space="sell", optimize=True)
    time_exit_fitness = DecimalParameter(0.05, 0.30, default=0.15, decimals=2, space="sell", optimize=True)

    # Exit — regime
    regime_exit_risk = DecimalParameter(0.60, 0.95, default=0.80, decimals=2, space="sell", optimize=True)

    # Exit — profit preserve
    profit_preserve_profit = DecimalParameter(0.01, 0.08, default=0.03, decimals=2, space="sell", optimize=True)
    profit_preserve_fitness = DecimalParameter(0.15, 0.50, default=0.30, decimals=2, space="sell", optimize=True)

    # Stoploss tuning
    trailing_profit_threshold = DecimalParameter(0.005, 0.05, default=0.02, decimals=3, space="sell", optimize=True)
    trailing_profit_ratio = DecimalParameter(0.30, 0.70, default=0.50, decimals=2, space="sell", optimize=True)
    transition_tighten_factor = DecimalParameter(0.50, 0.90, default=0.70, decimals=2, space="sell", optimize=True)
    sl_hard_floor = DecimalParameter(-0.30, -0.10, default=-0.20, decimals=2, space="sell", optimize=True)

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
            peer_closes = fetch_peer_closes(self.dp, metadata["pair"])
            if len(peer_closes) >= 2:
                peer_returns = compute_peer_returns(peer_closes)
                dataframe["ohio_feat_correlation_stress"] = compute_correlation_stress(
                    peer_returns
                ).reindex(dataframe.index)
                dataframe["ohio_feat_breadth_dispersion"] = compute_breadth_dispersion(
                    peer_returns
                ).reindex(dataframe.index)
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

        return dataframe

    # ------------------------------------------------------------------
    # populate_entry_trend — Fitness-based Entry Signals
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Generate entry signals based on ohio_active_mode + trend direction."""
        logger.info("ohio.populate_entry_trend | pair=%s", metadata["pair"])
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        # Entry condition: policy enabled
        enabled = dataframe["ohio_policy_enabled"] == True  # noqa: E712
        active = dataframe["ohio_active_mode"]
        trend = dataframe.get("ohio_stable_trend", 0.0)

        entry_mask = enabled.fillna(False)
        long_mask = entry_mask & (trend >= 0)
        short_mask = entry_mask & (trend < 0)

        dataframe.loc[long_mask, "enter_long"] = 1
        dataframe.loc[long_mask, "enter_tag"] = (
            "ohio_" + active[long_mask].astype(str)
        )
        dataframe.loc[short_mask, "enter_short"] = 1
        dataframe.loc[short_mask, "enter_tag"] = (
            "ohio_" + active[short_mask].astype(str) + "_short"
        )

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

        stake = compute_stake(
            policy=policy,
            fitness_score=fitness,
            base_stake=proposed_stake,
            min_stake=min_stake,
            max_stake=max_stake,
            dd_scale=dd_scale,
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
        """Mode-specific leverage via OHIO policy."""
        logger.info("ohio.leverage | pair=%s | entry_tag=%s", pair, entry_tag)

        mode = self._extract_mode(entry_tag)
        profile = self._get_profile_for_mode(mode) if mode else None
        if profile is None:
            return 1.0

        fitness = self._get_fitness_for_pair(pair)
        return compute_leverage(profile, fitness, max_leverage)

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

    # ------------------------------------------------------------------
    # custom_stoploss — Dynamic Stoploss
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
        """Mode-specific dynamic stoploss."""
        mode = get_trade_mode(trade)
        if mode is None:
            return None  # use default self.stoploss

        profile = self._get_profile_for_mode(mode)
        if profile is None:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        last = dataframe.iloc[-1]
        policy = self._build_policy_from_row(last)
        transition_risk = float(last.get("ohio_meta_transition_risk", 0.0))
        fitness = self._get_best_fitness(last)

        atr_ratio = float(last.get("ohio_feat_atr_ratio_14", 0.0))
        atr_baseline = float(last.get("ohio_feat_atr_baseline", 0.0))

        # Default to 1.0 if baseline unavailable (warmup/NaN)
        if (
            atr_baseline > 0
            and atr_ratio > 0
            and not math.isnan(atr_baseline)
            and not math.isnan(atr_ratio)
        ):
            atr_scale = atr_ratio / atr_baseline
        else:
            atr_scale = 1.0

        return compute_stoploss(
            profile, policy, current_profit, transition_risk, fitness,
            atr_scale=atr_scale,
            atr_ratio=atr_ratio,
            params=self._exit_params(),
        )

    # ------------------------------------------------------------------
    # custom_exit — Multi-condition Exit
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
        """Multi-condition exit engine."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        last = dataframe.iloc[-1]
        bars_since = self._bars_since_entry(trade, current_time)
        fitness = self._get_best_fitness(last)
        transition_risk = float(last.get("ohio_meta_transition_risk", 0.0))

        return compute_exit(
            bars_since_entry=bars_since,
            current_profit=current_profit,
            fitness_score=fitness,
            transition_risk=transition_risk,
            is_kill_switch=self._kill_switch.active,
            params=self._exit_params(),
        )

    # ==================================================================
    # Helper methods
    # ==================================================================

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
            return entry_tag[5:]
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
