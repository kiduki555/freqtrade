"""OhioMomentum4H — Pure momentum strategy on 4H timeframe.

Independent alpha source: Donchian channel breakouts with ATR-based exits.
Complements OhioThinStrategy (1H regime-based) by capturing larger trend moves.

Design principles:
- 4H timeframe: less noise, captures multi-day trends
- Single signal: Donchian(20) breakout + 1D trend filter
- ATR-based stops and targets
- No regime classification (simplicity)

Expected to excel in:
- Strong directional moves (bull runs, crashes)
- Multi-day trending periods

Expected to underperform in:
- Choppy markets (4H whipsaw)
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.parameters import DecimalParameter, IntParameter


class OhioMomentum4H(IStrategy):
    """Pure momentum on 4H via Donchian breakout with 1D trend filter."""

    INTERFACE_VERSION = 3

    timeframe = "4h"
    startup_candle_count = 300  # ~50 days @ 4H for 1D SMA50

    stoploss = -0.10
    minimal_roi = {"0": 0.15, "48": 0.10, "168": 0.05, "336": 0.02, "720": 0}

    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    process_only_new_candles = True
    can_short = True
    position_adjustment_enable = False

    # --- Hyperopt-optimized parameters (training 2024-08 to 2025-11, +26.31%) ---
    donchian_window = IntParameter(20, 60, default=34, space="buy", optimize=True)
    adx_threshold = IntParameter(15, 35, default=34, space="buy", optimize=True)
    volume_mult = DecimalParameter(1.0, 2.5, default=1.5, decimals=1, space="buy", optimize=True)

    # Stop multipliers (hyperopt-optimized)
    atr_initial_mult = DecimalParameter(1.5, 4.0, default=3.1, decimals=1, space="sell", optimize=True)
    atr_trail_mult = DecimalParameter(1.0, 2.5, default=1.5, decimals=1, space="sell", optimize=True)
    stop_floor = DecimalParameter(-0.12, -0.04, default=-0.08, decimals=2, space="sell", optimize=True)

    def informative_pairs(self) -> list[tuple[str, str]]:
        whitelist = self.dp.current_whitelist() if self.dp else []
        return [(pair, "1d") for pair in whitelist]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Compute Donchian channels, ATR, ADX, and 1D trend."""
        dc_win = self.donchian_window.value
        dataframe["donchian_upper_40"] = dataframe["high"].rolling(dc_win).max().shift(1)
        dataframe["donchian_lower_40"] = dataframe["low"].rolling(dc_win).min().shift(1)

        # ATR for stops — Wilder's
        period = 14
        alpha = 1.0 / period
        high = dataframe["high"]
        low = dataframe["low"]
        close = dataframe["close"]
        hl = high - low
        hpc = (high - close.shift(1)).abs()
        lpc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        dataframe["atr_14"] = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        dataframe["atr_ratio"] = dataframe["atr_14"] / dataframe["close"]

        # ADX for trend strength filter
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm_s = pd.Series(plus_dm, index=dataframe.index, dtype=float)
        minus_dm_s = pd.Series(minus_dm, index=dataframe.index, dtype=float)
        tr_smooth = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        pdm_smooth = plus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        mdm_smooth = minus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        plus_di = 100 * pdm_smooth / tr_smooth.replace(0, np.nan)
        minus_di = 100 * mdm_smooth / tr_smooth.replace(0, np.nan)
        di_sum = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
        dataframe["adx_14"] = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

        # Volume filter
        dataframe["vol_sma_20"] = dataframe["volume"].rolling(20).mean()

        # 1D trend filter
        try:
            df_1d = self.dp.get_pair_dataframe(metadata["pair"], "1d")
            if df_1d is not None and len(df_1d) > 50:
                ema_10_1d = df_1d["close"].ewm(span=10, adjust=False).mean()
                ema_20_1d = df_1d["close"].ewm(span=20, adjust=False).mean()

                trend_1d = pd.Series("choppy", index=df_1d.index, dtype=object)
                trend_1d.loc[ema_10_1d > ema_20_1d] = "bull"
                trend_1d.loc[ema_10_1d < ema_20_1d] = "bear"

                df_1d_feat = df_1d[["date"]].copy()
                df_1d_feat["trend_1d"] = trend_1d.values

                if "date" in dataframe.columns:
                    dataframe = pd.merge_asof(
                        dataframe.sort_values("date"),
                        df_1d_feat.sort_values("date"),
                        on="date",
                        direction="backward",
                    )
        except Exception:
            dataframe["trend_1d"] = "choppy"

        if "trend_1d" not in dataframe.columns:
            dataframe["trend_1d"] = "choppy"

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enter on Donchian(40) breakout with ADX>25 and 1D trend filter."""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""

        close = dataframe["close"]
        volume_ok = dataframe["volume"] > (self.volume_mult.value * dataframe["vol_sma_20"])
        trend_strong = dataframe["adx_14"] > self.adx_threshold.value

        # Fresh breakout — crosses level this bar
        break_up = (close > dataframe["donchian_upper_40"]) & (
            close.shift(1) <= dataframe["donchian_upper_40"]
        )
        break_down = (close < dataframe["donchian_lower_40"]) & (
            close.shift(1) >= dataframe["donchian_lower_40"]
        )

        bull_1d = dataframe["trend_1d"] == "bull"
        bear_1d = dataframe["trend_1d"] == "bear"

        long_mask = break_up & bull_1d & volume_ok & trend_strong
        short_mask = break_down & bear_1d & volume_ok & trend_strong

        dataframe.loc[long_mask, "enter_long"] = 1
        dataframe.loc[long_mask, "enter_tag"] = "momentum_breakout_long"
        dataframe.loc[short_mask, "enter_short"] = 1
        dataframe.loc[short_mask, "enter_tag"] = "momentum_breakout_short"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit via custom_stoploss and minimal_roi."""
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

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
        """ATR-based stoploss: 2.5x ATR initial, tighten after profit."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return -0.05

        last = dataframe.iloc[-1]
        atr_ratio = float(last.get("atr_ratio", 0.03))

        one_r = 2.0 * atr_ratio

        # Profit > 2R: trail at atr_trail_mult
        if current_profit >= 2.0 * one_r:
            return -(self.atr_trail_mult.value * atr_ratio)
        # Profit > 1R: move to breakeven
        if current_profit >= one_r:
            return -(0.5 * atr_ratio)

        # Initial: atr_initial_mult, floor stop_floor
        return max(-(self.atr_initial_mult.value * atr_ratio), self.stop_floor.value)

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
        """1x leverage — momentum strategy doesn't use leverage."""
        return 1.0
