"""BB Dip Buy entry — Signal 2 of independent entry system.

Buys when price dips below BB lower band with RSI confirmation.
Shorts when price spikes above BB upper band.
Source: CombinedBinHAndCluc pattern (proven on Freqtrade).
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "bb_rsi_oversold": 35.0,
    "bb_rsi_overbought": 65.0,
}


class BBDipEntry:
    """Enter on Bollinger Band dip with RSI and EMA(50) trend context."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        close = dataframe["close"]
        high = dataframe["high"]
        low = dataframe["low"]

        # BB bands
        bb_lower = dataframe.get(
            "ohio_feat_bb_lower_20",
            close.rolling(20).mean() - 2 * close.rolling(20).std(),
        )
        bb_upper = dataframe.get(
            "ohio_feat_bb_upper_20",
            close.rolling(20).mean() + 2 * close.rolling(20).std(),
        )

        # RSI (currently oversold)
        rsi = dataframe.get(
            "ohio_feat_rsi_14", pd.Series(50.0, index=dataframe.index),
        )

        # EMA(50) as trend context
        ema_50 = close.ewm(span=50, adjust=False).mean()

        rsi_threshold_low = params.get(
            "bb_rsi_oversold", DEFAULT_PARAMS["bb_rsi_oversold"],
        )
        rsi_threshold_high = params.get(
            "bb_rsi_overbought", DEFAULT_PARAMS["bb_rsi_overbought"],
        )

        # 4H trend filter: only buy dips in non-bearish higher TF
        sma_slope_4h = dataframe.get(
            "ohio_4h_sma_slope", pd.Series(0.0, index=dataframe.index),
        )
        not_bearish_4h = sma_slope_4h > -0.005

        # Volume confirmation
        volume = dataframe["volume"]
        vol_sma_20 = dataframe.get(
            "ohio_feat_volume_sma_20", volume.rolling(20).mean(),
        )
        vol_peak = volume > vol_sma_20

        # Long: BB dip + RSI oversold + below EMA50 + 4H not bearish + volume
        # (removed shadow requirement — too restrictive)
        long_mask = (
            (close < bb_lower)
            & (rsi < rsi_threshold_low)
            & (close < ema_50)
            & not_bearish_4h
            & vol_peak
        )

        # Short: BB top + RSI overbought + above EMA50 + volume
        short_mask = (
            (close > bb_upper)
            & (rsi > rsi_threshold_high)
            & (close > ema_50)
            & vol_peak
        )

        dataframe["enter_long"] = long_mask.astype(int)
        dataframe["enter_short"] = short_mask.astype(int)
        return dataframe
