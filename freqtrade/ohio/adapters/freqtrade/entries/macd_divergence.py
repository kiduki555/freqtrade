"""MACD Divergence entry — Signal 3 of independent entry system.

Enters on MACD histogram divergence from price, confirmed by volume.
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "macd_vol_mult": 1.2,
}


class MACDDivergenceEntry:
    """Enter on MACD histogram divergence or MACD crossover with volume confirmation."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        close = dataframe["close"]
        volume = dataframe.get("volume", pd.Series(0.0, index=dataframe.index))

        # MACD computation
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        # Volume filter
        vol_sma = volume.rolling(20).mean()
        vol_mult = params.get("macd_vol_mult", DEFAULT_PARAMS["macd_vol_mult"])
        vol_ok = volume > (vol_mult * vol_sma)

        # Bullish divergence: price making lower lows, MACD histogram making higher lows
        price_lower = close < close.shift(1)
        hist_higher = histogram > histogram.shift(1)
        hist_negative = histogram < 0  # histogram below zero (oversold zone)
        bullish_div = price_lower & hist_higher & hist_negative & vol_ok

        # Bearish divergence: price making higher highs, MACD histogram making lower highs
        price_higher = close > close.shift(1)
        hist_lower = histogram < histogram.shift(1)
        hist_positive = histogram > 0  # histogram above zero (overbought zone)
        bearish_div = price_higher & hist_lower & hist_positive & vol_ok

        # Also allow simple MACD crossover as secondary signal
        macd_cross_up = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        macd_cross_down = (macd < signal) & (macd.shift(1) >= signal.shift(1))

        dataframe["enter_long"] = (bullish_div | macd_cross_up).astype(int)
        dataframe["enter_short"] = (bearish_div | macd_cross_down).astype(int)
        return dataframe
