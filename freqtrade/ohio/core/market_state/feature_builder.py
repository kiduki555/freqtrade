"""Feature builder: extract 18 primitive per-symbol features from OHLCV 1h candles.

These raw features feed into 7 factor calculators (trend, volatility, downside,
liquidity, relative_strength, correlation, breadth).  All computations are
vectorized via pandas/numpy — no Python loops over rows.

All output columns are prefixed with ``ohio_feat_`` for namespace isolation.

Usage (inside populate_indicators)::

    from freqtrade.ohio.core.market_state.feature_builder import FeatureBuilder

    builder = FeatureBuilder()
    dataframe = builder.compute(dataframe)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Expected output columns (used for validation and documentation)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    # Trend (5)
    "ohio_feat_log_return_24",
    "ohio_feat_log_return_72",
    "ohio_feat_ma_slope_20",
    "ohio_feat_adx_14",
    "ohio_feat_efficiency_ratio_24",
    # Volatility (3)
    "ohio_feat_realized_vol_24",
    "ohio_feat_atr_ratio_14",
    "ohio_feat_parkinson_vol_24",
    # Downside (5)
    "ohio_feat_sortino_downside_24",
    "ohio_feat_max_drawdown_24",
    "ohio_feat_cvar_24",
    "ohio_feat_lower_shadow_ratio_24",
    "ohio_feat_negative_return_ratio_24",
    # Liquidity (3)
    "ohio_feat_bid_ask_approx",
    "ohio_feat_volume_ratio_24",
    "ohio_feat_obv_slope_24",
    # Placeholders (2)
    "ohio_feat_rs_raw",
    "ohio_feat_cross_asset_placeholder",
]


class FeatureBuilder:
    """Extract primitive per-symbol features from an OHLCV DataFrame.

    Designed to be called from ``populate_indicators()``.  The DataFrame
    may contain warmup bars — features for those bars will be NaN (not
    filled), which is intentional; the downstream Normalizer handles NaN.

    Example::

        builder = FeatureBuilder()
        dataframe = builder.compute(dataframe)
    """

    # Wilder smoothing alpha for ADX(14): 1 / period
    _ADX_PERIOD: int = 14
    _ADX_ALPHA: float = 1.0 / 14

    def compute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Append 18 ``ohio_feat_*`` columns to *dataframe* and return it.

        Args:
            dataframe: Standard Freqtrade OHLCV DataFrame with columns
                       ``open``, ``high``, ``low``, ``close``, ``volume``.

        Returns:
            The same DataFrame with new feature columns appended in-place.
        """
        self._compute_trend(dataframe)
        self._compute_volatility(dataframe)
        self._compute_downside(dataframe)
        self._compute_liquidity(dataframe)
        self._compute_placeholders(dataframe)
        return dataframe

    # ------------------------------------------------------------------
    # Trend features
    # ------------------------------------------------------------------

    def _compute_trend(self, df: pd.DataFrame) -> None:
        close = df["close"]

        # Log returns over 24 and 72 bars
        df["ohio_feat_log_return_24"] = np.log(close / close.shift(24))
        df["ohio_feat_log_return_72"] = np.log(close / close.shift(72))

        # MA slope: (SMA20 - SMA20.shift(5)) / (5 * close)
        sma20 = close.rolling(window=20).mean()
        df["ohio_feat_ma_slope_20"] = (sma20 - sma20.shift(5)) / (5 * close)

        # ADX(14) using Wilder smoothing
        df["ohio_feat_adx_14"] = self._compute_adx(df)

        # Efficiency ratio over 24 bars:
        # abs(close - close.shift(24)) / sum(abs(close.diff()), 24)
        net_change = (close - close.shift(24)).abs()
        path_length = close.diff().abs().rolling(window=24).sum()
        df["ohio_feat_efficiency_ratio_24"] = net_change / path_length.replace(0, np.nan)

    def _compute_adx(self, df: pd.DataFrame) -> pd.Series:
        """Compute Wilder's ADX(14) fully vectorized.

        Uses Wilder's smoothing (EMA with alpha=1/14) applied to True Range,
        +DM, and -DM, then derives +DI and -DI, DX, and finally ADX.

        Returns:
            Series aligned with *df* index containing ADX values (NaN in
            warmup region).
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        period = self._ADX_PERIOD
        alpha = self._ADX_ALPHA

        # True Range components
        hl = high - low
        hpc = (high - close.shift(1)).abs()
        lpc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm_s = pd.Series(plus_dm, index=df.index, dtype=float)
        minus_dm_s = pd.Series(minus_dm, index=df.index, dtype=float)

        # Wilder smoothing: seed with SMA of first `period` bars, then EWM
        # pandas ewm with adjust=False and alpha=alpha replicates Wilder
        tr_smooth = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        pdm_smooth = plus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        mdm_smooth = minus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

        # +DI and -DI (in percent)
        plus_di = 100 * pdm_smooth / tr_smooth.replace(0, np.nan)
        minus_di = 100 * mdm_smooth / tr_smooth.replace(0, np.nan)

        # DX
        di_sum = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)

        # ADX = Wilder smoothed DX
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
        return adx

    # ------------------------------------------------------------------
    # Volatility features
    # ------------------------------------------------------------------

    def _compute_volatility(self, df: pd.DataFrame) -> None:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Realized vol annualized (8760 hours/year)
        pct_ret = close.pct_change()
        df["ohio_feat_realized_vol_24"] = pct_ret.rolling(window=24).std() * math.sqrt(8760)

        # ATR(14) / close
        hl = high - low
        hpc = (high - close.shift(1)).abs()
        lpc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        atr14 = tr.rolling(window=14).mean()
        df["ohio_feat_atr_ratio_14"] = atr14 / close.replace(0, np.nan)

        # Parkinson vol: sqrt(rolling_mean(ln(high/low)^2 / (4*ln2), 24))
        ln2 = math.log(2)
        log_hl_sq = np.log(high / low.replace(0, np.nan)) ** 2
        df["ohio_feat_parkinson_vol_24"] = np.sqrt(
            log_hl_sq.rolling(window=24).mean() / (4 * ln2)
        )

    # ------------------------------------------------------------------
    # Downside features
    # ------------------------------------------------------------------

    def _compute_downside(self, df: pd.DataFrame) -> None:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        ret = close.pct_change()

        # Sortino downside deviation (only negative returns)
        neg_ret_sq = pd.Series(np.where(ret < 0, ret**2, 0.0), index=ret.index)
        df["ohio_feat_sortino_downside_24"] = np.sqrt(
            neg_ret_sq.rolling(window=24).mean()
        )

        # Max drawdown over rolling 24-bar window
        # For each bar t, compute (close[t] - max(close[t-23..t])) / max(...)
        roll_max = close.rolling(window=24).max()
        df["ohio_feat_max_drawdown_24"] = (close - roll_max) / roll_max.replace(0, np.nan)

        # CVaR at 5%: rolling mean of returns below 5th percentile
        def _cvar_5(window: pd.Series) -> float:
            q5 = window.quantile(0.05)
            tail = window[window <= q5]
            return tail.mean() if len(tail) > 0 else float("nan")

        df["ohio_feat_cvar_24"] = ret.rolling(window=24).apply(_cvar_5, raw=False)

        # Lower shadow ratio: (min(open,close) - low) / (high - low + eps)
        body_bottom = pd.concat([open_, close], axis=1).min(axis=1)
        lower_shadow = (body_bottom - low) / (high - low + 1e-10)
        df["ohio_feat_lower_shadow_ratio_24"] = lower_shadow.rolling(window=24).mean()

        # Negative return ratio: fraction of bars with negative return
        is_neg = (ret < 0).astype(float)
        df["ohio_feat_negative_return_ratio_24"] = is_neg.rolling(window=24).mean()

    # ------------------------------------------------------------------
    # Liquidity features
    # ------------------------------------------------------------------

    def _compute_liquidity(self, df: pd.DataFrame) -> None:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Bid-ask spread proxy (instantaneous): 2*(H-L)/(H+L)
        mid = high + low
        df["ohio_feat_bid_ask_approx"] = 2 * (high - low) / mid.replace(0, np.nan)

        # Volume ratio: volume / rolling(24).mean()
        vol_mean = volume.rolling(window=24).mean()
        df["ohio_feat_volume_ratio_24"] = volume / vol_mean.replace(0, np.nan)

        # OBV slope over 24 bars
        # OBV = cumsum(volume * sign(close.diff()))
        sign_ret = np.sign(close.diff()).fillna(0)
        obv = (volume * sign_ret).cumsum()
        # Slope = (OBV - OBV.shift(24)) / 24, normalised by mean volume
        vol_mean_safe = vol_mean.replace(0, np.nan)
        obv_change = (obv - obv.shift(24)) / 24
        df["ohio_feat_obv_slope_24"] = obv_change / vol_mean_safe

    # ------------------------------------------------------------------
    # Placeholder features (filled by cross-asset provider in M4/FT-019)
    # ------------------------------------------------------------------

    def _compute_placeholders(self, df: pd.DataFrame) -> None:
        df["ohio_feat_rs_raw"] = np.nan
        df["ohio_feat_cross_asset_placeholder"] = np.nan
