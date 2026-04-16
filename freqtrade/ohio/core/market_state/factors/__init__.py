"""Factor Calculator — 7-axis StateVector computation.

Reads ohio_norm_* columns from a DataFrame and produces ohio_factor_* columns.
Individual factor modules are pure functions; this module orchestrates them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from freqtrade.ohio.core.domain.models import StateVector
from freqtrade.ohio.core.market_state.factors.breadth import compute_breadth
from freqtrade.ohio.core.market_state.factors.correlation import compute_correlation
from freqtrade.ohio.core.market_state.factors.downside import compute_downside
from freqtrade.ohio.core.market_state.factors.liquidity import compute_liquidity
from freqtrade.ohio.core.market_state.factors.relative_strength import compute_relative_strength
from freqtrade.ohio.core.market_state.factors.trend import compute_trend
from freqtrade.ohio.core.market_state.factors.volatility import compute_volatility

__all__ = ["FactorCalculator"]

_TREND_WEIGHTS = {
    "ohio_norm_log_return_24": 0.30,
    "ohio_norm_ma_slope_20": 0.22,
    "ohio_norm_adx_14": 0.17,
    "ohio_norm_efficiency_ratio_24": 0.16,
    "ohio_norm_hurst_168": 0.15,
}
_VOL_WEIGHTS = {
    "ohio_norm_realized_vol_24": 0.50,
    "ohio_norm_atr_ratio_14": 0.30,
    "ohio_norm_parkinson_vol_24": 0.20,
}
_DOWN_WEIGHTS = {
    "ohio_norm_sortino_downside_24": 0.25,
    "ohio_norm_max_drawdown_24": 0.25,
    "ohio_norm_cvar_24": 0.20,
    "ohio_norm_lower_shadow_ratio_24": 0.15,
    "ohio_norm_negative_return_ratio_24": 0.15,
}


def _get_col(df: pd.DataFrame, col: str, default: float = 0.5) -> pd.Series:
    """Return column or a constant series at the default value."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


class FactorCalculator:
    """Compute 7-axis state vector from normalized features."""

    def compute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add ohio_factor_* columns to DataFrame.

        Reads ohio_norm_* columns, computes 7 factors per row using
        vectorized arithmetic, and writes factor columns in place.
        Returns the modified DataFrame (same object).
        """
        df = dataframe

        # --- trend_persistence [-1, 1] — vectorized with numpy tanh ---
        raw_trend = sum(
            w * _get_col(df, col) for col, w in _TREND_WEIGHTS.items()
        )

        # Vol-adjusted trend: Sharpe-like signal preserving direction
        log_ret = _get_col(df, "ohio_norm_log_return_24")
        real_vol = _get_col(df, "ohio_norm_realized_vol_24")
        safe_vol = np.maximum(real_vol, 0.15)
        centered_ret = log_ret - 0.5
        vol_adj = centered_ret / safe_vol
        vol_adj_norm = np.clip(vol_adj + 0.5, 0.0, 1.0)

        # Blend: 75% original + 25% vol-adjusted (conservative blend)
        blended = 0.75 * raw_trend + 0.25 * vol_adj_norm
        df["ohio_factor_trend"] = np.tanh(2.0 * (blended - 0.5))

        # --- volatility_level [0, 1] ---
        raw_vol = sum(w * _get_col(df, col) for col, w in _VOL_WEIGHTS.items())
        df["ohio_factor_volatility"] = np.clip(raw_vol, 0.0, 1.0)

        # --- downside_pressure [0, 1] ---
        raw_down = sum(w * _get_col(df, col) for col, w in _DOWN_WEIGHTS.items())
        df["ohio_factor_downside"] = np.clip(raw_down, 0.0, 1.0)

        # --- liquidity_stress [0, 1] ---
        spread = _get_col(df, "ohio_norm_bid_ask_approx")
        vol_inv = 1.0 - _get_col(df, "ohio_norm_volume_ratio_24")
        obv_inv = 1.0 - _get_col(df, "ohio_norm_obv_slope_24")
        df["ohio_factor_liquidity"] = np.clip(
            0.40 * spread + 0.35 * vol_inv + 0.25 * obv_inv, 0.0, 1.0
        )

        # --- relative_strength [0, 1] — vectorized with NaN → 0.5 default ---
        rs_raw = _get_col(df, "ohio_norm_rs_raw", default=np.nan)
        df["ohio_factor_relative_strength"] = np.clip(
            rs_raw.where(rs_raw.notna(), 0.5), 0.0, 1.0
        )

        # --- correlation_stress [0, 1] ---
        corr = _get_col(df, "ohio_norm_correlation_stress", default=np.nan)
        df["ohio_factor_correlation"] = np.clip(
            corr.where(corr.notna(), 0.5), 0.0, 1.0
        )

        # --- breadth_dispersion [0, 1] ---
        breadth = _get_col(df, "ohio_norm_breadth_dispersion", default=np.nan)
        df["ohio_factor_breadth"] = np.clip(
            breadth.where(breadth.notna(), 0.5), 0.0, 1.0
        )

        return df

    def compute_vector(self, row: dict[str, float]) -> StateVector:
        """Compute StateVector from a single row's normalized features."""
        return StateVector(
            trend_persistence=compute_trend(row),
            volatility_level=compute_volatility(row),
            downside_pressure=compute_downside(row),
            liquidity_stress=compute_liquidity(row),
            relative_strength=compute_relative_strength(row),
            correlation_stress=compute_correlation(row),
            breadth_dispersion=compute_breadth(row),
        )
