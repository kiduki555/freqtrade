"""Normalizer: transform ohio_feat_* columns to [0, 1] via percentile rank.

Pipeline position: FeatureBuilder → **Normalizer** → FactorCalculator

Primary method: rolling percentile rank over a 180-day window (4320 x 1h bars).
Fallback for early bars: z-score rescaled to [0, 1] using a 48-bar warm-up window.

Usage::

    from freqtrade.ohio.core.market_state.normalizer import Normalizer

    normalizer = Normalizer()
    dataframe = normalizer.normalize(dataframe)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Percentile-rank warm-up window (minimum bars for z-score fallback)
_FALLBACK_WINDOW: int = 48


class Normalizer:
    """Percentile-rank normalization of raw features to [0, 1].

    Primary: rolling percentile rank over ``window`` bars (default 4320 = 180 days x 24h).
    Fallback: z-score → clip(0.5 + z/6, 0, 1) for bars before ``window`` is saturated.

    Args:
        window: Rolling window size in bars. Default 4320 matches OhioConfig.
        method: Normalization method. Only ``"percentile_rank"`` is supported.
    """

    def __init__(self, window: int = 4320, method: str = "percentile_rank") -> None:
        self.window = window
        self.method = method

    def normalize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize all non-placeholder ``ohio_feat_*`` columns to ``ohio_norm_*``.

        For each ``ohio_feat_*`` column that is not entirely NaN:

        1. Apply ``Series.rolling(window).rank(pct=True)`` (vectorized percentile rank).
        2. Fill remaining NaN bars (warm-up region) with z-score fallback using a
           48-bar minimum window, clipped to [0, 1].

        Args:
            dataframe: DataFrame that already contains ``ohio_feat_*`` columns.

        Returns:
            The same DataFrame with ``ohio_norm_*`` columns appended.
        """
        feat_cols = [c for c in dataframe.columns if c.startswith("ohio_feat_")]
        for col in feat_cols:
            series = dataframe[col]
            # Skip placeholder columns (entirely NaN)
            if series.isna().all():
                continue
            norm_col = col.replace("ohio_feat_", "ohio_norm_", 1)
            dataframe[norm_col] = self._normalize_series(series)
        return dataframe

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_series(self, series: pd.Series) -> pd.Series:
        """Return a [0, 1]-normalized copy of *series*."""
        pct_rank = series.rolling(window=self.window, min_periods=1).rank(pct=True)

        # Identify bars where the window had only 1 unique non-NaN value:
        # rank(pct=True) returns NaN only when *all* values in the window are NaN.
        # That means pct_rank is fully populated for any bar with ≥1 non-NaN value.
        # However, for very sparse early windows the percentile rank may still be
        # well-defined (single value → rank = 1.0 not 0.5), so we apply the
        # z-score fallback only where pct_rank itself is NaN (all-NaN window).
        needs_fallback = pct_rank.isna()

        if needs_fallback.any():
            fallback = self._zscore_fallback(series)
            pct_rank = pct_rank.where(~needs_fallback, other=fallback)

        return pct_rank.clip(0.0, 1.0)

    @staticmethod
    def _zscore_fallback(series: pd.Series) -> pd.Series:
        """Rescale via rolling z-score: clip(0.5 + z/6, 0, 1).

        Maps +-3 std-devs to [0, 1].  Uses a 48-bar minimum rolling window.
        """
        roll = series.rolling(window=_FALLBACK_WINDOW, min_periods=2)
        mean = roll.mean()
        std = roll.std(ddof=1)

        # Avoid division by zero for constant windows
        safe_std = std.replace(0.0, np.nan)
        z = (series - mean) / safe_std

        # Constant window → z is NaN → fall back to 0.5 (neutral rank)
        z = z.fillna(0.0)
        return (0.5 + z / 6.0).clip(0.0, 1.0)
