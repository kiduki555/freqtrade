"""MTF Merger — merges 4h timeframe features into the primary 1h DataFrame.

FT-006: MTF Merger for the OHIO Market State Engine.

Pipeline position:
    OHLCV → FeatureBuilder → Normalizer → MTFMerger → FactorCalculator → Stabilizer → Meta

No Freqtrade imports — pure ohio/core module.
"""
from __future__ import annotations

import pandas as pd


_OHLCV_COLS = ("open", "high", "low", "close", "volume")
_FEAT_PREFIX = "ohio_feat_"


class MTFMerger:
    """Merges multi-timeframe features into the primary (1h) DataFrame.

    The 4h DataFrame has features computed on 4h bars.
    These need to be forward-filled onto 1h bars so each 1h bar
    sees the latest COMPLETED 4h bar's features (no lookahead).
    """

    def merge(
        self,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame | None,
        suffix: str = "_4h",
    ) -> pd.DataFrame:
        """Merge 4h features into 1h DataFrame.

        Args:
            df_1h: Primary 1h OHLCV + features. Must have 'date' column.
            df_4h: 4h OHLCV + features. Must have 'date' column.
                   Pass None or an empty DataFrame to get NaN columns.
            suffix: Suffix appended to all 4h columns (default ``"_4h"``).

        Returns:
            df_1h with 4h feature columns added (forward-filled, no lookahead).
        """
        feature_cols = self._feature_cols(df_4h)
        ohlcv_cols = [c for c in _OHLCV_COLS if df_4h is not None and c in df_4h.columns]
        cols_to_merge = feature_cols + ohlcv_cols

        if df_4h is None or df_4h.empty or not cols_to_merge:
            return self._add_nan_columns(df_1h, feature_cols, ohlcv_cols, suffix)

        df_4h_slim = df_4h[["date", *cols_to_merge]].copy()
        rename_map = {c: f"{c}{suffix}" for c in cols_to_merge}
        df_4h_slim = df_4h_slim.rename(columns=rename_map)

        df_1h_sorted = df_1h.sort_values("date").reset_index(drop=True)
        df_4h_sorted = df_4h_slim.sort_values("date").reset_index(drop=True)

        merged = pd.merge_asof(
            df_1h_sorted,
            df_4h_sorted,
            on="date",
            direction="backward",
        )

        # Restore original row order.
        merged = merged.set_index(df_1h_sorted.index)
        return merged.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_cols(df_4h: pd.DataFrame | None) -> list[str]:
        if df_4h is None:
            return []
        # Read from schema even when there are no rows — allows NaN column
        # generation for callers that pass an empty-but-typed DataFrame.
        return [c for c in df_4h.columns if c.startswith(_FEAT_PREFIX)]

    @staticmethod
    def _add_nan_columns(
        df_1h: pd.DataFrame,
        feature_cols: list[str],
        ohlcv_cols: list[str],
        suffix: str,
    ) -> pd.DataFrame:
        result = df_1h.copy()
        for col in feature_cols + ohlcv_cols:
            result[f"{col}{suffix}"] = float("nan")
        return result
