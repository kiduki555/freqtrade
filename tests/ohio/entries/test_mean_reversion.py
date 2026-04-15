"""Tests for MeanReversionEntry strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.entries.mean_reversion import (
    MeanReversionEntry,
    DEFAULT_PARAMS as MR_DEFAULTS,
)


@pytest.fixture
def strategy() -> MeanReversionEntry:
    return MeanReversionEntry()


def _make_oversold(n: int = 50) -> pd.DataFrame:
    close = np.full(n, 95.0)
    return pd.DataFrame({
        "close": close,
        "ohio_feat_zscore_20": np.full(n, -2.5),
        "ohio_feat_rsi_14": np.full(n, 25.0),
        "ohio_feat_hurst_168": np.full(n, 0.35),
        "ohio_feat_bb_lower_20": np.full(n, 96.0),
        "ohio_feat_bb_upper_20": np.full(n, 104.0),
    })


def _make_overbought(n: int = 50) -> pd.DataFrame:
    close = np.full(n, 105.0)
    return pd.DataFrame({
        "close": close,
        "ohio_feat_zscore_20": np.full(n, 2.5),
        "ohio_feat_rsi_14": np.full(n, 75.0),
        "ohio_feat_hurst_168": np.full(n, 0.35),
        "ohio_feat_bb_upper_20": np.full(n, 104.0),
        "ohio_feat_bb_lower_20": np.full(n, 96.0),
    })


def _make_trending(n: int = 50) -> pd.DataFrame:
    close = 100.0 + np.arange(n) * 0.5
    return pd.DataFrame({
        "close": close,
        "ohio_feat_zscore_20": np.full(n, -2.5),
        "ohio_feat_rsi_14": np.full(n, 25.0),
        "ohio_feat_hurst_168": np.full(n, 0.65),  # trending → blocks MR
        "ohio_feat_bb_lower_20": close * 0.98,
        "ohio_feat_bb_upper_20": close * 1.02,
    })


class TestMeanReversionEntry:
    def test_oversold_enters_long(self, strategy: MeanReversionEntry) -> None:
        df = _make_oversold()
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_long"].sum() > 0

    def test_overbought_enters_short(self, strategy: MeanReversionEntry) -> None:
        df = _make_overbought()
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_short"].sum() > 0

    def test_trending_blocks_entry(self, strategy: MeanReversionEntry) -> None:
        df = _make_trending()
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_long"].sum() == 0
        assert result["enter_short"].sum() == 0

    def test_hurst_gate_required(self, strategy: MeanReversionEntry) -> None:
        df = _make_oversold()
        df["ohio_feat_hurst_168"] = 0.60
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_custom_zscore_threshold(self, strategy: MeanReversionEntry) -> None:
        df = _make_oversold()
        df["ohio_feat_zscore_20"] = -1.8  # below default 2.0
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_long"].sum() == 0

        custom = {**MR_DEFAULTS, "mr_zscore_entry": 1.5}
        result2 = strategy.compute_entries(df, custom)
        assert result2["enter_long"].sum() > 0

    def test_rsi_gate(self, strategy: MeanReversionEntry) -> None:
        df = _make_oversold()
        df["ohio_feat_rsi_14"] = 50.0  # not oversold
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_no_simultaneous_entries(self, strategy: MeanReversionEntry) -> None:
        df = _make_oversold()
        result = strategy.compute_entries(df, MR_DEFAULTS)
        both = (result["enter_long"] == 1) & (result["enter_short"] == 1)
        assert not both.any()
