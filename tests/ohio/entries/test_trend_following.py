"""Tests for TrendFollowingEntry strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.entries.trend_following import (
    TrendFollowingEntry,
    DEFAULT_PARAMS as TF_DEFAULTS,
)


@pytest.fixture
def strategy() -> TrendFollowingEntry:
    return TrendFollowingEntry()


def _make_trending_up(n: int = 50) -> pd.DataFrame:
    close = 100.0 + np.arange(n) * 0.5
    return pd.DataFrame({
        "close": close,
        "ohio_stable_trend": np.full(n, 0.3),
        "ohio_feat_adx_14": np.full(n, 35.0),
        "ohio_feat_hurst_168": np.full(n, 0.65),
        "ohio_feat_kama_slope": np.full(n, 0.003),
    })


def _make_choppy(n: int = 50) -> pd.DataFrame:
    close = 100.0 + np.sin(np.arange(n) * 0.5) * 2
    return pd.DataFrame({
        "close": close,
        "ohio_stable_trend": np.full(n, 0.05),
        "ohio_feat_adx_14": np.full(n, 12.0),
        "ohio_feat_hurst_168": np.full(n, 0.35),
        "ohio_feat_kama_slope": np.full(n, 0.0001),
    })


class TestTrendFollowingEntry:
    def test_strong_uptrend_enters_long(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert result["enter_long"].sum() > 0, "Should enter long in uptrend"
        assert result["enter_short"].sum() == 0, "Should not short in uptrend"

    def test_choppy_market_no_entry(self, strategy: TrendFollowingEntry) -> None:
        df = _make_choppy()
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert result["enter_long"].sum() == 0
        assert result["enter_short"].sum() == 0

    def test_strong_downtrend_enters_short(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        df["ohio_stable_trend"] = -0.3
        df["ohio_feat_kama_slope"] = -0.003
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert result["enter_short"].sum() > 0
        assert result["enter_long"].sum() == 0

    def test_custom_adx_threshold(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        df["ohio_feat_adx_14"] = 20.0  # below default 25
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert result["enter_long"].sum() == 0

        custom = {**TF_DEFAULTS, "tf_adx_threshold": 15.0}
        result2 = strategy.compute_entries(df, custom)
        assert result2["enter_long"].sum() > 0

    def test_hurst_below_threshold_blocks(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        df["ohio_feat_hurst_168"] = 0.40  # below 0.55
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_no_simultaneous_entries(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        result = strategy.compute_entries(df, TF_DEFAULTS)
        both = (result["enter_long"] == 1) & (result["enter_short"] == 1)
        assert not both.any()

    def test_returns_dataframe(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert isinstance(result, pd.DataFrame)
