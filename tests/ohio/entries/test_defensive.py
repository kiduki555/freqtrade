"""Tests for DefensiveEntry strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.entries.defensive import (
    DefensiveEntry,
    DEFAULT_PARAMS as DEF_DEFAULTS,
)


@pytest.fixture
def strategy() -> DefensiveEntry:
    return DefensiveEntry()


def _make_calm_market(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({
        "close": np.full(n, 100.0),
        "ohio_stable_trend": np.full(n, 0.05),
        "ohio_feat_adx_14": np.full(n, 15.0),
    })


def _make_volatile_market(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({
        "close": np.full(n, 100.0),
        "ohio_stable_trend": np.full(n, 0.3),
        "ohio_feat_adx_14": np.full(n, 40.0),
    })


class TestDefensiveEntry:
    def test_calm_market_enters_long(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_long"].sum() > 0

    def test_volatile_market_blocked(self, strategy: DefensiveEntry) -> None:
        df = _make_volatile_market()
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_negative_trend_enters_short(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        df["ohio_stable_trend"] = -0.05
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_short"].sum() > 0
        assert result["enter_long"].sum() == 0

    def test_flat_trend_blocked(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        df["ohio_stable_trend"] = 0.001  # below def_min_trend_abs=0.03
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_custom_adx_threshold(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        df["ohio_feat_adx_14"] = 25.0
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_long"].sum() == 0  # default max=20

        custom = {**DEF_DEFAULTS, "def_adx_max": 30.0}
        result2 = strategy.compute_entries(df, custom)
        assert result2["enter_long"].sum() > 0

    def test_custom_min_trend(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        df["ohio_stable_trend"] = 0.02  # below default 0.03
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_long"].sum() == 0

        custom = {**DEF_DEFAULTS, "def_min_trend_abs": 0.01}
        result2 = strategy.compute_entries(df, custom)
        assert result2["enter_long"].sum() > 0

    def test_no_simultaneous_entries(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        both = (result["enter_long"] == 1) & (result["enter_short"] == 1)
        assert not both.any()

    def test_returns_dataframe(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert isinstance(result, pd.DataFrame)
