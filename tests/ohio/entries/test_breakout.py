"""Tests for BreakoutEntry strategy — squeeze release + Donchian break."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.entries.breakout import (
    BreakoutEntry,
    DEFAULT_PARAMS as BO_DEFAULTS,
)


@pytest.fixture
def strategy() -> BreakoutEntry:
    return BreakoutEntry()


def _make_df(
    close: float | np.ndarray,
    squeeze: np.ndarray,
    don_upper: float = 103.0,
    don_lower: float = 97.0,
    length: int = 50,
) -> pd.DataFrame:
    """Build a minimal dataframe for breakout tests."""
    close_arr = np.full(length, close) if isinstance(close, (int, float)) else close
    return pd.DataFrame({
        "close": close_arr,
        "ohio_feat_squeeze_count": squeeze,
        "ohio_feat_donchian_upper_20": np.full(length, don_upper),
        "ohio_feat_donchian_lower_20": np.full(length, don_lower),
    })


class TestBreakoutEntry:
    def test_squeeze_release_long(self, strategy: BreakoutEntry) -> None:
        """Squeeze ends + close above Donchian upper -> enter long."""
        squeeze = np.zeros(50)
        squeeze[23] = 8.0   # squeeze active on bar 23
        squeeze[24] = 0.0   # squeeze released on bar 24
        df = _make_df(close=105.0, squeeze=squeeze)
        result = strategy.compute_entries(df, BO_DEFAULTS)
        # Bar 24: shift(1)=8 >= 6 AND current=0 < 6 -> released; close > upper
        assert result["enter_long"].iloc[24] == 1
        assert result["enter_short"].iloc[24] == 0

    def test_squeeze_release_short(self, strategy: BreakoutEntry) -> None:
        """Squeeze ends + close below Donchian lower -> enter short."""
        squeeze = np.zeros(50)
        squeeze[23] = 8.0
        squeeze[24] = 0.0
        df = _make_df(close=95.0, squeeze=squeeze)
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_short"].iloc[24] == 1
        assert result["enter_long"].iloc[24] == 0

    def test_during_squeeze_no_entry(self, strategy: BreakoutEntry) -> None:
        """While squeeze is still active, no entry even with Donchian break."""
        squeeze = np.full(50, 8.0)  # squeeze active every bar
        df = _make_df(close=105.0, squeeze=squeeze)
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_long"].sum() == 0
        assert result["enter_short"].sum() == 0

    def test_no_squeeze_history_no_entry(self, strategy: BreakoutEntry) -> None:
        """Without any prior squeeze, no entries even with Donchian break."""
        squeeze = np.zeros(50)
        df = _make_df(close=105.0, squeeze=squeeze)
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_release_but_no_donchian_break(self, strategy: BreakoutEntry) -> None:
        """Squeeze released but close is between channels -> no entry."""
        squeeze = np.zeros(50)
        squeeze[23] = 8.0
        squeeze[24] = 0.0
        df = _make_df(close=100.0, squeeze=squeeze)  # between 97 and 103
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_long"].iloc[24] == 0
        assert result["enter_short"].iloc[24] == 0

    def test_only_first_bar_after_release(self, strategy: BreakoutEntry) -> None:
        """Entry fires only on the release bar, not subsequent bars."""
        squeeze = np.zeros(50)
        squeeze[20:25] = 8.0   # squeeze bars 20-24
        squeeze[25:] = 0.0     # released at bar 25
        df = _make_df(close=105.0, squeeze=squeeze)
        result = strategy.compute_entries(df, BO_DEFAULTS)
        # Bar 25: shift(1)=8 >= 6 AND current=0 < 6 -> released
        assert result["enter_long"].iloc[25] == 1
        # Bar 26: shift(1)=0 < 6 -> NOT released
        assert result["enter_long"].iloc[26] == 0

    def test_custom_squeeze_min(self, strategy: BreakoutEntry) -> None:
        """Respects custom bo_squeeze_min_bars threshold."""
        squeeze = np.zeros(50)
        squeeze[23] = 4.0   # below default (6) but above custom (3)
        squeeze[24] = 0.0
        df = _make_df(close=105.0, squeeze=squeeze)

        # Default min=6: squeeze_count 4 < 6 -> no release
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_long"].iloc[24] == 0

        custom = {**BO_DEFAULTS, "bo_squeeze_min_bars": 3.0}
        result2 = strategy.compute_entries(df, custom)
        assert result2["enter_long"].iloc[24] == 1

    def test_no_simultaneous_entries(self, strategy: BreakoutEntry) -> None:
        """Cannot have both long and short on the same bar."""
        squeeze = np.zeros(50)
        squeeze[23] = 8.0
        squeeze[24] = 0.0
        df = _make_df(close=100.0, squeeze=squeeze)  # between channels
        result = strategy.compute_entries(df, BO_DEFAULTS)
        both = (result["enter_long"] == 1) & (result["enter_short"] == 1)
        assert not both.any()

    def test_missing_columns_use_defaults(self, strategy: BreakoutEntry) -> None:
        """Missing feature columns fall back to safe defaults (no crash)."""
        df = pd.DataFrame({
            "close": np.full(50, 105.0),
        })
        result = strategy.compute_entries(df, BO_DEFAULTS)
        # Should not raise; with default squeeze=0 everywhere, no entries
        assert result["enter_long"].sum() == 0
        assert result["enter_short"].sum() == 0
