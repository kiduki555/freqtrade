"""Tests for BreakoutEntry strategy."""
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


class TestBreakoutEntry:
    def test_squeeze_breakout_up(self, strategy: BreakoutEntry) -> None:
        """Squeeze + Donchian break up + volume → enter long."""
        df = pd.DataFrame({
            "close": np.full(50, 105.0),           # above donchian upper
            "volume": np.full(50, 2000.0),          # high volume
            "ohio_feat_squeeze_count": np.full(50, 8.0),  # squeeze active
            "ohio_feat_donchian_upper_20": np.full(50, 103.0),
            "ohio_feat_donchian_lower_20": np.full(50, 97.0),
            "ohio_feat_volume_sma_20": np.full(50, 1000.0),
        })
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_long"].sum() > 0
        assert result["enter_short"].sum() == 0

    def test_squeeze_breakout_down(self, strategy: BreakoutEntry) -> None:
        """Squeeze + Donchian break down + volume → enter short."""
        df = pd.DataFrame({
            "close": np.full(50, 95.0),             # below donchian lower
            "volume": np.full(50, 2000.0),
            "ohio_feat_squeeze_count": np.full(50, 8.0),
            "ohio_feat_donchian_upper_20": np.full(50, 103.0),
            "ohio_feat_donchian_lower_20": np.full(50, 97.0),
            "ohio_feat_volume_sma_20": np.full(50, 1000.0),
        })
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_short"].sum() > 0

    def test_no_squeeze_no_entry(self, strategy: BreakoutEntry) -> None:
        """Without squeeze, no entries even if price breaks Donchian."""
        df = pd.DataFrame({
            "close": np.full(50, 105.0),
            "volume": np.full(50, 2000.0),
            "ohio_feat_squeeze_count": np.zeros(50),  # no squeeze
            "ohio_feat_donchian_upper_20": np.full(50, 103.0),
            "ohio_feat_donchian_lower_20": np.full(50, 97.0),
            "ohio_feat_volume_sma_20": np.full(50, 1000.0),
        })
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_low_volume_blocks(self, strategy: BreakoutEntry) -> None:
        """Low volume blocks entry even with squeeze + break."""
        df = pd.DataFrame({
            "close": np.full(50, 105.0),
            "volume": np.full(50, 500.0),           # low volume
            "ohio_feat_squeeze_count": np.full(50, 8.0),
            "ohio_feat_donchian_upper_20": np.full(50, 103.0),
            "ohio_feat_donchian_lower_20": np.full(50, 97.0),
            "ohio_feat_volume_sma_20": np.full(50, 1000.0),  # 500 < 1.5 * 1000
        })
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_previous_bar_squeeze_counts(self, strategy: BreakoutEntry) -> None:
        """Squeeze on shift(1) should also trigger entry."""
        squeeze = np.zeros(50)
        squeeze[24] = 8.0  # squeeze on bar 24
        df = pd.DataFrame({
            "close": np.full(50, 105.0),
            "volume": np.full(50, 2000.0),
            "ohio_feat_squeeze_count": squeeze,
            "ohio_feat_donchian_upper_20": np.full(50, 103.0),
            "ohio_feat_donchian_lower_20": np.full(50, 97.0),
            "ohio_feat_volume_sma_20": np.full(50, 1000.0),
        })
        result = strategy.compute_entries(df, BO_DEFAULTS)
        # Bar 25 should see shift(1) = 8 >= 6, so enter_long
        assert result["enter_long"].iloc[25] == 1

    def test_custom_volume_mult(self, strategy: BreakoutEntry) -> None:
        df = pd.DataFrame({
            "close": np.full(50, 105.0),
            "volume": np.full(50, 1200.0),
            "ohio_feat_squeeze_count": np.full(50, 8.0),
            "ohio_feat_donchian_upper_20": np.full(50, 103.0),
            "ohio_feat_donchian_lower_20": np.full(50, 97.0),
            "ohio_feat_volume_sma_20": np.full(50, 1000.0),
        })
        # Default: 1200 > 1.5 * 1000 = 1500? No → blocked
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_long"].sum() == 0

        custom = {**BO_DEFAULTS, "bo_volume_mult": 1.1}
        result2 = strategy.compute_entries(df, custom)
        assert result2["enter_long"].sum() > 0

    def test_no_simultaneous_entries(self, strategy: BreakoutEntry) -> None:
        df = pd.DataFrame({
            "close": np.full(50, 100.0),  # exactly between channels
            "volume": np.full(50, 2000.0),
            "ohio_feat_squeeze_count": np.full(50, 8.0),
            "ohio_feat_donchian_upper_20": np.full(50, 103.0),
            "ohio_feat_donchian_lower_20": np.full(50, 97.0),
            "ohio_feat_volume_sma_20": np.full(50, 1000.0),
        })
        result = strategy.compute_entries(df, BO_DEFAULTS)
        both = (result["enter_long"] == 1) & (result["enter_short"] == 1)
        assert not both.any()
