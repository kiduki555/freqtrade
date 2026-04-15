"""Tests for EntryAdapter dispatcher."""
from __future__ import annotations

import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.entry_adapter import EntryAdapter

MODES = ["trend_following", "mean_reversion", "breakout", "defensive"]


@pytest.fixture()
def adapter() -> EntryAdapter:
    return EntryAdapter()


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "close": [100.0, 101.0, 99.0, 102.0, 98.0],
        "ohio_stable_trend": [0.5, 0.3, -0.2, 0.8, -0.6],
    })


class TestEntryAdapter:
    """EntryAdapter dispatcher tests."""

    def test_returns_dataframe(
        self, adapter: EntryAdapter, sample_df: pd.DataFrame,
    ) -> None:
        result = adapter.compute_entries(sample_df.copy(), "trend_following", {})
        assert isinstance(result, pd.DataFrame)

    def test_adds_entry_columns(
        self, adapter: EntryAdapter, sample_df: pd.DataFrame,
    ) -> None:
        result = adapter.compute_entries(sample_df.copy(), "trend_following", {})
        assert "enter_long" in result.columns
        assert "enter_short" in result.columns

    def test_unknown_mode_fallback(
        self, adapter: EntryAdapter, sample_df: pd.DataFrame,
    ) -> None:
        result = adapter.compute_entries(sample_df.copy(), "nonexistent_mode", {})
        assert "enter_long" in result.columns
        assert "enter_short" in result.columns
        # Fallback uses >= 0 for long, so trend=0.5 -> long=1, trend=-0.2 -> short=1
        assert result["enter_long"].iloc[0] == 1
        assert result["enter_short"].iloc[2] == 1

    @pytest.mark.parametrize("mode", MODES)
    def test_all_four_modes_supported(
        self, adapter: EntryAdapter, sample_df: pd.DataFrame, mode: str,
    ) -> None:
        result = adapter.compute_entries(sample_df.copy(), mode, {})
        assert "enter_long" in result.columns
        assert "enter_short" in result.columns

    @pytest.mark.parametrize("mode", MODES)
    def test_entries_are_binary(
        self, adapter: EntryAdapter, sample_df: pd.DataFrame, mode: str,
    ) -> None:
        result = adapter.compute_entries(sample_df.copy(), mode, {})
        assert set(result["enter_long"].unique()).issubset({0, 1})
        assert set(result["enter_short"].unique()).issubset({0, 1})

    @pytest.mark.parametrize("mode", MODES)
    def test_no_simultaneous_long_short(
        self, adapter: EntryAdapter, sample_df: pd.DataFrame, mode: str,
    ) -> None:
        result = adapter.compute_entries(sample_df.copy(), mode, {})
        both = (result["enter_long"] == 1) & (result["enter_short"] == 1)
        assert not both.any(), "No row should have both enter_long=1 and enter_short=1"
