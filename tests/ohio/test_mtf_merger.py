"""Tests for MTFMerger (FT-006).

Covers: basic merge, suffix, forward-fill, no-lookahead, column presence,
        1h preservation, empty/None df_4h, custom suffix, date alignment,
        and OHLCV inclusion.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from freqtrade.ohio.core.market_state.mtf_merger import MTFMerger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_1h(n: int = 100) -> pd.DataFrame:
    """100 hourly bars starting 2024-01-01 00:00 UTC."""
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
            "ohio_feat_rsi": range(n),
            "ohio_feat_atr": range(n, 2 * n),
        }
    )


def _make_4h(n: int = 25) -> pd.DataFrame:
    """25 4-hourly bars starting 2024-01-01 00:00 UTC."""
    dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": 200.0,
            "high": 202.0,
            "low": 198.0,
            "close": 201.0,
            "volume": 5000.0,
            "ohio_feat_trend": range(n),
            "ohio_feat_vol": range(n, 2 * n),
        }
    )


@pytest.fixture()
def df_1h() -> pd.DataFrame:
    return _make_1h()


@pytest.fixture()
def df_4h() -> pd.DataFrame:
    return _make_4h()


@pytest.fixture()
def merger() -> MTFMerger:
    return MTFMerger()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicMerge:
    def test_row_count_matches_1h(
        self, merger: MTFMerger, df_1h: pd.DataFrame, df_4h: pd.DataFrame
    ) -> None:
        """Merged DataFrame must have the same number of rows as df_1h."""
        result = merger.merge(df_1h, df_4h)
        assert len(result) == len(df_1h)

    def test_4h_feature_columns_have_suffix(
        self, merger: MTFMerger, df_1h: pd.DataFrame, df_4h: pd.DataFrame
    ) -> None:
        """Every ohio_feat_ column from df_4h must appear with the suffix."""
        result = merger.merge(df_1h, df_4h)
        assert "ohio_feat_trend_4h" in result.columns
        assert "ohio_feat_vol_4h" in result.columns
        # Raw 4h names must NOT be present as plain (un-suffixed) columns.
        assert "ohio_feat_trend" not in result.columns
        assert "ohio_feat_vol" not in result.columns

    def test_forward_fill_between_4h_bars(
        self, merger: MTFMerger, df_1h: pd.DataFrame, df_4h: pd.DataFrame
    ) -> None:
        """1h bars between two 4h bars must carry the same 4h feature value."""
        result = merger.merge(df_1h, df_4h)
        # Bars at index 1, 2, 3 (hours 01:00, 02:00, 03:00) fall between the
        # first (00:00) and second (04:00) 4h bar — they should all share
        # the first 4h bar's ohio_feat_trend_4h value.
        val = result.loc[0, "ohio_feat_trend_4h"]  # 00:00 aligns exactly
        assert result.loc[1, "ohio_feat_trend_4h"] == val
        assert result.loc[2, "ohio_feat_trend_4h"] == val
        assert result.loc[3, "ohio_feat_trend_4h"] == val

    def test_no_lookahead_before_first_4h_bar(self) -> None:
        """1h bars before the first 4h bar must have NaN for 4h features."""
        merger = MTFMerger()
        df_1h = _make_1h(10)
        df_4h_offset = _make_4h(3).copy()
        # Shift 4h by 2 hours so bars 0 and 1 have no backward 4h bar.
        df_4h_offset["date"] = df_4h_offset["date"] + pd.Timedelta(hours=2)
        result = merger.merge(df_1h, df_4h_offset)
        assert math.isnan(result.loc[0, "ohio_feat_trend_4h"])
        assert math.isnan(result.loc[1, "ohio_feat_trend_4h"])
        # Bar at index 2 aligns with the shifted 4h bar — should NOT be NaN.
        assert not math.isnan(result.loc[2, "ohio_feat_trend_4h"])

    def test_all_4h_feature_columns_present(
        self, merger: MTFMerger, df_1h: pd.DataFrame, df_4h: pd.DataFrame
    ) -> None:
        """All ohio_feat_ columns from df_4h must be in the result."""
        expected = {f"{c}_4h" for c in df_4h.columns if c.startswith("ohio_feat_")}
        result = merger.merge(df_1h, df_4h)
        assert expected.issubset(result.columns)

    def test_original_1h_columns_preserved(
        self, merger: MTFMerger, df_1h: pd.DataFrame, df_4h: pd.DataFrame
    ) -> None:
        """1h columns must be present and their values must be unchanged."""
        result = merger.merge(df_1h, df_4h)
        for col in df_1h.columns:
            assert col in result.columns
        pd.testing.assert_series_equal(
            result["ohio_feat_rsi"].reset_index(drop=True),
            df_1h["ohio_feat_rsi"].reset_index(drop=True),
        )


class TestEdgeCases:
    def test_empty_df_4h_returns_nan_columns(
        self, merger: MTFMerger, df_1h: pd.DataFrame
    ) -> None:
        """An empty df_4h (schema only) must add NaN-filled columns."""
        df_4h_empty = pd.DataFrame(
            columns=["date", "ohio_feat_trend", "open", "high", "low", "close", "volume"]
        )
        result = merger.merge(df_1h, df_4h_empty)
        assert len(result) == len(df_1h)
        # The empty df declares ohio_feat_trend → NaN col with suffix expected.
        assert "ohio_feat_trend_4h" in result.columns
        assert all(math.isnan(v) for v in result["ohio_feat_trend_4h"])

    def test_none_df_4h_handled_gracefully(
        self, merger: MTFMerger, df_1h: pd.DataFrame
    ) -> None:
        """None df_4h must return df_1h unchanged (no crash, no extra cols)."""
        result = merger.merge(df_1h, None)
        assert len(result) == len(df_1h)
        for col in result.columns:
            assert not col.endswith("_4h")

    def test_custom_suffix(
        self, merger: MTFMerger, df_1h: pd.DataFrame, df_4h: pd.DataFrame
    ) -> None:
        """Custom suffix must be applied to all 4h columns."""
        result = merger.merge(df_1h, df_4h, suffix="_higher")
        assert "ohio_feat_trend_higher" in result.columns
        assert "ohio_feat_trend_4h" not in result.columns

    def test_date_alignment_specific_timestamps(
        self, merger: MTFMerger, df_1h: pd.DataFrame, df_4h: pd.DataFrame
    ) -> None:
        """Each 1h bar must map to the correct 4h bar via backward asof join."""
        result = merger.merge(df_1h, df_4h)
        # The 4h bar at index k covers hours [4k, 4k+4).
        # ohio_feat_trend is k for the k-th 4h bar.
        assert result.loc[3, "ohio_feat_trend_4h"] == 0
        assert result.loc[4, "ohio_feat_trend_4h"] == 1
        assert result.loc[7, "ohio_feat_trend_4h"] == 1
        assert result.loc[8, "ohio_feat_trend_4h"] == 2

    def test_ohlcv_from_4h_included_with_suffix(
        self, merger: MTFMerger, df_1h: pd.DataFrame, df_4h: pd.DataFrame
    ) -> None:
        """OHLCV columns from df_4h must appear in the result with the suffix."""
        result = merger.merge(df_1h, df_4h)
        for col in ("open", "high", "low", "close", "volume"):
            assert f"{col}_4h" in result.columns

    def test_ohlcv_values_from_4h_are_correct(
        self, merger: MTFMerger, df_1h: pd.DataFrame, df_4h: pd.DataFrame
    ) -> None:
        """4h OHLCV values must match the source df_4h values after merge."""
        result = merger.merge(df_1h, df_4h)
        # Rows 0-3 map to first 4h bar; close_4h should be 201.0.
        assert result.loc[0, "close_4h"] == pytest.approx(201.0)
        assert result.loc[3, "close_4h"] == pytest.approx(201.0)

    def test_1h_ohlcv_not_overwritten(
        self, merger: MTFMerger, df_1h: pd.DataFrame, df_4h: pd.DataFrame
    ) -> None:
        """Original 1h OHLCV (no suffix) must retain its original values."""
        result = merger.merge(df_1h, df_4h)
        assert result.loc[0, "close"] == pytest.approx(100.5)
        assert result.loc[0, "open"] == pytest.approx(100.0)
