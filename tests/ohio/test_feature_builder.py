"""Tests for FeatureBuilder — 19 primitive per-symbol OHLCV features.

Covers:
- Column presence and naming conventions
- Numeric correctness for key features
- Range constraints (ADX in [0,100], efficiency ratio in [0,1], vol >= 0)
- NaN in warmup region
- Placeholder columns are NaN
- Edge cases (flat price, constant volume)
- Return type contract
- No Freqtrade imports in the implementation module
"""

from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.core.market_state.feature_builder import (
    FEATURE_COLUMNS,
    FeatureBuilder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _make_ohlcv(
    n: int = 200,
    *,
    base_price: float = 100.0,
    drift: float = 0.0001,
    sigma: float = 0.01,
    volume_mean: float = 1_000.0,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with *n* bars.

    Prices follow a log-normal random walk; high/low are constructed from the
    intra-bar range so that ``high >= max(open, close)`` and
    ``low <= min(open, close)`` always holds.
    """
    log_returns = _RNG.normal(drift, sigma, size=n)
    closes = base_price * np.exp(np.cumsum(log_returns))
    opens = np.roll(closes, 1)
    opens[0] = base_price

    noise = _RNG.uniform(0.001, 0.005, size=n)
    body_top = np.maximum(opens, closes)
    body_bot = np.minimum(opens, closes)
    highs = body_top * (1 + noise)
    lows = body_bot * (1 - noise)

    volumes = _RNG.exponential(volume_mean, size=n)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    )


def _make_flat(n: int = 200, price: float = 100.0, volume: float = 500.0) -> pd.DataFrame:
    """Flat price series — all OHLCV identical (zero-volatility edge case)."""
    return pd.DataFrame(
        {
            "open": np.full(n, price),
            "high": np.full(n, price),
            "low": np.full(n, price),
            "close": np.full(n, price),
            "volume": np.full(n, volume),
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def builder() -> FeatureBuilder:
    return FeatureBuilder()


@pytest.fixture(scope="module")
def df_normal(builder: FeatureBuilder) -> pd.DataFrame:
    """200-bar synthetic OHLCV with all features computed."""
    df = _make_ohlcv(200)
    return builder.compute(df)


@pytest.fixture(scope="module")
def df_flat(builder: FeatureBuilder) -> pd.DataFrame:
    """Flat price series with features computed."""
    df = _make_flat(200)
    return builder.compute(df)


# ---------------------------------------------------------------------------
# Test 1 — return type
# ---------------------------------------------------------------------------


def test_compute_returns_dataframe(builder: FeatureBuilder) -> None:
    df = _make_ohlcv(200)
    result = builder.compute(df)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Test 2 — all expected columns present
# ---------------------------------------------------------------------------


def test_all_feature_columns_present(df_normal: pd.DataFrame) -> None:
    for col in FEATURE_COLUMNS:
        assert col in df_normal.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Test 3 — ohio_feat_ prefix on every feature column
# ---------------------------------------------------------------------------


def test_feature_column_prefix(df_normal: pd.DataFrame) -> None:
    feat_cols = [c for c in df_normal.columns if c.startswith("ohio_feat_")]
    assert len(feat_cols) == len(FEATURE_COLUMNS), (
        f"Expected {len(FEATURE_COLUMNS)} feature columns, got {len(feat_cols)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — log_return_24 correctness
# ---------------------------------------------------------------------------


def test_log_return_24_correctness(builder: FeatureBuilder) -> None:
    df = _make_ohlcv(50)
    result = builder.compute(df)

    close = df["close"]
    expected_at_30 = math.log(close.iloc[30] / close.iloc[6])
    actual_at_30 = result["ohio_feat_log_return_24"].iloc[30]
    assert math.isclose(actual_at_30, expected_at_30, rel_tol=1e-9), (
        f"Expected {expected_at_30}, got {actual_at_30}"
    )


# ---------------------------------------------------------------------------
# Test 5 — log_return_72 is NaN for first 72 bars
# ---------------------------------------------------------------------------


def test_log_return_72_nan_warmup(df_normal: pd.DataFrame) -> None:
    series = df_normal["ohio_feat_log_return_72"]
    assert series.iloc[:72].isna().all(), "First 72 bars of log_return_72 should be NaN"
    assert series.iloc[72:].notna().any(), "Should have non-NaN values after bar 72"


# ---------------------------------------------------------------------------
# Test 6 — efficiency_ratio in [0, 1]
# ---------------------------------------------------------------------------


def test_efficiency_ratio_in_unit_interval(df_normal: pd.DataFrame) -> None:
    valid = df_normal["ohio_feat_efficiency_ratio_24"].dropna()
    assert (valid >= 0).all(), "Efficiency ratio must be >= 0"
    assert (valid <= 1.0 + 1e-9).all(), "Efficiency ratio must be <= 1"


# ---------------------------------------------------------------------------
# Test 7 — ADX in [0, 100]
# ---------------------------------------------------------------------------


def test_adx_in_valid_range(df_normal: pd.DataFrame) -> None:
    valid = df_normal["ohio_feat_adx_14"].dropna()
    assert (valid >= 0).all(), "ADX must be >= 0"
    assert (valid <= 100 + 1e-9).all(), "ADX must be <= 100"


# ---------------------------------------------------------------------------
# Test 8 — realized_vol non-negative
# ---------------------------------------------------------------------------


def test_realized_vol_non_negative(df_normal: pd.DataFrame) -> None:
    valid = df_normal["ohio_feat_realized_vol_24"].dropna()
    assert (valid >= 0).all(), "Realized vol must be non-negative"


# ---------------------------------------------------------------------------
# Test 9 — parkinson_vol non-negative
# ---------------------------------------------------------------------------


def test_parkinson_vol_non_negative(df_normal: pd.DataFrame) -> None:
    valid = df_normal["ohio_feat_parkinson_vol_24"].dropna()
    assert (valid >= 0).all(), "Parkinson vol must be non-negative"


# ---------------------------------------------------------------------------
# Test 10 — volume_ratio = 1.0 when volume is constant
# ---------------------------------------------------------------------------


def test_volume_ratio_constant_volume(builder: FeatureBuilder) -> None:
    df = _make_flat(100, volume=500.0)
    result = builder.compute(df)
    ratio = result["ohio_feat_volume_ratio_24"].dropna()
    assert (ratio - 1.0).abs().max() < 1e-9, (
        "volume_ratio should be exactly 1.0 when volume is constant"
    )


# ---------------------------------------------------------------------------
# Test 11 — NaN in warmup for log_return_24 (first 24 bars)
# ---------------------------------------------------------------------------


def test_log_return_24_nan_warmup(df_normal: pd.DataFrame) -> None:
    series = df_normal["ohio_feat_log_return_24"]
    assert series.iloc[:24].isna().all(), "First 24 bars of log_return_24 should be NaN"


# ---------------------------------------------------------------------------
# Test 12 — placeholder columns are all NaN
# ---------------------------------------------------------------------------


def test_rs_raw_is_all_nan(df_normal: pd.DataFrame) -> None:
    assert df_normal["ohio_feat_rs_raw"].isna().all(), "ohio_feat_rs_raw must be all NaN"


def test_cross_asset_features_are_all_nan(df_normal: pd.DataFrame) -> None:
    assert df_normal["ohio_feat_correlation_stress"].isna().all(), (
        "ohio_feat_correlation_stress must be all NaN when cross-asset provider is absent"
    )
    assert df_normal["ohio_feat_breadth_dispersion"].isna().all(), (
        "ohio_feat_breadth_dispersion must be all NaN when cross-asset provider is absent"
    )


# ---------------------------------------------------------------------------
# Test 13 — flat price edge case (zero-volatility)
# ---------------------------------------------------------------------------


def test_flat_price_realized_vol_is_zero_or_nan(df_flat: pd.DataFrame) -> None:
    """Realized vol on a flat series should be 0 (after warmup) or NaN."""
    valid = df_flat["ohio_feat_realized_vol_24"].dropna()
    assert (valid.abs() < 1e-9).all(), "Realized vol must be 0 for flat prices"


def test_flat_price_parkinson_vol_is_zero_or_nan(df_flat: pd.DataFrame) -> None:
    """Parkinson vol on a flat series: ln(high/low)=ln(1)=0, so vol=0."""
    valid = df_flat["ohio_feat_parkinson_vol_24"].dropna()
    assert (valid.abs() < 1e-9).all(), "Parkinson vol must be 0 for flat prices"


# ---------------------------------------------------------------------------
# Test 14 — atr_ratio_14 non-negative
# ---------------------------------------------------------------------------


def test_atr_ratio_non_negative(df_normal: pd.DataFrame) -> None:
    valid = df_normal["ohio_feat_atr_ratio_14"].dropna()
    assert (valid >= 0).all(), "ATR ratio must be non-negative"


# ---------------------------------------------------------------------------
# Test 15 — negative_return_ratio in [0, 1]
# ---------------------------------------------------------------------------


def test_negative_return_ratio_in_unit_interval(df_normal: pd.DataFrame) -> None:
    valid = df_normal["ohio_feat_negative_return_ratio_24"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Test 16 — bid_ask_approx is non-negative (H >= L always)
# ---------------------------------------------------------------------------


def test_bid_ask_approx_non_negative(df_normal: pd.DataFrame) -> None:
    valid = df_normal["ohio_feat_bid_ask_approx"].dropna()
    assert (valid >= 0).all(), "bid_ask_approx must be non-negative"


# ---------------------------------------------------------------------------
# Test 17 — no Freqtrade imports in feature_builder module
# ---------------------------------------------------------------------------


def test_no_freqtrade_imports_in_feature_builder() -> None:
    """Verify feature_builder.py has no import of freqtrade (except ohio itself)."""
    module_path = Path(
        importlib.util.find_spec(  # type: ignore[union-attr]
            "freqtrade.ohio.core.market_state.feature_builder"
        ).origin
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    freqtrade_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("freqtrade") and not alias.name.startswith(
                    "freqtrade.ohio"
                ):
                    freqtrade_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("freqtrade") and not mod.startswith("freqtrade.ohio"):
                freqtrade_imports.append(mod)

    assert not freqtrade_imports, (
        f"feature_builder.py must not import freqtrade internals; found: {freqtrade_imports}"
    )


# ---------------------------------------------------------------------------
# Test 18 — ma_slope_20 requires 25 bars (20 for SMA + 5 shift)
# ---------------------------------------------------------------------------


def test_ma_slope_warmup(df_normal: pd.DataFrame) -> None:
    series = df_normal["ohio_feat_ma_slope_20"]
    # Bars 0..23 (20-bar SMA needs 20, then shift(5) needs 5 more = 24 NaN bars)
    assert series.iloc[:24].isna().all(), "ma_slope_20 warmup bars should be NaN"
    assert series.iloc[24:].notna().any(), "ma_slope_20 should produce values after warmup"


# ---------------------------------------------------------------------------
# Test 19 — max_drawdown_24 is <= 0 (close always <= rolling max)
# ---------------------------------------------------------------------------


def test_max_drawdown_non_positive(df_normal: pd.DataFrame) -> None:
    valid = df_normal["ohio_feat_max_drawdown_24"].dropna()
    assert (valid <= 1e-9).all(), "max drawdown must be <= 0 (price <= rolling max)"


# ---------------------------------------------------------------------------
# Test 20 — feature_builder can handle 500-bar dataframe without error
# ---------------------------------------------------------------------------


def test_large_dataframe_no_exception(builder: FeatureBuilder) -> None:
    df = _make_ohlcv(500)
    result = builder.compute(df)
    assert len(result) == 500
    for col in FEATURE_COLUMNS:
        assert col in result.columns


# ---------------------------------------------------------------------------
# Test 21 — ATR baseline (168h rolling median of atr_ratio_14)
# ---------------------------------------------------------------------------


class TestAtrBaseline:
    def test_atr_baseline_computed(self) -> None:
        """ohio_feat_atr_baseline is the 168h rolling median of atr_ratio_14."""
        builder = FeatureBuilder()
        df = _make_ohlcv(200)
        df = builder.compute(df)
        assert "ohio_feat_atr_baseline" in df.columns
        assert not pd.isna(df["ohio_feat_atr_baseline"].iloc[-1])

    def test_atr_baseline_nan_during_warmup(self) -> None:
        """First 168 bars should have NaN baseline."""
        builder = FeatureBuilder()
        df = _make_ohlcv(200)
        df = builder.compute(df)
        assert pd.isna(df["ohio_feat_atr_baseline"].iloc[100])

    def test_atr_baseline_non_negative(self) -> None:
        """ATR baseline must be non-negative (it is a ratio of positive values)."""
        builder = FeatureBuilder()
        df = _make_ohlcv(200)
        df = builder.compute(df)
        valid = df["ohio_feat_atr_baseline"].dropna()
        assert (valid >= 0).all(), "ATR baseline must be non-negative"

    def test_atr_baseline_in_feature_columns(self) -> None:
        """ohio_feat_atr_baseline must be listed in FEATURE_COLUMNS."""
        assert "ohio_feat_atr_baseline" in FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Test 22 — Hurst exponent via R/S analysis
# ---------------------------------------------------------------------------


class TestHurstExponent:
    def test_hurst_column_exists(self, df_normal: pd.DataFrame) -> None:
        """ohio_feat_hurst_168 must be present after compute()."""
        assert "ohio_feat_hurst_168" in df_normal.columns

    def test_hurst_in_feature_columns(self) -> None:
        assert "ohio_feat_hurst_168" in FEATURE_COLUMNS

    def test_hurst_nan_during_warmup(self, df_normal: pd.DataFrame) -> None:
        """First 168 bars should be NaN (window=168)."""
        series = df_normal["ohio_feat_hurst_168"]
        assert series.iloc[:168].isna().all(), "Hurst warmup (168 bars) should be NaN"

    def test_hurst_has_values_after_warmup(self, df_normal: pd.DataFrame) -> None:
        series = df_normal["ohio_feat_hurst_168"]
        assert series.iloc[168:].notna().any(), "Should have values after warmup"

    def test_hurst_range(self) -> None:
        """Hurst exponent should typically be in [0, 1] for financial series."""
        builder = FeatureBuilder()
        df = _make_ohlcv(400)
        df = builder.compute(df)
        valid = df["ohio_feat_hurst_168"].dropna()
        assert len(valid) > 0, "Should have computed Hurst values"
        assert (valid >= 0.0).all(), f"Hurst min={valid.min():.4f}, expected >= 0"
        assert (valid <= 1.2).all(), f"Hurst max={valid.max():.4f}, expected <= 1.2"

    def test_hurst_trending_series(self) -> None:
        """Strong uptrend should produce Hurst > 0.5 (persistent)."""
        n = 400
        closes = 100.0 + np.arange(n) * 0.5  # strong linear uptrend
        noise = np.random.default_rng(42).normal(0, 0.05, n)
        closes = closes + noise
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        df = pd.DataFrame({
            "open": opens,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": np.full(n, 1000.0),
        })
        builder = FeatureBuilder()
        df = builder.compute(df)
        valid = df["ohio_feat_hurst_168"].dropna()
        median_h = valid.median()
        assert median_h > 0.5, f"Trending series Hurst median={median_h:.3f}, expected > 0.5"

    def test_hurst_flat_series_is_nan(self, df_flat: pd.DataFrame) -> None:
        """Flat price → std=0 → Hurst should be NaN (invalid)."""
        valid = df_flat["ohio_feat_hurst_168"].dropna()
        # Flat price has zero returns → std < 1e-12 → skip → all NaN
        assert len(valid) == 0, "Flat series should produce all-NaN Hurst"


# ---------------------------------------------------------------------------
# Test 23 — Entry indicator features (12 new columns)
# ---------------------------------------------------------------------------

NEW_FEATURE_COLUMNS = [
    "ohio_feat_zscore_20",
    "ohio_feat_rsi_14",
    "ohio_feat_kama_10",
    "ohio_feat_kama_slope",
    "ohio_feat_bb_upper_20",
    "ohio_feat_bb_lower_20",
    "ohio_feat_kc_upper_20",
    "ohio_feat_kc_lower_20",
    "ohio_feat_donchian_upper_20",
    "ohio_feat_donchian_lower_20",
    "ohio_feat_squeeze_count",
    "ohio_feat_volume_sma_20",
]


class TestEntryIndicatorFeatures:
    """Tests for the 12 entry strategy indicator features."""

    def test_new_columns_present(self, df_normal: pd.DataFrame) -> None:
        for col in NEW_FEATURE_COLUMNS:
            assert col in df_normal.columns, f"Missing column: {col}"

    def test_new_columns_in_feature_columns(self) -> None:
        for col in NEW_FEATURE_COLUMNS:
            assert col in FEATURE_COLUMNS, f"{col} not in FEATURE_COLUMNS"

    def test_zscore_centered_near_zero(self, df_normal: pd.DataFrame) -> None:
        valid = df_normal["ohio_feat_zscore_20"].dropna()
        assert abs(valid.mean()) < 1.0, "Z-score should be roughly centered"

    def test_rsi_in_range(self, df_normal: pd.DataFrame) -> None:
        valid = df_normal["ohio_feat_rsi_14"].dropna()
        assert (valid >= 0).all(), "RSI must be >= 0"
        assert (valid <= 100 + 1e-9).all(), "RSI must be <= 100"

    def test_rsi_warmup(self, df_normal: pd.DataFrame) -> None:
        series = df_normal["ohio_feat_rsi_14"]
        # close.diff() produces NaN at index 0; ewm(min_periods=14) needs 14
        # valid values, so first non-NaN RSI appears at index 13 (0-based).
        assert series.iloc[:13].isna().all(), "RSI first 13 bars should be NaN"

    def test_kama_follows_price(self, df_normal: pd.DataFrame) -> None:
        valid_idx = df_normal["ohio_feat_kama_10"].dropna().index
        valid_kama = df_normal.loc[valid_idx, "ohio_feat_kama_10"]
        valid_close = df_normal.loc[valid_idx, "close"]
        corr = valid_kama.corr(valid_close)
        assert corr > 0.9, f"KAMA should track close, corr={corr:.3f}"

    def test_bb_upper_above_lower(self, df_normal: pd.DataFrame) -> None:
        mask = df_normal["ohio_feat_bb_upper_20"].notna()
        upper = df_normal.loc[mask, "ohio_feat_bb_upper_20"]
        lower = df_normal.loc[mask, "ohio_feat_bb_lower_20"]
        assert (upper >= lower).all(), "BB upper must be >= lower"

    def test_kc_upper_above_lower(self, df_normal: pd.DataFrame) -> None:
        mask = df_normal["ohio_feat_kc_upper_20"].notna()
        upper = df_normal.loc[mask, "ohio_feat_kc_upper_20"]
        lower = df_normal.loc[mask, "ohio_feat_kc_lower_20"]
        assert (upper >= lower).all(), "KC upper must be >= lower"

    def test_donchian_upper_above_lower(self, df_normal: pd.DataFrame) -> None:
        mask = df_normal["ohio_feat_donchian_upper_20"].notna()
        upper = df_normal.loc[mask, "ohio_feat_donchian_upper_20"]
        lower = df_normal.loc[mask, "ohio_feat_donchian_lower_20"]
        assert (upper >= lower).all(), "Donchian upper must be >= lower"

    def test_squeeze_count_non_negative(self, df_normal: pd.DataFrame) -> None:
        valid = df_normal["ohio_feat_squeeze_count"].dropna()
        assert (valid >= 0).all(), "Squeeze count must be >= 0"

    def test_volume_sma_positive(self, df_normal: pd.DataFrame) -> None:
        valid = df_normal["ohio_feat_volume_sma_20"].dropna()
        assert (valid > 0).all(), "Volume SMA must be positive"

    def test_flat_price_zscore_zero(self, df_flat: pd.DataFrame) -> None:
        valid = df_flat["ohio_feat_zscore_20"].dropna()
        # Flat price → close == SMA → zscore = 0, but std = 0 → NaN
        # So valid might be empty, which is also acceptable
        if len(valid) > 0:
            assert (valid.abs() < 1e-9).all(), "Z-score should be 0 for flat prices"
