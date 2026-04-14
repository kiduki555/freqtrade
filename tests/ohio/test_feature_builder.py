"""Tests for FeatureBuilder — 18 primitive per-symbol OHLCV features.

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


def test_cross_asset_placeholder_is_all_nan(df_normal: pd.DataFrame) -> None:
    assert df_normal["ohio_feat_cross_asset_placeholder"].isna().all(), (
        "ohio_feat_cross_asset_placeholder must be all NaN"
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
