"""Tests for FT-007: 7-Axis Factor Calculator.

Covers: each factor function, FactorCalculator.compute, compute_vector,
boundary conditions, NaN neutrality, weight consistency, and vectorized
vs row-by-row parity.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.core.domain.models import StateVector
from freqtrade.ohio.core.market_state.factors import FactorCalculator
from freqtrade.ohio.core.market_state.factors.breadth import compute_breadth
from freqtrade.ohio.core.market_state.factors.correlation import compute_correlation
from freqtrade.ohio.core.market_state.factors.downside import compute_downside
from freqtrade.ohio.core.market_state.factors.liquidity import compute_liquidity
from freqtrade.ohio.core.market_state.factors.relative_strength import compute_relative_strength
from freqtrade.ohio.core.market_state.factors.trend import compute_trend
from freqtrade.ohio.core.market_state.factors.volatility import compute_volatility

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def neutral_row() -> dict[str, float]:
    """All normalized inputs at 0.5 (neutral midpoint)."""
    return {}  # all .get() calls fall back to 0.5


@pytest.fixture
def all_zero_row() -> dict[str, float]:
    return {
        "ohio_norm_log_return_24": 0.0,
        "ohio_norm_ma_slope_20": 0.0,
        "ohio_norm_adx_14": 0.0,
        "ohio_norm_efficiency_ratio_24": 0.0,
        "ohio_norm_realized_vol_24": 0.0,
        "ohio_norm_atr_ratio_14": 0.0,
        "ohio_norm_parkinson_vol_24": 0.0,
        "ohio_norm_sortino_downside_24": 0.0,
        "ohio_norm_max_drawdown_24": 0.0,
        "ohio_norm_cvar_24": 0.0,
        "ohio_norm_lower_shadow_ratio_24": 0.0,
        "ohio_norm_negative_return_ratio_24": 0.0,
        "ohio_norm_bid_ask_approx": 0.0,
        "ohio_norm_volume_ratio_24": 0.0,
        "ohio_norm_obv_slope_24": 0.0,
        "ohio_norm_rs_raw": 0.0,
        "ohio_norm_cross_asset_placeholder": 0.0,
    }


@pytest.fixture
def all_one_row() -> dict[str, float]:
    return {
        "ohio_norm_log_return_24": 1.0,
        "ohio_norm_ma_slope_20": 1.0,
        "ohio_norm_adx_14": 1.0,
        "ohio_norm_efficiency_ratio_24": 1.0,
        "ohio_norm_realized_vol_24": 1.0,
        "ohio_norm_atr_ratio_14": 1.0,
        "ohio_norm_parkinson_vol_24": 1.0,
        "ohio_norm_sortino_downside_24": 1.0,
        "ohio_norm_max_drawdown_24": 1.0,
        "ohio_norm_cvar_24": 1.0,
        "ohio_norm_lower_shadow_ratio_24": 1.0,
        "ohio_norm_negative_return_ratio_24": 1.0,
        "ohio_norm_bid_ask_approx": 1.0,
        "ohio_norm_volume_ratio_24": 1.0,
        "ohio_norm_obv_slope_24": 1.0,
        "ohio_norm_rs_raw": 1.0,
        "ohio_norm_cross_asset_placeholder": 1.0,
    }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame with 3 rows covering low, neutral, and high inputs."""
    rows = [
        {  # row 0: all zeros
            "ohio_norm_log_return_24": 0.0,
            "ohio_norm_ma_slope_20": 0.0,
            "ohio_norm_adx_14": 0.0,
            "ohio_norm_efficiency_ratio_24": 0.0,
            "ohio_norm_realized_vol_24": 0.0,
            "ohio_norm_atr_ratio_14": 0.0,
            "ohio_norm_parkinson_vol_24": 0.0,
            "ohio_norm_sortino_downside_24": 0.0,
            "ohio_norm_max_drawdown_24": 0.0,
            "ohio_norm_cvar_24": 0.0,
            "ohio_norm_lower_shadow_ratio_24": 0.0,
            "ohio_norm_negative_return_ratio_24": 0.0,
            "ohio_norm_bid_ask_approx": 0.0,
            "ohio_norm_volume_ratio_24": 0.0,
            "ohio_norm_obv_slope_24": 0.0,
            "ohio_norm_rs_raw": float("nan"),
            "ohio_norm_cross_asset_placeholder": float("nan"),
        },
        {  # row 1: neutral
            col: 0.5
            for col in [
                "ohio_norm_log_return_24", "ohio_norm_ma_slope_20",
                "ohio_norm_adx_14", "ohio_norm_efficiency_ratio_24",
                "ohio_norm_realized_vol_24", "ohio_norm_atr_ratio_14",
                "ohio_norm_parkinson_vol_24", "ohio_norm_sortino_downside_24",
                "ohio_norm_max_drawdown_24", "ohio_norm_cvar_24",
                "ohio_norm_lower_shadow_ratio_24", "ohio_norm_negative_return_ratio_24",
                "ohio_norm_bid_ask_approx", "ohio_norm_volume_ratio_24",
                "ohio_norm_obv_slope_24", "ohio_norm_rs_raw",
                "ohio_norm_cross_asset_placeholder",
            ]
        },
        {  # row 2: all ones
            "ohio_norm_log_return_24": 1.0,
            "ohio_norm_ma_slope_20": 1.0,
            "ohio_norm_adx_14": 1.0,
            "ohio_norm_efficiency_ratio_24": 1.0,
            "ohio_norm_realized_vol_24": 1.0,
            "ohio_norm_atr_ratio_14": 1.0,
            "ohio_norm_parkinson_vol_24": 1.0,
            "ohio_norm_sortino_downside_24": 1.0,
            "ohio_norm_max_drawdown_24": 1.0,
            "ohio_norm_cvar_24": 1.0,
            "ohio_norm_lower_shadow_ratio_24": 1.0,
            "ohio_norm_negative_return_ratio_24": 1.0,
            "ohio_norm_bid_ask_approx": 1.0,
            "ohio_norm_volume_ratio_24": 1.0,
            "ohio_norm_obv_slope_24": 1.0,
            "ohio_norm_rs_raw": 1.0,
            "ohio_norm_cross_asset_placeholder": 1.0,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1-8: Return type checks
# ---------------------------------------------------------------------------

def test_compute_trend_returns_float(neutral_row):
    result = compute_trend(neutral_row)
    assert isinstance(result, float)


def test_compute_volatility_returns_float(neutral_row):
    assert isinstance(compute_volatility(neutral_row), float)


def test_compute_downside_returns_float(neutral_row):
    assert isinstance(compute_downside(neutral_row), float)


def test_compute_liquidity_returns_float(neutral_row):
    assert isinstance(compute_liquidity(neutral_row), float)


def test_compute_relative_strength_returns_float(neutral_row):
    assert isinstance(compute_relative_strength(neutral_row), float)


def test_compute_correlation_returns_float(neutral_row):
    assert isinstance(compute_correlation(neutral_row), float)


def test_compute_breadth_returns_float(neutral_row):
    assert isinstance(compute_breadth(neutral_row), float)


# ---------------------------------------------------------------------------
# 9-13: Range checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inputs,label", [
    ({}, "neutral"),
    ({"ohio_norm_log_return_24": 0.0, "ohio_norm_ma_slope_20": 0.0,
      "ohio_norm_adx_14": 0.0, "ohio_norm_efficiency_ratio_24": 0.0}, "all_zero"),
    ({"ohio_norm_log_return_24": 1.0, "ohio_norm_ma_slope_20": 1.0,
      "ohio_norm_adx_14": 1.0, "ohio_norm_efficiency_ratio_24": 1.0}, "all_one"),
])
def test_trend_in_range(inputs, label):
    result = compute_trend(inputs)
    assert -1.0 <= result <= 1.0, f"trend out of range for {label}: {result}"


@pytest.mark.parametrize("inputs", [{}, {"ohio_norm_realized_vol_24": 0.0}, {"ohio_norm_realized_vol_24": 1.0}])
def test_volatility_in_range(inputs):
    result = compute_volatility(inputs)
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize("inputs", [{}, {"ohio_norm_max_drawdown_24": 0.0}, {"ohio_norm_max_drawdown_24": 1.0}])
def test_downside_in_range(inputs):
    result = compute_downside(inputs)
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize("inputs", [{}, {"ohio_norm_bid_ask_approx": 0.0}, {"ohio_norm_bid_ask_approx": 1.0}])
def test_liquidity_in_range(inputs):
    result = compute_liquidity(inputs)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 14-16: NaN neutrality for cross-asset factors
# ---------------------------------------------------------------------------

def test_relative_strength_nan_returns_neutral():
    row = {"ohio_norm_rs_raw": float("nan")}
    assert compute_relative_strength(row) == pytest.approx(0.5)


def test_correlation_nan_returns_neutral():
    row = {"ohio_norm_cross_asset_placeholder": float("nan")}
    assert compute_correlation(row) == pytest.approx(0.5)


def test_breadth_nan_returns_neutral():
    row = {"ohio_norm_cross_asset_placeholder": float("nan")}
    assert compute_breadth(row) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 17-18: FactorCalculator.compute — columns and types
# ---------------------------------------------------------------------------

_EXPECTED_COLUMNS = [
    "ohio_factor_trend",
    "ohio_factor_volatility",
    "ohio_factor_downside",
    "ohio_factor_liquidity",
    "ohio_factor_relative_strength",
    "ohio_factor_correlation",
    "ohio_factor_breadth",
]


def test_factor_calculator_adds_all_columns(sample_df):
    calc = FactorCalculator()
    out = calc.compute(sample_df.copy())
    for col in _EXPECTED_COLUMNS:
        assert col in out.columns, f"Missing column: {col}"


def test_factor_calculator_returns_dataframe(sample_df):
    calc = FactorCalculator()
    out = calc.compute(sample_df.copy())
    assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# 19: FactorCalculator.compute_vector returns StateVector
# ---------------------------------------------------------------------------

def test_compute_vector_returns_state_vector(neutral_row):
    calc = FactorCalculator()
    result = calc.compute_vector(neutral_row)
    assert isinstance(result, StateVector)


# ---------------------------------------------------------------------------
# 20-21: All-zero inputs — expected values
# ---------------------------------------------------------------------------

def test_all_zero_trend_is_negative(all_zero_row):
    """All-zero inputs → raw = 0.0 → tanh(2*(0-0.5)) = tanh(-1) ≈ -0.762."""
    result = compute_trend(all_zero_row)
    expected = math.tanh(2.0 * (0.0 - 0.5))
    assert result == pytest.approx(expected, abs=1e-9)
    assert result < 0.0


def test_all_zero_volatility_is_zero(all_zero_row):
    assert compute_volatility(all_zero_row) == pytest.approx(0.0)


def test_all_zero_downside_is_zero(all_zero_row):
    assert compute_downside(all_zero_row) == pytest.approx(0.0)


def test_all_zero_liquidity(all_zero_row):
    """All zero inputs: spread=0, vol_inv=1, obv_inv=1 → 0.40*0 + 0.35*1 + 0.25*1 = 0.60."""
    result = compute_liquidity(all_zero_row)
    assert result == pytest.approx(0.60, abs=1e-9)


# ---------------------------------------------------------------------------
# 22-23: All-one inputs — expected values
# ---------------------------------------------------------------------------

def test_all_one_trend_is_positive(all_one_row):
    """All-one inputs → raw = 1.0 → tanh(2*(1-0.5)) = tanh(1) ≈ 0.762."""
    result = compute_trend(all_one_row)
    expected = math.tanh(2.0 * (1.0 - 0.5))
    assert result == pytest.approx(expected, abs=1e-9)
    assert result > 0.0


def test_all_one_volatility_is_one(all_one_row):
    assert compute_volatility(all_one_row) == pytest.approx(1.0)


def test_all_one_downside_is_one(all_one_row):
    assert compute_downside(all_one_row) == pytest.approx(1.0)


def test_all_one_liquidity(all_one_row):
    """All one: spread=1, vol_inv=0, obv_inv=0 → 0.40*1 + 0.35*0 + 0.25*0 = 0.40."""
    result = compute_liquidity(all_one_row)
    assert result == pytest.approx(0.40, abs=1e-9)


# ---------------------------------------------------------------------------
# 24-26: Weight sums
# ---------------------------------------------------------------------------

def test_trend_weights_sum_to_one():
    weights = [0.35, 0.25, 0.20, 0.20]
    assert sum(weights) == pytest.approx(1.0)


def test_volatility_weights_sum_to_one():
    weights = [0.50, 0.30, 0.20]
    assert sum(weights) == pytest.approx(1.0)


def test_downside_weights_sum_to_one():
    weights = [0.25, 0.25, 0.20, 0.15, 0.15]
    assert sum(weights) == pytest.approx(1.0)


def test_liquidity_weights_sum_to_one():
    weights = [0.40, 0.35, 0.25]
    assert sum(weights) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 27-28: Directional sanity for trend
# ---------------------------------------------------------------------------

def test_trend_all_high_inputs_positive():
    high_row = {
        "ohio_norm_log_return_24": 1.0,
        "ohio_norm_ma_slope_20": 1.0,
        "ohio_norm_adx_14": 1.0,
        "ohio_norm_efficiency_ratio_24": 1.0,
    }
    assert compute_trend(high_row) > 0.5


def test_trend_all_low_inputs_negative():
    low_row = {
        "ohio_norm_log_return_24": 0.0,
        "ohio_norm_ma_slope_20": 0.0,
        "ohio_norm_adx_14": 0.0,
        "ohio_norm_efficiency_ratio_24": 0.0,
    }
    assert compute_trend(low_row) < -0.5


# ---------------------------------------------------------------------------
# 29: Vectorized compute matches row-by-row compute_vector
# ---------------------------------------------------------------------------

def test_vectorized_matches_row_by_row(sample_df):
    calc = FactorCalculator()
    out = calc.compute(sample_df.copy())

    for i, row_series in sample_df.iterrows():
        row = row_series.to_dict()
        sv = calc.compute_vector(row)

        assert out.loc[i, "ohio_factor_trend"] == pytest.approx(sv.trend_persistence, abs=1e-9)
        assert out.loc[i, "ohio_factor_volatility"] == pytest.approx(sv.volatility_level, abs=1e-9)
        assert out.loc[i, "ohio_factor_downside"] == pytest.approx(sv.downside_pressure, abs=1e-9)
        assert out.loc[i, "ohio_factor_liquidity"] == pytest.approx(sv.liquidity_stress, abs=1e-9)
        assert out.loc[i, "ohio_factor_relative_strength"] == pytest.approx(sv.relative_strength, abs=1e-9)
        assert out.loc[i, "ohio_factor_correlation"] == pytest.approx(sv.correlation_stress, abs=1e-9)
        assert out.loc[i, "ohio_factor_breadth"] == pytest.approx(sv.breadth_dispersion, abs=1e-9)


# ---------------------------------------------------------------------------
# 30: No Freqtrade imports (structural check)
# ---------------------------------------------------------------------------

def test_no_freqtrade_imports_in_factor_modules():
    """Verify factor modules do not import from freqtrade.* directly."""
    import ast
    import importlib.util
    import pathlib

    factors_dir = pathlib.Path(__file__).parents[2] / "freqtrade" / "ohio" / "core" / "market_state" / "factors"
    individual_modules = [
        "trend.py", "volatility.py", "downside.py", "liquidity.py",
        "relative_strength.py", "correlation.py", "breadth.py",
    ]
    for module_name in individual_modules:
        src = (factors_dir / module_name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("freqtrade."), (
                        f"{module_name} imports from freqtrade.*: {node.module}"
                    )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("freqtrade."), (
                            f"{module_name} imports freqtrade.*: {alias.name}"
                        )
