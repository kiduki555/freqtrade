"""Test cross-asset factor wiring: correlation_stress and breadth_dispersion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.cross_asset_provider import (
    compute_breadth_dispersion,
    compute_correlation_stress,
    compute_peer_returns,
)
from freqtrade.ohio.core.market_state.factors import FactorCalculator
from freqtrade.ohio.core.market_state.factors.breadth import compute_breadth
from freqtrade.ohio.core.market_state.factors.correlation import compute_correlation
from freqtrade.ohio.core.market_state.feature_builder import FEATURE_COLUMNS, FeatureBuilder
from freqtrade.ohio.core.market_state.normalizer import Normalizer


class TestFeatureColumns:
    """FEATURE_COLUMNS should list the new cross-asset columns."""

    def test_correlation_stress_in_feature_columns(self):
        assert "ohio_feat_correlation_stress" in FEATURE_COLUMNS

    def test_breadth_dispersion_in_feature_columns(self):
        assert "ohio_feat_breadth_dispersion" in FEATURE_COLUMNS

    def test_old_placeholder_removed(self):
        assert "ohio_feat_cross_asset_placeholder" not in FEATURE_COLUMNS


class TestFeatureBuilderPlaceholders:
    """FeatureBuilder should create NaN columns for cross-asset if not already set."""

    def test_creates_nan_columns_when_missing(self):
        df = _make_ohlcv(100)
        builder = FeatureBuilder()
        result = builder.compute(df)
        assert "ohio_feat_correlation_stress" in result.columns
        assert "ohio_feat_breadth_dispersion" in result.columns
        assert result["ohio_feat_correlation_stress"].isna().all()
        assert result["ohio_feat_breadth_dispersion"].isna().all()

    def test_preserves_existing_columns(self):
        df = _make_ohlcv(100)
        df["ohio_feat_correlation_stress"] = 0.7
        df["ohio_feat_breadth_dispersion"] = 0.3
        builder = FeatureBuilder()
        result = builder.compute(df)
        assert (result["ohio_feat_correlation_stress"] == 0.7).all()
        assert (result["ohio_feat_breadth_dispersion"] == 0.3).all()


class TestFactorCalculatorReadsNewColumns:
    """FactorCalculator reads ohio_norm_correlation_stress and ohio_norm_breadth_dispersion."""

    def test_uses_real_values_when_present(self):
        df = _make_ohlcv(200)
        builder = FeatureBuilder()
        df = builder.compute(df)
        # Inject monotonically increasing cross-asset features BEFORE normalization.
        # A rising series → percentile rank climbs toward 1.0 over time, deviating
        # clearly from the 0.5 NaN-fallback default.
        n = len(df)
        df["ohio_feat_correlation_stress"] = np.linspace(0.1, 0.9, n)
        df["ohio_feat_breadth_dispersion"] = np.linspace(0.1, 0.9, n)
        normalizer = Normalizer(window=100)
        df = normalizer.normalize(df)
        calculator = FactorCalculator()
        df = calculator.compute(df)
        # Factor values in the tail (where the window is saturated) should be
        # clearly above 0.5 because values are monotonically rising.
        corr_tail = df["ohio_factor_correlation"].iloc[150:]
        breadth_tail = df["ohio_factor_breadth"].iloc[150:]
        assert corr_tail.mean() > 0.7, (
            f"correlation factor not elevated after wiring: {corr_tail.mean()}"
        )
        assert breadth_tail.mean() > 0.7, (
            f"breadth factor not elevated after wiring: {breadth_tail.mean()}"
        )

    def test_defaults_to_05_when_nan(self):
        df = _make_ohlcv(100)
        builder = FeatureBuilder()
        df = builder.compute(df)
        # Leave correlation/breadth as NaN (no cross-asset data)
        normalizer = Normalizer(window=50)
        df = normalizer.normalize(df)
        calculator = FactorCalculator()
        df = calculator.compute(df)
        # Should default to 0.5
        assert df["ohio_factor_correlation"].iloc[-1] == pytest.approx(0.5)
        assert df["ohio_factor_breadth"].iloc[-1] == pytest.approx(0.5)


class TestCorrelationModule:
    """compute_correlation should read ohio_norm_correlation_stress."""

    def test_reads_new_key(self):
        row = {"ohio_norm_correlation_stress": 0.75}
        assert compute_correlation(row) == pytest.approx(0.75)

    def test_defaults_to_05_when_missing(self):
        assert compute_correlation({}) == pytest.approx(0.5)


class TestBreadthModule:
    """compute_breadth should read ohio_norm_breadth_dispersion."""

    def test_reads_new_key(self):
        row = {"ohio_norm_breadth_dispersion": 0.3}
        assert compute_breadth(row) == pytest.approx(0.3)

    def test_defaults_to_05_when_missing(self):
        assert compute_breadth({}) == pytest.approx(0.5)


class TestCrossAssetComputation:
    """Verify cross-asset computations produce distinct values."""

    def test_correlation_stress_varies_with_input(self):
        # Perfectly correlated peers → stress near 1.0
        idx = pd.date_range("2024-01-01", periods=200, freq="h")
        base = pd.Series(np.cumsum(np.random.randn(200) * 0.01) + 100, index=idx)
        peer_closes = {
            "BTC/USDT": base,
            "ETH/USDT": base * 1.1,  # perfectly correlated
            "SOL/USDT": base * 0.5,  # perfectly correlated
        }
        returns = compute_peer_returns(peer_closes, window=24)
        stress = compute_correlation_stress(returns, window=72)
        # High correlation should yield high stress
        valid = stress.dropna()
        assert len(valid) > 50
        assert valid.iloc[-20:].mean() > 0.7

    def test_breadth_dispersion_varies(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="h")
        np.random.seed(42)
        peer_closes = {
            "BTC/USDT": pd.Series(
                np.cumsum(np.random.randn(200) * 0.02) + 100, index=idx
            ),
            "ETH/USDT": pd.Series(
                np.cumsum(np.random.randn(200) * 0.05) + 50, index=idx
            ),
            "SOL/USDT": pd.Series(
                np.cumsum(np.random.randn(200) * 0.03) + 30, index=idx
            ),
        }
        returns = compute_peer_returns(peer_closes, window=24)
        dispersion = compute_breadth_dispersion(returns, window=24)
        valid = dispersion.dropna()
        assert len(valid) > 50
        assert valid.mean() > 0.0  # should have some dispersion


def _make_ohlcv(n: int) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.uniform(1e6, 1e7, n),
    })
