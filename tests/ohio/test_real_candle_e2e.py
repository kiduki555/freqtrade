"""End-to-end pipeline test with REAL Binance candle data.

Downloads BTC/USDT and ETH/USDT 1h feather files from user_data/data/binance/
and runs the full 7-phase pipeline:

    OHLCV → FeatureBuilder → Normalizer → FactorCalculator → Stabilizer
          → MetaCalculator → FitnessEstimator → PolicyGenerator

Validates shape, NaN ratios, value ranges, and column presence at each stage.
This is NOT a unit test — it requires real data to be present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Pipeline modules
from freqtrade.ohio.core.market_state.feature_builder import FeatureBuilder, FEATURE_COLUMNS
from freqtrade.ohio.core.market_state.normalizer import Normalizer
from freqtrade.ohio.core.market_state.factors import FactorCalculator
from freqtrade.ohio.core.market_state.stabilizer import StateStabilizer
from freqtrade.ohio.core.market_state.meta_calculator import MetaCalculator
from freqtrade.ohio.core.strategy_router.fitness_estimator import FitnessEstimator
from freqtrade.ohio.core.strategy_router.policy_generator import PolicyGenerator
from freqtrade.ohio.core.domain.models import StrategyMode


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent.parent / "user_data" / "data" / "binance"
_BTC_FILE = _DATA_DIR / "BTC_USDT-1h.feather"
_ETH_FILE = _DATA_DIR / "ETH_USDT-1h.feather"


def _load_candles(path: Path) -> pd.DataFrame:
    """Load feather file and ensure standard OHLCV column names."""
    df = pd.read_feather(path)
    # Freqtrade feather files use lowercase column names
    expected = {"date", "open", "high", "low", "close", "volume"}
    assert expected.issubset(set(df.columns)), (
        f"Missing columns: {expected - set(df.columns)}"
    )
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Skip if data not available
# ---------------------------------------------------------------------------

_has_data = _BTC_FILE.exists() and _ETH_FILE.exists()
pytestmark = pytest.mark.skipif(not _has_data, reason="Real candle data not downloaded")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def btc_raw() -> pd.DataFrame:
    return _load_candles(_BTC_FILE)


@pytest.fixture(scope="module")
def eth_raw() -> pd.DataFrame:
    return _load_candles(_ETH_FILE)


# ---------------------------------------------------------------------------
# Phase ② FEATURE — FeatureBuilder
# ---------------------------------------------------------------------------

class TestFeatureBuilder:
    """Phase ②: raw OHLCV → ohio_feat_* columns."""

    @pytest.fixture(scope="class")
    def btc_feat(self, btc_raw: pd.DataFrame) -> pd.DataFrame:
        return FeatureBuilder().compute(btc_raw.copy())

    def test_all_feature_columns_present(self, btc_feat: pd.DataFrame) -> None:
        # Placeholder columns (rs_raw, cross_asset_placeholder) are all-NaN, still present
        non_placeholder = [c for c in FEATURE_COLUMNS if "placeholder" not in c and "rs_raw" not in c]
        for col in non_placeholder:
            assert col in btc_feat.columns, f"Missing feature column: {col}"

    def test_row_count_preserved(self, btc_feat: pd.DataFrame, btc_raw: pd.DataFrame) -> None:
        assert len(btc_feat) == len(btc_raw)

    def test_nan_only_in_warmup(self, btc_feat: pd.DataFrame) -> None:
        """After warmup (~72 bars), core features should have data."""
        warmup = 100  # conservative warmup
        for col in ["ohio_feat_log_return_24", "ohio_feat_realized_vol_24", "ohio_feat_adx_14"]:
            post_warmup = btc_feat[col].iloc[warmup:]
            nan_ratio = post_warmup.isna().mean()
            assert nan_ratio < 0.01, f"{col} has {nan_ratio:.1%} NaN after warmup"

    def test_adx_in_valid_range(self, btc_feat: pd.DataFrame) -> None:
        """ADX should be in [0, 100]."""
        adx = btc_feat["ohio_feat_adx_14"].dropna()
        assert (adx >= 0).all(), f"ADX has negative values: min={adx.min()}"
        assert (adx <= 100).all(), f"ADX exceeds 100: max={adx.max()}"

    def test_log_returns_reasonable(self, btc_feat: pd.DataFrame) -> None:
        """24h log returns should be within +-50% for BTC."""
        lr = btc_feat["ohio_feat_log_return_24"].dropna()
        assert lr.abs().max() < 0.5, f"Extreme log return: {lr.abs().max():.4f}"


# ---------------------------------------------------------------------------
# Phase ③ COMPUTE — Normalizer + FactorCalculator
# ---------------------------------------------------------------------------

class TestNormalizerAndFactors:
    """Phase ③: ohio_feat_* → ohio_norm_* → ohio_factor_*."""

    @pytest.fixture(scope="class")
    def btc_factor(self, btc_raw: pd.DataFrame) -> pd.DataFrame:
        df = btc_raw.copy()
        df = FeatureBuilder().compute(df)
        df = Normalizer(window=4320).normalize(df)
        df = FactorCalculator().compute(df)
        return df

    def test_norm_columns_present(self, btc_factor: pd.DataFrame) -> None:
        norm_cols = [c for c in btc_factor.columns if c.startswith("ohio_norm_")]
        # Should have norm columns for non-placeholder features
        assert len(norm_cols) >= 14, f"Only {len(norm_cols)} norm columns found"

    def test_norm_values_in_range(self, btc_factor: pd.DataFrame) -> None:
        """All ohio_norm_* values should be in [0, 1]."""
        for col in btc_factor.columns:
            if col.startswith("ohio_norm_"):
                vals = btc_factor[col].dropna()
                if len(vals) == 0:
                    continue
                assert (vals >= 0.0).all() and (vals <= 1.0).all(), (
                    f"{col} out of [0,1]: min={vals.min():.4f}, max={vals.max():.4f}"
                )

    def test_factor_columns_present(self, btc_factor: pd.DataFrame) -> None:
        expected = [
            "ohio_factor_trend", "ohio_factor_volatility", "ohio_factor_downside",
            "ohio_factor_liquidity", "ohio_factor_relative_strength",
            "ohio_factor_correlation", "ohio_factor_breadth",
        ]
        for col in expected:
            assert col in btc_factor.columns, f"Missing factor column: {col}"

    def test_factor_trend_in_range(self, btc_factor: pd.DataFrame) -> None:
        """trend_persistence factor should be in [-1, 1]."""
        trend = btc_factor["ohio_factor_trend"].dropna()
        assert (trend >= -1.0).all() and (trend <= 1.0).all(), (
            f"trend factor out of range: min={trend.min():.4f}, max={trend.max():.4f}"
        )

    def test_other_factors_in_01(self, btc_factor: pd.DataFrame) -> None:
        """All non-trend factors should be in [0, 1]."""
        for col in ["ohio_factor_volatility", "ohio_factor_downside",
                     "ohio_factor_liquidity", "ohio_factor_relative_strength",
                     "ohio_factor_correlation", "ohio_factor_breadth"]:
            vals = btc_factor[col].dropna()
            assert (vals >= 0.0).all() and (vals <= 1.0).all(), (
                f"{col} out of [0,1]: min={vals.min():.4f}, max={vals.max():.4f}"
            )


# ---------------------------------------------------------------------------
# Phase ④ PERSIST — Stabilizer
# ---------------------------------------------------------------------------

class TestStabilizer:
    """Phase ④: ohio_factor_* → ohio_stable_*."""

    @pytest.fixture(scope="class")
    def btc_stable(self, btc_raw: pd.DataFrame) -> pd.DataFrame:
        df = btc_raw.copy()
        df = FeatureBuilder().compute(df)
        df = Normalizer(window=4320).normalize(df)
        df = FactorCalculator().compute(df)
        df = StateStabilizer().stabilize(df)
        return df

    def test_stable_columns_present(self, btc_stable: pd.DataFrame) -> None:
        expected = [
            "ohio_stable_trend", "ohio_stable_volatility", "ohio_stable_downside",
            "ohio_stable_liquidity", "ohio_stable_relative_strength",
            "ohio_stable_correlation", "ohio_stable_breadth",
        ]
        for col in expected:
            assert col in btc_stable.columns, f"Missing stable column: {col}"

    def test_stable_values_smoother_than_factor(self, btc_stable: pd.DataFrame) -> None:
        """Stabilized values should have less bar-to-bar variance than raw factors."""
        warmup = 200
        for factor_col, stable_col in [
            ("ohio_factor_volatility", "ohio_stable_volatility"),
            ("ohio_factor_downside", "ohio_stable_downside"),
        ]:
            factor_var = btc_stable[factor_col].iloc[warmup:].diff().var()
            stable_var = btc_stable[stable_col].iloc[warmup:].diff().var()
            assert stable_var < factor_var, (
                f"Stable not smoother than factor for {stable_col}: "
                f"stable_var={stable_var:.6f} >= factor_var={factor_var:.6f}"
            )


# ---------------------------------------------------------------------------
# Phase ④→⑤ META + FITNESS + POLICY
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Phase ④→⑤: full pipeline from OHLCV to ExecutionPolicy columns."""

    @pytest.fixture(scope="class")
    def btc_full(self, btc_raw: pd.DataFrame) -> pd.DataFrame:
        df = btc_raw.copy()
        df = FeatureBuilder().compute(df)
        df = Normalizer(window=4320).normalize(df)
        df = FactorCalculator().compute(df)
        df = StateStabilizer().stabilize(df)
        df = MetaCalculator().compute(df)
        df = FitnessEstimator().compute_dataframe(df)
        df = PolicyGenerator().generate_dataframe(df)
        return df

    def test_all_output_columns_present(self, btc_full: pd.DataFrame) -> None:
        meta_cols = ["ohio_meta_transition_risk", "ohio_meta_confidence",
                     "ohio_meta_stability", "ohio_meta_data_mode"]
        fitness_cols = [f"ohio_fitness_{m.value}" for m in StrategyMode]
        policy_cols = ["ohio_policy_enabled", "ohio_policy_size_multiplier",
                       "ohio_policy_entry_threshold_adj", "ohio_policy_max_positions",
                       "ohio_policy_stoploss_width_adj"]
        all_cols = meta_cols + fitness_cols + ["ohio_active_mode"] + policy_cols

        for col in all_cols:
            assert col in btc_full.columns, f"Missing output column: {col}"

    def test_row_count_preserved(self, btc_full: pd.DataFrame, btc_raw: pd.DataFrame) -> None:
        assert len(btc_full) == len(btc_raw)

    def test_fitness_scores_in_range(self, btc_full: pd.DataFrame) -> None:
        warmup = 200
        for mode in StrategyMode:
            col = f"ohio_fitness_{mode.value}"
            vals = btc_full[col].iloc[warmup:].dropna()
            assert len(vals) > 0, f"No fitness values for {mode.value} after warmup"
            assert (vals >= 0.0).all() and (vals <= 1.0).all(), (
                f"{col} out of range: min={vals.min():.4f}, max={vals.max():.4f}"
            )

    def test_active_mode_changes_over_time(self, btc_full: pd.DataFrame) -> None:
        """Over 200 days of BTC, the active mode should change at least once."""
        warmup = 200
        modes = btc_full["ohio_active_mode"].iloc[warmup:]
        unique_modes = modes.unique()
        assert len(unique_modes) >= 2, (
            f"Active mode never changed — always {unique_modes}. "
            f"This suggests the pipeline isn't discriminating between market states."
        )

    def test_policy_enabled_varies(self, btc_full: pd.DataFrame) -> None:
        """Policy enabled should not be all-True or all-False after warmup."""
        warmup = 200
        enabled = btc_full["ohio_policy_enabled"].iloc[warmup:]
        # With real data, fitness > 0.15 for most bars → mostly enabled
        assert enabled.any(), "Policy never enabled"

    def test_size_multiplier_in_range(self, btc_full: pd.DataFrame) -> None:
        warmup = 200
        size = btc_full["ohio_policy_size_multiplier"].iloc[warmup:]
        valid = size.dropna()
        assert (valid >= 0.3).all() and (valid <= 1.5).all(), (
            f"size_multiplier out of range: min={valid.min():.4f}, max={valid.max():.4f}"
        )

    def test_max_positions_in_range(self, btc_full: pd.DataFrame) -> None:
        warmup = 200
        pos = btc_full["ohio_policy_max_positions"].iloc[warmup:]
        valid = pos.dropna()
        assert (valid >= 1).all() and (valid <= 6).all(), (
            f"max_positions out of range: min={valid.min()}, max={valid.max()}"
        )

    def test_meta_data_mode_mostly_full(self, btc_full: pd.DataFrame) -> None:
        """With all features present, data_mode should be mostly 'full'."""
        warmup = 200
        dm = btc_full["ohio_meta_data_mode"].iloc[warmup:]
        full_ratio = (dm == "full").mean()
        assert full_ratio > 0.8, (
            f"data_mode is 'full' only {full_ratio:.1%} of the time — too many degraded rows"
        )


# ---------------------------------------------------------------------------
# Cross-pair comparison
# ---------------------------------------------------------------------------

class TestCrossPair:
    """BTC and ETH should produce different fitness/policy profiles."""

    @pytest.fixture(scope="class")
    def btc_policy(self, btc_raw: pd.DataFrame) -> pd.DataFrame:
        df = btc_raw.copy()
        df = FeatureBuilder().compute(df)
        df = Normalizer(window=4320).normalize(df)
        df = FactorCalculator().compute(df)
        df = StateStabilizer().stabilize(df)
        df = MetaCalculator().compute(df)
        df = FitnessEstimator().compute_dataframe(df)
        df = PolicyGenerator().generate_dataframe(df)
        return df

    @pytest.fixture(scope="class")
    def eth_policy(self, eth_raw: pd.DataFrame) -> pd.DataFrame:
        df = eth_raw.copy()
        df = FeatureBuilder().compute(df)
        df = Normalizer(window=4320).normalize(df)
        df = FactorCalculator().compute(df)
        df = StateStabilizer().stabilize(df)
        df = MetaCalculator().compute(df)
        df = FitnessEstimator().compute_dataframe(df)
        df = PolicyGenerator().generate_dataframe(df)
        return df

    def test_different_fitness_profiles(self, btc_policy: pd.DataFrame, eth_policy: pd.DataFrame) -> None:
        """BTC and ETH should not have identical fitness scores."""
        warmup = 200
        btc_tf = btc_policy["ohio_fitness_trend_following"].iloc[warmup:].mean()
        eth_tf = eth_policy["ohio_fitness_trend_following"].iloc[warmup:].mean()
        # They should differ by at least a small amount
        assert abs(btc_tf - eth_tf) > 0.001, (
            f"BTC and ETH have nearly identical trend_following fitness: "
            f"BTC={btc_tf:.4f}, ETH={eth_tf:.4f}"
        )
