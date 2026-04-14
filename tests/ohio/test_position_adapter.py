"""Tests for FT-013 PositionAdapter.

Covers fractional_kelly, compute_stake, compute_leverage,
boundary conditions, and the no-Freqtrade-import constraint.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from freqtrade.ohio.adapters.freqtrade.position_adapter import (
    compute_leverage,
    compute_stake,
    fractional_kelly,
)
from freqtrade.ohio.core.domain.models import ExecutionPolicy, StrategyMode


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes (no Freqtrade dependency)
# ---------------------------------------------------------------------------

def _policy(enabled: bool = True, size_multiplier: float = 1.0) -> ExecutionPolicy:
    return ExecutionPolicy(
        strategy_mode=StrategyMode.TREND_FOLLOWING,
        enabled=enabled,
        size_multiplier=size_multiplier,
        entry_threshold_adj=0.0,
        max_positions=3,
        stoploss_width_adj=0.0,
    )


class _FakeProfile:
    """Minimal stand-in for StrategyProfile with only leverage_range."""

    def __init__(self, leverage_range: tuple[float, float]) -> None:
        self.leverage_range = leverage_range


# ---------------------------------------------------------------------------
# 1–4  fractional_kelly
# ---------------------------------------------------------------------------

class TestFractionalKelly:
    def test_known_output(self) -> None:
        """win_prob=0.6, rr=2.5, fraction=0.25 → deterministic value."""
        result = fractional_kelly(0.6, risk_reward=2.5, fraction=0.25)
        # kelly_full = 0.6 - 0.4/2.5 = 0.6 - 0.16 = 0.44
        # fractional = 0.44 * 0.25 = 0.11
        assert result == pytest.approx(0.11)

    def test_below_breakeven(self) -> None:
        """win_prob=0.3 with rr=2.5 → negative full Kelly → clamped to 0."""
        # kelly_full = 0.3 - 0.7/2.5 = 0.3 - 0.28 = 0.02 (positive!)
        # Actually 0.3 is above breakeven for rr=2.5. Use rr=1.0 instead:
        # win_prob=0.3, rr=1.0: kelly_full = 0.3 - 0.7 = -0.4 → 0.0
        result = fractional_kelly(0.3, risk_reward=1.0, fraction=0.25)
        assert result == 0.0

    def test_win_prob_zero(self) -> None:
        """win_prob=0.0 → always negative → 0.0."""
        result = fractional_kelly(0.0, risk_reward=2.5, fraction=0.25)
        assert result == 0.0

    def test_win_prob_one(self) -> None:
        """win_prob=1.0 → maximum Kelly fraction."""
        result = fractional_kelly(1.0, risk_reward=2.5, fraction=0.25)
        # kelly_full = 1.0 - 0.0/2.5 = 1.0
        # fractional = 1.0 * 0.25 = 0.25
        assert result == pytest.approx(0.25)

    def test_risk_reward_zero_raises(self) -> None:
        """risk_reward=0 → division by zero → ValueError."""
        with pytest.raises(ValueError, match="risk_reward must be > 0"):
            fractional_kelly(0.5, risk_reward=0.0)

    def test_risk_reward_negative_raises(self) -> None:
        """risk_reward < 0 → ValueError."""
        with pytest.raises(ValueError, match="risk_reward must be > 0"):
            fractional_kelly(0.5, risk_reward=-1.0)

    def test_win_prob_out_of_range_raises(self) -> None:
        """win_prob outside [0, 1] → ValueError."""
        with pytest.raises(ValueError, match="win_prob must be in"):
            fractional_kelly(1.5, risk_reward=2.5)
        with pytest.raises(ValueError, match="win_prob must be in"):
            fractional_kelly(-0.1, risk_reward=2.5)


# ---------------------------------------------------------------------------
# 5–11  compute_stake
# ---------------------------------------------------------------------------

class TestComputeStake:
    def test_disabled_policy_returns_min(self) -> None:
        result = compute_stake(
            _policy(enabled=False),
            fitness_score=0.9,
            base_stake=1000.0,
            min_stake=10.0,
            max_stake=500.0,
        )
        assert result == 10.0

    def test_high_fitness_large_stake(self) -> None:
        result = compute_stake(
            _policy(enabled=True, size_multiplier=1.0),
            fitness_score=0.9,
            base_stake=1000.0,
            min_stake=10.0,
            max_stake=500.0,
        )
        # kelly = fractional_kelly(0.9, 2.5, 0.25)
        # kelly_full = 0.9 - 0.1/2.5 = 0.9 - 0.04 = 0.86
        # kelly = 0.86 * 0.25 = 0.215
        # stake = 1000 * 0.215 * 1.0 * 1.0 = 215.0
        assert result == pytest.approx(215.0)

    def test_low_fitness_small_stake(self) -> None:
        result = compute_stake(
            _policy(enabled=True, size_multiplier=1.0),
            fitness_score=0.45,
            base_stake=1000.0,
            min_stake=10.0,
            max_stake=500.0,
        )
        # kelly_full = 0.45 - 0.55/2.5 = 0.45 - 0.22 = 0.23
        # kelly = 0.23 * 0.25 = 0.0575
        # stake = 1000 * 0.0575 * 1.0 * 1.0 = 57.5
        assert result == pytest.approx(57.5)

    def test_clamped_to_min_stake(self) -> None:
        result = compute_stake(
            _policy(enabled=True, size_multiplier=0.01),
            fitness_score=0.5,
            base_stake=100.0,
            min_stake=10.0,
            max_stake=500.0,
        )
        # Very small multiplier → result below min_stake → clamped
        assert result == 10.0

    def test_clamped_to_max_stake(self) -> None:
        result = compute_stake(
            _policy(enabled=True, size_multiplier=10.0),
            fitness_score=0.95,
            base_stake=10000.0,
            min_stake=10.0,
            max_stake=500.0,
        )
        assert result == 500.0

    def test_dd_scale_halves_result(self) -> None:
        full = compute_stake(
            _policy(enabled=True),
            fitness_score=0.8,
            base_stake=1000.0,
            min_stake=1.0,
            max_stake=10000.0,
            dd_scale=1.0,
        )
        half = compute_stake(
            _policy(enabled=True),
            fitness_score=0.8,
            base_stake=1000.0,
            min_stake=1.0,
            max_stake=10000.0,
            dd_scale=0.5,
        )
        assert half == pytest.approx(full * 0.5)

    def test_dd_scale_zero_returns_min(self) -> None:
        result = compute_stake(
            _policy(enabled=True),
            fitness_score=0.9,
            base_stake=1000.0,
            min_stake=10.0,
            max_stake=500.0,
            dd_scale=0.0,
        )
        assert result == 10.0


# ---------------------------------------------------------------------------
# 12–18  compute_leverage
# ---------------------------------------------------------------------------

class TestComputeLeverage:
    def test_fitness_zero_returns_lev_min(self) -> None:
        result = compute_leverage(
            _FakeProfile(leverage_range=(1.0, 5.0)),  # type: ignore[arg-type]
            fitness_score=0.0,
            max_leverage=10.0,
        )
        assert result == 1.0

    def test_fitness_one_returns_lev_max(self) -> None:
        result = compute_leverage(
            _FakeProfile(leverage_range=(1.0, 5.0)),  # type: ignore[arg-type]
            fitness_score=1.0,
            max_leverage=10.0,
        )
        assert result == 5.0

    def test_fitness_half_returns_midpoint(self) -> None:
        result = compute_leverage(
            _FakeProfile(leverage_range=(2.0, 6.0)),  # type: ignore[arg-type]
            fitness_score=0.5,
            max_leverage=10.0,
        )
        assert result == 4.0

    def test_clamped_to_max_leverage(self) -> None:
        result = compute_leverage(
            _FakeProfile(leverage_range=(1.0, 20.0)),  # type: ignore[arg-type]
            fitness_score=1.0,
            max_leverage=5.0,
        )
        assert result == 5.0

    def test_never_below_one(self) -> None:
        result = compute_leverage(
            _FakeProfile(leverage_range=(0.5, 3.0)),  # type: ignore[arg-type]
            fitness_score=0.0,
            max_leverage=10.0,
        )
        assert result == 1.0

    def test_defensive_profile_always_one(self) -> None:
        for fitness in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = compute_leverage(
                _FakeProfile(leverage_range=(1.0, 1.0)),  # type: ignore[arg-type]
                fitness_score=fitness,
                max_leverage=10.0,
            )
            assert result == 1.0


# ---------------------------------------------------------------------------
# 18  No Freqtrade imports (AST check)
# ---------------------------------------------------------------------------

class TestNoFreqtradeImports:
    def test_no_freqtrade_framework_imports(self) -> None:
        """Verify position_adapter.py imports nothing from freqtrade
        except the ohio subpackage."""
        src = pathlib.Path(__file__).resolve().parents[2] / (
            "freqtrade" / pathlib.Path("ohio/adapters/freqtrade/position_adapter.py")
        )
        tree = ast.parse(src.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("freqtrade") and not node.module.startswith(
                    "freqtrade.ohio"
                ):
                    pytest.fail(
                        f"Forbidden import: 'from {node.module} import ...'"
                    )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("freqtrade") and not alias.name.startswith(
                        "freqtrade.ohio"
                    ):
                        pytest.fail(f"Forbidden import: 'import {alias.name}'")
