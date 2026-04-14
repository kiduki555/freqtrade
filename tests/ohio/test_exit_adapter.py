"""Tests for FT-015: ExitAdapter — compute_stoploss & compute_exit."""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from freqtrade.ohio.adapters.freqtrade.exit_adapter import compute_exit, compute_stoploss


# ---------------------------------------------------------------------------
# Lightweight fakes (no Freqtrade imports needed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeProfile:
    stoploss_range: tuple[float, float]


@dataclass(frozen=True)
class FakePolicy:
    enabled: bool = True
    size_multiplier: float = 0.9
    stoploss_width_adj: float = 0.0


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _trend_profile() -> FakeProfile:
    """Trend-following: wide stop [-0.12, -0.08]."""
    return FakeProfile(stoploss_range=(-0.12, -0.08))


def _defensive_profile() -> FakeProfile:
    """Defensive: tight stop [-0.05, -0.03]."""
    return FakeProfile(stoploss_range=(-0.05, -0.03))


def _default_policy(**overrides) -> FakePolicy:
    return FakePolicy(**overrides)


# ===================================================================
# compute_stoploss
# ===================================================================

class TestComputeStoplossTrendFollowing:
    """Trend-following profile -> base stoploss around -0.12 to -0.08."""

    def test_base_within_range(self):
        sl = compute_stoploss(_trend_profile(), _default_policy(), 0.0, 0.0)
        assert -0.12 <= sl <= -0.08

    def test_low_fitness_widens(self):
        # fitness_score=0 -> lerp returns sl_wide
        sl = compute_stoploss(
            _trend_profile(),
            _default_policy(),
            0.0, 0.0,
            fitness_score=0.0,
        )
        assert sl == pytest.approx(-0.12, abs=1e-9)

    def test_high_fitness_tightens(self):
        # fitness_score=1 -> lerp returns sl_tight
        sl = compute_stoploss(
            _trend_profile(),
            _default_policy(),
            0.0, 0.0,
            fitness_score=1.0,
        )
        assert sl == pytest.approx(-0.08, abs=1e-9)


class TestComputeStoplossDefensive:
    """Defensive profile -> tight -0.05 to -0.03."""

    def test_base_within_range(self):
        sl = compute_stoploss(_defensive_profile(), _default_policy(), 0.0, 0.0)
        assert -0.05 <= sl <= -0.03


class TestComputeStoplossPolicyDisabled:
    def test_returns_hard_floor(self):
        sl = compute_stoploss(
            _trend_profile(),
            _default_policy(enabled=False),
            0.0, 0.0,
        )
        assert sl == -0.20


class TestComputeStoplossWidthAdj:
    def test_positive_adj_widens(self):
        """Positive adj moves stop more negative (wider)."""
        base = compute_stoploss(
            _trend_profile(), _default_policy(stoploss_width_adj=0.0), 0.0, 0.0,
        )
        wider = compute_stoploss(
            _trend_profile(), _default_policy(stoploss_width_adj=-0.01), 0.0, 0.0,
        )
        assert wider < base  # more negative = wider

    def test_negative_adj_tightens(self):
        """Negative adj (toward 0) tightens."""
        base = compute_stoploss(
            _trend_profile(), _default_policy(stoploss_width_adj=0.0), 0.0, 0.0,
        )
        tighter = compute_stoploss(
            _trend_profile(), _default_policy(stoploss_width_adj=0.01), 0.0, 0.0,
        )
        assert tighter > base  # less negative = tighter


class TestComputeStoplossTransitionRisk:
    def test_high_risk_tightens(self):
        normal = compute_stoploss(_trend_profile(), _default_policy(), 0.0, 0.5)
        tight = compute_stoploss(_trend_profile(), _default_policy(), 0.0, 0.8)
        assert tight > normal  # closer to zero


class TestComputeStoplossTrailing:
    def test_trailing_at_2pct(self):
        sl = compute_stoploss(_trend_profile(), _default_policy(), 0.02, 0.0)
        # At exactly 2%, profit lock not triggered (needs > 0.02)
        sl_above = compute_stoploss(_trend_profile(), _default_policy(), 0.025, 0.0)
        # trailing should tighten: max(base, -(0.025*0.5)) = max(base, -0.0125)
        assert sl_above >= -0.0125

    def test_trailing_at_10pct(self):
        sl = compute_stoploss(_trend_profile(), _default_policy(), 0.10, 0.0)
        # trailing: max(base, -(0.10*0.5)) = max(base, -0.05)
        # base is around -0.10, trailing lock at -0.05 wins
        assert sl == pytest.approx(-0.05, abs=0.005)


class TestComputeStoplossClamp:
    def test_never_wider_than_hard_floor(self):
        # Extreme: very wide range + negative adj
        profile = FakeProfile(stoploss_range=(-0.30, -0.25))
        sl = compute_stoploss(profile, _default_policy(stoploss_width_adj=-0.01), 0.0, 0.0)
        assert sl >= -0.20

    def test_never_tighter_than_minus_001(self):
        # Extreme trailing profit should be clamped
        sl = compute_stoploss(
            _defensive_profile(),
            _default_policy(stoploss_width_adj=0.01, size_multiplier=1.5),
            0.50,  # 50% profit -> trailing = -0.25, but base is tight
            0.0,
        )
        assert sl <= -0.01


# ===================================================================
# compute_exit
# ===================================================================

class TestComputeExitKillSwitch:
    def test_kill_switch_active(self):
        assert compute_exit(5, 0.01, 0.50, 0.30, True) == "ohio_kill_switch"


class TestComputeExitTimeStop:
    def test_triggers_at_12_bars_low_fitness(self):
        assert compute_exit(12, 0.0, 0.10, 0.0, False) == "ohio_time_exit"

    def test_no_trigger_at_11_bars(self):
        assert compute_exit(11, 0.0, 0.10, 0.0, False) is None

    def test_no_trigger_high_fitness(self):
        assert compute_exit(12, 0.0, 0.50, 0.0, False) is None


class TestComputeExitRegime:
    def test_triggers_at_085(self):
        assert compute_exit(5, 0.0, 0.50, 0.85, False) == "ohio_regime_exit"

    def test_no_trigger_at_075(self):
        assert compute_exit(5, 0.0, 0.50, 0.75, False) is None


class TestComputeExitProfitPreserve:
    def test_triggers_at_4pct_low_fitness(self):
        assert compute_exit(5, 0.04, 0.25, 0.0, False) == "ohio_profit_preserve"

    def test_no_trigger_at_2pct(self):
        assert compute_exit(5, 0.02, 0.25, 0.0, False) is None

    def test_no_trigger_high_fitness(self):
        assert compute_exit(5, 0.04, 0.50, 0.0, False) is None


class TestComputeExitNormal:
    def test_no_exit(self):
        assert compute_exit(5, 0.01, 0.60, 0.30, False) is None


class TestComputeExitPriority:
    def test_kill_beats_all(self):
        # All conditions true simultaneously
        result = compute_exit(15, 0.05, 0.10, 0.90, True)
        assert result == "ohio_kill_switch"

    def test_time_beats_regime_and_profit(self):
        result = compute_exit(15, 0.05, 0.10, 0.90, False)
        assert result == "ohio_time_exit"

    def test_regime_beats_profit(self):
        result = compute_exit(5, 0.05, 0.10, 0.90, False)
        assert result == "ohio_regime_exit"


# ===================================================================
# No Freqtrade imports (AST check)
# ===================================================================

class TestNoFreqtradeImports:
    def test_exit_adapter_has_no_ft_imports(self):
        src_path = (
            Path(__file__).resolve().parent.parent.parent
            / "freqtrade" / "ohio" / "adapters" / "freqtrade" / "exit_adapter.py"
        )
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("freqtrade."), (
                        f"Forbidden freqtrade import: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("freqtrade."):
                    # Allow ohio subpackage imports
                    assert node.module.startswith("freqtrade.ohio"), (
                        f"Forbidden freqtrade import: from {node.module}"
                    )
