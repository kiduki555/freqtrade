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
class FakeProfileV2:
    """Profile with ATR + Chandelier fields."""
    stoploss_range: tuple[float, float]
    atr_stop_direction: str = "tighten"
    atr_scale_cap: float = 2.0
    chandelier_enabled: bool = False
    chandelier_multiplier: float = 2.5
    chandelier_activation: float = 0.02


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


def _trend_profile_v2() -> FakeProfileV2:
    return FakeProfileV2(
        stoploss_range=(-0.12, -0.08),
        atr_stop_direction="tighten",
        chandelier_enabled=True,
        chandelier_multiplier=2.5,
        chandelier_activation=0.02,
    )


def _mr_profile_v2() -> FakeProfileV2:
    return FakeProfileV2(
        stoploss_range=(-0.08, -0.05),
        atr_stop_direction="widen",
        chandelier_enabled=False,
    )


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

class TestProfileNewFields:
    def test_trend_following_has_atr_fields(self):
        from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles
        from freqtrade.ohio.core.domain.models import StrategyMode
        profiles = load_default_profiles()
        tf = profiles[StrategyMode.TREND_FOLLOWING]
        assert tf.atr_stop_direction == "tighten"
        assert tf.atr_scale_cap == 2.0
        assert tf.chandelier_enabled is True
        assert tf.chandelier_multiplier == 2.5
        assert tf.chandelier_activation == 0.02

    def test_mean_reversion_widens(self):
        from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles
        from freqtrade.ohio.core.domain.models import StrategyMode
        profiles = load_default_profiles()
        mr = profiles[StrategyMode.MEAN_REVERSION]
        assert mr.atr_stop_direction == "widen"
        assert mr.chandelier_enabled is False

    def test_breakout_chandelier(self):
        from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles
        from freqtrade.ohio.core.domain.models import StrategyMode
        profiles = load_default_profiles()
        bo = profiles[StrategyMode.BREAKOUT]
        assert bo.chandelier_enabled is True
        assert bo.chandelier_multiplier == 2.0
        assert bo.chandelier_activation == 0.015

    def test_defensive_no_chandelier(self):
        from freqtrade.ohio.core.strategy_router.strategy_profile import load_default_profiles
        from freqtrade.ohio.core.domain.models import StrategyMode
        profiles = load_default_profiles()
        d = profiles[StrategyMode.DEFENSIVE]
        assert d.atr_stop_direction == "tighten"
        assert d.chandelier_enabled is False


class TestAtrScalingTighten:
    def test_high_vol_tightens_trend(self):
        """atr_scale=1.5 + trend_following (tighten) → stop closer to zero."""
        base = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=1.0,
        )
        tightened = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=1.5,
        )
        assert tightened > base  # closer to zero = tighter


class TestAtrScalingWiden:
    def test_high_vol_widens_mr(self):
        """atr_scale=1.5 + mean_reversion (widen) → stop further from zero."""
        base = compute_stoploss(
            _mr_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=1.0,
        )
        widened = compute_stoploss(
            _mr_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=1.5,
        )
        assert widened < base  # further from zero = wider


class TestAtrScaleClamped:
    def test_extreme_scale_clamped(self):
        """atr_scale=5.0 should behave same as atr_scale=2.0 (cap)."""
        at_cap = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=2.0,
        )
        extreme = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.0,
            fitness_score=0.5, atr_scale=5.0,
        )
        assert extreme == pytest.approx(at_cap, abs=1e-9)


class TestBackwardCompat:
    def test_scale_1_matches_original(self):
        """atr_scale=1.0 produces identical result to no-ATR call."""
        original = compute_stoploss(
            _trend_profile(), _default_policy(), 0.0, 0.3, fitness_score=0.5,
        )
        with_atr = compute_stoploss(
            _trend_profile_v2(), _default_policy(), 0.0, 0.3,
            fitness_score=0.5, atr_scale=1.0,
        )
        assert with_atr == pytest.approx(original, abs=1e-9)


class TestChandelierActivation:
    def test_below_activation_uses_base(self):
        """Profit below chandelier_activation → no chandelier, no profit lock, just base."""
        sl = compute_stoploss(
            _trend_profile_v2(), _default_policy(),
            current_profit=0.01,  # below 0.02 activation
            transition_risk=0.0,
            fitness_score=0.5,
            atr_scale=1.0,
            atr_ratio=0.012,
        )
        base = compute_stoploss(
            _trend_profile_v2(), _default_policy(),
            current_profit=0.0,
            transition_risk=0.0,
            fitness_score=0.5,
            atr_scale=1.0,
            atr_ratio=0.012,
        )
        assert sl == pytest.approx(base, abs=1e-9)

    def test_above_activation_uses_chandelier(self):
        """Profit above activation → chandelier trailing active."""
        sl = compute_stoploss(
            _trend_profile_v2(), _default_policy(),
            current_profit=0.05,  # above 0.02 activation
            transition_risk=0.0,
            fitness_score=0.5,
            atr_scale=1.0,
            atr_ratio=0.012,  # chandelier = -(2.5 * 0.012) = -0.03
        )
        # Chandelier SL = -0.03
        # Base SL (trend, fitness 0.5) ~= -0.10
        # max(-0.10, -0.03) = -0.03 (chandelier wins, tighter)
        assert sl == pytest.approx(-0.03, abs=0.005)


class TestChandelierTighterWins:
    def test_chandelier_overrides_base_when_tighter(self):
        """When chandelier SL is tighter than base, chandelier is used."""
        sl = compute_stoploss(
            _trend_profile_v2(), _default_policy(),
            current_profit=0.05,
            transition_risk=0.0,
            fitness_score=0.0,   # low fitness → wide base (sl_wide = -0.12)
            atr_scale=1.0,
            atr_ratio=0.010,     # chandelier = -(2.5 * 0.01) = -0.025
        )
        # max(-0.12, -0.025) = -0.025 → chandelier wins
        assert sl == pytest.approx(-0.025, abs=0.005)


class TestChandelierDisabledMR:
    def test_mr_uses_profit_lock_not_chandelier(self):
        """MR mode (chandelier_enabled=False) uses profit lock at 2%+."""
        sl = compute_stoploss(
            _mr_profile_v2(), _default_policy(),
            current_profit=0.10,  # 10% profit
            transition_risk=0.0,
            fitness_score=0.5,
            atr_scale=1.0,
            atr_ratio=0.012,
        )
        # MR: chandelier disabled → profit lock: -(0.10 * 0.50) = -0.05
        # base_sl ~= -0.065
        # max(-0.065, -0.05) = -0.05 → profit lock
        assert sl == pytest.approx(-0.05, abs=0.005)


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
