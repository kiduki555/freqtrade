"""Tests for FT-016 DrawdownController.

Covers tier classification, boundary conditions, properties, reset,
edge cases, and the no-Freqtrade-import constraint.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from freqtrade.ohio.core.portfolio_risk.drawdown_controller import (
    DrawdownConfig,
    DrawdownController,
    DrawdownTier,
)


# ---------------------------------------------------------------------------
# 1–2  Config
# ---------------------------------------------------------------------------

class TestDrawdownConfig:
    def test_default_values(self) -> None:
        cfg = DrawdownConfig()
        assert cfg.reduce_pct == 0.05
        assert cfg.block_pct == 0.10
        assert cfg.kill_pct == 0.15

    def test_custom_values(self) -> None:
        cfg = DrawdownConfig(reduce_pct=0.03, block_pct=0.08, kill_pct=0.12)
        assert cfg.reduce_pct == 0.03
        assert cfg.block_pct == 0.08
        assert cfg.kill_pct == 0.12


# ---------------------------------------------------------------------------
# 3–5  Initial state & peak tracking
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_initial_tier_is_normal(self) -> None:
        ctrl = DrawdownController()
        assert ctrl.tier == DrawdownTier.NORMAL
        assert ctrl.drawdown == 0.0

    def test_first_update_sets_peak(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        assert ctrl.peak_equity == 100_000

    def test_rising_equity_updates_peak(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        ctrl.update(110_000)
        assert ctrl.peak_equity == 110_000


# ---------------------------------------------------------------------------
# 6–9  Tier classification
# ---------------------------------------------------------------------------

class TestTierClassification:
    def test_3pct_drawdown_is_normal(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        tier = ctrl.update(97_000)  # 3% DD
        assert tier == DrawdownTier.NORMAL

    def test_7pct_drawdown_is_reduce(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        tier = ctrl.update(93_000)  # 7% DD
        assert tier == DrawdownTier.REDUCE

    def test_12pct_drawdown_is_block(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        tier = ctrl.update(88_000)  # 12% DD
        assert tier == DrawdownTier.BLOCK

    def test_16pct_drawdown_is_kill(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        tier = ctrl.update(84_000)  # 16% DD
        assert tier == DrawdownTier.KILL


# ---------------------------------------------------------------------------
# 10–11  Recovery
# ---------------------------------------------------------------------------

class TestRecovery:
    def test_recovery_from_reduce_to_normal(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        ctrl.update(93_000)  # REDUCE
        assert ctrl.tier == DrawdownTier.REDUCE
        tier = ctrl.update(100_000)  # back to peak
        assert tier == DrawdownTier.NORMAL

    def test_recovery_from_block_to_normal(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        ctrl.update(88_000)  # BLOCK
        assert ctrl.tier == DrawdownTier.BLOCK
        tier = ctrl.update(100_000)
        assert tier == DrawdownTier.NORMAL


# ---------------------------------------------------------------------------
# 12–13  Properties per tier
# ---------------------------------------------------------------------------

class TestProperties:
    @pytest.mark.parametrize(
        ("tier", "expected_scale"),
        [
            (DrawdownTier.NORMAL, 1.0),
            (DrawdownTier.REDUCE, 0.5),
            (DrawdownTier.BLOCK, 0.0),
            (DrawdownTier.KILL, 0.0),
        ],
    )
    def test_size_scale(self, tier: DrawdownTier, expected_scale: float) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        # Drive to desired tier
        equity_map = {
            DrawdownTier.NORMAL: 100_000,
            DrawdownTier.REDUCE: 93_000,
            DrawdownTier.BLOCK: 88_000,
            DrawdownTier.KILL: 84_000,
        }
        ctrl.update(equity_map[tier])
        assert ctrl.size_scale == expected_scale

    @pytest.mark.parametrize(
        ("tier", "expected_allowed"),
        [
            (DrawdownTier.NORMAL, True),
            (DrawdownTier.REDUCE, True),
            (DrawdownTier.BLOCK, False),
            (DrawdownTier.KILL, False),
        ],
    )
    def test_entry_allowed(self, tier: DrawdownTier, expected_allowed: bool) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        equity_map = {
            DrawdownTier.NORMAL: 100_000,
            DrawdownTier.REDUCE: 93_000,
            DrawdownTier.BLOCK: 88_000,
            DrawdownTier.KILL: 84_000,
        }
        ctrl.update(equity_map[tier])
        assert ctrl.entry_allowed is expected_allowed


# ---------------------------------------------------------------------------
# 14  Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_peak_and_restores_normal(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        ctrl.update(84_000)  # KILL
        assert ctrl.tier == DrawdownTier.KILL

        ctrl.reset(90_000)
        assert ctrl.peak_equity == 90_000
        assert ctrl.drawdown == 0.0
        assert ctrl.tier == DrawdownTier.NORMAL


# ---------------------------------------------------------------------------
# 15–16  Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_equity(self) -> None:
        ctrl = DrawdownController()
        tier = ctrl.update(0.0)
        # peak=0, equity=0 → drawdown=0 → NORMAL
        assert tier == DrawdownTier.NORMAL
        assert ctrl.drawdown == 0.0

    def test_negative_equity_raises(self) -> None:
        ctrl = DrawdownController()
        with pytest.raises(ValueError, match="must be >= 0"):
            ctrl.update(-1.0)


# ---------------------------------------------------------------------------
# 17–19  Exact boundaries (>= threshold → next tier)
# ---------------------------------------------------------------------------

class TestExactBoundaries:
    def test_exactly_5pct_is_reduce(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        tier = ctrl.update(95_000)  # exactly 5%
        assert tier == DrawdownTier.REDUCE

    def test_exactly_10pct_is_block(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        tier = ctrl.update(90_000)  # exactly 10%
        assert tier == DrawdownTier.BLOCK

    def test_exactly_15pct_is_kill(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        tier = ctrl.update(85_000)  # exactly 15%
        assert tier == DrawdownTier.KILL


# ---------------------------------------------------------------------------
# 20  Sequential mixed updates
# ---------------------------------------------------------------------------

class TestSequentialUpdates:
    def test_mixed_up_down_sequence(self) -> None:
        ctrl = DrawdownController()
        ctrl.update(100_000)
        assert ctrl.tier == DrawdownTier.NORMAL

        ctrl.update(105_000)  # new peak
        assert ctrl.peak_equity == 105_000

        ctrl.update(99_750)  # 5% of 105k → REDUCE
        assert ctrl.tier == DrawdownTier.REDUCE

        ctrl.update(102_000)  # recovery, but still below peak
        # DD = (105k - 102k)/105k ≈ 2.86% → NORMAL
        assert ctrl.tier == DrawdownTier.NORMAL

        ctrl.update(89_250)  # 15% of 105k → KILL
        assert ctrl.tier == DrawdownTier.KILL


# ---------------------------------------------------------------------------
# 21  No Freqtrade imports (AST check)
# ---------------------------------------------------------------------------

class TestNoFreqtradeImports:
    def test_module_has_no_freqtrade_imports(self) -> None:
        src = pathlib.Path(__file__).resolve().parents[2] / (
            "freqtrade/ohio/core/portfolio_risk/drawdown_controller.py"
        )
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("freqtrade"), (
                        f"Forbidden import: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("freqtrade"), (
                        f"Forbidden import from: {node.module}"
                    )
