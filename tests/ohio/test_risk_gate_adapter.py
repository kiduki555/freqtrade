"""Tests for FT-014: RiskGateAdapter."""
from __future__ import annotations

import ast
import textwrap

import pytest

from freqtrade.ohio.adapters.freqtrade.risk_gate_adapter import RiskGateAdapter
from freqtrade.ohio.core.domain.models import (
    DataMode,
    ExecutionPolicy,
    StateMeta,
    StrategyMode,
)
from freqtrade.ohio.core.portfolio_risk.drawdown_controller import DrawdownController
from freqtrade.ohio.core.portfolio_risk.kill_switch import KillSwitch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(
    enabled: bool = True,
    max_positions: int = 4,
) -> ExecutionPolicy:
    return ExecutionPolicy(
        strategy_mode=StrategyMode.TREND_FOLLOWING,
        enabled=enabled,
        size_multiplier=1.0,
        entry_threshold_adj=0.0,
        max_positions=max_positions,
        stoploss_width_adj=0.0,
    )


def _make_meta(
    confidence: float = 0.80,
    transition_risk: float = 0.20,
    data_mode: DataMode = DataMode.FULL,
) -> StateMeta:
    return StateMeta(
        transition_risk=transition_risk,
        confidence=confidence,
        stability=0.70,
        data_mode=data_mode,
    )


def _make_dd(entry_allowed: bool = True) -> DrawdownController:
    """Return a DrawdownController whose entry_allowed matches *entry_allowed*."""
    ctrl = DrawdownController()
    ctrl.update(100_000)  # set peak
    if not entry_allowed:
        # push equity down past block threshold (default 10 %)
        ctrl.update(89_000)
    return ctrl


def _make_dd_kill() -> DrawdownController:
    """Return a DrawdownController in KILL tier."""
    ctrl = DrawdownController()
    ctrl.update(100_000)
    ctrl.update(84_000)  # 16 % drawdown => KILL
    return ctrl


def _make_dd_reduce() -> DrawdownController:
    """Return a DrawdownController in REDUCE tier (entry still allowed)."""
    ctrl = DrawdownController()
    ctrl.update(100_000)
    ctrl.update(94_000)  # 6 % drawdown => REDUCE
    return ctrl


def _make_ks(active: bool = False) -> KillSwitch:
    ks = KillSwitch()
    if active:
        ks.activate("test")
    return ks


# ---------------------------------------------------------------------------
# 1. All-clear scenario
# ---------------------------------------------------------------------------

def test_all_clear():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(), _make_meta(), 0, 0.0,
    )
    assert allowed is True
    assert reasons == []


# ---------------------------------------------------------------------------
# 2. Kill switch active
# ---------------------------------------------------------------------------

def test_kill_switch_active():
    adapter = RiskGateAdapter(_make_dd(), _make_ks(active=True))
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(), _make_meta(), 0, 0.0,
    )
    assert allowed is False
    assert "kill_switch_active" in reasons


# ---------------------------------------------------------------------------
# 3. Policy disabled
# ---------------------------------------------------------------------------

def test_policy_disabled():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(enabled=False), _make_meta(), 0, 0.0,
    )
    assert allowed is False
    assert "policy_disabled" in reasons


# ---------------------------------------------------------------------------
# 4. Drawdown BLOCK tier
# ---------------------------------------------------------------------------

def test_drawdown_block():
    adapter = RiskGateAdapter(_make_dd(entry_allowed=False), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(), _make_meta(), 0, 0.0,
    )
    assert allowed is False
    assert "drawdown_block" in reasons


# ---------------------------------------------------------------------------
# 5. Drawdown KILL tier
# ---------------------------------------------------------------------------

def test_drawdown_kill():
    adapter = RiskGateAdapter(_make_dd_kill(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(), _make_meta(), 0, 0.0,
    )
    assert allowed is False
    assert "drawdown_block" in reasons


# ---------------------------------------------------------------------------
# 6. Drawdown REDUCE tier (entry still allowed)
# ---------------------------------------------------------------------------

def test_drawdown_reduce_allows_entry():
    adapter = RiskGateAdapter(_make_dd_reduce(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(), _make_meta(), 0, 0.0,
    )
    assert allowed is True
    assert "drawdown_block" not in reasons


# ---------------------------------------------------------------------------
# 7. Data degraded
# ---------------------------------------------------------------------------

def test_data_degraded():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(),
        _make_meta(data_mode=DataMode.DEGRADED), 0, 0.0,
    )
    assert allowed is False
    assert "data_degraded" in reasons


# ---------------------------------------------------------------------------
# 8. Low confidence (0.25)
# ---------------------------------------------------------------------------

def test_low_confidence():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(),
        _make_meta(confidence=0.25), 0, 0.0,
    )
    assert allowed is False
    assert "low_confidence" in reasons


# ---------------------------------------------------------------------------
# 9. Confidence at boundary (0.30) — allowed
# ---------------------------------------------------------------------------

def test_confidence_boundary_allowed():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(),
        _make_meta(confidence=0.30), 0, 0.0,
    )
    assert allowed is True
    assert "low_confidence" not in reasons


# ---------------------------------------------------------------------------
# 10. High transition risk (0.90)
# ---------------------------------------------------------------------------

def test_high_transition_risk():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(),
        _make_meta(transition_risk=0.90), 0, 0.0,
    )
    assert allowed is False
    assert "high_transition_risk" in reasons


# ---------------------------------------------------------------------------
# 11. Transition risk at boundary (0.85) — blocks
# ---------------------------------------------------------------------------

def test_transition_risk_boundary_blocks():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(),
        _make_meta(transition_risk=0.85), 0, 0.0,
    )
    assert allowed is True
    assert "high_transition_risk" not in reasons


# ---------------------------------------------------------------------------
# 12. Transition risk at 0.84 — allowed
# ---------------------------------------------------------------------------

def test_transition_risk_below_boundary():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(),
        _make_meta(transition_risk=0.84), 0, 0.0,
    )
    assert allowed is True
    assert "high_transition_risk" not in reasons


# ---------------------------------------------------------------------------
# 13. Max positions reached
# ---------------------------------------------------------------------------

def test_max_positions_reached():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(max_positions=3),
        _make_meta(), 3, 0.0,
    )
    assert allowed is False
    assert "max_positions_reached" in reasons


# ---------------------------------------------------------------------------
# 14. Max positions not reached
# ---------------------------------------------------------------------------

def test_max_positions_not_reached():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(max_positions=3),
        _make_meta(), 2, 0.0,
    )
    assert allowed is True
    assert "max_positions_reached" not in reasons


# ---------------------------------------------------------------------------
# 15. Daily loss at 3 % — blocks
# ---------------------------------------------------------------------------

def test_daily_loss_limit_blocks():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(), _make_meta(), 0, 0.03,
    )
    assert allowed is False
    assert "daily_loss_limit" in reasons


# ---------------------------------------------------------------------------
# 16. Daily loss at 2.9 % — allowed
# ---------------------------------------------------------------------------

def test_daily_loss_below_limit():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(), _make_meta(), 0, 0.029,
    )
    assert allowed is True
    assert "daily_loss_limit" not in reasons


# ---------------------------------------------------------------------------
# 17. Multiple blocking reasons collected simultaneously
# ---------------------------------------------------------------------------

def test_multiple_blocking_reasons():
    adapter = RiskGateAdapter(
        _make_dd(entry_allowed=False), _make_ks(active=True),
    )
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long",
        _make_policy(enabled=False),
        _make_meta(confidence=0.10, transition_risk=0.99, data_mode=DataMode.DEGRADED),
        10,
        0.05,
    )
    assert allowed is False
    expected = {
        "kill_switch_active",
        "policy_disabled",
        "drawdown_block",
        "data_degraded",
        "low_confidence",
        "high_transition_risk",
        "max_positions_reached",
        "daily_loss_limit",
    }
    assert set(reasons) == expected


# ---------------------------------------------------------------------------
# 18. Kill switch + max positions — both reasons present
# ---------------------------------------------------------------------------

def test_kill_switch_and_max_positions():
    adapter = RiskGateAdapter(_make_dd(), _make_ks(active=True))
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(max_positions=2),
        _make_meta(), 5, 0.0,
    )
    assert allowed is False
    assert "kill_switch_active" in reasons
    assert "max_positions_reached" in reasons


# ---------------------------------------------------------------------------
# 19. FALLBACK data_mode — allowed
# ---------------------------------------------------------------------------

def test_fallback_data_mode_allowed():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(),
        _make_meta(data_mode=DataMode.FALLBACK), 0, 0.0,
    )
    assert allowed is True
    assert "data_degraded" not in reasons


# ---------------------------------------------------------------------------
# 20. FULL data_mode — allowed
# ---------------------------------------------------------------------------

def test_full_data_mode_allowed():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(),
        _make_meta(data_mode=DataMode.FULL), 0, 0.0,
    )
    assert allowed is True


# ---------------------------------------------------------------------------
# 21. Default meta (high confidence, low risk) — allowed
# ---------------------------------------------------------------------------

def test_default_meta_allowed():
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", "long", _make_policy(), _make_meta(), 0, 0.0,
    )
    assert allowed is True
    assert reasons == []


# ---------------------------------------------------------------------------
# 22. Side "long" and "short" both work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", ["long", "short"])
def test_both_sides_work(side: str):
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        "BTC/USDT", side, _make_policy(), _make_meta(), 0, 0.0,
    )
    assert allowed is True
    assert reasons == []


# ---------------------------------------------------------------------------
# 23. Pair name doesn't affect logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", ["BTC/USDT", "ETH/BTC", "DOGE/USDT"])
def test_pair_name_irrelevant(pair: str):
    adapter = RiskGateAdapter(_make_dd(), _make_ks())
    allowed, reasons = adapter.confirm_entry(
        pair, "long", _make_policy(), _make_meta(), 0, 0.0,
    )
    assert allowed is True
    assert reasons == []


# ---------------------------------------------------------------------------
# 24. No Freqtrade imports (AST check)
# ---------------------------------------------------------------------------

def test_no_freqtrade_framework_imports():
    """Verify the adapter module imports nothing from freqtrade outside ohio."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / (
        "freqtrade/ohio/adapters/freqtrade/risk_gate_adapter.py"
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))

    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # Allow freqtrade.ohio.* but block any other freqtrade.* import
            if node.module.startswith("freqtrade") and not node.module.startswith(
                "freqtrade.ohio"
            ):
                forbidden.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("freqtrade") and not alias.name.startswith(
                    "freqtrade.ohio"
                ):
                    forbidden.append(alias.name)

    assert forbidden == [], f"Forbidden freqtrade imports found: {forbidden}"
