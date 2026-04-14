"""Tests for KillSwitch — one-way trading halt latch."""

from __future__ import annotations

import ast
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from freqtrade.ohio.core.portfolio_risk.kill_switch import KillSwitch

# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------


def test_initial_state_is_inactive() -> None:
    ks = KillSwitch()
    assert ks.active is False
    assert ks.reason is None
    assert ks.activated_at is None


# ---------------------------------------------------------------------------
# 2. Activate
# ---------------------------------------------------------------------------


def test_activate_sets_active_reason_and_time() -> None:
    ks = KillSwitch()
    before = datetime.now(tz=timezone.utc)
    ks.activate("max drawdown exceeded")
    after = datetime.now(tz=timezone.utc)

    assert ks.active is True
    assert ks.reason == "max drawdown exceeded"
    assert before <= ks.activated_at <= after  # type: ignore[operator]


# ---------------------------------------------------------------------------
# 3. Double activate — idempotent
# ---------------------------------------------------------------------------


def test_double_activate_preserves_original_reason_and_time() -> None:
    ks = KillSwitch()
    ks.activate("first reason")
    original_reason = ks.reason
    original_time = ks.activated_at

    time.sleep(0.01)  # ensure clock ticks
    ks.activate("second reason")

    assert ks.reason == original_reason
    assert ks.activated_at == original_time


# ---------------------------------------------------------------------------
# 4. No reset / deactivate
# ---------------------------------------------------------------------------


def test_no_reset_or_deactivate_method() -> None:
    ks = KillSwitch()
    assert not hasattr(ks, "reset")
    assert not hasattr(ks, "deactivate")


# ---------------------------------------------------------------------------
# 5. __repr__
# ---------------------------------------------------------------------------


def test_repr_inactive() -> None:
    ks = KillSwitch()
    assert repr(ks) == "KillSwitch(active=False)"


def test_repr_active() -> None:
    ks = KillSwitch()
    ks.activate("test")
    r = repr(ks)
    assert "active=True" in r
    assert "reason='test'" in r
    assert "activated_at=" in r


# ---------------------------------------------------------------------------
# 6. Empty-string reason
# ---------------------------------------------------------------------------


def test_activate_with_empty_string() -> None:
    ks = KillSwitch()
    ks.activate("")
    assert ks.active is True
    assert ks.reason == ""


# ---------------------------------------------------------------------------
# 7. activated_at type
# ---------------------------------------------------------------------------


def test_activated_at_is_datetime() -> None:
    ks = KillSwitch()
    ks.activate("check type")
    assert isinstance(ks.activated_at, datetime)


# ---------------------------------------------------------------------------
# 8. Multiple instances are independent
# ---------------------------------------------------------------------------


def test_multiple_instances_independent() -> None:
    a = KillSwitch()
    b = KillSwitch()
    a.activate("only a")

    assert a.active is True
    assert b.active is False


# ---------------------------------------------------------------------------
# 9. No Freqtrade imports (AST check)
# ---------------------------------------------------------------------------


def test_no_freqtrade_imports_in_module() -> None:
    src = Path(__file__).resolve().parents[2] / (
        "freqtrade/ohio/core/portfolio_risk/kill_switch.py"
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
                    f"Forbidden import: {node.module}"
                )
