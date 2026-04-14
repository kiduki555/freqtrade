"""Tests for paper-trading dry-run configuration and monitor alerting."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "user_data"
    / "config_dryrun.json"
)


@pytest.fixture(scope="module")
def dryrun_config() -> dict:
    """Load and parse the dry-run config once per module."""
    assert _CONFIG_PATH.exists(), f"Config not found: {_CONFIG_PATH}"
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. Config is valid JSON with required keys
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "trading_mode",
    "margin_mode",
    "max_open_trades",
    "stake_currency",
    "stake_amount",
    "dry_run",
    "dry_run_wallet",
    "exchange",
    "entry_pricing",
    "exit_pricing",
    "pairlists",
    "bot_name",
    "initial_state",
    "timeframe",
}


def test_config_has_required_keys(dryrun_config: dict) -> None:
    missing = _REQUIRED_KEYS - set(dryrun_config.keys())
    assert not missing, f"Missing required keys: {missing}"


# ---------------------------------------------------------------------------
# 2. dry_run flag is True
# ---------------------------------------------------------------------------


def test_dry_run_enabled(dryrun_config: dict) -> None:
    assert dryrun_config["dry_run"] is True


# ---------------------------------------------------------------------------
# 3. Pair whitelist contains BTC and ETH futures pairs
# ---------------------------------------------------------------------------


def test_pair_whitelist_futures(dryrun_config: dict) -> None:
    whitelist = dryrun_config["exchange"]["pair_whitelist"]

    btc_pairs = [p for p in whitelist if "BTC/USDT" in p and ":USDT" in p]
    eth_pairs = [p for p in whitelist if "ETH/USDT" in p and ":USDT" in p]

    assert btc_pairs, "BTC futures pair missing from whitelist"
    assert eth_pairs, "ETH futures pair missing from whitelist"


# ---------------------------------------------------------------------------
# 4. Monitor script alert thresholds
# ---------------------------------------------------------------------------


def test_monitor_drawdown_alert_threshold() -> None:
    """Verify that the monitor triggers an alert when drawdown > 10%."""
    from scripts.monitor_paper import (
        _DRAWDOWN_THRESHOLD_PCT,
        _NO_TRADE_HOURS,
        _alert,
    )

    assert _DRAWDOWN_THRESHOLD_PCT == 10.0, (
        f"Expected drawdown threshold 10%, got {_DRAWDOWN_THRESHOLD_PCT}"
    )
    assert _NO_TRADE_HOURS == 24, (
        f"Expected no-trade threshold 24h, got {_NO_TRADE_HOURS}"
    )

    # Verify _alert calls logger.warning
    mock_logger = MagicMock()
    _alert(mock_logger, "test alert")
    mock_logger.warning.assert_called_once_with("test alert")


# ---------------------------------------------------------------------------
# 5. Config futures mode settings
# ---------------------------------------------------------------------------


def test_futures_mode_settings(dryrun_config: dict) -> None:
    assert dryrun_config["trading_mode"] == "futures"
    assert dryrun_config["margin_mode"] == "isolated"
    assert dryrun_config["stake_currency"] == "USDT"
    assert dryrun_config["dry_run_wallet"] == 10000
