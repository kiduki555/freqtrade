"""Paper trading monitor for OhioThinStrategy dry-run bot.

Periodically polls the Freqtrade REST API, logs status to
``user_data/logs/paper_monitor.log``, and prints colour-coded
alerts to stdout when thresholds are breached.

Usage::

    python scripts/monitor_paper.py
    python scripts/monitor_paper.py --api-url http://127.0.0.1:8080 --interval 60
"""

from __future__ import annotations

import argparse
import os
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RED = "\033[91m"
_RESET = "\033[0m"

_DRAWDOWN_THRESHOLD_PCT = 10.0
_NO_TRADE_HOURS = 24

_LOG_DIR = Path(__file__).resolve().parent.parent / "user_data" / "logs"
_LOG_FILE = _LOG_DIR / "paper_monitor.log"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("paper_monitor")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


# ---------------------------------------------------------------------------
# Alert helper
# ---------------------------------------------------------------------------

def _alert(logger: logging.Logger, message: str) -> None:
    """Log a warning and print a red-highlighted message to stdout."""
    logger.warning(message)
    print(f"{_RED}[ALERT] {message}{_RESET}", flush=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

class FreqtradeClient:
    """Minimal wrapper around the Freqtrade REST API."""

    def __init__(self, base_url: str, username: str, password: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.auth = (username, password)
        self._token: str | None = None

    # ------------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self._base_url}/api/v1{path}"

    def _login(self) -> None:
        resp = self._session.post(
            self._url("/token/login"),
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data.get("access_token")
        if self._token:
            self._session.headers["Authorization"] = f"Bearer {self._token}"

    def _get(self, path: str) -> Any:
        if self._token is None:
            self._login()
        resp = self._session.get(self._url(path))
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def ping(self) -> dict:
        return self._get("/ping")

    def status(self) -> list[dict]:
        return self._get("/status")

    def profit(self) -> dict:
        return self._get("/profit")

    def trades(self, limit: int = 50) -> dict:
        return self._get(f"/trades?limit={limit}")

    def show_config(self) -> dict:
        return self._get("/show_config")


# ---------------------------------------------------------------------------
# Core monitoring loop
# ---------------------------------------------------------------------------

def _check_once(client: FreqtradeClient, logger: logging.Logger) -> None:
    """Run a single monitoring check cycle."""

    # --- Bot status ---
    try:
        config = client.show_config()
    except Exception:
        _alert(logger, "Bot is unreachable or stopped unexpectedly")
        return

    state = config.get("state", "unknown")
    bot_name = config.get("bot_name", "?")
    logger.info("Bot '%s' state: %s", bot_name, state)

    if state != "running":
        _alert(logger, f"Bot state is '{state}' (expected 'running')")

    # --- Open trades ---
    try:
        open_trades = client.status()
    except Exception:
        open_trades = []

    logger.info("Open trades: %d", len(open_trades))
    for trade in open_trades:
        pair = trade.get("pair", "?")
        profit_pct = trade.get("profit_pct", 0.0)
        duration = trade.get("trade_duration", "?")
        logger.info(
            "  %s | profit: %.2f%% | duration: %s min",
            pair,
            profit_pct,
            duration,
        )

    # --- Profit ---
    try:
        profit_data = client.profit()
    except Exception:
        profit_data = {}

    closed_today = profit_data.get("closed_trade_count", 0)
    profit_all = profit_data.get("profit_all_coin", 0.0)
    profit_pct = profit_data.get("profit_all_percent", 0.0)
    logger.info(
        "Closed trades (total): %d | Total profit: %.4f USDT (%.2f%%)",
        closed_today,
        profit_all,
        profit_pct,
    )

    # --- Drawdown ---
    current_drawdown = abs(min(profit_pct, 0.0))
    logger.info("Current drawdown estimate: %.2f%%", current_drawdown)

    if current_drawdown > _DRAWDOWN_THRESHOLD_PCT:
        _alert(
            logger,
            f"Drawdown {current_drawdown:.2f}% exceeds threshold "
            f"({_DRAWDOWN_THRESHOLD_PCT}%)",
        )

    # --- No-trade check ---
    try:
        trades_data = client.trades(limit=1)
        trades_list = trades_data.get("trades", [])
        if trades_list:
            last_trade_ts = trades_list[0].get("open_date", "")
            if last_trade_ts:
                last_dt = datetime.fromisoformat(
                    last_trade_ts.replace("Z", "+00:00"),
                )
                hours_since = (
                    datetime.now(timezone.utc) - last_dt
                ).total_seconds() / 3600
                if hours_since > _NO_TRADE_HOURS:
                    _alert(
                        logger,
                        f"No new trades in {hours_since:.1f}h "
                        f"(threshold: {_NO_TRADE_HOURS}h)",
                    )
        else:
            _alert(logger, "No trades found at all — bot may not be trading")
    except Exception as exc:
        logger.warning("Could not check last trade time: %s", exc)


def run_monitor(
    api_url: str,
    username: str,
    password: str,
    interval: int,
) -> None:
    """Entry-point: run the monitoring loop until interrupted."""

    logger = _setup_logging()
    client = FreqtradeClient(api_url, username, password)

    logger.info(
        "Starting paper monitor (url=%s, interval=%ds)",
        api_url,
        interval,
    )

    while True:
        try:
            _check_once(client, logger)
        except requests.ConnectionError:
            _alert(logger, f"Connection failed to {api_url} — retrying in {interval}s")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.error("Unexpected error: %s", exc, exc_info=True)

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Monitor stopped by user (Ctrl+C)")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor a Freqtrade paper-trading bot",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8080",
        help="Freqtrade REST API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("FREQTRADE_API_USER", "freqtrader"),
        help="API username (env: FREQTRADE_API_USER, default: freqtrader)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("FREQTRADE_API_PASS", "freqtrader"),
        help="API password (env: FREQTRADE_API_PASS, default: freqtrader)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: %(default)s)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_monitor(
        api_url=args.api_url,
        username=args.username,
        password=args.password,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
