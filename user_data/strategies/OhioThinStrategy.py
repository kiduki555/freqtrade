"""Wrapper so Freqtrade's strategy resolver can discover OhioThinStrategy.

Freqtrade scans .py files for class definitions that subclass IStrategy.
A bare re-export is not detected, so we create a thin subclass here.

startup_candle_count is reduced from 4320 to 2000 for paper/live trading
because Binance API provides ~2494 candles max (500 per request × 5).
Backtest uses the full 4320 via local feather data.
"""
from freqtrade.ohio.adapters.freqtrade.thin_strategy import (
    OhioThinStrategy as _Base,
)


class OhioThinStrategy(_Base):
    """Discoverable wrapper — all logic lives in the base class."""

    # Binance API limit: ~2494 candles for 1h.
    # 2000 bars = ~83 days warmup (vs 180 days in backtest).
    startup_candle_count = 2000
