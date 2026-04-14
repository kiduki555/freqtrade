"""Validate that a dataframe has enough rows for OHIO pipeline warmup."""

from __future__ import annotations

_HOURS_PER_DAY = 24


def validate_warmup(
    dataframe_length: int,
    startup_candle_count: int = 4320,
) -> tuple[bool, str]:
    """Check if dataframe has enough rows for OHIO warmup.

    Args:
        dataframe_length: Number of rows in the dataframe.
        startup_candle_count: Required warmup candles (default 4320 = 180 days @ 1h).

    Returns:
        (is_valid, message) tuple.
    """
    days_available = dataframe_length / _HOURS_PER_DAY
    days_required = startup_candle_count / _HOURS_PER_DAY

    if dataframe_length >= startup_candle_count:
        return (
            True,
            f"OK: {dataframe_length} rows ({days_available:.0f} days) "
            f">= {startup_candle_count} required ({days_required:.0f} days)",
        )

    return (
        False,
        f"INSUFFICIENT: need {startup_candle_count} rows ({days_required:.0f} days), "
        f"got {dataframe_length} rows ({days_available:.0f} days)",
    )
