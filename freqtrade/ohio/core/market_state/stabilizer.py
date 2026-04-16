"""Stabilizer: regime persistence via EMA smoothing + jump gate + dwell controller.

Pipeline position: FactorCalculator → **Stabilizer** → Meta

Prevents noisy factor oscillations from causing rapid strategy switching.

3-stage pipeline:
    raw ohio_factor_* columns
    → Stage 1: EMA Smoothing  (pandas ewm, vectorized)
    → Stage 2: Jump Penalty Gate + Dwell Controller  (sequential, Numba JIT)
    → stabilized ohio_stable_* columns

Usage::

    from freqtrade.ohio.core.market_state.stabilizer import StateStabilizer

    stabilizer = StateStabilizer()
    dataframe = stabilizer.stabilize(dataframe)
"""

from __future__ import annotations

import numpy as np
import numba
import pandas as pd


# ---------------------------------------------------------------------------
# Column mapping: factor → stable
# ---------------------------------------------------------------------------

_FACTOR_COLS = [
    "ohio_factor_trend",
    "ohio_factor_volatility",
    "ohio_factor_downside",
    "ohio_factor_liquidity",
    "ohio_factor_relative_strength",
    "ohio_factor_correlation",
    "ohio_factor_breadth",
]

_STABLE_COLS = [c.replace("ohio_factor_", "ohio_stable_") for c in _FACTOR_COLS]

_FACTOR_TO_STABLE: dict[str, str] = dict(zip(_FACTOR_COLS, _STABLE_COLS))

# Short axis names used in StreamingStateStabilizer dicts
_AXIS_NAMES = [c.replace("ohio_factor_", "") for c in _FACTOR_COLS]


# ---------------------------------------------------------------------------
# Stage 2: Numba JIT — jump gate + dwell controller (single axis)
# ---------------------------------------------------------------------------


@numba.njit
def _jump_dwell_pass(
    smoothed: np.ndarray,
    jump_threshold: float,
    min_dwell: int,
) -> np.ndarray:
    """Apply jump gate + dwell controller on a single axis.

    PRECONDITION: ``smoothed`` must NOT contain NaN. The caller is
    responsible for stripping leading NaN before calling and
    re-inserting them after. NaN values inside this function will
    silently cascade through the entire output.

    Args:
        smoothed: EMA-smoothed values, shape (n,). Must be NaN-free.
        jump_threshold: Minimum delta to accept an axis update.
        min_dwell: Minimum bars between accepted changes.

    Returns:
        Stabilized array, same shape as ``smoothed``.
    """
    n = len(smoothed)
    stabilized = np.empty(n)
    stabilized[0] = smoothed[0]
    bars_since_change = min_dwell  # allow first change immediately

    for t in range(1, n):
        delta = abs(smoothed[t] - stabilized[t - 1])
        if bars_since_change >= min_dwell and delta > jump_threshold:
            stabilized[t] = smoothed[t]
            bars_since_change = 0
        else:
            stabilized[t] = stabilized[t - 1]
            bars_since_change += 1

    return stabilized


# ---------------------------------------------------------------------------
# Batch mode stabilizer
# ---------------------------------------------------------------------------


class StateStabilizer:
    """Batch mode stabilizer for backtesting.

    Processes entire time series at once.

    Args:
        ema_alpha: EMA smoothing factor. Smaller = more smoothing.
        jump_threshold: Minimum EMA delta to allow a state update.
        min_dwell: Minimum bars to hold a state before allowing another change.
    """

    def __init__(
        self,
        ema_alpha: float = 0.30,
        jump_threshold: float = 0.05,
        min_dwell: int = 12,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.jump_threshold = jump_threshold
        self.min_dwell = min_dwell

    def stabilize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply 3-stage stabilization to ohio_factor_* columns.

        Stage 1: EMA smoothing via pandas ewm (vectorized).
        Stage 2: Jump gate + dwell controller via Numba JIT (sequential).

        Args:
            dataframe: DataFrame containing ohio_factor_* columns.

        Returns:
            Same DataFrame with ohio_stable_* columns appended.
        """
        result = dataframe.copy()

        for factor_col, stable_col in _FACTOR_TO_STABLE.items():
            if factor_col not in dataframe.columns:
                continue

            series = dataframe[factor_col]

            # Rows that are fully NaN produce NaN output without entering Numba
            if series.isna().all():
                result[stable_col] = np.nan
                continue

            # Stage 1: EMA smoothing (vectorized)
            smoothed = series.ewm(alpha=self.ema_alpha, adjust=False).mean()

            # Stage 2: jump gate + dwell — operate on numpy for Numba
            # _jump_dwell_pass requires NaN-free input; strip leading NaN first
            smoothed_arr = smoothed.to_numpy(dtype=np.float64, na_value=np.nan)
            valid_mask = ~np.isnan(smoothed_arr)
            first_valid = int(np.argmax(valid_mask)) if valid_mask.any() else -1

            if first_valid == -1:
                result[stable_col] = np.nan
                continue

            trimmed = smoothed_arr[first_valid:]
            stable_trimmed = _jump_dwell_pass(
                trimmed,
                self.jump_threshold,
                self.min_dwell,
            )

            stable_arr = np.full(len(smoothed_arr), np.nan)
            stable_arr[first_valid:] = stable_trimmed

            # Re-apply NaN for positions where original series was NaN
            nan_mask = series.isna().to_numpy()
            stable_arr[nan_mask] = np.nan

            result[stable_col] = stable_arr

        return result


# ---------------------------------------------------------------------------
# Streaming (bar-by-bar) stabilizer — must be bit-exact with StateStabilizer
# ---------------------------------------------------------------------------


class StreamingStateStabilizer:
    """Bar-by-bar stabilizer for live/paper trading.

    Maintains internal state between calls.
    Output MUST be bit-exact with StateStabilizer for the same input sequence.

    EMA formula: ema[t] = alpha * value[t] + (1 - alpha) * ema[t-1]
    Jump/dwell logic mirrors _jump_dwell_pass exactly.

    Args:
        ema_alpha: EMA smoothing factor. Must match StateStabilizer.
        jump_threshold: Minimum EMA delta to allow a state update.
        min_dwell: Minimum bars to hold a state before allowing another change.
    """

    def __init__(
        self,
        ema_alpha: float = 0.30,
        jump_threshold: float = 0.05,
        min_dwell: int = 12,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.jump_threshold = jump_threshold
        self.min_dwell = min_dwell

        self._ema_state: dict[str, float] = {}
        self._stable_state: dict[str, float] = {}
        self._dwell_counter: dict[str, int] = {}
        self._initialized: bool = False

    def update(self, factors: dict[str, float]) -> dict[str, float]:
        """Process one bar's factor values.

        Args:
            factors: Axis values keyed by short name, e.g.
                     {"trend": 0.3, "volatility": 0.7, ...}

        Returns:
            Stabilized axis values with the same keys.
        """
        result: dict[str, float] = {}

        for axis in _AXIS_NAMES:
            value = factors.get(axis, np.nan)

            if np.isnan(value):
                result[axis] = np.nan
                continue

            if axis not in self._ema_state:
                # First bar for this axis — initialize state
                self._ema_state[axis] = value
                self._stable_state[axis] = value
                # min_dwell bars already elapsed → allow immediate change next bar
                self._dwell_counter[axis] = self.min_dwell
                result[axis] = value
                continue

            # Stage 1: incremental EMA
            ema_prev = self._ema_state[axis]
            ema_new = self.ema_alpha * value + (1.0 - self.ema_alpha) * ema_prev
            self._ema_state[axis] = ema_new

            # Stage 2: jump gate + dwell
            stable_prev = self._stable_state[axis]
            bars_since = self._dwell_counter[axis]
            delta = abs(ema_new - stable_prev)

            if bars_since >= self.min_dwell and delta > self.jump_threshold:
                self._stable_state[axis] = ema_new
                self._dwell_counter[axis] = 0
            else:
                self._dwell_counter[axis] = bars_since + 1

            result[axis] = self._stable_state[axis]

        self._initialized = True
        return result
