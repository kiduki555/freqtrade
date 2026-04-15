# Mode-Specific Entry Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-logic entry with mode-specific strategies (TF/MR/BO/DEF), each with hyperopt-tunable parameters.

**Architecture:** EntryAdapter dispatcher + 4 strategy modules implementing EntryStrategy Protocol. Feature builder extended with 12 new indicators. 14 new hyperopt parameters in thin_strategy.

**Tech Stack:** Python 3.11+, pandas, numpy, pytest, Freqtrade IStrategy

**Spec:** `docs/superpowers/specs/2026-04-15-mode-specific-entry-strategies-design.md`

---

### Task 1: Add 12 New Indicator Features to feature_builder.py

**Files:**
- Modify: `freqtrade/ohio/core/market_state/feature_builder.py`
- Test: `tests/ohio/test_feature_builder.py`

- [ ] **Step 1: Write failing tests for new features**

Add to `tests/ohio/test_feature_builder.py`:

```python
# --- New features for mode-specific entry strategies ---

NEW_FEATURE_COLUMNS = [
    "ohio_feat_zscore_20",
    "ohio_feat_rsi_14",
    "ohio_feat_kama_10",
    "ohio_feat_kama_slope",
    "ohio_feat_bb_upper_20",
    "ohio_feat_bb_lower_20",
    "ohio_feat_kc_upper_20",
    "ohio_feat_kc_lower_20",
    "ohio_feat_donchian_upper_20",
    "ohio_feat_donchian_lower_20",
    "ohio_feat_squeeze_count",
    "ohio_feat_volume_sma_20",
]


class TestNewIndicatorFeatures:
    """Tests for the 12 new indicator features."""

    def test_new_columns_present(self, df_normal: pd.DataFrame) -> None:
        for col in NEW_FEATURE_COLUMNS:
            assert col in df_normal.columns, f"Missing column: {col}"

    def test_new_columns_in_feature_columns(self) -> None:
        for col in NEW_FEATURE_COLUMNS:
            assert col in FEATURE_COLUMNS, f"{col} not in FEATURE_COLUMNS"

    def test_zscore_centered_near_zero(self, df_normal: pd.DataFrame) -> None:
        valid = df_normal["ohio_feat_zscore_20"].dropna()
        assert abs(valid.mean()) < 1.0, "Z-score should be roughly centered"

    def test_rsi_in_range(self, df_normal: pd.DataFrame) -> None:
        valid = df_normal["ohio_feat_rsi_14"].dropna()
        assert (valid >= 0).all(), "RSI must be >= 0"
        assert (valid <= 100).all(), "RSI must be <= 100"

    def test_rsi_warmup(self, df_normal: pd.DataFrame) -> None:
        series = df_normal["ohio_feat_rsi_14"]
        assert series.iloc[:14].isna().all(), "RSI first 14 bars should be NaN"

    def test_kama_follows_price(self, df_normal: pd.DataFrame) -> None:
        valid_kama = df_normal["ohio_feat_kama_10"].dropna()
        valid_close = df_normal["close"].iloc[valid_kama.index]
        corr = valid_kama.corr(valid_close)
        assert corr > 0.9, f"KAMA should track close, corr={corr:.3f}"

    def test_bb_upper_above_lower(self, df_normal: pd.DataFrame) -> None:
        mask = df_normal["ohio_feat_bb_upper_20"].notna()
        upper = df_normal.loc[mask, "ohio_feat_bb_upper_20"]
        lower = df_normal.loc[mask, "ohio_feat_bb_lower_20"]
        assert (upper >= lower).all(), "BB upper must be >= lower"

    def test_kc_upper_above_lower(self, df_normal: pd.DataFrame) -> None:
        mask = df_normal["ohio_feat_kc_upper_20"].notna()
        upper = df_normal.loc[mask, "ohio_feat_kc_upper_20"]
        lower = df_normal.loc[mask, "ohio_feat_kc_lower_20"]
        assert (upper >= lower).all(), "KC upper must be >= lower"

    def test_donchian_upper_above_lower(self, df_normal: pd.DataFrame) -> None:
        mask = df_normal["ohio_feat_donchian_upper_20"].notna()
        upper = df_normal.loc[mask, "ohio_feat_donchian_upper_20"]
        lower = df_normal.loc[mask, "ohio_feat_donchian_lower_20"]
        assert (upper >= lower).all(), "Donchian upper must be >= lower"

    def test_squeeze_count_non_negative(self, df_normal: pd.DataFrame) -> None:
        valid = df_normal["ohio_feat_squeeze_count"].dropna()
        assert (valid >= 0).all(), "Squeeze count must be >= 0"

    def test_volume_sma_positive(self, df_normal: pd.DataFrame) -> None:
        valid = df_normal["ohio_feat_volume_sma_20"].dropna()
        assert (valid > 0).all(), "Volume SMA must be positive"

    def test_flat_price_zscore_zero(self, df_flat: pd.DataFrame) -> None:
        valid = df_flat["ohio_feat_zscore_20"].dropna()
        assert (valid.abs() < 1e-9).all(), "Z-score should be 0 for flat prices"

    def test_flat_price_rsi_50(self, df_flat: pd.DataFrame) -> None:
        """Flat price (close unchanged) should produce RSI near 50."""
        valid = df_flat["ohio_feat_rsi_14"].dropna()
        # Flat = zero change = avg_gain == avg_loss == 0 → RSI undefined or 50
        # Implementation should handle zero div → NaN or 50
        if len(valid) > 0:
            assert ((valid - 50).abs() < 1.0).all() or valid.isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Business/ohio-backend-v2/freqtrade && python -m pytest tests/ohio/test_feature_builder.py::TestNewIndicatorFeatures -v`
Expected: FAIL — columns not found

- [ ] **Step 3: Implement 12 new features in feature_builder.py**

Add to `FEATURE_COLUMNS` list and implement:

```python
# In FEATURE_COLUMNS list, add after existing entries:
    # Entry strategy indicators (12)
    "ohio_feat_zscore_20",
    "ohio_feat_rsi_14",
    "ohio_feat_kama_10",
    "ohio_feat_kama_slope",
    "ohio_feat_bb_upper_20",
    "ohio_feat_bb_lower_20",
    "ohio_feat_kc_upper_20",
    "ohio_feat_kc_lower_20",
    "ohio_feat_donchian_upper_20",
    "ohio_feat_donchian_lower_20",
    "ohio_feat_squeeze_count",
    "ohio_feat_volume_sma_20",
```

Add `self._compute_entry_indicators(dataframe)` call in `compute()`.

Implement `_compute_entry_indicators`:

```python
def _compute_entry_indicators(self, df: pd.DataFrame) -> None:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- Z-score(20) ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["ohio_feat_zscore_20"] = (close - sma20) / std20.replace(0, np.nan)

    # --- RSI(14) via Wilder smoothing ---
    df["ohio_feat_rsi_14"] = self._compute_rsi(close, period=14)

    # --- KAMA(10, fast=2, slow=30) ---
    kama = self._compute_kama(close, er_period=10, fast_period=2, slow_period=30)
    df["ohio_feat_kama_10"] = kama
    df["ohio_feat_kama_slope"] = (kama - kama.shift(5)) / close

    # --- Bollinger Bands(20, 2.0) ---
    df["ohio_feat_bb_upper_20"] = sma20 + 2.0 * std20
    df["ohio_feat_bb_lower_20"] = sma20 - 2.0 * std20

    # --- Keltner Channel(20, 1.5) ---
    ema20 = close.ewm(span=20, adjust=False).mean()
    hl = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    atr20 = tr.rolling(20).mean()
    df["ohio_feat_kc_upper_20"] = ema20 + 1.5 * atr20
    df["ohio_feat_kc_lower_20"] = ema20 - 1.5 * atr20

    # --- Donchian Channel(20) ---
    df["ohio_feat_donchian_upper_20"] = high.rolling(20).max()
    df["ohio_feat_donchian_lower_20"] = low.rolling(20).min()

    # --- Bollinger Squeeze count ---
    bb_inside_kc = (
        (df["ohio_feat_bb_upper_20"] < df["ohio_feat_kc_upper_20"])
        & (df["ohio_feat_bb_lower_20"] > df["ohio_feat_kc_lower_20"])
    )
    # Count consecutive True bars
    groups = (~bb_inside_kc).cumsum()
    squeeze_count = bb_inside_kc.groupby(groups).cumsum()
    df["ohio_feat_squeeze_count"] = squeeze_count.where(bb_inside_kc, 0)

    # --- Volume SMA(20) ---
    df["ohio_feat_volume_sma_20"] = volume.rolling(20).mean()

@staticmethod
def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

@staticmethod
def _compute_kama(
    close: pd.Series, er_period: int = 10, fast_period: int = 2, slow_period: int = 30,
) -> pd.Series:
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    direction = (close - close.shift(er_period)).abs()
    volatility = close.diff().abs().rolling(er_period).sum()
    er = direction / volatility.replace(0, np.nan)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = close.copy()
    kama.iloc[:er_period] = np.nan
    for i in range(er_period, len(close)):
        if np.isnan(kama.iloc[i - 1]):
            kama.iloc[i] = close.iloc[i]
        else:
            kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i - 1])
    return kama
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Business/ohio-backend-v2/freqtrade && python -m pytest tests/ohio/test_feature_builder.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add freqtrade/ohio/core/market_state/feature_builder.py tests/ohio/test_feature_builder.py
git commit -m "feat: add 12 entry indicator features (zscore, RSI, KAMA, BB, KC, Donchian, squeeze)"
```

---

### Task 2: Create EntryAdapter Protocol and Dispatcher

**Files:**
- Create: `freqtrade/ohio/adapters/freqtrade/entry_adapter.py`
- Create: `freqtrade/ohio/adapters/freqtrade/entries/__init__.py`
- Test: `tests/ohio/test_entry_adapter.py`

- [ ] **Step 1: Write failing test**

Create `tests/ohio/test_entry_adapter.py`:

```python
"""Tests for EntryAdapter — mode dispatch + fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.entry_adapter import EntryAdapter


@pytest.fixture
def adapter() -> EntryAdapter:
    return EntryAdapter()


def _make_row_df(n: int = 50, mode: str = "trend_following") -> pd.DataFrame:
    """Minimal dataframe with required columns for entry adapter."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": rng.exponential(1000, n),
        "ohio_active_mode": mode,
        "ohio_stable_trend": rng.uniform(-0.5, 0.5, n),
        "ohio_feat_adx_14": rng.uniform(10, 50, n),
        "ohio_feat_hurst_168": rng.uniform(0.3, 0.7, n),
        "ohio_feat_kama_slope": rng.uniform(-0.01, 0.01, n),
        "ohio_feat_zscore_20": rng.uniform(-3, 3, n),
        "ohio_feat_rsi_14": rng.uniform(10, 90, n),
        "ohio_feat_bb_upper_20": close * 1.02,
        "ohio_feat_bb_lower_20": close * 0.98,
        "ohio_feat_kc_upper_20": close * 1.03,
        "ohio_feat_kc_lower_20": close * 0.97,
        "ohio_feat_donchian_upper_20": close * 1.025,
        "ohio_feat_donchian_lower_20": close * 0.975,
        "ohio_feat_squeeze_count": rng.integers(0, 10, n),
        "ohio_feat_volume_sma_20": 1000.0,
    })
    return df


class TestEntryAdapter:
    def test_returns_dataframe(self, adapter: EntryAdapter) -> None:
        df = _make_row_df(50, "trend_following")
        params = {}
        result = adapter.compute_entries(df, "trend_following", params)
        assert isinstance(result, pd.DataFrame)

    def test_adds_entry_columns(self, adapter: EntryAdapter) -> None:
        df = _make_row_df(50, "trend_following")
        result = adapter.compute_entries(df, "trend_following", {})
        assert "enter_long" in result.columns
        assert "enter_short" in result.columns

    def test_unknown_mode_fallback(self, adapter: EntryAdapter) -> None:
        df = _make_row_df(50, "unknown_mode")
        result = adapter.compute_entries(df, "unknown_mode", {})
        # Fallback uses trend-sign logic
        assert "enter_long" in result.columns

    def test_all_four_modes_supported(self, adapter: EntryAdapter) -> None:
        for mode in ["trend_following", "mean_reversion", "breakout", "defensive"]:
            df = _make_row_df(50, mode)
            result = adapter.compute_entries(df, mode, {})
            assert "enter_long" in result.columns

    def test_entries_are_binary(self, adapter: EntryAdapter) -> None:
        df = _make_row_df(50, "trend_following")
        result = adapter.compute_entries(df, "trend_following", {})
        assert set(result["enter_long"].unique()).issubset({0, 1})
        assert set(result["enter_short"].unique()).issubset({0, 1})

    def test_no_simultaneous_long_short(self, adapter: EntryAdapter) -> None:
        for mode in ["trend_following", "mean_reversion", "breakout", "defensive"]:
            df = _make_row_df(100, mode)
            result = adapter.compute_entries(df, mode, {})
            both = (result["enter_long"] == 1) & (result["enter_short"] == 1)
            assert not both.any(), f"{mode}: simultaneous long+short found"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Business/ohio-backend-v2/freqtrade && python -m pytest tests/ohio/test_entry_adapter.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement EntryAdapter**

Create `freqtrade/ohio/adapters/freqtrade/entries/__init__.py` (empty).

Create `freqtrade/ohio/adapters/freqtrade/entry_adapter.py`:

```python
"""EntryAdapter — dispatch entry signals to mode-specific strategies.

Each mode has a dedicated entry strategy implementing vectorized
computation. Unknown modes fall back to trend-sign logic.
"""
from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)


class EntryStrategy(Protocol):
    """Protocol for mode-specific entry strategy."""

    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        """Add enter_long/enter_short columns. Must be vectorized."""
        ...


class EntryAdapter:
    """Dispatch entry signal computation to mode-specific strategies."""

    def __init__(self) -> None:
        from freqtrade.ohio.adapters.freqtrade.entries.trend_following import (
            TrendFollowingEntry,
        )
        from freqtrade.ohio.adapters.freqtrade.entries.mean_reversion import (
            MeanReversionEntry,
        )
        from freqtrade.ohio.adapters.freqtrade.entries.breakout import (
            BreakoutEntry,
        )
        from freqtrade.ohio.adapters.freqtrade.entries.defensive import (
            DefensiveEntry,
        )

        self._strategies: dict[str, EntryStrategy] = {
            "trend_following": TrendFollowingEntry(),
            "mean_reversion": MeanReversionEntry(),
            "breakout": BreakoutEntry(),
            "defensive": DefensiveEntry(),
        }

    def compute_entries(
        self,
        dataframe: pd.DataFrame,
        mode: str,
        params: dict[str, float],
    ) -> pd.DataFrame:
        strategy = self._strategies.get(mode)
        if strategy is None:
            logger.warning("Unknown mode %r, using trend-sign fallback", mode)
            return self._fallback_entries(dataframe)
        return strategy.compute_entries(dataframe, params)

    @staticmethod
    def _fallback_entries(dataframe: pd.DataFrame) -> pd.DataFrame:
        trend = dataframe.get(
            "ohio_stable_trend", pd.Series(0.0, index=dataframe.index)
        )
        dataframe["enter_long"] = (trend >= 0).astype(int)
        dataframe["enter_short"] = (trend < 0).astype(int)
        return dataframe
```

- [ ] **Step 4: Run tests**

Run: `cd /c/Business/ohio-backend-v2/freqtrade && python -m pytest tests/ohio/test_entry_adapter.py -v`
Expected: FAIL — entries submodules not implemented yet (expected at this stage)

- [ ] **Step 5: Commit**

```bash
git add freqtrade/ohio/adapters/freqtrade/entry_adapter.py \
        freqtrade/ohio/adapters/freqtrade/entries/__init__.py \
        tests/ohio/test_entry_adapter.py
git commit -m "feat: add EntryAdapter dispatcher with Protocol + fallback"
```

---

### Task 3: Implement TrendFollowingEntry

**Files:**
- Create: `freqtrade/ohio/adapters/freqtrade/entries/trend_following.py`
- Create: `tests/ohio/entries/__init__.py`
- Create: `tests/ohio/entries/test_trend_following.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ohio/entries/__init__.py` (empty).

Create `tests/ohio/entries/test_trend_following.py`:

```python
"""Tests for TrendFollowingEntry strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.entries.trend_following import (
    TrendFollowingEntry,
    DEFAULT_PARAMS as TF_DEFAULTS,
)


@pytest.fixture
def strategy() -> TrendFollowingEntry:
    return TrendFollowingEntry()


def _make_trending_up(n: int = 50) -> pd.DataFrame:
    close = 100.0 + np.arange(n) * 0.5
    return pd.DataFrame({
        "close": close,
        "ohio_stable_trend": np.full(n, 0.3),  # positive trend
        "ohio_feat_adx_14": np.full(n, 35.0),  # strong trend
        "ohio_feat_hurst_168": np.full(n, 0.65),  # persistent
        "ohio_feat_kama_slope": np.full(n, 0.003),  # positive KAMA slope
    })


def _make_choppy(n: int = 50) -> pd.DataFrame:
    close = 100.0 + np.sin(np.arange(n) * 0.5) * 2
    return pd.DataFrame({
        "close": close,
        "ohio_stable_trend": np.full(n, 0.05),
        "ohio_feat_adx_14": np.full(n, 12.0),  # weak ADX
        "ohio_feat_hurst_168": np.full(n, 0.35),  # mean-reverting
        "ohio_feat_kama_slope": np.full(n, 0.0001),  # flat
    })


class TestTrendFollowingEntry:
    def test_strong_uptrend_enters_long(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert result["enter_long"].sum() > 0, "Should enter long in uptrend"
        assert result["enter_short"].sum() == 0, "Should not short in uptrend"

    def test_choppy_market_no_entry(self, strategy: TrendFollowingEntry) -> None:
        df = _make_choppy()
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert result["enter_long"].sum() == 0, "Should not enter in choppy market"
        assert result["enter_short"].sum() == 0

    def test_strong_downtrend_enters_short(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        df["ohio_stable_trend"] = -0.3
        df["ohio_feat_kama_slope"] = -0.003
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert result["enter_short"].sum() > 0
        assert result["enter_long"].sum() == 0

    def test_custom_params_override(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        df["ohio_feat_adx_14"] = 20.0  # below default 25 threshold
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert result["enter_long"].sum() == 0, "ADX too low for default"

        custom = {**TF_DEFAULTS, "tf_adx_threshold": 15.0}
        result2 = strategy.compute_entries(df, custom)
        assert result2["enter_long"].sum() > 0, "Should enter with lower threshold"

    def test_no_simultaneous_entries(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        result = strategy.compute_entries(df, TF_DEFAULTS)
        both = (result["enter_long"] == 1) & (result["enter_short"] == 1)
        assert not both.any()

    def test_returns_dataframe(self, strategy: TrendFollowingEntry) -> None:
        df = _make_trending_up()
        result = strategy.compute_entries(df, TF_DEFAULTS)
        assert isinstance(result, pd.DataFrame)
```

- [ ] **Step 2: Implement TrendFollowingEntry**

Create `freqtrade/ohio/adapters/freqtrade/entries/trend_following.py`:

```python
"""Trend Following entry strategy — TSMOM + KAMA + ADX + Hurst.

Sources:
- Huang et al. 2024 (SSRN 4825389): Vol-weighted TSMOM
- Sepp (SSRN 3167787): Volatility scaling
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "tf_adx_threshold": 25.0,
    "tf_hurst_threshold": 0.55,
    "tf_kama_slope_threshold": 0.001,
    "tf_vol_scale_target": 1.0,
}


class TrendFollowingEntry:
    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        adx_th = params.get("tf_adx_threshold", DEFAULT_PARAMS["tf_adx_threshold"])
        hurst_th = params.get("tf_hurst_threshold", DEFAULT_PARAMS["tf_hurst_threshold"])
        kama_th = params.get("tf_kama_slope_threshold", DEFAULT_PARAMS["tf_kama_slope_threshold"])

        trend = dataframe.get("ohio_stable_trend", pd.Series(0.0, index=dataframe.index))
        adx = dataframe.get("ohio_feat_adx_14", pd.Series(0.0, index=dataframe.index))
        hurst = dataframe.get("ohio_feat_hurst_168", pd.Series(0.5, index=dataframe.index))
        kama_slope = dataframe.get("ohio_feat_kama_slope", pd.Series(0.0, index=dataframe.index))

        base_gate = (adx > adx_th) & (hurst > hurst_th)

        long_mask = base_gate & (kama_slope > kama_th) & (trend > 0)
        short_mask = base_gate & (kama_slope < -kama_th) & (trend < 0)

        dataframe["enter_long"] = long_mask.astype(int)
        dataframe["enter_short"] = short_mask.astype(int)
        return dataframe
```

- [ ] **Step 3: Run tests**

Run: `cd /c/Business/ohio-backend-v2/freqtrade && python -m pytest tests/ohio/entries/test_trend_following.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add freqtrade/ohio/adapters/freqtrade/entries/trend_following.py \
        tests/ohio/entries/__init__.py tests/ohio/entries/test_trend_following.py
git commit -m "feat: add TrendFollowingEntry (TSMOM + KAMA + ADX + Hurst)"
```

---

### Task 4: Implement MeanReversionEntry

**Files:**
- Create: `freqtrade/ohio/adapters/freqtrade/entries/mean_reversion.py`
- Create: `tests/ohio/entries/test_mean_reversion.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ohio/entries/test_mean_reversion.py`:

```python
"""Tests for MeanReversionEntry strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.entries.mean_reversion import (
    MeanReversionEntry,
    DEFAULT_PARAMS as MR_DEFAULTS,
)


@pytest.fixture
def strategy() -> MeanReversionEntry:
    return MeanReversionEntry()


def _make_oversold(n: int = 50) -> pd.DataFrame:
    close = np.full(n, 95.0)  # below SMA
    return pd.DataFrame({
        "close": close,
        "ohio_feat_zscore_20": np.full(n, -2.5),
        "ohio_feat_rsi_14": np.full(n, 25.0),
        "ohio_feat_hurst_168": np.full(n, 0.35),
        "ohio_feat_bb_lower_20": np.full(n, 96.0),  # close <= bb_lower
        "ohio_feat_bb_upper_20": np.full(n, 104.0),
    })


def _make_overbought(n: int = 50) -> pd.DataFrame:
    close = np.full(n, 105.0)
    return pd.DataFrame({
        "close": close,
        "ohio_feat_zscore_20": np.full(n, 2.5),
        "ohio_feat_rsi_14": np.full(n, 75.0),
        "ohio_feat_hurst_168": np.full(n, 0.35),
        "ohio_feat_bb_upper_20": np.full(n, 104.0),  # close >= bb_upper
        "ohio_feat_bb_lower_20": np.full(n, 96.0),
    })


def _make_trending(n: int = 50) -> pd.DataFrame:
    close = 100.0 + np.arange(n) * 0.5
    return pd.DataFrame({
        "close": close,
        "ohio_feat_zscore_20": np.full(n, -2.5),
        "ohio_feat_rsi_14": np.full(n, 25.0),
        "ohio_feat_hurst_168": np.full(n, 0.65),  # trending → MR blocked
        "ohio_feat_bb_lower_20": close * 0.98,
        "ohio_feat_bb_upper_20": close * 1.02,
    })


class TestMeanReversionEntry:
    def test_oversold_enters_long(self, strategy: MeanReversionEntry) -> None:
        df = _make_oversold()
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_long"].sum() > 0

    def test_overbought_enters_short(self, strategy: MeanReversionEntry) -> None:
        df = _make_overbought()
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_short"].sum() > 0

    def test_trending_blocks_entry(self, strategy: MeanReversionEntry) -> None:
        df = _make_trending()
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_long"].sum() == 0
        assert result["enter_short"].sum() == 0

    def test_hurst_gate_required(self, strategy: MeanReversionEntry) -> None:
        df = _make_oversold()
        df["ohio_feat_hurst_168"] = 0.60  # above mr_hurst_max
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_custom_zscore_threshold(self, strategy: MeanReversionEntry) -> None:
        df = _make_oversold()
        df["ohio_feat_zscore_20"] = -1.8  # below default 2.0
        result = strategy.compute_entries(df, MR_DEFAULTS)
        assert result["enter_long"].sum() == 0

        custom = {**MR_DEFAULTS, "mr_zscore_entry": 1.5}
        result2 = strategy.compute_entries(df, custom)
        assert result2["enter_long"].sum() > 0

    def test_no_simultaneous_entries(self, strategy: MeanReversionEntry) -> None:
        df = _make_oversold()
        result = strategy.compute_entries(df, MR_DEFAULTS)
        both = (result["enter_long"] == 1) & (result["enter_short"] == 1)
        assert not both.any()
```

- [ ] **Step 2: Implement MeanReversionEntry**

Create `freqtrade/ohio/adapters/freqtrade/entries/mean_reversion.py`:

```python
"""Mean Reversion entry strategy — Z-score + Hurst gate + BB + RSI.

Sources:
- Avellaneda & Lee 2010: Statistical arbitrage z-score framework
- Beluska & Vojtko (SSRN 4955617): Hurst exponent as MR filter
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "mr_zscore_entry": 2.0,
    "mr_rsi_oversold": 30.0,
    "mr_rsi_overbought": 70.0,
    "mr_hurst_max": 0.45,
}


class MeanReversionEntry:
    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        zs_th = params.get("mr_zscore_entry", DEFAULT_PARAMS["mr_zscore_entry"])
        rsi_os = params.get("mr_rsi_oversold", DEFAULT_PARAMS["mr_rsi_oversold"])
        rsi_ob = params.get("mr_rsi_overbought", DEFAULT_PARAMS["mr_rsi_overbought"])
        hurst_max = params.get("mr_hurst_max", DEFAULT_PARAMS["mr_hurst_max"])

        zscore = dataframe.get("ohio_feat_zscore_20", pd.Series(0.0, index=dataframe.index))
        rsi = dataframe.get("ohio_feat_rsi_14", pd.Series(50.0, index=dataframe.index))
        hurst = dataframe.get("ohio_feat_hurst_168", pd.Series(0.5, index=dataframe.index))
        close = dataframe["close"]
        bb_lower = dataframe.get("ohio_feat_bb_lower_20", pd.Series(0.0, index=dataframe.index))
        bb_upper = dataframe.get("ohio_feat_bb_upper_20", pd.Series(float("inf"), index=dataframe.index))

        hurst_gate = hurst < hurst_max

        long_mask = hurst_gate & (zscore < -zs_th) & (rsi < rsi_os) & (close <= bb_lower)
        short_mask = hurst_gate & (zscore > zs_th) & (rsi > rsi_ob) & (close >= bb_upper)

        dataframe["enter_long"] = long_mask.astype(int)
        dataframe["enter_short"] = short_mask.astype(int)
        return dataframe
```

- [ ] **Step 3: Run tests and commit**

Run: `cd /c/Business/ohio-backend-v2/freqtrade && python -m pytest tests/ohio/entries/test_mean_reversion.py -v`

```bash
git add freqtrade/ohio/adapters/freqtrade/entries/mean_reversion.py \
        tests/ohio/entries/test_mean_reversion.py
git commit -m "feat: add MeanReversionEntry (Z-score + Hurst gate + BB + RSI)"
```

---

### Task 5: Implement BreakoutEntry

**Files:**
- Create: `freqtrade/ohio/adapters/freqtrade/entries/breakout.py`
- Create: `tests/ohio/entries/test_breakout.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ohio/entries/test_breakout.py`:

```python
"""Tests for BreakoutEntry strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.entries.breakout import (
    BreakoutEntry,
    DEFAULT_PARAMS as BO_DEFAULTS,
)


@pytest.fixture
def strategy() -> BreakoutEntry:
    return BreakoutEntry()


def _make_squeeze_breakout_up(n: int = 50) -> pd.DataFrame:
    close = np.full(n, 100.0)
    close[-5:] = 105.0  # breakout up
    return pd.DataFrame({
        "close": close,
        "volume": np.concatenate([np.full(n - 5, 500.0), np.full(5, 2000.0)]),
        "ohio_feat_squeeze_count": np.concatenate([np.full(n - 5, 8.0), np.zeros(5)]),
        "ohio_feat_donchian_upper_20": np.full(n, 103.0),
        "ohio_feat_donchian_lower_20": np.full(n, 97.0),
        "ohio_feat_volume_sma_20": np.full(n, 1000.0),
    })


def _make_no_squeeze(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({
        "close": np.full(n, 100.0),
        "volume": np.full(n, 1000.0),
        "ohio_feat_squeeze_count": np.zeros(n),  # no squeeze
        "ohio_feat_donchian_upper_20": np.full(n, 103.0),
        "ohio_feat_donchian_lower_20": np.full(n, 97.0),
        "ohio_feat_volume_sma_20": np.full(n, 1000.0),
    })


class TestBreakoutEntry:
    def test_squeeze_breakout_up(self, strategy: BreakoutEntry) -> None:
        df = _make_squeeze_breakout_up()
        # Set squeeze_count on the breakout bars to meet threshold
        # Squeeze happened BEFORE breakout — check the shift logic
        df["ohio_feat_squeeze_count"] = np.concatenate([
            np.arange(1, n + 1) if (n := 45) else [],
            np.full(5, 0),  # squeeze ended
        ])
        # Actually need to rethink: squeeze_count > threshold on PREVIOUS bar
        # Use shift(1) in impl. For test, set squeeze_count on prior bars
        df = pd.DataFrame({
            "close": np.concatenate([np.full(45, 100.0), np.full(5, 105.0)]),
            "volume": np.concatenate([np.full(45, 500.0), np.full(5, 2000.0)]),
            "ohio_feat_squeeze_count": np.concatenate([np.full(45, 8.0), np.full(5, 0.0)]),
            "ohio_feat_donchian_upper_20": np.full(50, 103.0),
            "ohio_feat_donchian_lower_20": np.full(50, 97.0),
            "ohio_feat_volume_sma_20": np.full(50, 1000.0),
        })
        result = strategy.compute_entries(df, BO_DEFAULTS)
        # Last 5 bars break above Donchian with volume — should have entries
        assert result["enter_long"].iloc[-5:].sum() > 0

    def test_no_squeeze_no_entry(self, strategy: BreakoutEntry) -> None:
        df = _make_no_squeeze()
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_long"].sum() == 0
        assert result["enter_short"].sum() == 0

    def test_low_volume_blocks(self, strategy: BreakoutEntry) -> None:
        df = pd.DataFrame({
            "close": np.full(50, 105.0),
            "volume": np.full(50, 500.0),  # low volume
            "ohio_feat_squeeze_count": np.full(50, 8.0),
            "ohio_feat_donchian_upper_20": np.full(50, 103.0),
            "ohio_feat_donchian_lower_20": np.full(50, 97.0),
            "ohio_feat_volume_sma_20": np.full(50, 1000.0),  # volume < 1.5x SMA
        })
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_breakout_down_enters_short(self, strategy: BreakoutEntry) -> None:
        df = pd.DataFrame({
            "close": np.full(50, 95.0),  # below donchian_lower
            "volume": np.full(50, 2000.0),
            "ohio_feat_squeeze_count": np.full(50, 8.0),
            "ohio_feat_donchian_upper_20": np.full(50, 103.0),
            "ohio_feat_donchian_lower_20": np.full(50, 97.0),
            "ohio_feat_volume_sma_20": np.full(50, 1000.0),
        })
        result = strategy.compute_entries(df, BO_DEFAULTS)
        assert result["enter_short"].sum() > 0
```

- [ ] **Step 2: Implement BreakoutEntry**

Create `freqtrade/ohio/adapters/freqtrade/entries/breakout.py`:

```python
"""Breakout entry strategy — Bollinger Squeeze + Donchian + Volume.

Sources:
- Arda (SSRN 5775962): Volatility contraction → expansion
- Wen et al. (SSRN 4080253): Crypto technical analysis efficiency
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "bo_squeeze_min_bars": 6.0,
    "bo_volume_mult": 1.5,
    "bo_donchian_window": 20.0,
}


class BreakoutEntry:
    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        sq_min = int(params.get("bo_squeeze_min_bars", DEFAULT_PARAMS["bo_squeeze_min_bars"]))
        vol_mult = params.get("bo_volume_mult", DEFAULT_PARAMS["bo_volume_mult"])

        close = dataframe["close"]
        volume = dataframe.get("volume", pd.Series(0.0, index=dataframe.index))
        squeeze = dataframe.get("ohio_feat_squeeze_count", pd.Series(0.0, index=dataframe.index))
        don_upper = dataframe.get("ohio_feat_donchian_upper_20", pd.Series(float("inf"), index=dataframe.index))
        don_lower = dataframe.get("ohio_feat_donchian_lower_20", pd.Series(0.0, index=dataframe.index))
        vol_sma = dataframe.get("ohio_feat_volume_sma_20", pd.Series(1.0, index=dataframe.index))

        # Squeeze must have been active (use current or shift(1) for "just released")
        squeeze_gate = (squeeze >= sq_min) | (squeeze.shift(1) >= sq_min)
        volume_gate = volume > (vol_mult * vol_sma)

        long_mask = squeeze_gate & volume_gate & (close > don_upper)
        short_mask = squeeze_gate & volume_gate & (close < don_lower)

        dataframe["enter_long"] = long_mask.astype(int)
        dataframe["enter_short"] = short_mask.astype(int)
        return dataframe
```

- [ ] **Step 3: Run tests and commit**

Run: `cd /c/Business/ohio-backend-v2/freqtrade && python -m pytest tests/ohio/entries/test_breakout.py -v`

```bash
git add freqtrade/ohio/adapters/freqtrade/entries/breakout.py \
        tests/ohio/entries/test_breakout.py
git commit -m "feat: add BreakoutEntry (Bollinger Squeeze + Donchian + volume)"
```

---

### Task 6: Implement DefensiveEntry

**Files:**
- Create: `freqtrade/ohio/adapters/freqtrade/entries/defensive.py`
- Create: `tests/ohio/entries/test_defensive.py`

- [ ] **Step 1: Write failing tests**

Create `tests/ohio/entries/test_defensive.py`:

```python
"""Tests for DefensiveEntry strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freqtrade.ohio.adapters.freqtrade.entries.defensive import (
    DefensiveEntry,
    DEFAULT_PARAMS as DEF_DEFAULTS,
)


@pytest.fixture
def strategy() -> DefensiveEntry:
    return DefensiveEntry()


def _make_calm_market(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({
        "close": np.full(n, 100.0),
        "ohio_stable_trend": np.full(n, 0.05),  # mild positive
        "ohio_feat_adx_14": np.full(n, 15.0),  # low ADX = calm
    })


def _make_volatile_market(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({
        "close": np.full(n, 100.0),
        "ohio_stable_trend": np.full(n, 0.3),
        "ohio_feat_adx_14": np.full(n, 40.0),  # high ADX = trending
    })


class TestDefensiveEntry:
    def test_calm_market_enters(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_long"].sum() > 0

    def test_volatile_market_blocked(self, strategy: DefensiveEntry) -> None:
        df = _make_volatile_market()
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_negative_trend_enters_short(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        df["ohio_stable_trend"] = -0.05
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_short"].sum() > 0

    def test_flat_trend_blocked(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        df["ohio_stable_trend"] = 0.001  # below min_trend_abs
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_long"].sum() == 0

    def test_custom_adx_threshold(self, strategy: DefensiveEntry) -> None:
        df = _make_calm_market()
        df["ohio_feat_adx_14"] = 25.0
        result = strategy.compute_entries(df, DEF_DEFAULTS)
        assert result["enter_long"].sum() == 0  # default max=20

        custom = {**DEF_DEFAULTS, "def_adx_max": 30.0}
        result2 = strategy.compute_entries(df, custom)
        assert result2["enter_long"].sum() > 0
```

- [ ] **Step 2: Implement DefensiveEntry**

Create `freqtrade/ohio/adapters/freqtrade/entries/defensive.py`:

```python
"""Defensive entry strategy — vol-targeting + low-ADX gate.

Sources:
- Moreira & Muir 2017: Volatility targeting
- Faber 2007: Tactical asset allocation
"""
from __future__ import annotations

import pandas as pd


DEFAULT_PARAMS: dict[str, float] = {
    "def_adx_max": 20.0,
    "def_vol_scale": 0.15,
    "def_min_trend_abs": 0.03,
}


class DefensiveEntry:
    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float],
    ) -> pd.DataFrame:
        adx_max = params.get("def_adx_max", DEFAULT_PARAMS["def_adx_max"])
        min_trend = params.get("def_min_trend_abs", DEFAULT_PARAMS["def_min_trend_abs"])

        trend = dataframe.get("ohio_stable_trend", pd.Series(0.0, index=dataframe.index))
        adx = dataframe.get("ohio_feat_adx_14", pd.Series(50.0, index=dataframe.index))

        calm_gate = adx < adx_max

        long_mask = calm_gate & (trend > min_trend)
        short_mask = calm_gate & (trend < -min_trend)

        dataframe["enter_long"] = long_mask.astype(int)
        dataframe["enter_short"] = short_mask.astype(int)
        return dataframe
```

- [ ] **Step 3: Run tests and commit**

Run: `cd /c/Business/ohio-backend-v2/freqtrade && python -m pytest tests/ohio/entries/test_defensive.py -v`

```bash
git add freqtrade/ohio/adapters/freqtrade/entries/defensive.py \
        tests/ohio/entries/test_defensive.py
git commit -m "feat: add DefensiveEntry (vol-targeting + low-ADX gate)"
```

---

### Task 7: Add Hyperopt Parameters + Wire EntryAdapter into thin_strategy.py

**Files:**
- Modify: `freqtrade/ohio/adapters/freqtrade/thin_strategy.py`
- Modify: `tests/ohio/test_thin_strategy.py`

- [ ] **Step 1: Add 14 new hyperopt parameters to OhioThinStrategy**

Add after existing buy-space params:

```python
# --- Mode-specific entry parameters (buy space) ---

# Trend Following
tf_adx_threshold = DecimalParameter(15.0, 40.0, default=25.0, decimals=1, space="buy", optimize=True)
tf_hurst_threshold = DecimalParameter(0.45, 0.70, default=0.55, decimals=2, space="buy", optimize=True)
tf_kama_slope_threshold = DecimalParameter(0.0001, 0.005, default=0.001, decimals=4, space="buy", optimize=True)
tf_vol_scale_target = DecimalParameter(0.5, 2.0, default=1.0, decimals=1, space="buy", optimize=True)

# Mean Reversion
mr_zscore_entry = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=True)
mr_rsi_oversold = IntParameter(20, 40, default=30, space="buy", optimize=True)
mr_rsi_overbought = IntParameter(60, 80, default=70, space="buy", optimize=True)
mr_hurst_max = DecimalParameter(0.35, 0.55, default=0.45, decimals=2, space="buy", optimize=True)

# Breakout
bo_squeeze_min_bars = IntParameter(3, 10, default=6, space="buy", optimize=True)
bo_volume_mult = DecimalParameter(1.2, 2.5, default=1.5, decimals=1, space="buy", optimize=True)
bo_donchian_window = IntParameter(10, 30, default=20, space="buy", optimize=True)

# Defensive
def_adx_max = DecimalParameter(15.0, 30.0, default=20.0, decimals=1, space="buy", optimize=True)
def_vol_scale = DecimalParameter(0.05, 0.30, default=0.15, decimals=2, space="buy", optimize=True)
def_min_trend_abs = DecimalParameter(0.01, 0.10, default=0.03, decimals=2, space="buy", optimize=True)
```

- [ ] **Step 2: Add `_entry_params()` helper and instantiate EntryAdapter**

In `__init__`:
```python
self._entry_adapter = EntryAdapter()
```

Add helper:
```python
def _entry_params(self) -> dict[str, float]:
    return {
        "tf_adx_threshold": self.tf_adx_threshold.value,
        "tf_hurst_threshold": self.tf_hurst_threshold.value,
        "tf_kama_slope_threshold": self.tf_kama_slope_threshold.value,
        "tf_vol_scale_target": self.tf_vol_scale_target.value,
        "mr_zscore_entry": self.mr_zscore_entry.value,
        "mr_rsi_oversold": float(self.mr_rsi_oversold.value),
        "mr_rsi_overbought": float(self.mr_rsi_overbought.value),
        "mr_hurst_max": self.mr_hurst_max.value,
        "bo_squeeze_min_bars": float(self.bo_squeeze_min_bars.value),
        "bo_volume_mult": self.bo_volume_mult.value,
        "bo_donchian_window": float(self.bo_donchian_window.value),
        "def_adx_max": self.def_adx_max.value,
        "def_vol_scale": self.def_vol_scale.value,
        "def_min_trend_abs": self.def_min_trend_abs.value,
    }
```

- [ ] **Step 3: Rewrite populate_entry_trend to use EntryAdapter**

Replace the current single-logic entry with per-mode dispatch:

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    logger.info("ohio.populate_entry_trend | pair=%s", metadata["pair"])
    dataframe["enter_long"] = 0
    dataframe["enter_short"] = 0
    dataframe["enter_tag"] = ""

    enabled = dataframe["ohio_policy_enabled"] == True  # noqa: E712
    active = dataframe["ohio_active_mode"]
    trend = dataframe.get("ohio_stable_trend", pd.Series(0.0, index=dataframe.index))

    # Fitness gate
    threshold = self.disabled_threshold.value
    entry_adj = dataframe.get(
        "ohio_policy_entry_threshold_adj",
        pd.Series(0.0, index=dataframe.index),
    )
    fitness = pd.Series(np.nan, index=dataframe.index)
    for mode in ["trend_following", "mean_reversion", "breakout", "defensive"]:
        mask = active == mode
        col = f"ohio_fitness_{mode}"
        if col in dataframe.columns:
            fitness = fitness.where(~mask, dataframe[col])
    fitness_gate = fitness.fillna(0.0) >= (threshold + entry_adj)

    # Regime cooldown gate
    cooldown = self.regime_cooldown_bars.value
    cooldown_gate = pd.Series(True, index=dataframe.index)
    if cooldown > 0:
        mode_changed = active != active.shift(1)
        regime_group = mode_changed.cumsum()
        regime_age = regime_group.groupby(regime_group).cumcount()
        cooldown_gate = regime_age >= cooldown

    base_mask = enabled.fillna(False) & fitness_gate & cooldown_gate

    # --- Mode-specific entry via EntryAdapter ---
    entry_params = self._entry_params_dict()
    for mode in ["trend_following", "mean_reversion", "breakout", "defensive"]:
        mode_rows = active == mode
        if not mode_rows.any():
            continue
        mode_df = dataframe.loc[mode_rows].copy()
        mode_df = self._entry_adapter.compute_entries(mode_df, mode, entry_params)
        mode_mask = base_mask & mode_rows

        long_mask = mode_mask & (mode_df["enter_long"] == 1).reindex(dataframe.index, fill_value=False)
        short_mask = mode_mask & (mode_df["enter_short"] == 1).reindex(dataframe.index, fill_value=False)

        dataframe.loc[long_mask, "enter_long"] = 1
        dataframe.loc[long_mask, "enter_tag"] = f"ohio_{mode}"
        dataframe.loc[short_mask, "enter_short"] = 1
        dataframe.loc[short_mask, "enter_tag"] = f"ohio_{mode}_short"

    return dataframe
```

- [ ] **Step 4: Update tests and run full suite**

Run: `cd /c/Business/ohio-backend-v2/freqtrade && python -m pytest tests/ohio/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add freqtrade/ohio/adapters/freqtrade/thin_strategy.py tests/ohio/test_thin_strategy.py
git commit -m "feat: wire EntryAdapter + 14 mode-specific hyperopt params into thin_strategy"
```

---

### Task 8: Full Integration Test + Run All Tests

**Files:**
- All test files

- [ ] **Step 1: Run full ohio test suite**

```bash
cd /c/Business/ohio-backend-v2/freqtrade && python -m pytest tests/ohio/ -v --tb=short
```

- [ ] **Step 2: Fix any failures**

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "test: fix integration issues for mode-specific entry strategies"
```
