# Mode-Specific Entry Strategies Design

**Date:** 2026-04-15
**Status:** Approved
**Author:** Claude (collaborative design with user)

---

## Problem

Current OHIO pipeline classifies 4 regimes (trend_following, mean_reversion, breakout, defensive) but uses **identical entry logic** for all modes: `trend_sign >= 0 → long, < 0 → short`. This means:

- Mean reversion doesn't fade extremes — it follows trends like TF
- Breakout doesn't detect squeeze/expansion patterns
- Defensive doesn't reduce exposure
- All modes produce the same signals, defeating the purpose of regime classification

## Solution

**Adapter pattern** — new `EntryAdapter` dispatches to mode-specific entry strategy modules, each with its own hyperopt-tunable parameters.

## Architecture

```
thin_strategy.py::populate_entry_trend()
  └─ EntryAdapter.compute_entries(mode, dataframe, profile)
       ├─ TrendFollowingEntry   (TSMOM + KAMA + vol-scale)
       ├─ MeanReversionEntry    (Z-score + Hurst gate + BB + RSI)
       ├─ BreakoutEntry         (Bollinger Squeeze + Donchian + volume)
       └─ DefensiveEntry        (vol-target + low-ADX gate)
```

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `adapters/freqtrade/entry_adapter.py` | Protocol + dispatcher, maps mode → strategy |
| `adapters/freqtrade/entries/__init__.py` | Package init |
| `adapters/freqtrade/entries/trend_following.py` | TSMOM + KAMA entry logic |
| `adapters/freqtrade/entries/mean_reversion.py` | Z-score + Hurst gate entry |
| `adapters/freqtrade/entries/breakout.py` | Squeeze + Donchian entry |
| `adapters/freqtrade/entries/defensive.py` | Vol-targeting minimal entry |
| `tests/ohio/entries/__init__.py` | Test package |
| `tests/ohio/entries/test_trend_following.py` | TF entry tests |
| `tests/ohio/entries/test_mean_reversion.py` | MR entry tests |
| `tests/ohio/entries/test_breakout.py` | BO entry tests |
| `tests/ohio/entries/test_defensive.py` | DEF entry tests |
| `tests/ohio/test_entry_adapter.py` | Dispatcher tests |

### Modified Files

| File | Changes |
|------|---------|
| `core/market_state/feature_builder.py` | Add 12 new indicator features |
| `adapters/freqtrade/thin_strategy.py` | Add mode-specific hyperopt params, wire EntryAdapter |
| `tests/ohio/test_feature_builder.py` | Tests for new features |

## New Features (feature_builder.py)

12 new `ohio_feat_*` columns:

| Column | Formula | Used By |
|--------|---------|---------|
| `ohio_feat_zscore_20` | `(close - SMA(20)) / std(20)` | MR |
| `ohio_feat_rsi_14` | Wilder RSI(14) | MR |
| `ohio_feat_kama_10` | KAMA(10, fast=2, slow=30) | TF |
| `ohio_feat_kama_slope` | `(kama - kama.shift(5)) / close` | TF |
| `ohio_feat_bb_upper_20` | `SMA(20) + 2*std(20)` | MR, BO |
| `ohio_feat_bb_lower_20` | `SMA(20) - 2*std(20)` | MR, BO |
| `ohio_feat_kc_upper_20` | `EMA(20) + 1.5*ATR(20)` | BO |
| `ohio_feat_kc_lower_20` | `EMA(20) - 1.5*ATR(20)` | BO |
| `ohio_feat_donchian_upper_20` | `max(high, 20)` | BO |
| `ohio_feat_donchian_lower_20` | `min(low, 20)` | BO |
| `ohio_feat_squeeze_count` | Consecutive bars BB inside KC | BO |
| `ohio_feat_volume_sma_20` | `SMA(volume, 20)` | BO |

## Entry Strategy Protocol

```python
class EntryStrategy(Protocol):
    def compute_entries(
        self, dataframe: pd.DataFrame, params: dict[str, float]
    ) -> pd.DataFrame:
        """Add enter_long_X and enter_short_X columns. Pure vectorized."""
        ...
```

Each strategy operates on the full dataframe (vectorized, no row iteration) and returns boolean mask columns.

## Mode-Specific Hyperopt Parameters

### Trend Following (space="buy")

| Parameter | Type | Range | Default | Purpose |
|-----------|------|-------|---------|---------|
| `tf_adx_threshold` | Decimal | 15-40 | 25 | Min ADX for entry |
| `tf_hurst_threshold` | Decimal | 0.45-0.70 | 0.55 | Min Hurst for persistence |
| `tf_kama_slope_threshold` | Decimal | 0.0001-0.005 | 0.001 | Min KAMA slope |
| `tf_vol_scale_target` | Decimal | 0.5-2.0 | 1.0 | Target vol for position sizing |

### Mean Reversion (space="buy")

| Parameter | Type | Range | Default | Purpose |
|-----------|------|-------|---------|---------|
| `mr_zscore_entry` | Decimal | 1.5-3.0 | 2.0 | Z-score threshold for entry |
| `mr_rsi_oversold` | Int | 20-40 | 30 | RSI oversold level |
| `mr_rsi_overbought` | Int | 60-80 | 70 | RSI overbought level |
| `mr_hurst_max` | Decimal | 0.35-0.55 | 0.45 | Max Hurst (anti-trend gate) |

### Breakout (space="buy")

| Parameter | Type | Range | Default | Purpose |
|-----------|------|-------|---------|---------|
| `bo_squeeze_min_bars` | Int | 3-10 | 6 | Min squeeze duration |
| `bo_volume_mult` | Decimal | 1.2-2.5 | 1.5 | Vol > X * SMA(20) |
| `bo_donchian_window` | Int | 10-30 | 20 | Donchian channel period |

### Defensive (space="buy")

| Parameter | Type | Range | Default | Purpose |
|-----------|------|-------|---------|---------|
| `def_adx_max` | Decimal | 15-30 | 20 | Max ADX (avoid trends) |
| `def_vol_scale` | Decimal | 0.05-0.30 | 0.15 | Position size fraction |
| `def_min_trend_abs` | Decimal | 0.01-0.10 | 0.03 | Min |trend| for direction |

Total: 14 new hyperopt parameters (all in "buy" space for separate stage optimization).

## Entry Logic Detail

### Trend Following

```
long:  KAMA_slope > tf_kama_slope_threshold
       AND ADX > tf_adx_threshold
       AND Hurst > tf_hurst_threshold
       AND ohio_stable_trend > 0

short: KAMA_slope < -tf_kama_slope_threshold
       AND ADX > tf_adx_threshold
       AND Hurst > tf_hurst_threshold
       AND ohio_stable_trend < 0
```

### Mean Reversion

```
long:  zscore < -mr_zscore_entry
       AND RSI < mr_rsi_oversold
       AND Hurst < mr_hurst_max
       AND close <= bb_lower (touch)

short: zscore > mr_zscore_entry
       AND RSI > mr_rsi_overbought
       AND Hurst < mr_hurst_max
       AND close >= bb_upper (touch)
```

### Breakout

```
long:  squeeze_count >= bo_squeeze_min_bars
       AND close > donchian_upper
       AND volume > bo_volume_mult * volume_sma_20

short: squeeze_count >= bo_squeeze_min_bars
       AND close < donchian_lower
       AND volume > bo_volume_mult * volume_sma_20
```

### Defensive

```
long:  ADX < def_adx_max
       AND ohio_stable_trend > def_min_trend_abs
       (position scaled by def_vol_scale)

short: ADX < def_adx_max
       AND ohio_stable_trend < -def_min_trend_abs
       (position scaled by def_vol_scale)
```

## Fallback

If `EntryAdapter` encounters an unknown mode or all conditions fail, it falls back to the existing trend-sign logic (backward compatible).

## Integration with Existing Pipeline

- `populate_entry_trend()` calls `EntryAdapter` instead of raw trend-sign logic
- Existing fitness gate, regime cooldown gate, and direction gate remain
- Mode-specific entry replaces only the final long/short decision
- Exit logic (exit_adapter.py) unchanged
- Position sizing (position_adapter.py) unchanged

## Academic Sources

| Mode | Primary Source | Key Insight |
|------|---------------|-------------|
| TF | Huang et al. 2024 (SSRN 4825389) | Vol-weighted TSMOM, Sharpe 2.17 for crypto |
| TF | Sepp (SSRN 3167787) | Vol-scaling improves trend signals |
| MR | Avellaneda & Lee 2010 | Statistical arbitrage z-score framework |
| MR | Beluska & Vojtko (SSRN 4955617) | Hurst exponent as MR filter |
| BO | Arda (SSRN 5775962) | Vol contraction → expansion pattern |
| BO | Wen et al. (SSRN 4080253) | Crypto technical analysis efficiency |
| DEF | Moreira & Muir 2017 | Vol-targeting for risk-adjusted returns |
| DEF | Faber 2007 | Tactical asset allocation |
