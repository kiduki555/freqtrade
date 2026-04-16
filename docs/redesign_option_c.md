# Option C: Hybrid Redesign — Ohio Risk Engine + Independent Entry Signals

## Architecture Change

```
Before (Regime-Gated):
  OHLCV → Feature → Normalize → Factor → Stabilize → Meta → Fitness → Mode Select
  → Mode-Specific Entry → Execute

After (Independent Entries + Ohio Risk Engine):
  OHLCV → Feature → Normalize → Factor → Stabilize → Meta → Fitness
  → Risk Engine (sizing, exposure, exit speed)
  
  OHLCV → Multiple Independent Entry Signals (OR logic)
  → Risk Engine gates (sizing, max positions) → Execute
```

## Entry Signals (Independent, OR-combined)

### Signal 1: EMA Crossover + Daily Trend (Trend-based)
- EMA(13) crosses above EMA(48) → long
- EMA(13) crosses below EMA(48) → short  
- MUST confirm with 4H SMA(50) direction
- Source: Mesicek & Vojtko (SSRN 5748642), Bitwise Trendwise ETFs

### Signal 2: BB Dip Buy (Mean Reversion at 1H)
- Close < BB lower band (20, 2.0)
- RSI(14) < 35
- 4H RSI(14) < 70 (not overbought on higher TF)
- BTC 24h return > -5% (no crash)
- Source: CombinedBinHAndCluc pattern

### Signal 3: MACD Divergence + Volume
- MACD histogram increasing while price decreasing (bullish divergence) → long
- MACD histogram decreasing while price increasing (bearish divergence) → short
- Volume > 1.2x SMA(20)
- Source: Classic divergence trading

### Signal 4: Squeeze Release (Breakout)
- BB squeeze ends (BB was inside KC, now expanding)
- Enter direction of first candle after squeeze release
- Donchian(20) break confirms direction
- Source: Existing breakout logic, simplified from 5 conditions to 2

## Ohio Risk Engine (Repurposed)

### Position Sizing (from StateVector)
- High confidence + high fitness → full size
- Low confidence or transition risk → half size
- correlation_stress > 0.7 → reduce max positions to 3

### Exit Speed (from active mode)
- Defensive regime → tight profit target (2%), fast time exit
- Trending regime → wide trailing (3x ATR), let winners run
- High volatility → widen stops to 3x ATR

### Exposure Control
- ohio_factor_downside > 0.6 → max 2 open longs
- ohio_factor_volatility > 0.7 → reduce all position sizes by 50%
- Kill switch unchanged

## Exit Logic (Simplified)

### Primary: ATR Trailing Stop
- Initial stop: 2.5x ATR from entry
- After +1x ATR profit: tighten to 2.0x ATR
- After +2x ATR profit: tighten to 1.5x ATR

### Secondary: ROI Table (keep current hyperopt values)
- {"0": 0.189, "442": 0.081, "1102": 0.032, "1934": 0}

### Remove
- profit_preserve exit (kills winners)
- regime_exit (already problematic)
- partial exits (destroys compounding)
- time_exit (arbitrary)

## Implementation Plan

### Phase 1: New Entry Signals
- Create entries/ema_cross.py (Signal 1)
- Create entries/bb_dip.py (Signal 2)  
- Create entries/macd_divergence.py (Signal 3)
- Simplify entries/breakout.py (Signal 4)
- Delete entries/mean_reversion.py
- Delete entries/trend_following.py
- Delete entries/defensive.py

### Phase 2: Decouple Entries from Regime
- Modify thin_strategy.populate_entry_trend: OR-combine all signals
- Remove mode-based entry gating
- Keep entry_tag per signal for tracking

### Phase 3: Repurpose Ohio as Risk Engine
- Modify custom_stake_amount: use StateVector for sizing
- Modify custom_stoploss: use regime for stop width
- Modify custom_exit: use regime for exit speed
- Remove regime-gated entry logic

### Phase 4: Exit Overhaul
- Implement pure ATR trailing in compute_stoploss
- Remove profit_preserve, regime_exit, time_exit, partial exits
- Keep ROI table

### Phase 5: Add Multi-TF Confirmation
- Add 4H and 1D informative pairs (already partially done)
- Wire 4H/1D trend direction into entry signals
- Add BTC health filter (BTC 24h return)
