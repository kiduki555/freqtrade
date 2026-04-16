# Portfolio Deployment Guide

## Overview

The recommended production setup is a **20/80 Ohio/Momentum4H** portfolio.

- **Target CAGR**: 20%
- **Target Sharpe**: 1.67
- **Expected max drawdown**: 6-10%

## Architecture

Freqtrade runs one strategy per bot. To deploy the portfolio, run **two separate bot instances** with different capital allocations.

```
Bot 1 (Ohio):      $2,000 capital (20% of $10k) → runs OhioThinStrategy on 1H
Bot 2 (Momentum):  $8,000 capital (80% of $10k) → runs OhioMomentum4H on 4H
```

## Backtested Performance (2024-08 to 2026-04)

Market change: -37.34% (bear dominant)

| Strategy | Profit | CAGR | Sharpe | Max DD |
|----------|--------|------|--------|--------|
| Ohio alone | +2.70% | 1.61% | 0.67 | 2.31% |
| Momentum4H alone | +43.98% | 24.46% | 1.65 | 10.22% |
| **20/80 Portfolio** | **+35.72%** | **20.11%** | **1.67** | **6.05%** |

## Deployment Steps

### 1. Directory Structure

```bash
cd C:/Business/ohio-backend-v2/freqtrade
```

### 2. Paper Trading (Dry Run)

Two terminal windows (or use systemd / pm2 for persistence):

**Terminal 1 — Ohio bot (1H)**:
```bash
freqtrade trade \
  --config user_data/config_portfolio.json \
  --strategy OhioThinStrategy \
  --dry-run-wallet 2000 \
  --db-url sqlite:///user_data/ohio_bot.sqlite
```

**Terminal 2 — Momentum4H bot (4H)**:
```bash
freqtrade trade \
  --config user_data/config_portfolio.json \
  --strategy OhioMomentum4H \
  --dry-run-wallet 8000 \
  --db-url sqlite:///user_data/momentum_bot.sqlite
```

Each bot will have its own SQLite DB and won't interfere with the other.

**Important**: change `api_server.listen_port` to different values for each bot
(e.g., 8080 for Ohio, 8081 for Momentum) if running both API servers.

### 3. Live Trading

Same as above but set `"dry_run": false` in config and provide real exchange API
keys. Ensure:
- Sub-accounts or separate exchange accounts recommended for clean bookkeeping.
- Initial wallet sizes match intended allocation (2k + 8k).
- Each bot's API server on different ports.

### 4. Monitoring

Use Freqtrade's FreqUI or API endpoints to monitor both bots. Key metrics:
- Combined P&L
- Individual strategy drawdowns
- Correlation of returns (should be moderate — ~0.3-0.5)

## Periodic Re-Hyperopt

Re-run hyperopt every 3-6 months with fresh data:

```bash
freqtrade hyperopt \
  --strategy OhioMomentum4H \
  --config user_data/config_portfolio.json \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --timerange <last-15-months> \
  --epochs 250 --min-trades 100 \
  --spaces buy sell
```

Validate OOS (last 1-2 months) before applying new parameters.

## Risk Management

- **Position sizing**: handled internally by each strategy
- **Max simultaneous trades**: 10 per bot (configurable via `max_open_trades`)
- **Kill switch**: enabled via OhioThinStrategy's risk gate
- **Circuit breaker**: if combined DD > 15%, pause both bots manually

## Known Limitations

1. **Bull run weakness**: Momentum4H loses ~1% in parabolic bull runs (e.g., +88% BTC moves)
2. **Correlation risk**: Both bots trade same pairs — true diversification limited
3. **Compounding**: Each bot compounds independently; occasional rebalancing needed

## Future Improvements

1. **15m timeframe strategy** to catch bull-run pullbacks (currently no strategy excels in parabolic bulls)
2. **Correlation-based rebalancing** when both bots drawdown simultaneously
3. **Regime-aware capital allocation** (more to Ohio in bull, more to Momentum in bear)
