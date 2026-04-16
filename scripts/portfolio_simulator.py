"""Portfolio simulator for multiple independent strategies.

Given individual strategy returns from backtest, computes combined portfolio
performance with different allocation ratios.

Each strategy runs with its own capital fraction and compounds independently.
"""
from __future__ import annotations

import sys
from pathlib import Path


# Hard-coded results from independent backtests (OOS 2024-08 to 2026-04)
STRATEGIES = {
    "OhioThinStrategy": {
        "profit_pct": 2.70,  # % return over 20 months
        "sharpe": 0.67,
        "max_dd": 2.31,
        "trades": 236,
    },
    "OhioMomentum4H": {
        "profit_pct": 36.23,
        "sharpe": 1.09,
        "max_dd": 10.50,
        "trades": 1675,
    },
}

ALLOCATIONS = [
    ("Ohio 100%", {"OhioThinStrategy": 1.0, "OhioMomentum4H": 0.0}),
    ("Momentum 100%", {"OhioThinStrategy": 0.0, "OhioMomentum4H": 1.0}),
    ("50/50", {"OhioThinStrategy": 0.5, "OhioMomentum4H": 0.5}),
    ("30/70 (more Momentum)", {"OhioThinStrategy": 0.3, "OhioMomentum4H": 0.7}),
    ("40/60", {"OhioThinStrategy": 0.4, "OhioMomentum4H": 0.6}),
    ("70/30 (more Ohio)", {"OhioThinStrategy": 0.7, "OhioMomentum4H": 0.3}),
    ("20/80", {"OhioThinStrategy": 0.2, "OhioMomentum4H": 0.8}),
]


def compute_portfolio(alloc: dict[str, float]) -> dict:
    """Compute blended portfolio metrics.

    Simple weighted average — assumes strategies are uncorrelated (Sharpe boost)
    or correlated (no boost). For crypto multi-signal strats, correlation is
    typically 0.2-0.5.
    """
    total_profit = 0.0
    weighted_sharpe = 0.0
    weighted_dd = 0.0

    for strat, w in alloc.items():
        if w == 0:
            continue
        s = STRATEGIES[strat]
        total_profit += w * s["profit_pct"]
        weighted_sharpe += w * s["sharpe"]
        weighted_dd += w * s["max_dd"]

    # Assume low correlation (0.3) → diversification boost of ~15-20% on Sharpe
    diversification_factor = 1.0
    if len([w for w in alloc.values() if w > 0]) > 1:
        # Approximate boost for uncorrelated strategies
        diversification_factor = 1.15

    boosted_sharpe = weighted_sharpe * diversification_factor

    # DD is typically LOWER than weighted average due to offsetting drawdowns
    # Approximate: 70% of weighted DD for uncorrelated strats
    dd_reduction = 0.7 if diversification_factor > 1.0 else 1.0
    realized_dd = weighted_dd * dd_reduction

    cagr = ((1 + total_profit / 100) ** (12 / 20) - 1) * 100  # annualized

    return {
        "profit_pct": total_profit,
        "cagr": cagr,
        "weighted_sharpe": weighted_sharpe,
        "boosted_sharpe_approx": boosted_sharpe,
        "realized_dd_approx": realized_dd,
    }


def main():
    print("=" * 80)
    print("PORTFOLIO ALLOCATION ANALYSIS (OOS 2024-08 to 2026-04)")
    print("=" * 80)
    print(f"\nMarket change: -37.34%\n")

    print(f"{'Allocation':<25} {'20m Profit':>12} {'CAGR':>10} {'Sharpe':>10} {'DD':>8}")
    print("-" * 75)

    for name, alloc in ALLOCATIONS:
        m = compute_portfolio(alloc)
        print(
            f"{name:<25} "
            f"{m['profit_pct']:>11.2f}% "
            f"{m['cagr']:>9.2f}% "
            f"{m['boosted_sharpe_approx']:>9.2f} "
            f"{m['realized_dd_approx']:>7.2f}%"
        )

    print("\nBest by criteria:")
    best_profit = max(ALLOCATIONS, key=lambda x: compute_portfolio(x[1])["profit_pct"])
    best_sharpe = max(ALLOCATIONS, key=lambda x: compute_portfolio(x[1])["boosted_sharpe_approx"])
    best_calmar = max(ALLOCATIONS, key=lambda x: compute_portfolio(x[1])["profit_pct"] / max(compute_portfolio(x[1])["realized_dd_approx"], 0.01))

    print(f"  Max Profit: {best_profit[0]} → {compute_portfolio(best_profit[1])['profit_pct']:.2f}%")
    print(f"  Max Sharpe: {best_sharpe[0]} → {compute_portfolio(best_sharpe[1])['boosted_sharpe_approx']:.2f}")
    print(f"  Max Calmar: {best_calmar[0]}")

    print("\nNotes:")
    print("- Diversification Sharpe boost (×1.15) assumes correlation ~0.3")
    print("- DD reduction (×0.7) assumes offsetting drawdowns")
    print("- Real-world: validate with live or simulated combined wallet")


if __name__ == "__main__":
    main()
