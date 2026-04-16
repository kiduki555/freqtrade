"""GT-Score inspired robust loss function for OHIO strategy optimization.

Combines multiple metrics to resist overfitting:
- Mean profit (performance)
- Z-score vs zero (statistical significance gate)
- R-squared of cumulative returns (consistency)
- Downside deviation (risk)
- Trade count regularization
- Drawdown penalty

Reference: "The GT-Score" (arXiv 2602.00080)
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


class OhioRobustHyperOptLoss(IHyperOptLoss):
    """GT-Score inspired loss: penalizes overfitting via statistical significance gate."""

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: dict | None = None,
        processed: dict | None = None,
        backtest_stats: dict | None = None,
        *,
        starting_balance: float = 0,
        **kwargs,
    ) -> float:
        # Reject insufficient trade counts
        if trade_count < 30:
            return 1e5

        profit_ratios = results["profit_ratio"]
        mean_profit = profit_ratios.mean()
        std_profit = profit_ratios.std()

        # 1. Statistical significance: z-score of mean profit
        z_score = (mean_profit * math.sqrt(trade_count)) / std_profit if std_profit > 0 else 0.0
        log_z = math.log(max(z_score, 0.01))

        # 2. Consistency: R-squared of cumulative returns
        cum_returns = profit_ratios.cumsum().values
        x = np.arange(len(cum_returns))
        if len(x) > 1:
            corr_matrix = np.corrcoef(x, cum_returns)
            corr = corr_matrix[0, 1]
            r_squared = corr ** 2 if not np.isnan(corr) else 0.0
        else:
            r_squared = 0.0

        # 3. Downside deviation (Sortino-style)
        negative = profit_ratios[profit_ratios < 0]
        downside_dev = float(np.sqrt((negative ** 2).mean())) if len(negative) > 0 else 0.001

        # 4. Trade count regularization (target: 50+ trades)
        trade_penalty = min(1.0, trade_count / 50.0)

        # 5. Drawdown penalty
        if backtest_stats and starting_balance > 0:
            max_dd = backtest_stats.get("max_drawdown_abs", 0)
            dd_ratio = max_dd / starting_balance
        else:
            dd_ratio = 0.0

        # GT-Score composite
        if mean_profit <= 0 or log_z <= 0:
            # Negative or insignificant: just use raw loss scaled by trade count
            score = mean_profit * trade_penalty
        else:
            score = (mean_profit * log_z * r_squared * trade_penalty) / (downside_dev + 0.001)
            score *= (1.0 - dd_ratio * 0.5)  # drawdown penalty

        # Return negative (hyperopt minimizes)
        return -score
