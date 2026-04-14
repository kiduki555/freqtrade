"""FT-014: RiskGateAdapter for the OHIO Market State Engine.

Pre-entry risk gate that checks multiple blocking conditions before allowing
a trade entry.  Returns all blocking reasons for debugging visibility.

No Freqtrade framework imports — only ohio/core domain types.
"""
from __future__ import annotations

from freqtrade.ohio.core.domain.models import DataMode, ExecutionPolicy, StateMeta
from freqtrade.ohio.core.portfolio_risk.drawdown_controller import DrawdownController
from freqtrade.ohio.core.portfolio_risk.kill_switch import KillSwitch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DAILY_LOSS_LIMIT: float = 0.03          # 3 % daily loss cap
_MIN_CONFIDENCE: float = 0.30            # minimum acceptable confidence
_MAX_TRANSITION_RISK: float = 0.85       # maximum acceptable transition risk


# ---------------------------------------------------------------------------
# RiskGateAdapter
# ---------------------------------------------------------------------------

class RiskGateAdapter:
    """Evaluate pre-entry risk rules and return (allowed, blocking_reasons).

    All rules are evaluated — the adapter does **not** short-circuit so that
    every blocking reason is surfaced for debugging.
    """

    def __init__(
        self,
        dd_controller: DrawdownController,
        kill_switch: KillSwitch,
    ) -> None:
        self._dd_controller = dd_controller
        self._kill_switch = kill_switch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def confirm_entry(
        self,
        pair: str,
        side: str,
        policy: ExecutionPolicy,
        meta: StateMeta,
        open_trade_count: int,
        daily_loss_pct: float,
    ) -> tuple[bool, list[str]]:
        """Check all risk rules and return ``(allowed, reasons)``.

        Parameters
        ----------
        pair:
            Trading pair symbol (e.g. ``"BTC/USDT"``).
        side:
            ``"long"`` or ``"short"``.
        policy:
            Current execution policy from the strategy router.
        meta:
            Inference quality metadata for the current state.
        open_trade_count:
            Number of currently open positions.
        daily_loss_pct:
            Today's realised loss as a positive fraction (e.g. 0.02 = 2 %).

        Returns
        -------
        tuple[bool, list[str]]
            ``(True, [])`` when entry is permitted, otherwise
            ``(False, [<rule_name>, ...])`` listing every rule that blocked.
        """
        reasons: list[str] = []

        # 1. kill switch
        if self._kill_switch.active:
            reasons.append("kill_switch_active")

        # 2. policy disabled
        if not policy.enabled:
            reasons.append("policy_disabled")

        # 3. drawdown block
        if not self._dd_controller.entry_allowed:
            reasons.append("drawdown_block")

        # 4. data degraded
        if meta.data_mode == DataMode.DEGRADED:
            reasons.append("data_degraded")

        # 5. low confidence
        if meta.confidence < _MIN_CONFIDENCE:
            reasons.append("low_confidence")

        # 6. high transition risk
        if meta.transition_risk > _MAX_TRANSITION_RISK:
            reasons.append("high_transition_risk")

        # 7. max positions reached
        if open_trade_count >= policy.max_positions:
            reasons.append("max_positions_reached")

        # 8. daily loss limit
        if daily_loss_pct >= _DAILY_LOSS_LIMIT:
            reasons.append("daily_loss_limit")

        allowed = len(reasons) == 0
        return allowed, reasons
