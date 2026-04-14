"""One-way kill switch — once activated, stays active for the process lifetime."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class KillSwitch:
    """One-way latch that halts all trading when activated.

    There is intentionally **no** ``reset()`` or ``deactivate()`` method.
    The only way to clear the kill switch is to restart the bot
    (i.e. create a new ``KillSwitch`` instance).
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._reason: str | None = None
        self._activated_at: datetime | None = None

    # -- read-only properties ------------------------------------------------

    @property
    def active(self) -> bool:
        """Return ``True`` when the kill switch is engaged."""
        return self._active

    @property
    def reason(self) -> str | None:
        """Return the human-readable reason, or ``None`` if inactive."""
        return self._reason

    @property
    def activated_at(self) -> datetime | None:
        """Return the UTC timestamp of activation, or ``None`` if inactive."""
        return self._activated_at

    # -- mutation ------------------------------------------------------------

    def activate(self, reason: str) -> None:
        """Engage the kill switch.  Subsequent calls are no-ops."""
        if self._active:
            logger.warning(
                "KillSwitch already active (reason=%r) — ignoring duplicate activate",
                self._reason,
            )
            return
        self._active = True
        self._reason = reason
        self._activated_at = datetime.now(tz=timezone.utc)
        logger.critical("KillSwitch ACTIVATED: %s", reason)

    # -- dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        if self._active:
            return (
                f"KillSwitch(active=True, reason={self._reason!r}, "
                f"activated_at={self._activated_at!r})"
            )
        return "KillSwitch(active=False)"
