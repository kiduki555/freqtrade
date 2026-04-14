"""FT-018: Trade metadata persistence helpers.

Saves and retrieves OHIO pipeline metadata on Freqtrade Trade objects
via ``trade.set_custom_data`` / ``trade.get_custom_data``.

Design constraints
------------------
* No imports from the ``freqtrade`` package (except ``freqtrade.ohio``).
* ``trade`` is typed as ``object`` so the module stays decoupled from
  Freqtrade internals.  Attribute access uses ``# type: ignore``.
"""
from __future__ import annotations

from freqtrade.ohio.core.domain.models import StateMeta, StrategyFitness


_OHIO_PREFIX = "ohio_"


def _extract_mode(entry_tag: str | None) -> str:
    """Extract strategy mode name from an entry tag.

    >>> _extract_mode("ohio_trend_following")
    'trend_following'
    >>> _extract_mode("breakout")
    'breakout'
    >>> _extract_mode(None)
    'unknown'
    """
    if entry_tag is None:
        return "unknown"
    if entry_tag.startswith(_OHIO_PREFIX):
        return entry_tag[len(_OHIO_PREFIX):]
    return entry_tag


def _fitness_to_dict(fitness: StrategyFitness) -> dict[str, float]:
    """Convert a StrategyFitness to a JSON-serializable dict."""
    return {
        "trend_following": fitness.trend_following,
        "mean_reversion": fitness.mean_reversion,
        "breakout": fitness.breakout,
        "defensive": fitness.defensive,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_entry_metadata(
    trade: object,
    entry_tag: str | None,
    fitness_scores: StrategyFitness,
    meta_snapshot: StateMeta,
) -> None:
    """Persist OHIO entry metadata on a trade.

    Parameters
    ----------
    trade:
        A Freqtrade ``Trade`` instance (typed as ``object`` to avoid the
        framework import).
    entry_tag:
        The ``enter_tag`` string set by ``populate_entry_trend``.
    fitness_scores:
        ``StrategyFitness`` produced by the pipeline for this candle.
    meta_snapshot:
        ``StateMeta`` produced by the pipeline for this candle.
    """
    mode = _extract_mode(entry_tag)

    trade.set_custom_data("strategy_mode", mode)  # type: ignore[union-attr]
    trade.set_custom_data("entry_fitness", _fitness_to_dict(fitness_scores))  # type: ignore[union-attr]
    trade.set_custom_data("entry_confidence", meta_snapshot.confidence)  # type: ignore[union-attr]
    trade.set_custom_data("entry_transition_risk", meta_snapshot.transition_risk)  # type: ignore[union-attr]


def get_trade_mode(trade: object) -> str | None:
    """Return the saved strategy mode, or ``None`` if not set."""
    return trade.get_custom_data("strategy_mode")  # type: ignore[union-attr]


def get_entry_fitness(trade: object) -> dict[str, float] | None:
    """Return the saved entry fitness dict, or ``None`` if not set."""
    return trade.get_custom_data("entry_fitness")  # type: ignore[union-attr]


def get_entry_confidence(trade: object) -> float | None:
    """Return the saved entry confidence, or ``None`` if not set."""
    return trade.get_custom_data("entry_confidence")  # type: ignore[union-attr]
