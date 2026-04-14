"""Tests for OHIO domain models (FT-002).

Verifies immutability, construction, and behaviour of all frozen dataclasses.
No Freqtrade imports — domain objects are pure Python.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from freqtrade.ohio.core.domain.models import (
    DataMode,
    ExecutionIntent,
    ExecutionPolicy,
    MarketStateSnapshot,
    StateMeta,
    StateVector,
    StrategyFitness,
    StrategyMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_vector(**kwargs: float) -> StateVector:
    defaults = dict(
        trend_persistence=0.3,
        volatility_level=0.5,
        downside_pressure=0.2,
        liquidity_stress=0.1,
        relative_strength=0.6,
        correlation_stress=0.4,
        breadth_dispersion=0.3,
    )
    defaults.update(kwargs)
    return StateVector(**defaults)


def _make_state_meta(**kwargs: object) -> StateMeta:
    defaults = dict(
        transition_risk=0.2,
        confidence=0.9,
        stability=0.8,
        data_mode=DataMode.FULL,
    )
    defaults.update(kwargs)
    return StateMeta(**defaults)


def _make_fitness(**kwargs: float) -> StrategyFitness:
    defaults = dict(
        trend_following=0.7,
        mean_reversion=0.3,
        breakout=0.4,
        defensive=0.2,
    )
    defaults.update(kwargs)
    return StrategyFitness(**defaults)


def _make_policy(**kwargs: object) -> ExecutionPolicy:
    defaults = dict(
        strategy_mode=StrategyMode.TREND_FOLLOWING,
        enabled=True,
        size_multiplier=1.0,
        entry_threshold_adj=0.0,
        max_positions=3,
        stoploss_width_adj=0.0,
    )
    defaults.update(kwargs)
    return ExecutionPolicy(**defaults)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestDataMode:
    def test_full_value(self):
        assert DataMode.FULL == "full"

    def test_fallback_value(self):
        assert DataMode.FALLBACK == "fallback"

    def test_degraded_value(self):
        assert DataMode.DEGRADED == "degraded"

    def test_is_str_subclass(self):
        assert isinstance(DataMode.FULL, str)


class TestStrategyMode:
    def test_trend_following_value(self):
        assert StrategyMode.TREND_FOLLOWING == "trend_following"

    def test_mean_reversion_value(self):
        assert StrategyMode.MEAN_REVERSION == "mean_reversion"

    def test_breakout_value(self):
        assert StrategyMode.BREAKOUT == "breakout"

    def test_defensive_value(self):
        assert StrategyMode.DEFENSIVE == "defensive"

    def test_is_str_subclass(self):
        assert isinstance(StrategyMode.BREAKOUT, str)


# ---------------------------------------------------------------------------
# StateVector tests
# ---------------------------------------------------------------------------

class TestStateVector:
    def test_construction_with_valid_values(self):
        sv = _make_state_vector()
        assert sv.trend_persistence == pytest.approx(0.3)
        assert sv.volatility_level == pytest.approx(0.5)
        assert sv.downside_pressure == pytest.approx(0.2)
        assert sv.liquidity_stress == pytest.approx(0.1)
        assert sv.relative_strength == pytest.approx(0.6)
        assert sv.correlation_stress == pytest.approx(0.4)
        assert sv.breadth_dispersion == pytest.approx(0.3)

    def test_negative_trend_persistence_allowed(self):
        sv = _make_state_vector(trend_persistence=-0.8)
        assert sv.trend_persistence == pytest.approx(-0.8)

    def test_frozen_raises_on_mutation(self):
        sv = _make_state_vector()
        with pytest.raises(dataclasses.FrozenInstanceError):
            sv.trend_persistence = 0.9  # type: ignore[misc]

    def test_frozen_raises_on_new_attr(self):
        sv = _make_state_vector()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            sv.nonexistent = 1.0  # type: ignore[attr-defined]

    def test_equality_by_value(self):
        sv1 = _make_state_vector()
        sv2 = _make_state_vector()
        assert sv1 == sv2

    def test_inequality_on_different_values(self):
        sv1 = _make_state_vector(trend_persistence=0.1)
        sv2 = _make_state_vector(trend_persistence=0.9)
        assert sv1 != sv2


# ---------------------------------------------------------------------------
# StateMeta tests
# ---------------------------------------------------------------------------

class TestStateMeta:
    def test_construction(self):
        meta = _make_state_meta()
        assert meta.confidence == pytest.approx(0.9)
        assert meta.data_mode is DataMode.FULL

    def test_frozen_raises_on_mutation(self):
        meta = _make_state_meta()
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.confidence = 0.1  # type: ignore[misc]

    def test_data_mode_degraded(self):
        meta = _make_state_meta(data_mode=DataMode.DEGRADED)
        assert meta.data_mode is DataMode.DEGRADED


# ---------------------------------------------------------------------------
# StrategyFitness tests
# ---------------------------------------------------------------------------

class TestStrategyFitness:
    def test_best_mode_returns_highest(self):
        fitness = _make_fitness(
            trend_following=0.9,
            mean_reversion=0.3,
            breakout=0.5,
            defensive=0.1,
        )
        assert fitness.best_mode is StrategyMode.TREND_FOLLOWING

    def test_best_mode_defensive(self):
        fitness = _make_fitness(
            trend_following=0.1,
            mean_reversion=0.2,
            breakout=0.3,
            defensive=0.95,
        )
        assert fitness.best_mode is StrategyMode.DEFENSIVE

    def test_best_mode_breakout(self):
        fitness = _make_fitness(
            trend_following=0.4,
            mean_reversion=0.4,
            breakout=0.8,
            defensive=0.2,
        )
        assert fitness.best_mode is StrategyMode.BREAKOUT

    def test_best_mode_mean_reversion(self):
        fitness = _make_fitness(
            trend_following=0.3,
            mean_reversion=0.85,
            breakout=0.5,
            defensive=0.1,
        )
        assert fitness.best_mode is StrategyMode.MEAN_REVERSION

    def test_score_for_trend_following(self):
        fitness = _make_fitness(trend_following=0.77)
        assert fitness.score_for(StrategyMode.TREND_FOLLOWING) == pytest.approx(0.77)

    def test_score_for_mean_reversion(self):
        fitness = _make_fitness(mean_reversion=0.55)
        assert fitness.score_for(StrategyMode.MEAN_REVERSION) == pytest.approx(0.55)

    def test_score_for_breakout(self):
        fitness = _make_fitness(breakout=0.42)
        assert fitness.score_for(StrategyMode.BREAKOUT) == pytest.approx(0.42)

    def test_score_for_defensive(self):
        fitness = _make_fitness(defensive=0.18)
        assert fitness.score_for(StrategyMode.DEFENSIVE) == pytest.approx(0.18)

    def test_frozen_raises_on_mutation(self):
        fitness = _make_fitness()
        with pytest.raises(dataclasses.FrozenInstanceError):
            fitness.trend_following = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExecutionPolicy tests
# ---------------------------------------------------------------------------

class TestExecutionPolicy:
    def test_default_construction(self):
        policy = _make_policy()
        assert policy.strategy_mode is StrategyMode.TREND_FOLLOWING
        assert policy.enabled is True
        assert policy.size_multiplier == pytest.approx(1.0)
        assert policy.entry_threshold_adj == pytest.approx(0.0)
        assert policy.max_positions == 3
        assert policy.stoploss_width_adj == pytest.approx(0.0)

    def test_disabled_policy(self):
        policy = _make_policy(enabled=False, size_multiplier=0.3)
        assert policy.enabled is False
        assert policy.size_multiplier == pytest.approx(0.3)

    def test_frozen_raises_on_mutation(self):
        policy = _make_policy()
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.enabled = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MarketStateSnapshot tests
# ---------------------------------------------------------------------------

class TestMarketStateSnapshot:
    def _make_snapshot(self) -> MarketStateSnapshot:
        sv = _make_state_vector()
        meta = _make_state_meta()
        fitness = _make_fitness()
        policy = _make_policy()
        return MarketStateSnapshot(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            timeframe="1h",
            state_vector=sv,
            meta=meta,
            fitness=fitness,
            active_mode=StrategyMode.TREND_FOLLOWING,
            policy=policy,
        )

    def test_holds_all_components(self):
        snap = self._make_snapshot()
        assert snap.symbol == "BTC/USDT"
        assert snap.timeframe == "1h"
        assert snap.active_mode is StrategyMode.TREND_FOLLOWING
        assert isinstance(snap.state_vector, StateVector)
        assert isinstance(snap.meta, StateMeta)
        assert isinstance(snap.fitness, StrategyFitness)
        assert isinstance(snap.policy, ExecutionPolicy)

    def test_frozen_raises_on_mutation(self):
        snap = self._make_snapshot()
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.symbol = "ETH/USDT"  # type: ignore[misc]

    def test_nested_state_vector_accessible(self):
        snap = self._make_snapshot()
        assert snap.state_vector.trend_persistence == pytest.approx(0.3)

    def test_nested_meta_data_mode(self):
        snap = self._make_snapshot()
        assert snap.meta.data_mode is DataMode.FULL


# ---------------------------------------------------------------------------
# ExecutionIntent tests
# ---------------------------------------------------------------------------

class TestExecutionIntent:
    def test_required_fields(self):
        intent = ExecutionIntent(
            symbol="ETH/USDT",
            mode=StrategyMode.BREAKOUT,
            direction="long",
            size_multiplier=1.2,
            leverage=2.0,
            stoploss=-0.05,
        )
        assert intent.symbol == "ETH/USDT"
        assert intent.mode is StrategyMode.BREAKOUT
        assert intent.direction == "long"
        assert intent.size_multiplier == pytest.approx(1.2)
        assert intent.leverage == pytest.approx(2.0)
        assert intent.stoploss == pytest.approx(-0.05)

    def test_optional_defaults(self):
        intent = ExecutionIntent(
            symbol="BTC/USDT",
            mode=StrategyMode.DEFENSIVE,
            direction="short",
            size_multiplier=0.5,
            leverage=1.0,
            stoploss=-0.03,
        )
        assert intent.take_profit is None
        assert intent.max_hold_bars is None
        assert intent.entry_reason == ""

    def test_optional_fields_provided(self):
        intent = ExecutionIntent(
            symbol="BTC/USDT",
            mode=StrategyMode.TREND_FOLLOWING,
            direction="long",
            size_multiplier=1.0,
            leverage=3.0,
            stoploss=-0.04,
            take_profit=0.12,
            max_hold_bars=24,
            entry_reason="trend_breakout",
        )
        assert intent.take_profit == pytest.approx(0.12)
        assert intent.max_hold_bars == 24
        assert intent.entry_reason == "trend_breakout"

    def test_frozen_raises_on_mutation(self):
        intent = ExecutionIntent(
            symbol="BTC/USDT",
            mode=StrategyMode.DEFENSIVE,
            direction="long",
            size_multiplier=1.0,
            leverage=1.0,
            stoploss=-0.02,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            intent.symbol = "ETH/USDT"  # type: ignore[misc]
