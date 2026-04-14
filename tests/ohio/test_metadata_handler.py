"""Tests for FT-018: metadata_handler."""
from __future__ import annotations

import ast
import pathlib

import pytest

from freqtrade.ohio.adapters.freqtrade.metadata_handler import (
    get_entry_confidence,
    get_entry_fitness,
    get_trade_mode,
    save_entry_metadata,
)
from freqtrade.ohio.core.domain.models import DataMode, StateMeta, StrategyFitness


# ---------------------------------------------------------------------------
# Lightweight fake — dict-backed Trade stand-in
# ---------------------------------------------------------------------------

class FakeTrade:
    """Minimal Trade double with custom-data storage."""

    def __init__(self) -> None:
        self._custom_data: dict[str, object] = {}

    def set_custom_data(self, key: str, value: object) -> None:
        self._custom_data[key] = value

    def get_custom_data(self, key: str) -> object | None:
        return self._custom_data.get(key)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fitness() -> StrategyFitness:
    return StrategyFitness(
        trend_following=0.85,
        mean_reversion=0.30,
        breakout=0.55,
        defensive=0.10,
    )


@pytest.fixture()
def meta() -> StateMeta:
    return StateMeta(
        transition_risk=0.25,
        confidence=0.90,
        stability=0.80,
        data_mode=DataMode.FULL,
    )


@pytest.fixture()
def trade() -> FakeTrade:
    return FakeTrade()


# ---------------------------------------------------------------------------
# save_entry_metadata stores all 4 keys
# ---------------------------------------------------------------------------

class TestSaveEntryMetadata:
    def test_stores_all_four_keys(
        self, trade: FakeTrade, fitness: StrategyFitness, meta: StateMeta
    ) -> None:
        save_entry_metadata(trade, "ohio_trend_following", fitness, meta)

        assert trade.get_custom_data("strategy_mode") == "trend_following"
        assert trade.get_custom_data("entry_fitness") == {
            "trend_following": 0.85,
            "mean_reversion": 0.30,
            "breakout": 0.55,
            "defensive": 0.10,
        }
        assert trade.get_custom_data("entry_confidence") == 0.90
        assert trade.get_custom_data("entry_transition_risk") == 0.25

    def test_overwrite_on_duplicate_save(
        self, trade: FakeTrade, fitness: StrategyFitness, meta: StateMeta
    ) -> None:
        save_entry_metadata(trade, "ohio_trend_following", fitness, meta)

        new_fitness = StrategyFitness(
            trend_following=0.10,
            mean_reversion=0.90,
            breakout=0.20,
            defensive=0.50,
        )
        new_meta = StateMeta(
            transition_risk=0.70,
            confidence=0.40,
            stability=0.30,
            data_mode=DataMode.DEGRADED,
        )
        save_entry_metadata(trade, "ohio_defensive", new_fitness, new_meta)

        assert trade.get_custom_data("strategy_mode") == "defensive"
        assert trade.get_custom_data("entry_confidence") == 0.40
        assert trade.get_custom_data("entry_transition_risk") == 0.70
        assert trade.get_custom_data("entry_fitness") == {
            "trend_following": 0.10,
            "mean_reversion": 0.90,
            "breakout": 0.20,
            "defensive": 0.50,
        }


# ---------------------------------------------------------------------------
# strategy_mode extraction from various entry_tags
# ---------------------------------------------------------------------------

class TestStrategyModeExtraction:
    @pytest.mark.parametrize(
        ("entry_tag", "expected"),
        [
            ("ohio_trend_following", "trend_following"),
            ("ohio_defensive", "defensive"),
            ("ohio_breakout", "breakout"),
            ("ohio_mean_reversion", "mean_reversion"),
        ],
    )
    def test_ohio_prefix_stripped(
        self,
        trade: FakeTrade,
        fitness: StrategyFitness,
        meta: StateMeta,
        entry_tag: str,
        expected: str,
    ) -> None:
        save_entry_metadata(trade, entry_tag, fitness, meta)
        assert trade.get_custom_data("strategy_mode") == expected

    def test_no_ohio_prefix_stored_as_is(
        self, trade: FakeTrade, fitness: StrategyFitness, meta: StateMeta
    ) -> None:
        save_entry_metadata(trade, "custom_tag", fitness, meta)
        assert trade.get_custom_data("strategy_mode") == "custom_tag"

    def test_none_entry_tag_stored_as_unknown(
        self, trade: FakeTrade, fitness: StrategyFitness, meta: StateMeta
    ) -> None:
        save_entry_metadata(trade, None, fitness, meta)
        assert trade.get_custom_data("strategy_mode") == "unknown"


# ---------------------------------------------------------------------------
# get_trade_mode
# ---------------------------------------------------------------------------

class TestGetTradeMode:
    def test_returns_correct_mode(
        self, trade: FakeTrade, fitness: StrategyFitness, meta: StateMeta
    ) -> None:
        save_entry_metadata(trade, "ohio_breakout", fitness, meta)
        assert get_trade_mode(trade) == "breakout"

    def test_returns_none_when_not_set(self, trade: FakeTrade) -> None:
        assert get_trade_mode(trade) is None


# ---------------------------------------------------------------------------
# get_entry_fitness
# ---------------------------------------------------------------------------

class TestGetEntryFitness:
    def test_returns_correct_dict(
        self, trade: FakeTrade, fitness: StrategyFitness, meta: StateMeta
    ) -> None:
        save_entry_metadata(trade, "ohio_trend_following", fitness, meta)
        result = get_entry_fitness(trade)
        assert result == {
            "trend_following": 0.85,
            "mean_reversion": 0.30,
            "breakout": 0.55,
            "defensive": 0.10,
        }

    def test_returns_none_when_not_set(self, trade: FakeTrade) -> None:
        assert get_entry_fitness(trade) is None


# ---------------------------------------------------------------------------
# get_entry_confidence
# ---------------------------------------------------------------------------

class TestGetEntryConfidence:
    def test_returns_correct_float(
        self, trade: FakeTrade, fitness: StrategyFitness, meta: StateMeta
    ) -> None:
        save_entry_metadata(trade, "ohio_trend_following", fitness, meta)
        assert get_entry_confidence(trade) == pytest.approx(0.90)

    def test_returns_none_when_not_set(self, trade: FakeTrade) -> None:
        assert get_entry_confidence(trade) is None


# ---------------------------------------------------------------------------
# AST import guard — metadata_handler must NOT import from freqtrade
# (except freqtrade.ohio subpackage)
# ---------------------------------------------------------------------------

class TestNoFreqtradeImports:
    def test_no_direct_freqtrade_imports(self) -> None:
        source_path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "freqtrade"
            / "ohio"
            / "adapters"
            / "freqtrade"
            / "metadata_handler.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("freqtrade") and not alias.name.startswith(
                        "freqtrade.ohio"
                    ):
                        violations.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("freqtrade") and not node.module.startswith(
                    "freqtrade.ohio"
                ):
                    violations.append(node.module)

        assert violations == [], (
            f"metadata_handler.py must not import from freqtrade "
            f"(except freqtrade.ohio). Found: {violations}"
        )
