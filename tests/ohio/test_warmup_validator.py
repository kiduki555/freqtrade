"""Tests for OHIO warmup data validator."""

from __future__ import annotations

import ast
import pathlib

from freqtrade.ohio.adapters.freqtrade.warmup_validator import validate_warmup

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "freqtrade"
    / "ohio"
    / "adapters"
    / "freqtrade"
    / "warmup_validator.py"
)


class TestValidateWarmup:
    """validate_warmup behaviour."""

    def test_exact_minimum_passes(self) -> None:
        ok, msg = validate_warmup(4320)
        assert ok is True
        assert "OK" in msg

    def test_above_minimum_passes(self) -> None:
        ok, msg = validate_warmup(5000)
        assert ok is True
        assert "OK" in msg

    def test_below_minimum_fails(self) -> None:
        ok, msg = validate_warmup(4319)
        assert ok is False
        assert "INSUFFICIENT" in msg

    def test_zero_rows_fails(self) -> None:
        ok, msg = validate_warmup(0)
        assert ok is False
        assert "INSUFFICIENT" in msg

    def test_custom_startup_candle_count(self) -> None:
        ok, _ = validate_warmup(100, startup_candle_count=100)
        assert ok is True

        ok, _ = validate_warmup(99, startup_candle_count=100)
        assert ok is False

    def test_message_contains_row_count(self) -> None:
        _, msg = validate_warmup(3000)
        assert "3000" in msg

    def test_message_contains_day_count(self) -> None:
        _, msg = validate_warmup(4320)
        # 4320 / 24 = 180
        assert "180" in msg

    def test_no_freqtrade_imports(self) -> None:
        """The validator must not import anything from freqtrade."""
        tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("freqtrade"), (
                        f"Forbidden import: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    assert not node.module.startswith("freqtrade"), (
                        f"Forbidden import from: {node.module}"
                    )
