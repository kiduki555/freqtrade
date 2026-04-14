"""Tests for FT-010: Strategy Profile YAML + Pydantic Loader.

Covers: profile loading, validation, immutability, idempotency,
        weight sums, range invariants, axis coverage, and error handling.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from freqtrade.ohio.core.domain.models import StrategyMode
from freqtrade.ohio.core.strategy_router.strategy_profile import (
    AxisPreference,
    STATE_VECTOR_AXES,
    StrategyProfile,
    load_default_profiles,
    load_profiles,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_MODES = {
    StrategyMode.TREND_FOLLOWING,
    StrategyMode.MEAN_REVERSION,
    StrategyMode.BREAKOUT,
    StrategyMode.DEFENSIVE,
}

_WEIGHT_SUM_TOLERANCE = 0.05  # weights should sum to 1.0 ± this


def _make_valid_profile_data(**overrides: object) -> dict:
    """Return a minimal valid raw profile dict."""
    data: dict = {
        "name": "Test Profile",
        "mode": "trend_following",
        "description": "Test description.",
        "preferences": {
            axis: {"weight": round(1.0 / 7, 4), "ideal": 0.5}
            for axis in STATE_VECTOR_AXES
        },
        "stoploss_range": [-0.10, -0.05],
        "leverage_range": [1.0, 2.0],
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# 1. load_default_profiles returns all 4 modes
# ---------------------------------------------------------------------------

class TestLoadDefaultProfiles:
    def test_returns_all_four_modes(self) -> None:
        profiles = load_default_profiles()
        assert set(profiles.keys()) == ALL_MODES

    def test_each_value_is_strategy_profile(self) -> None:
        profiles = load_default_profiles()
        for mode, profile in profiles.items():
            assert isinstance(profile, StrategyProfile), f"Expected StrategyProfile for {mode}"

    def test_profile_mode_matches_key(self) -> None:
        profiles = load_default_profiles()
        for mode, profile in profiles.items():
            assert profile.mode == mode, f"Profile mode mismatch for {mode}"


# ---------------------------------------------------------------------------
# 2. Weights sum to approximately 1.0 per profile
# ---------------------------------------------------------------------------

class TestWeightSums:
    @pytest.mark.parametrize("mode", list(ALL_MODES))
    def test_weights_sum_to_one(self, mode: StrategyMode) -> None:
        profiles = load_default_profiles()
        profile = profiles[mode]
        total = sum(pref.weight for pref in profile.preferences.values())
        assert abs(total - 1.0) <= _WEIGHT_SUM_TOLERANCE, (
            f"{mode.value} weights sum to {total:.4f}, expected 1.0 ± {_WEIGHT_SUM_TOLERANCE}"
        )


# ---------------------------------------------------------------------------
# 3. Stoploss range is valid (negative values, min < max)
# ---------------------------------------------------------------------------

class TestStoplossRange:
    @pytest.mark.parametrize("mode", list(ALL_MODES))
    def test_stoploss_range_is_negative(self, mode: StrategyMode) -> None:
        profiles = load_default_profiles()
        sl_min, sl_max = profiles[mode].stoploss_range
        assert sl_min < 0, f"{mode.value}: stoploss min must be negative, got {sl_min}"
        assert sl_max < 0, f"{mode.value}: stoploss max must be negative, got {sl_max}"

    @pytest.mark.parametrize("mode", list(ALL_MODES))
    def test_stoploss_range_min_less_than_max(self, mode: StrategyMode) -> None:
        profiles = load_default_profiles()
        sl_min, sl_max = profiles[mode].stoploss_range
        assert sl_min < sl_max, (
            f"{mode.value}: stoploss min ({sl_min}) must be < max ({sl_max})"
        )


# ---------------------------------------------------------------------------
# 4. Leverage range is valid (positive, min <= max)
# ---------------------------------------------------------------------------

class TestLeverageRange:
    @pytest.mark.parametrize("mode", list(ALL_MODES))
    def test_leverage_range_is_positive(self, mode: StrategyMode) -> None:
        profiles = load_default_profiles()
        lev_min, lev_max = profiles[mode].leverage_range
        assert lev_min > 0, f"{mode.value}: leverage min must be positive, got {lev_min}"
        assert lev_max > 0, f"{mode.value}: leverage max must be positive, got {lev_max}"

    @pytest.mark.parametrize("mode", list(ALL_MODES))
    def test_leverage_range_min_lte_max(self, mode: StrategyMode) -> None:
        profiles = load_default_profiles()
        lev_min, lev_max = profiles[mode].leverage_range
        assert lev_min <= lev_max, (
            f"{mode.value}: leverage min ({lev_min}) must be <= max ({lev_max})"
        )


# ---------------------------------------------------------------------------
# 5. All 7 axis names present in each profile's preferences
# ---------------------------------------------------------------------------

class TestAxisCoverage:
    @pytest.mark.parametrize("mode", list(ALL_MODES))
    def test_all_seven_axes_present(self, mode: StrategyMode) -> None:
        profiles = load_default_profiles()
        profile = profiles[mode]
        assert set(profile.preferences.keys()) == STATE_VECTOR_AXES, (
            f"{mode.value}: axes mismatch. "
            f"Missing: {STATE_VECTOR_AXES - set(profile.preferences.keys())}, "
            f"Extra: {set(profile.preferences.keys()) - STATE_VECTOR_AXES}"
        )


# ---------------------------------------------------------------------------
# 6. Invalid YAML raises clear error
# ---------------------------------------------------------------------------

class TestInvalidYaml:
    def test_malformed_yaml_raises_value_error(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad_mode.yaml"
        bad_yaml.write_text("name: [unclosed bracket\nmode: trend_following", encoding="utf-8")

        with pytest.raises(ValueError, match="Failed to parse YAML"):
            load_profiles(tmp_path)

    def test_non_mapping_yaml_raises_value_error(self, tmp_path: Path) -> None:
        list_yaml = tmp_path / "list_profile.yaml"
        list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_profiles(tmp_path)

    def test_nonexistent_dir_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent_dir"
        with pytest.raises(FileNotFoundError):
            load_profiles(missing)


# ---------------------------------------------------------------------------
# 7. Missing required field raises validation error
# ---------------------------------------------------------------------------

class TestMissingRequiredField:
    def test_missing_name_raises_validation_error(self, tmp_path: Path) -> None:
        data = _make_valid_profile_data()
        del data["name"]
        yaml_file = tmp_path / "broken.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid strategy profile"):
            load_profiles(tmp_path)

    def test_missing_mode_raises_validation_error(self, tmp_path: Path) -> None:
        data = _make_valid_profile_data()
        del data["mode"]
        yaml_file = tmp_path / "broken.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid strategy profile"):
            load_profiles(tmp_path)

    def test_missing_stoploss_range_raises_validation_error(self, tmp_path: Path) -> None:
        data = _make_valid_profile_data()
        del data["stoploss_range"]
        yaml_file = tmp_path / "broken.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid strategy profile"):
            load_profiles(tmp_path)

    def test_missing_axis_in_preferences_raises_validation_error(self, tmp_path: Path) -> None:
        data = _make_valid_profile_data()
        del data["preferences"]["trend_persistence"]
        yaml_file = tmp_path / "broken.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid strategy profile"):
            load_profiles(tmp_path)

    def test_positive_stoploss_raises_validation_error(self, tmp_path: Path) -> None:
        data = _make_valid_profile_data(stoploss_range=[0.05, 0.10])
        yaml_file = tmp_path / "broken.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid strategy profile"):
            load_profiles(tmp_path)


# ---------------------------------------------------------------------------
# 8. Custom config_dir loading works
# ---------------------------------------------------------------------------

class TestCustomConfigDir:
    def test_load_from_custom_dir(self, tmp_path: Path) -> None:
        data = _make_valid_profile_data(
            name="Custom Breakout",
            mode="breakout",
            description="Custom breakout profile.",
            stoploss_range=[-0.15, -0.10],
            leverage_range=[2.0, 4.0],
        )
        yaml_file = tmp_path / "breakout.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")

        profiles = load_profiles(tmp_path)
        assert StrategyMode.BREAKOUT in profiles
        profile = profiles[StrategyMode.BREAKOUT]
        assert profile.name == "Custom Breakout"

    def test_empty_dir_returns_empty_dict(self, tmp_path: Path) -> None:
        profiles = load_profiles(tmp_path)
        assert profiles == {}

    def test_multiple_yaml_files_loaded(self, tmp_path: Path) -> None:
        for mode in ["trend_following", "defensive"]:
            data = _make_valid_profile_data(
                name=f"Test {mode}",
                mode=mode,
                description=f"Test profile for {mode}.",
            )
            (tmp_path / f"{mode}.yaml").write_text(yaml.dump(data), encoding="utf-8")

        profiles = load_profiles(tmp_path)
        assert len(profiles) == 2
        assert StrategyMode.TREND_FOLLOWING in profiles
        assert StrategyMode.DEFENSIVE in profiles


# ---------------------------------------------------------------------------
# 9. StrategyProfile is immutable (frozen pydantic config)
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_strategy_profile_is_immutable(self) -> None:
        profiles = load_default_profiles()
        profile = profiles[StrategyMode.TREND_FOLLOWING]
        # Pydantic v2 frozen models raise ValidationError on attribute assignment
        with pytest.raises((TypeError, ValidationError)):
            profile.name = "Mutated Name"  # type: ignore[misc]

    def test_axis_preference_is_immutable(self) -> None:
        pref = AxisPreference(weight=0.5, ideal=0.5)
        # Pydantic v2 frozen models raise ValidationError on attribute assignment
        with pytest.raises((TypeError, ValidationError)):
            pref.weight = 0.9  # type: ignore[misc]

    def test_profile_model_config_frozen(self) -> None:
        assert StrategyProfile.model_config.get("frozen") is True

    def test_axis_preference_model_config_frozen(self) -> None:
        assert AxisPreference.model_config.get("frozen") is True


# ---------------------------------------------------------------------------
# 10. load_default_profiles is idempotent
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_load_default_profiles_idempotent(self) -> None:
        profiles_1 = load_default_profiles()
        profiles_2 = load_default_profiles()
        assert set(profiles_1.keys()) == set(profiles_2.keys())
        for mode in ALL_MODES:
            p1, p2 = profiles_1[mode], profiles_2[mode]
            assert p1.name == p2.name
            assert p1.mode == p2.mode
            assert p1.stoploss_range == p2.stoploss_range
            assert p1.leverage_range == p2.leverage_range

    def test_successive_loads_return_equal_profiles(self) -> None:
        first = load_default_profiles()
        second = load_default_profiles()
        for mode in ALL_MODES:
            assert first[mode].model_dump() == second[mode].model_dump()


# ---------------------------------------------------------------------------
# 11. No Freqtrade framework imports in strategy_profile.py (AST check)
# ---------------------------------------------------------------------------

class TestNoFreqtradeImports:
    _MODULE_PATH = (
        Path(__file__).parent.parent.parent
        / "freqtrade"
        / "ohio"
        / "core"
        / "strategy_router"
        / "strategy_profile.py"
    )

    def test_no_freqtrade_framework_imports(self) -> None:
        source = self._MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        forbidden: list[str] = []
        allowed_prefix = "freqtrade.ohio.core.domain"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("freqtrade") and not alias.name.startswith(
                        allowed_prefix
                    ):
                        forbidden.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("freqtrade") and not module.startswith(allowed_prefix):
                    forbidden.append(module)

        assert not forbidden, (
            f"strategy_profile.py must not import Freqtrade framework modules. "
            f"Found: {forbidden}"
        )

    def test_module_file_exists(self) -> None:
        assert self._MODULE_PATH.exists(), (
            f"strategy_profile.py not found at {self._MODULE_PATH}"
        )
