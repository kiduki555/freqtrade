"""FT-010: Strategy Profile YAML + Pydantic Loader.

Loads per-strategy-mode profiles that define how the 7-axis StateVector
maps to a fitness score for each StrategyMode.

No Freqtrade framework imports — only domain models and stdlib/pydantic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from freqtrade.ohio.core.domain.models import StrategyMode

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "ohio" / "config" / "strategies"

STATE_VECTOR_AXES = frozenset(
    {
        "trend_persistence",
        "volatility_level",
        "downside_pressure",
        "liquidity_stress",
        "relative_strength",
        "correlation_stress",
        "breadth_dispersion",
    }
)


class AxisPreference(BaseModel):
    """How a strategy mode relates to one StateVector axis.

    Fitness contribution = weight * (1 - |axis_value - ideal| / scale)
    where scale normalises the axis range.
    """

    model_config = ConfigDict(frozen=True)

    weight: float  # importance [0, 1]
    ideal: float   # ideal axis value for this strategy

    @field_validator("weight")
    @classmethod
    def weight_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {v}")
        return v

    @field_validator("ideal")
    @classmethod
    def ideal_in_range(cls, v: float) -> float:
        """Ideal must be in [-1, 1] (covers trend_persistence [-1,1] and all [0,1] axes)."""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"ideal must be in [-1, 1], got {v}")
        return v


class StrategyProfile(BaseModel):
    """Complete profile for one strategy mode."""

    model_config = ConfigDict(frozen=True)

    name: str
    mode: StrategyMode
    description: str
    preferences: dict[str, AxisPreference]  # key = axis name
    meta_confidence_weight: float = 0.15
    meta_stability_weight: float = 0.10
    stoploss_range: tuple[float, float]  # (min, max) — both negative; TODO(FT-015): consumed by ExitAdapter
    leverage_range: tuple[float, float]  # (min, max) — both positive; TODO(FT-017): consumed by EntryAdapter

    # ATR scaling direction for dynamic stoploss (regime-adaptive)
    atr_stop_direction: Literal["tighten", "widen"] = "tighten"
    atr_scale_cap: float = 2.0
    # Chandelier trailing stop parameters
    chandelier_enabled: bool = False
    chandelier_multiplier: float = 2.5
    chandelier_activation: float = 0.02
    # Mode-specific hard floor for stoploss (overrides global hard_floor)
    hard_floor: float = -0.20

    @field_validator("atr_scale_cap")
    @classmethod
    def atr_scale_cap_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"atr_scale_cap must be positive, got {v}")
        return v

    @field_validator("chandelier_multiplier")
    @classmethod
    def chandelier_multiplier_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"chandelier_multiplier must be positive, got {v}")
        return v

    @field_validator("chandelier_activation")
    @classmethod
    def chandelier_activation_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"chandelier_activation must be in [0, 1], got {v}")
        return v

    @field_validator("preferences")
    @classmethod
    def all_axes_present(cls, v: dict[str, AxisPreference]) -> dict[str, AxisPreference]:
        missing = STATE_VECTOR_AXES - set(v.keys())
        if missing:
            raise ValueError(f"Missing axis preferences: {sorted(missing)}")
        extra = set(v.keys()) - STATE_VECTOR_AXES
        if extra:
            raise ValueError(f"Unknown axis preferences: {sorted(extra)}")
        return v

    @model_validator(mode="after")
    def validate_ranges(self) -> "StrategyProfile":
        sl_min, sl_max = self.stoploss_range
        if sl_min >= 0 or sl_max >= 0:
            raise ValueError(
                f"stoploss_range values must be negative, got ({sl_min}, {sl_max})"
            )
        if sl_min >= sl_max:
            raise ValueError(
                f"stoploss_range min must be less than max, got ({sl_min}, {sl_max})"
            )
        lev_min, lev_max = self.leverage_range
        if lev_min <= 0 or lev_max <= 0:
            raise ValueError(
                f"leverage_range values must be positive, got ({lev_min}, {lev_max})"
            )
        if lev_min > lev_max:
            raise ValueError(
                f"leverage_range min must be <= max, got ({lev_min}, {lev_max})"
            )
        return self


def _parse_profile(data: dict[str, Any], source: str) -> StrategyProfile:
    """Parse a raw YAML dict into a validated StrategyProfile.

    Args:
        data: Raw YAML data dict.
        source: Human-readable source label for error messages.

    Returns:
        Validated StrategyProfile instance.

    Raises:
        ValueError: If validation fails.
    """
    try:
        raw_prefs = data.get("preferences", {})
        parsed_prefs = {
            axis: AxisPreference(**vals) for axis, vals in raw_prefs.items()
        }
        profile_data = {**data, "preferences": parsed_prefs}
        return StrategyProfile(**profile_data)
    except Exception as exc:
        raise ValueError(f"Invalid strategy profile in {source!r}: {exc}") from exc


def load_profiles(config_dir: Path | None = None) -> dict[StrategyMode, StrategyProfile]:
    """Load all strategy profiles from YAML files in config_dir.

    Args:
        config_dir: Directory containing YAML profile files.
                    Defaults to freqtrade/ohio/config/strategies/.

    Returns:
        Mapping of StrategyMode -> StrategyProfile for all loaded profiles.

    Raises:
        FileNotFoundError: If config_dir does not exist.
        ValueError: If any YAML file fails validation.
    """
    resolved_dir = config_dir or _DEFAULT_CONFIG_DIR

    if not resolved_dir.exists():
        raise FileNotFoundError(f"Strategy config directory not found: {resolved_dir}")

    profiles: dict[StrategyMode, StrategyProfile] = {}
    yaml_files = sorted(resolved_dir.glob("*.yaml"))

    if not yaml_files:
        logger.warning("No YAML strategy profiles found in %s", resolved_dir)
        return profiles

    for yaml_path in yaml_files:
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML file {yaml_path.name!r}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError(
                f"Strategy profile {yaml_path.name!r} must be a YAML mapping, got {type(raw).__name__}"
            )

        profile = _parse_profile(raw, yaml_path.name)
        profiles[profile.mode] = profile
        logger.debug("Loaded strategy profile: %s (%s)", profile.name, profile.mode.value)

    return profiles


def load_default_profiles() -> dict[StrategyMode, StrategyProfile]:
    """Load the four built-in strategy profiles from the package config directory.

    Returns:
        Mapping of StrategyMode -> StrategyProfile for all four built-in modes.

    Raises:
        ValueError: If any built-in profile fails validation (indicates a bug).
    """
    return load_profiles(_DEFAULT_CONFIG_DIR)
