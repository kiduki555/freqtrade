"""Default configuration values for the OHIO Market State Engine."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OhioConfig:
    # Normalization
    normalization_window_days: int = 180
    normalization_method: str = "percentile_rank"

    # Stabilizer (tuned via M5 FT-023 A/B test — top-10 stability combos)
    ema_alpha: float = 0.15
    jump_threshold: float = 0.10
    min_dwell: int = 3

    # Strategy Router (raised from 0.15 — M5 FT-022 showed 100% entry rate)
    fitness_disabled_threshold: float = 0.40

    # Risk
    max_drawdown_pct: float = 0.15
    drawdown_reduce_threshold: float = 0.05
    drawdown_block_threshold: float = 0.10

    # Warmup
    startup_candle_count: int = 4320  # 180 days @ 1h
