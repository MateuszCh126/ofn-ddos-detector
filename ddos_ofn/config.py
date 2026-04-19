"""Configuration objects for the OFN DDoS detector."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BuilderConfig:
    """Controls how OFNs are built from packet measurements."""

    n_points: int = 256
    window_size: int = 4
    history_size: int = 16
    min_spread: float = 0.2
    trend_epsilon: float = 0.15
    anomaly_clip: float = 8.0
    min_baseline_scale: float = 1.0
    neutral_contribution: float = 0.25


@dataclass(slots=True)
class DetectorConfig:
    """Thresholds and hysteresis for the global detector."""

    alert_threshold: float = 4.0
    clear_threshold: float = 2.0
    alert_windows: int = 2
    clear_windows: int = 2
    min_positive_routers: int = 4
    min_total_score: float = 0.0


@dataclass(slots=True)
class SimulationConfig:
    """Synthetic scenario generation parameters."""

    routers: int = 30
    steps: int = 160
    seed: int | None = 7
    baseline_low: float = 80.0
    baseline_high: float = 160.0
    noise_std: float = 4.0
    attack_fraction: float = 0.7
    attack_scale: float = 5.0
    pulse_scale: float = 6.0
    flash_scale: float = 2.0
    attack_start: int = 80
    attack_duration: int = 40


@dataclass(slots=True)
class GAConfig:
    """Genetic algorithm hyperparameters used to tune detector settings."""

    population_size: int = 36
    generations: int = 24
    mutation_rate: float = 0.12
    mutation_sigma: float = 0.18
    crossover_rate: float = 0.75
    tournament_k: int = 3
    elite_count: int = 4
    weight_bounds: tuple[float, float] = (0.1, 3.0)
    alert_threshold_bounds: tuple[float, float] = (1.0, 10.0)
    clear_ratio_bounds: tuple[float, float] = (0.25, 0.9)
    positive_fraction_bounds: tuple[float, float] = (0.05, 0.8)
    hysteresis_bounds: tuple[int, int] = (1, 5)
    seed: int | None = 13
