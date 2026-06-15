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
    trend_epsilon: float = 2.2
    anomaly_clip: float = 8.0
    min_baseline_scale: float = 1.0
    neutral_contribution: float = 0.25
    feature_aggregation: str = "weighted_mean"
    # Baseline estimation:
    #   "global_floor" — per-router idle floor (low quantile over the whole
    #       series). Robust to sustained attacks that occupy most of the
    #       timeline; the default because it generalizes across datasets.
    #   "rolling" — legacy median/MAD over a trailing history window. Suitable
    #       for true streaming where the full series is not available up front.
    baseline_mode: str = "global_floor"
    floor_quantile: float = 0.4
    # Direction is driven by the anomaly *level* (sigmas above the floor), not
    # the instantaneous slope: a sustained-high router is "positive" even with a
    # flat trend. level_epsilon is the elevation (in robust sigmas) required to
    # count a router as positively contributing. Set high enough that pure noise
    # does not register — this is what makes the breadth gate (min_positive_
    # fraction) meaningfully separate narrow flash crowds from broad attacks.
    direction_mode: str = "level"  # "level" | "trend"
    level_epsilon: float = 1.5


@dataclass(slots=True)
class DetectorConfig:
    """Thresholds and hysteresis for the global detector."""

    alert_threshold: float = 1.0
    clear_threshold: float = 0.5
    alert_windows: int = 2
    clear_windows: int = 2
    min_positive_routers: int = 4
    min_total_score: float = 0.0
    # When > 0, the effective minimum number of positively-voting routers is
    # max(1, round(min_positive_fraction * n_routers)), making the rule scale
    # with the network size instead of a fixed absolute count. This is the
    # breadth gate: a broad attack lights up many routers, a narrow flash crowd
    # only a few. Lower it for single-victim deployments. Falls back to the
    # absolute min_positive_routers when 0.
    min_positive_fraction: float = 0.35
    # Threshold calibration:
    #   "absolute" — fixed alert_threshold / clear_threshold. Because the score
    #       is a count-invariant weighted mean of per-router robust z-scores
    #       (sigmas above the idle floor), a fixed value already generalizes
    #       across router counts and traffic scales — the default.
    #   "auto" — derive alert/clear from the score series' own idle floor:
    #       alert = floor + auto_alert_sigma * spread,
    #       clear = floor + auto_clear_sigma * spread. Self-calibrates to scale
    #       but can over-fire on pure noise and under-fire on bursty attacks.
    threshold_mode: str = "absolute"
    auto_alert_sigma: float = 3.0
    auto_clear_sigma: float = 1.5


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
    alert_threshold_bounds: tuple[float, float] = (0.5, 5.0)
    clear_ratio_bounds: tuple[float, float] = (0.25, 0.9)
    positive_fraction_bounds: tuple[float, float] = (0.05, 0.8)
    hysteresis_bounds: tuple[int, int] = (1, 5)
    seed: int | None = 13
