"""Genetic algorithm for tuning detector weights and thresholds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ddos_ofn.config import BuilderConfig, DetectorConfig, GAConfig
from ddos_ofn.detector import DDoSDetector
from ddos_ofn.metrics import evaluate_predictions
from ddos_ofn.schemas import GAResult, SimulationResult


def _decode_genome(
    genome: np.ndarray,
    router_ids: list[str],
    base_detector_config: DetectorConfig,
    ga_config: GAConfig,
) -> tuple[dict[str, float], DetectorConfig]:
    """Convert a numeric genome into weights and detector config."""

    router_count = len(router_ids)
    weights = np.clip(genome[:router_count], *ga_config.weight_bounds)
    alert_threshold = float(np.clip(genome[router_count], *ga_config.alert_threshold_bounds))
    clear_ratio = float(np.clip(genome[router_count + 1], *ga_config.clear_ratio_bounds))
    positive_fraction = float(np.clip(genome[router_count + 2], *ga_config.positive_fraction_bounds))
    alert_windows = int(np.clip(round(genome[router_count + 3]), *ga_config.hysteresis_bounds))
    clear_windows = int(np.clip(round(genome[router_count + 4]), *ga_config.hysteresis_bounds))
    clear_threshold = alert_threshold * clear_ratio
    min_positive_routers = max(1, int(round(positive_fraction * router_count)))

    detector_config = DetectorConfig(
        alert_threshold=alert_threshold,
        clear_threshold=clear_threshold,
        alert_windows=alert_windows,
        clear_windows=clear_windows,
        min_positive_routers=min_positive_routers,
        min_total_score=base_detector_config.min_total_score,
    )
    return dict(zip(router_ids, weights.tolist())), detector_config


def _scenario_cost(
    scenario: SimulationResult,
    builder_config: BuilderConfig,
    detector_config: DetectorConfig,
    weights: dict[str, float],
) -> float:
    """Evaluate one candidate on one scenario."""

    detector = DDoSDetector(builder_config, detector_config, weights=weights)
    trace = detector.run(scenario.traffic, scenario.router_ids, scenario.labels, scenario.name)
    metrics = evaluate_predictions(trace.labels, trace.predictions)

    if scenario.attack_slice is None:
        delay_term = 0.0
    else:
        attack_start, _ = scenario.attack_slice
        delay_term = metrics.detection_delay / max(1, scenario.traffic.shape[0] - attack_start)

    recall_error = 1.0 - metrics.recall
    return 0.55 * recall_error + 0.30 * metrics.false_positive_rate + 0.15 * delay_term


def evaluate_candidate(
    genome: np.ndarray,
    scenarios: Sequence[SimulationResult],
    builder_config: BuilderConfig,
    base_detector_config: DetectorConfig,
    ga_config: GAConfig,
) -> float:
    """Return the average candidate cost across scenarios."""

    router_ids = list(scenarios[0].router_ids)
    weights, detector_config = _decode_genome(genome, router_ids, base_detector_config, ga_config)
    costs = [
        _scenario_cost(scenario, builder_config, detector_config, weights)
        for scenario in scenarios
    ]
    return float(np.mean(costs))


def _initialize_population(router_count: int, ga_config: GAConfig) -> np.ndarray:
    """Sample an initial GA population."""

    rng = np.random.default_rng(ga_config.seed)
    genome_size = router_count + 5
    population = np.empty((ga_config.population_size, genome_size), dtype=np.float64)
    population[:, :router_count] = rng.uniform(
        ga_config.weight_bounds[0],
        ga_config.weight_bounds[1],
        size=(ga_config.population_size, router_count),
    )
    population[:, router_count] = rng.uniform(
        ga_config.alert_threshold_bounds[0],
        ga_config.alert_threshold_bounds[1],
        size=ga_config.population_size,
    )
    population[:, router_count + 1] = rng.uniform(
        ga_config.clear_ratio_bounds[0],
        ga_config.clear_ratio_bounds[1],
        size=ga_config.population_size,
    )
    population[:, router_count + 2] = rng.uniform(
        ga_config.positive_fraction_bounds[0],
        ga_config.positive_fraction_bounds[1],
        size=ga_config.population_size,
    )
    population[:, router_count + 3] = rng.integers(
        ga_config.hysteresis_bounds[0],
        ga_config.hysteresis_bounds[1] + 1,
        size=ga_config.population_size,
    )
    population[:, router_count + 4] = rng.integers(
        ga_config.hysteresis_bounds[0],
        ga_config.hysteresis_bounds[1] + 1,
        size=ga_config.population_size,
    )
    return population


def optimize_detector(
    scenarios: Sequence[SimulationResult],
    builder_config: BuilderConfig | None = None,
    detector_config: DetectorConfig | None = None,
    ga_config: GAConfig | None = None,
) -> GAResult:
    """Tune router weights and detector thresholds with a simple GA."""

    if not scenarios:
        raise ValueError("at least one scenario is required")

    builder_cfg = builder_config or BuilderConfig()
    detector_cfg = detector_config or DetectorConfig()
    ga_cfg = ga_config or GAConfig()
    router_ids = list(scenarios[0].router_ids)
    router_count = len(router_ids)

    rng = np.random.default_rng(ga_cfg.seed)
    population = _initialize_population(router_count, ga_cfg)
    fitness = np.array(
        [evaluate_candidate(genome, scenarios, builder_cfg, detector_cfg, ga_cfg) for genome in population],
        dtype=np.float64,
    )

    for _ in range(ga_cfg.generations):
        elite_count = min(ga_cfg.elite_count, len(population))
        elite_idx = np.argpartition(fitness, elite_count - 1)[:elite_count]
        elites = population[elite_idx].copy()

        selected = []
        for _ in range(len(population)):
            idx = rng.integers(0, len(population), size=ga_cfg.tournament_k)
            selected.append(population[idx[np.argmin(fitness[idx])]])
        selected_population = np.asarray(selected, dtype=np.float64)
        offspring = selected_population.copy()

        for row in range(0, len(offspring) - 1, 2):
            if rng.random() >= ga_cfg.crossover_rate:
                continue
            alpha = rng.random()
            first = offspring[row].copy()
            second = offspring[row + 1].copy()
            offspring[row] = alpha * first + (1.0 - alpha) * second
            offspring[row + 1] = (1.0 - alpha) * first + alpha * second

        mutation_mask = rng.random(offspring.shape) < ga_cfg.mutation_rate
        offspring += mutation_mask * rng.normal(0.0, ga_cfg.mutation_sigma, size=offspring.shape)
        offspring[:, router_count + 3 :] = np.round(offspring[:, router_count + 3 :])
        offspring[:elite_count] = elites

        population = offspring
        fitness = np.array(
            [evaluate_candidate(genome, scenarios, builder_cfg, detector_cfg, ga_cfg) for genome in population],
            dtype=np.float64,
        )

    best_idx = int(np.argmin(fitness))
    best_weights, best_detector_config = _decode_genome(population[best_idx], router_ids, detector_cfg, ga_cfg)
    return GAResult(
        best_fitness=float(fitness[best_idx]),
        weights=best_weights,
        alert_threshold=best_detector_config.alert_threshold,
        clear_threshold=best_detector_config.clear_threshold,
        min_positive_routers=best_detector_config.min_positive_routers,
        alert_windows=best_detector_config.alert_windows,
        clear_windows=best_detector_config.clear_windows,
    )
