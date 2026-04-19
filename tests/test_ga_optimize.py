from ddos_ofn.config import BuilderConfig, DetectorConfig, GAConfig, SimulationConfig
from ddos_ofn.datasets import build_train_validation_sets
from ddos_ofn.ga_optimize import evaluate_candidate, optimize_detector


def test_optimize_detector_returns_valid_configuration():
    train_set, _ = build_train_validation_sets(
        SimulationConfig(routers=8, steps=90, seed=3, attack_start=40, attack_duration=20)
    )
    result = optimize_detector(
        train_set,
        BuilderConfig(),
        DetectorConfig(),
        GAConfig(population_size=10, generations=4, elite_count=2, seed=17),
    )

    assert result.best_fitness >= 0.0
    assert len(result.weights) == 8
    assert result.alert_threshold > result.clear_threshold
    assert result.min_positive_routers >= 1


def test_evaluate_candidate_returns_finite_cost():
    train_set, _ = build_train_validation_sets(
        SimulationConfig(routers=6, steps=80, seed=4, attack_start=35, attack_duration=20)
    )
    router_count = len(train_set[0].router_ids)
    genome = [1.0] * router_count + [3.0, 0.5, 0.4, 2.0, 2.0]
    cost = evaluate_candidate(
        genome,
        train_set,
        BuilderConfig(),
        DetectorConfig(),
        GAConfig(seed=19),
    )

    assert cost >= 0.0
