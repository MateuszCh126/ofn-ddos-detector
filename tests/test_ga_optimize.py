from types import SimpleNamespace

import numpy as np
import pytest

from ddos_ofn.config import BuilderConfig, DetectorConfig, GAConfig, SimulationConfig
from ddos_ofn.datasets import build_train_validation_sets
from ddos_ofn.ga_optimize import _scenario_cost, evaluate_candidate, optimize_detector
from ddos_ofn.schemas import DetectionMetrics, SimulationResult


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


def test_scenario_cost_normalizes_delay_by_remaining_attack_horizon(monkeypatch):
    scenario = SimulationResult(
        name="ddos_ramp",
        router_ids=["router_00"],
        traffic=np.zeros((20, 1), dtype=np.float64),
        labels=np.zeros(20, dtype=np.int8),
        attack_slice=(5, 10),
    )

    class DummyDetector:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return SimpleNamespace(
                labels=scenario.labels,
                predictions=np.zeros_like(scenario.labels),
            )

    metrics = DetectionMetrics(
        recall=0.8,
        precision=1.0,
        f1=0.8888888888888888,
        false_positive_rate=0.1,
        detection_delay=10.0,
        true_positives=8,
        false_positives=1,
        true_negatives=9,
        false_negatives=2,
    )

    monkeypatch.setattr("ddos_ofn.ga_optimize.DDoSDetector", DummyDetector)
    monkeypatch.setattr("ddos_ofn.ga_optimize.evaluate_predictions", lambda _labels, _predictions: metrics)

    cost = _scenario_cost(
        scenario,
        BuilderConfig(),
        DetectorConfig(),
        {"router_00": 1.0},
    )

    expected_delay_term = 10.0 / (20 - 5)
    expected_cost = 0.55 * (1.0 - metrics.recall) + 0.30 * metrics.false_positive_rate + 0.15 * expected_delay_term
    assert cost == pytest.approx(expected_cost)
