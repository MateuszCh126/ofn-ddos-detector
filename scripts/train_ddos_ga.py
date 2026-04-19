"""Train the detector with GA on synthetic scenarios."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddos_ofn import BuilderConfig, DetectorConfig, GAConfig, SimulationConfig
from ddos_ofn.datasets import build_train_validation_sets
from ddos_ofn.detector import DDoSDetector
from ddos_ofn.ga_optimize import optimize_detector
from ddos_ofn.metrics import evaluate_predictions


def main() -> None:
    simulation_config = SimulationConfig(routers=10, steps=96, seed=7, attack_start=48, attack_duration=24)
    train_set, valid_set = build_train_validation_sets(simulation_config)
    result = optimize_detector(
        train_set,
        BuilderConfig(),
        DetectorConfig(),
        GAConfig(population_size=8, generations=3, elite_count=2, seed=13),
    )

    validation = {}
    for scenario in valid_set:
        detector = DDoSDetector(
            BuilderConfig(),
            DetectorConfig(
                alert_threshold=result.alert_threshold,
                clear_threshold=result.clear_threshold,
                alert_windows=result.alert_windows,
                clear_windows=result.clear_windows,
                min_positive_routers=result.min_positive_routers,
            ),
            weights=result.weights,
        )
        trace = detector.run(scenario.traffic, scenario.router_ids, scenario.labels, scenario.name)
        metrics = evaluate_predictions(trace.labels, trace.predictions)
        validation[scenario.name] = {
            "recall": metrics.recall,
            "precision": metrics.precision,
            "f1": metrics.f1,
            "false_positive_rate": metrics.false_positive_rate,
            "detection_delay": metrics.detection_delay,
        }

    payload = {
        "best_fitness": result.best_fitness,
        "alert_threshold": result.alert_threshold,
        "clear_threshold": result.clear_threshold,
        "min_positive_routers": result.min_positive_routers,
        "alert_windows": result.alert_windows,
        "clear_windows": result.clear_windows,
        "weights": result.weights,
        "validation": validation,
    }

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "best_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
