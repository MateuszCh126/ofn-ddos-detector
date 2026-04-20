"""Train the detector with GA on synthetic scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddos_ofn import BuilderConfig, DetectorConfig, GAConfig, SimulationConfig
from ddos_ofn.datasets import build_real_train_validation_sets, build_train_validation_sets
from ddos_ofn.detector import DDoSDetector
from ddos_ofn.ga_optimize import optimize_detector
from ddos_ofn.metrics import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="append", default=None, help="Path to a labeled real CSV dataset. Repeat for multiple files.")
    parser.add_argument("--suite", choices=["basic", "extended"], default="basic")
    parser.add_argument("--csv-format", choices=["auto", "wide", "long"], default="auto")
    parser.add_argument("--step-column", type=str, default="step")
    parser.add_argument("--timestamp-column", type=str, default="timestamp")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--router-column", type=str, default="router_id")
    parser.add_argument("--value-column", type=str, default="packet_count")
    parser.add_argument("--feature-column", type=str, default="feature_name")
    parser.add_argument("--wide-feature-separator", type=str, default="__")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--min-segment-steps", type=int, default=16)
    parser.add_argument("--routers", type=int, default=10)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attack-start", type=int, default=48)
    parser.add_argument("--attack-duration", type=int, default=24)
    args = parser.parse_args()

    if args.csv:
        train_set, valid_set = build_real_train_validation_sets(
            args.csv,
            csv_format=args.csv_format,
            step_column=args.step_column,
            timestamp_column=args.timestamp_column,
            label_column=args.label_column,
            router_column=args.router_column,
            value_column=args.value_column,
            feature_column=args.feature_column,
            wide_feature_separator=args.wide_feature_separator,
            train_fraction=args.train_fraction,
            min_segment_steps=args.min_segment_steps,
        )
        data_source = "csv"
    else:
        simulation_config = SimulationConfig(
            routers=args.routers,
            steps=args.steps,
            seed=args.seed,
            attack_start=args.attack_start,
            attack_duration=args.attack_duration,
        )
        train_set, valid_set = build_train_validation_sets(simulation_config, suite=args.suite)
        data_source = "synthetic"

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
        trace = detector.run(
            scenario.traffic,
            scenario.router_ids,
            scenario.labels,
            scenario.name,
            feature_names=scenario.feature_names,
        )
        metrics = evaluate_predictions(trace.labels, trace.predictions)
        validation[scenario.name] = {
            "recall": metrics.recall,
            "precision": metrics.precision,
            "f1": metrics.f1,
            "false_positive_rate": metrics.false_positive_rate,
            "detection_delay": metrics.detection_delay,
        }

    payload = {
        "data_source": data_source,
        "suite": args.suite if data_source == "synthetic" else "csv",
        "best_fitness": result.best_fitness,
        "alert_threshold": result.alert_threshold,
        "clear_threshold": result.clear_threshold,
        "min_positive_routers": result.min_positive_routers,
        "alert_windows": result.alert_windows,
        "clear_windows": result.clear_windows,
        "weights": result.weights,
        "feature_names": list(train_set[0].feature_names),
        "validation": validation,
    }

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "best_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
