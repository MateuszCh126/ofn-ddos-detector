"""Benchmark OFN against simpler reference detectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddos_ofn import BuilderConfig, DDoSDetector, DetectorConfig, evaluate_predictions, generate_scenario
from ddos_ofn.comparators import run_ewma_detector, run_volume_threshold_detector
from ddos_ofn.config import SimulationConfig
from ddos_ofn.datasets import load_csv_scenarios


def _scenario_choices() -> list[str]:
    return ["normal", "ddos_ramp", "ddos_pulse", "ddos_low_and_slow", "ddos_rotating", "flash_crowd", "flash_cascade"]


def _load_scenarios(args: argparse.Namespace) -> list[object]:
    if args.csv:
        scenarios = load_csv_scenarios(
            args.csv,
            csv_format=args.csv_format,
            step_column=args.step_column,
            timestamp_column=args.timestamp_column,
            label_column=args.label_column,
            router_column=args.router_column,
            value_column=args.value_column,
            feature_column=args.feature_column,
            wide_feature_separator=args.wide_feature_separator,
        )
    else:
        cfg = SimulationConfig(
            routers=args.routers,
            steps=args.steps,
            seed=args.seed,
            attack_start=args.attack_start,
            attack_duration=args.attack_duration,
        )
        scenarios = [generate_scenario(name, cfg) for name in args.scenario]

    if any(not scenario.labels_present for scenario in scenarios):
        raise ValueError("benchmarking requires labels for all scenarios")
    return scenarios


def _metric_payload(trace: object) -> dict[str, float]:
    metrics = evaluate_predictions(trace.labels, trace.predictions)
    return {
        "recall": metrics.recall,
        "precision": metrics.precision,
        "f1": metrics.f1,
        "false_positive_rate": metrics.false_positive_rate,
        "detection_delay": metrics.detection_delay,
        "max_score": float(trace.scores.max()) if len(trace.scores) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", action="append", choices=_scenario_choices(), default=None)
    parser.add_argument("--csv", action="append", default=None)
    parser.add_argument("--csv-format", choices=["auto", "wide", "long"], default="auto")
    parser.add_argument("--step-column", type=str, default="step")
    parser.add_argument("--timestamp-column", type=str, default="timestamp")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--router-column", type=str, default="router_id")
    parser.add_argument("--value-column", type=str, default="packet_count")
    parser.add_argument("--feature-column", type=str, default="feature_name")
    parser.add_argument("--wide-feature-separator", type=str, default="__")
    parser.add_argument("--routers", type=int, default=12)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attack-start", type=int, default=48)
    parser.add_argument("--attack-duration", type=int, default=24)
    args = parser.parse_args()

    if args.csv is None and not args.scenario:
        args.scenario = ["normal", "ddos_ramp", "ddos_low_and_slow", "flash_crowd"]

    scenarios = _load_scenarios(args)

    results: dict[str, dict[str, dict[str, float]]] = {}
    model_averages: dict[str, dict[str, float]] = {}
    for scenario in scenarios:
        ofn_detector = DDoSDetector(BuilderConfig(), DetectorConfig())
        ofn_trace = ofn_detector.run(
            scenario.traffic,
            scenario.router_ids,
            scenario.labels,
            scenario.name,
            feature_names=scenario.feature_names,
        )
        volume_trace = run_volume_threshold_detector(scenario.traffic, scenario.labels, scenario_name=scenario.name)
        ewma_trace = run_ewma_detector(scenario.traffic, scenario.labels, scenario_name=scenario.name)

        results[scenario.name] = {
            "ofn": _metric_payload(ofn_trace),
            "volume_threshold": _metric_payload(volume_trace),
            "ewma": _metric_payload(ewma_trace),
        }

    for model_name in ("ofn", "volume_threshold", "ewma"):
        metrics_by_model = [scenario_metrics[model_name] for scenario_metrics in results.values()]
        model_averages[model_name] = {
            metric_name: float(sum(item[metric_name] for item in metrics_by_model) / len(metrics_by_model))
            for metric_name in metrics_by_model[0]
        }

    payload = {
        "scenario_count": len(scenarios),
        "results": results,
        "model_averages": model_averages,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
