"""Evaluate the detector on one synthetic scenario."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddos_ofn import BuilderConfig, DDoSDetector, DetectorConfig, SimulationConfig, evaluate_predictions, generate_scenario
from ddos_ofn.datasets import load_csv_scenario


def _json_number(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="ddos_ramp",
        choices=["normal", "ddos_ramp", "ddos_pulse", "ddos_low_and_slow", "ddos_rotating", "flash_crowd", "flash_cascade"],
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to a real CSV dataset")
    parser.add_argument("--csv-format", choices=["auto", "wide", "long"], default="auto")
    parser.add_argument("--name", type=str, default=None, help="Optional scenario name override for CSV input")
    parser.add_argument("--step-column", type=str, default="step")
    parser.add_argument("--timestamp-column", type=str, default="timestamp")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--router-column", type=str, default="router_id")
    parser.add_argument("--value-column", type=str, default="packet_count")
    parser.add_argument("--feature-column", type=str, default="feature_name")
    parser.add_argument("--wide-feature-separator", type=str, default="__")
    parser.add_argument("--routers", type=int, default=30)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attack-start", type=int, default=80)
    parser.add_argument("--attack-duration", type=int, default=40)
    args = parser.parse_args()

    if args.csv:
        simulation = load_csv_scenario(
            args.csv,
            name=args.name,
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
        simulation = generate_scenario(
            args.scenario,
            SimulationConfig(
                routers=args.routers,
                steps=args.steps,
                seed=args.seed,
                attack_start=args.attack_start,
                attack_duration=args.attack_duration,
            ),
        )

    detector = DDoSDetector(BuilderConfig(), DetectorConfig())
    trace = detector.run(
        simulation.traffic,
        simulation.router_ids,
        simulation.labels,
        simulation.name,
        feature_names=simulation.feature_names,
    )

    payload = {
        "scenario": simulation.name,
        "labels_present": simulation.labels_present,
        "routers": len(simulation.router_ids),
        "features": list(simulation.feature_names),
        "steps": int(simulation.traffic.shape[0]),
        "max_score": float(trace.scores.max()),
    }

    if simulation.labels_present:
        metrics = evaluate_predictions(trace.labels, trace.predictions)
        payload.update(
            {
                "recall": _json_number(metrics.recall),
                "precision": _json_number(metrics.precision),
                "f1": _json_number(metrics.f1),
                "false_positive_rate": _json_number(metrics.false_positive_rate),
                "detection_delay": _json_number(metrics.detection_delay),
            }
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
