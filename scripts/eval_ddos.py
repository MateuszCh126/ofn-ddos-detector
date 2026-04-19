"""Evaluate the detector on one synthetic scenario."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddos_ofn import BuilderConfig, DDoSDetector, DetectorConfig, SimulationConfig, evaluate_predictions, generate_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="ddos_ramp", choices=["normal", "ddos_ramp", "ddos_pulse", "flash_crowd"])
    parser.add_argument("--routers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    simulation = generate_scenario(
        args.scenario,
        SimulationConfig(routers=args.routers, seed=args.seed),
    )
    detector = DDoSDetector(BuilderConfig(), DetectorConfig())
    trace = detector.run(simulation.traffic, simulation.router_ids, simulation.labels, simulation.name)
    metrics = evaluate_predictions(trace.labels, trace.predictions)

    payload = {
        "scenario": simulation.name,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "f1": metrics.f1,
        "false_positive_rate": metrics.false_positive_rate,
        "detection_delay": metrics.detection_delay,
        "max_score": float(trace.scores.max()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
