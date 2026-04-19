"""Stream-like demo for one synthetic scenario."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddos_ofn import BuilderConfig, DDoSDetector, DetectorConfig, SimulationConfig, generate_scenario


def main() -> None:
    scenario = generate_scenario("ddos_ramp", SimulationConfig())
    detector = DDoSDetector(BuilderConfig(), DetectorConfig())
    trace = detector.run(scenario.traffic, scenario.router_ids, scenario.labels, scenario.name)

    last_alarm = False
    for snapshot in trace.snapshots:
        if snapshot.alarm != last_alarm:
            print(
                f"step={snapshot.step} alarm={int(snapshot.alarm)} "
                f"score={snapshot.score:.3f} pos={snapshot.positive_routers} neg={snapshot.negative_routers}"
            )
            last_alarm = snapshot.alarm


if __name__ == "__main__":
    main()
