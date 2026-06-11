import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddos_ofn import BuilderConfig, DetectorConfig, DDoSDetector, generate_scenario, SimulationConfig
import numpy as np

sim = generate_scenario("normal", SimulationConfig(routers=30, steps=160, seed=7))
for eps in [1.5, 2.0, 2.2]:
    detector = DDoSDetector(BuilderConfig(trend_epsilon=eps), DetectorConfig())
    trace = detector.run(sim.traffic, sim.router_ids, sim.labels, sim.name)
    positives = [snap.positive_routers for snap in trace.snapshots]
    print(f"For eps={eps}:")
    print("  Median of positive routers:", np.median(positives))
    print("  Mean of positive routers:", np.mean(positives))
    print("  Max of positive routers:", np.max(positives))
