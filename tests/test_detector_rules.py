from ddos_ofn.config import BuilderConfig, DetectorConfig, SimulationConfig
from ddos_ofn.detector import DDoSDetector
from ddos_ofn.simulation import generate_scenario


def test_detector_raises_alarm_for_synthetic_ddos_ramp():
    sim = generate_scenario(
        "ddos_ramp",
        SimulationConfig(routers=12, steps=120, seed=5, attack_start=50, attack_duration=30),
    )
    detector = DDoSDetector(
        BuilderConfig(history_size=12),
        DetectorConfig(alert_threshold=1.5, clear_threshold=0.8, alert_windows=2, clear_windows=2, min_positive_routers=3),
    )

    trace = detector.run(sim.traffic, sim.router_ids, sim.labels, sim.name)

    assert trace.predictions.max() == 1
    attack_start, _ = sim.attack_slice
    assert trace.predictions[attack_start:].sum() > 0


def test_detector_stays_quiet_for_normal_scenario():
    sim = generate_scenario("normal", SimulationConfig(routers=12, steps=100, seed=11))
    detector = DDoSDetector(
        BuilderConfig(history_size=12),
        DetectorConfig(alert_threshold=5.0, clear_threshold=2.0, alert_windows=3, clear_windows=2, min_positive_routers=5),
    )

    trace = detector.run(sim.traffic, sim.router_ids, sim.labels, sim.name)

    assert trace.predictions.sum() == 0
