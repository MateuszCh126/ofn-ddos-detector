from ddos_ofn.config import BuilderConfig, DetectorConfig, SimulationConfig
from ddos_ofn.detector import DDoSDetector
from ddos_ofn.simulation import generate_scenario
import numpy as np


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


def test_detector_requires_all_alert_conditions_to_be_true():
    detector = DDoSDetector(
        BuilderConfig(),
        DetectorConfig(
            alert_threshold=4.0,
            clear_threshold=2.0,
            alert_windows=1,
            clear_windows=1,
            min_positive_routers=2,
            min_total_score=6.0,
        ),
    )

    assert detector._update_alarm(score=5.0, positive_routers=3) is False
    assert detector._update_alarm(score=6.0, positive_routers=1) is False
    assert detector._update_alarm(score=6.0, positive_routers=2) is True


def test_detector_accepts_multifeature_router_tensor():
    traffic = np.array(
        [
            [[100.0, 500.0], [105.0, 510.0]],
            [[101.0, 505.0], [104.0, 515.0]],
            [[102.0, 510.0], [103.0, 520.0]],
            [[120.0, 650.0], [122.0, 660.0]],
            [[135.0, 760.0], [136.0, 780.0]],
            [[150.0, 860.0], [152.0, 880.0]],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    detector = DDoSDetector(
        BuilderConfig(history_size=3, window_size=4),
        DetectorConfig(alert_threshold=1.0, clear_threshold=0.5, alert_windows=1, clear_windows=1, min_positive_routers=1),
    )

    trace = detector.run(
        traffic,
        ["router_a", "router_b"],
        labels,
        "multifeature",
        feature_names=["packet_count", "byte_count"],
    )

    assert trace.predictions.max() == 1
    assert trace.scores.max() > 0.0
