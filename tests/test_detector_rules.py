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
        BuilderConfig(history_size=12, trend_epsilon=1.0),
        DetectorConfig(alert_threshold=1.5, clear_threshold=0.8, alert_windows=2, clear_windows=2, min_positive_routers=3),
    )

    trace = detector.run(sim.traffic, sim.router_ids, sim.labels, sim.name)

    assert trace.predictions.max() == 1
    attack_start, _ = sim.attack_slice
    assert trace.predictions[attack_start:].sum() > 0


def test_detector_stays_quiet_for_normal_scenario():
    sim = generate_scenario("normal", SimulationConfig(routers=12, steps=100, seed=11))
    detector = DDoSDetector(
        BuilderConfig(history_size=12, trend_epsilon=1.0),
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

    # below the absolute min_total_score floor -> no alarm even if above alert_threshold
    assert detector._update_alarm(5.0, 3, min_positive=2, alert_threshold=4.0, clear_threshold=2.0) is False
    detector.reset()
    # too few positive routers -> no alarm
    assert detector._update_alarm(6.0, 1, min_positive=2, alert_threshold=4.0, clear_threshold=2.0) is False
    detector.reset()
    # all conditions satisfied -> alarm
    assert detector._update_alarm(6.0, 2, min_positive=2, alert_threshold=4.0, clear_threshold=2.0) is True


def test_detector_accepts_multifeature_router_tensor():
    # 10 calm steps establish a clean idle floor, then a strong sustained burst
    # on both routers across both features. Enough benign history that the global
    # floor stays uncontaminated and the burst reads as a clear anomaly.
    rng = np.random.default_rng(0)
    calm = rng.normal([100.0, 500.0], [1.5, 6.0], size=(10, 2, 2))
    burst = np.tile(np.array([[300.0, 1500.0], [305.0, 1520.0]]), (3, 1, 1))
    traffic = np.concatenate([calm, burst], axis=0)
    labels = np.array([0] * 10 + [1, 1, 1], dtype=np.int8)
    detector = DDoSDetector(
        BuilderConfig(history_size=8, window_size=4),
        DetectorConfig(
            alert_threshold=1.0,
            clear_threshold=0.5,
            alert_windows=1,
            clear_windows=1,
            min_positive_fraction=0.0,
            min_positive_routers=1,
        ),
    )

    trace = detector.run(
        traffic,
        ["router_a", "router_b"],
        labels,
        "multifeature",
        feature_names=["packet_count", "byte_count"],
    )

    assert trace.predictions.shape == (13,)
    assert trace.predictions.max() == 1
    assert trace.scores.max() > 0.0
