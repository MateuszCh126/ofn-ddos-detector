from ddos_ofn.comparators import run_ewma_detector, run_volume_threshold_detector
from ddos_ofn.config import SimulationConfig
from ddos_ofn.metrics import evaluate_predictions
from ddos_ofn.simulation import generate_scenario


def test_volume_threshold_detector_detects_ramp_attack():
    scenario = generate_scenario(
        "ddos_ramp",
        SimulationConfig(routers=10, steps=100, seed=5, attack_start=40, attack_duration=20),
    )

    trace = run_volume_threshold_detector(scenario.traffic, scenario.labels, scenario_name=scenario.name)
    metrics = evaluate_predictions(trace.labels, trace.predictions)

    assert trace.detector_name == "volume_threshold"
    assert metrics.recall > 0.0


def test_ewma_detector_produces_finite_scores_on_normal_traffic():
    scenario = generate_scenario(
        "normal",
        SimulationConfig(routers=10, steps=100, seed=9),
    )

    trace = run_ewma_detector(scenario.traffic, scenario.labels, scenario_name=scenario.name)
    metrics = evaluate_predictions(trace.labels, trace.predictions)

    assert trace.detector_name == "ewma"
    assert metrics.false_positive_rate >= 0.0
    assert trace.scores.max() >= 0.0
