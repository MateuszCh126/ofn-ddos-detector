"""Robustness of the redesigned detector: contamination-resistant floor,
level-based direction, and generalization across router counts."""
import numpy as np
import pytest

from ddos_ofn.baseline import robust_floor_scale
from ddos_ofn.config import BuilderConfig, DetectorConfig, SimulationConfig
from ddos_ofn.detector import DDoSDetector
from ddos_ofn.metrics import evaluate_predictions
from ddos_ofn.ofn_builder import infer_level_direction
from ddos_ofn.simulation import generate_scenario


def _run(name, n, builder=None, detector=None):
    sim = generate_scenario(name, SimulationConfig(routers=n))
    det = DDoSDetector(builder or BuilderConfig(), detector or DetectorConfig())
    trace = det.run(sim.traffic, sim.router_ids, sim.labels, sim.name, feature_names=sim.feature_names)
    return evaluate_predictions(trace.labels, trace.predictions), trace


def test_robust_floor_resists_majority_high_contamination():
    # 60% of the series is "under attack" (high); with floor_quantile=0.3 (which
    # tolerates up to ~70% contamination) the floor still tracks the quiet 40%,
    # unlike a median which sits squarely in the attack regime.
    series = np.concatenate([np.full(40, 10.0), np.full(60, 100.0)])
    center, scale = robust_floor_scale(series, floor_quantile=0.3, min_scale=1.0)
    assert center < 50.0  # near the quiet floor, not the contaminated median (100)
    assert scale > 0.0
    assert np.median(series) == 100.0  # the median IS contaminated — contrast


def test_infer_level_direction_uses_elevation_not_slope():
    flat_high = np.full(4, 3.0)  # sustained high, zero slope -> still positive
    assert infer_level_direction(flat_high, flat_high, level_epsilon=1.5)[0] == 1
    below = np.full(4, 0.0)
    signed_below = np.full(4, -3.0)  # pinned below floor -> negative
    assert infer_level_direction(below, signed_below, level_epsilon=1.5)[0] == -1
    quiet = np.full(4, 0.2)
    assert infer_level_direction(quiet, quiet, level_epsilon=1.5)[0] == 0


@pytest.mark.parametrize("n", [8, 30, 120])
def test_detection_generalizes_across_router_counts(n):
    # A clear broad attack is caught and pure noise stays silent regardless of
    # the network size — the score is count-invariant, the breadth gate fractional.
    ramp, _ = _run("ddos_ramp", n)
    assert ramp.recall >= 0.6
    normal, _ = _run("normal", n)
    assert normal.false_positive_rate == 0.0


@pytest.mark.parametrize("n", [8, 30, 120])
def test_narrow_flash_crowd_is_not_flagged(n):
    # A flash crowd hits a minority of routers; the breadth gate must keep it
    # below the alarm across every network size.
    flash, _ = _run("flash_crowd", n)
    assert flash.false_positive_rate <= 0.05


def test_auto_threshold_mode_runs_and_detects():
    detector = DetectorConfig(threshold_mode="auto", auto_alert_sigma=2.0, auto_clear_sigma=1.0)
    metrics, trace = _run("ddos_ramp", 30, detector=detector)
    assert trace.scores.max() > 0.0
    assert metrics.recall > 0.0
