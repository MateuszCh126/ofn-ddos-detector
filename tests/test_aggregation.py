import numpy as np
import pytest

from ddos_ofn.aggregator import aggregate_router_signals
from ddos_ofn.config import BuilderConfig
from ddos_ofn.schemas import RouterOFN
from pyofn import singleton


def _router_signal(router_id: str, direction: int, suspicion: float) -> RouterOFN:
    ofn = singleton(suspicion, n=64)
    if direction < 0:
        ofn = -ofn
    return RouterOFN(
        router_id=router_id,
        raw_window=np.zeros(4),
        normalized_window=np.zeros(4),
        baseline_center=0.0,
        baseline_scale=1.0,
        anomaly_window=np.zeros(4),
        trend=float(direction),
        direction=direction,
        ofn=ofn,
        suspicion=suspicion,
    )


def test_aggregate_router_signals_adds_positive_and_subtracts_negative():
    cfg = BuilderConfig(n_points=64)
    signals = [
        _router_signal("r1", 1, 5.0),
        _router_signal("r2", 1, 3.0),
        _router_signal("r3", -1, 2.0),
    ]

    aggregated = aggregate_router_signals(signals, {"r1": 1.0, "r2": 1.0, "r3": 1.0}, cfg)

    assert aggregated.positive_routers == 2
    assert aggregated.negative_routers == 1
    assert aggregated.score > 0.0
    assert aggregated.raw_score > 0.0


def test_aggregate_router_signals_adds_neutral_with_reduced_weight():
    cfg = BuilderConfig(n_points=64, neutral_contribution=0.25)
    signals = [_router_signal("r1", 0, 4.0)]

    aggregated = aggregate_router_signals(signals, {"r1": 1.0}, cfg)

    assert aggregated.neutral_routers == 1
    assert aggregated.positive_routers == 0
    assert aggregated.negative_routers == 0
    assert aggregated.raw_score == pytest.approx(1.0)
    assert aggregated.score == pytest.approx(1.0)
