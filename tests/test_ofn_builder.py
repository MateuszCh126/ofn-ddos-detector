import numpy as np

from ddos_ofn.config import BuilderConfig
from ddos_ofn.ofn_builder import build_router_ofn


def test_build_router_ofn_positive_direction_for_rising_window():
    cfg = BuilderConfig()
    history = np.array([100.0, 102.0, 98.0, 101.0, 99.0, 100.0])
    window = np.array([105.0, 110.0, 118.0, 135.0])

    signal = build_router_ofn("router_a", window, history, cfg)

    assert signal.direction == 1
    assert signal.trend > 0.0
    assert signal.suspicion > 0.0
    assert signal.ofn.direction == 1


def test_build_router_ofn_negative_direction_for_falling_window():
    cfg = BuilderConfig()
    history = np.array([120.0, 121.0, 119.0, 122.0, 118.0, 120.0])
    window = np.array([150.0, 145.0, 132.0, 124.0])

    signal = build_router_ofn("router_b", window, history, cfg)

    assert signal.direction == -1
    assert signal.trend < 0.0
    assert signal.ofn.direction == -1
