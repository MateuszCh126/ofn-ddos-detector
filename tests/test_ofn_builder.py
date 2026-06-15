import numpy as np
import pytest

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


def test_build_router_ofn_negative_direction_for_below_floor_window():
    # Direction is level-based: a window pinned well BELOW the baseline floor is
    # negative (a quiet/recovering router), regardless of slope. A still-elevated
    # but falling window stays positive, which is the point of level semantics.
    cfg = BuilderConfig()
    history = np.array([150.0, 151.0, 149.0, 152.0, 148.0, 150.0])
    window = np.array([120.0, 115.0, 110.0, 105.0])

    signal = build_router_ofn("router_b", window, history, cfg)

    assert signal.direction == -1
    assert signal.ofn.direction == -1


def test_build_router_ofn_supports_multiple_features_per_router():
    cfg = BuilderConfig()
    history = np.array(
        [
            [100.0, 500.0],
            [101.0, 510.0],
            [99.0, 495.0],
            [100.0, 505.0],
            [102.0, 515.0],
            [98.0, 490.0],
        ]
    )
    window = np.array(
        [
            [105.0, 540.0],
            [110.0, 580.0],
            [118.0, 640.0],
            [130.0, 720.0],
        ]
    )

    signal = build_router_ofn(
        "router_multi",
        window,
        history,
        cfg,
        feature_names=["packet_count", "byte_count"],
    )

    assert signal.direction == 1
    assert signal.trend > 0.0
    assert signal.ofn.direction == 1
    assert signal.composite_window is not None
    assert signal.feature_names == ["packet_count", "byte_count"]


def test_build_router_ofn_rejects_negative_feature_weights():
    cfg = BuilderConfig()
    history = np.array(
        [
            [100.0, 500.0],
            [101.0, 510.0],
            [99.0, 495.0],
            [100.0, 505.0],
        ]
    )
    window = np.array(
        [
            [105.0, 540.0],
            [110.0, 580.0],
            [118.0, 640.0],
            [130.0, 720.0],
        ]
    )

    with pytest.raises(ValueError, match="non-negative"):
        build_router_ofn(
            "router_multi",
            window,
            history,
            cfg,
            feature_names=["packet_count", "byte_count"],
            feature_weights={"packet_count": 1.0, "byte_count": -0.2},
        )
