import numpy as np

from ddos_ofn.config import BuilderConfig
from ddos_ofn.ofn_builder import infer_direction


def test_infer_direction_switches_sign_correctly():
    epsilon = BuilderConfig().trend_epsilon

    rising = np.array([0.2, 0.5, 1.0, 1.8])
    falling = np.array([1.8, 1.0, 0.5, 0.2])
    flat = np.array([0.1, 0.15, 0.12, 0.18])

    assert infer_direction(rising, epsilon)[0] == 1
    assert infer_direction(falling, epsilon)[0] == -1
    assert infer_direction(flat, epsilon)[0] == 0
