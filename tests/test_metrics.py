import numpy as np

from ddos_ofn.metrics import detection_delay


def test_detection_delay_returns_zero_when_no_attack_is_present():
    labels = np.zeros(12, dtype=np.int8)
    predictions = np.zeros(12, dtype=np.int8)

    assert detection_delay(labels, predictions) == 0.0


def test_detection_delay_returns_remaining_horizon_when_alarm_never_fires():
    labels = np.zeros(12, dtype=np.int8)
    labels[5:9] = 1
    predictions = np.zeros(12, dtype=np.int8)

    assert detection_delay(labels, predictions) == 7.0
