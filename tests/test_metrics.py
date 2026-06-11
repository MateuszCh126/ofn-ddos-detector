import numpy as np

from ddos_ofn.metrics import detection_delay, evaluate_predictions


def test_detection_delay_returns_zero_when_no_attack_is_present():
    labels = np.zeros(12, dtype=np.int8)
    predictions = np.zeros(12, dtype=np.int8)

    assert detection_delay(labels, predictions) == 0.0


def test_detection_delay_returns_remaining_horizon_when_alarm_never_fires():
    labels = np.zeros(12, dtype=np.int8)
    labels[5:9] = 1
    predictions = np.zeros(12, dtype=np.int8)

    assert detection_delay(labels, predictions) == 7.0


def test_evaluate_predictions_marks_attack_metrics_undefined_without_positive_labels():
    labels = np.zeros(12, dtype=np.int8)
    predictions = np.zeros(12, dtype=np.int8)

    metrics = evaluate_predictions(labels, predictions)

    assert np.isnan(metrics.recall)
    assert np.isnan(metrics.precision)
    assert np.isnan(metrics.f1)
    assert metrics.false_positive_rate == 0.0


def test_evaluate_predictions_keeps_missed_attack_f1_at_zero():
    labels = np.zeros(12, dtype=np.int8)
    labels[5:9] = 1
    predictions = np.zeros(12, dtype=np.int8)

    metrics = evaluate_predictions(labels, predictions)

    assert metrics.recall == 0.0
    assert np.isnan(metrics.precision)
    assert metrics.f1 == 0.0
