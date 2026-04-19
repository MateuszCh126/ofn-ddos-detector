"""Evaluation metrics for the OFN detector."""

from __future__ import annotations

import numpy as np

from ddos_ofn.schemas import DetectionMetrics


def confusion_counts(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    """Return TP, FP, TN and FN counts."""

    y_true = np.asarray(labels, dtype=np.int8)
    y_pred = np.asarray(predictions, dtype=np.int8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def detection_delay(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Return the first-detection delay in steps for attack periods."""

    y_true = np.asarray(labels, dtype=np.int8)
    y_pred = np.asarray(predictions, dtype=np.int8)
    attack_steps = np.flatnonzero(y_true == 1)
    if attack_steps.size == 0:
        return 0.0

    first_attack = int(attack_steps[0])
    predicted_steps = np.flatnonzero((y_pred == 1) & (np.arange(y_pred.size) >= first_attack))
    if predicted_steps.size == 0:
        return float(y_pred.size - first_attack)
    return float(predicted_steps[0] - first_attack)


def evaluate_predictions(labels: np.ndarray, predictions: np.ndarray) -> DetectionMetrics:
    """Compute detection metrics for one scenario."""

    tp, fp, tn, fn = confusion_counts(labels, predictions)
    recall = tp / (tp + fn) if tp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    delay = detection_delay(labels, predictions)
    return DetectionMetrics(
        recall=recall,
        precision=precision,
        f1=f1,
        false_positive_rate=false_positive_rate,
        detection_delay=delay,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )
