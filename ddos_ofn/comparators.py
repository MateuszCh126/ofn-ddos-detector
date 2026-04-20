"""Reference baseline detectors for OFN benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ddos_ofn.baseline import robust_center_scale
from ddos_ofn.schemas import ComparatorTrace


def _collapse_router_features(traffic: np.ndarray) -> np.ndarray:
    matrix = np.asarray(traffic, dtype=np.float64)
    if matrix.ndim == 2:
        return matrix
    if matrix.ndim == 3:
        return np.mean(matrix, axis=2)
    raise ValueError("traffic must have shape (steps, routers) or (steps, routers, features)")


def _global_volume_series(traffic: np.ndarray) -> np.ndarray:
    router_matrix = _collapse_router_features(traffic)
    return np.sum(router_matrix, axis=1)


def _apply_hysteresis(
    scores: np.ndarray,
    *,
    alert_threshold: float,
    clear_threshold: float,
    alert_windows: int,
    clear_windows: int,
) -> np.ndarray:
    predictions = np.zeros(len(scores), dtype=np.int8)
    alarm = False
    alert_streak = 0
    clear_streak = 0

    for idx, score in enumerate(scores):
        if score >= alert_threshold:
            alert_streak += 1
            clear_streak = 0
        elif score <= clear_threshold:
            clear_streak += 1
            alert_streak = 0
        else:
            alert_streak = 0
            clear_streak = 0

        if not alarm and alert_streak >= alert_windows:
            alarm = True
        elif alarm and clear_streak >= clear_windows:
            alarm = False
        predictions[idx] = int(alarm)

    return predictions


@dataclass(slots=True)
class VolumeThresholdConfig:
    """Configuration for the total-volume z-score baseline."""

    history_size: int = 16
    min_scale: float = 1.0
    alert_threshold: float = 3.0
    clear_threshold: float = 1.5
    alert_windows: int = 2
    clear_windows: int = 2


@dataclass(slots=True)
class EWMAConfig:
    """Configuration for the EWMA anomaly detector."""

    alpha: float = 0.25
    min_std: float = 1.0
    alert_threshold: float = 2.8
    clear_threshold: float = 1.2
    alert_windows: int = 2
    clear_windows: int = 2


def run_volume_threshold_detector(
    traffic: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    scenario_name: str = "custom",
    config: VolumeThresholdConfig | None = None,
) -> ComparatorTrace:
    """Run a simple baseline on the summed traffic volume."""

    cfg = config or VolumeThresholdConfig()
    series = _global_volume_series(traffic)
    scores = np.zeros_like(series, dtype=np.float64)

    for step in range(len(series)):
        history_start = max(0, step - cfg.history_size)
        history = series[history_start:step]
        if history.size == 0:
            continue
        center, scale = robust_center_scale(history, min_scale=cfg.min_scale)
        scores[step] = max((series[step] - center) / max(scale, 1e-9), 0.0)

    predictions = _apply_hysteresis(
        scores,
        alert_threshold=cfg.alert_threshold,
        clear_threshold=cfg.clear_threshold,
        alert_windows=cfg.alert_windows,
        clear_windows=cfg.clear_windows,
    )
    final_labels = np.zeros(len(series), dtype=np.int8) if labels is None else np.asarray(labels, dtype=np.int8)
    return ComparatorTrace(
        detector_name="volume_threshold",
        scenario_name=scenario_name,
        labels=final_labels,
        predictions=predictions,
        scores=scores,
    )


def run_ewma_detector(
    traffic: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    scenario_name: str = "custom",
    config: EWMAConfig | None = None,
) -> ComparatorTrace:
    """Run an EWMA-based anomaly detector on total traffic volume."""

    cfg = config or EWMAConfig()
    if not 0.0 < cfg.alpha <= 1.0:
        raise ValueError("alpha must be in the range (0, 1]")

    series = _global_volume_series(traffic)
    scores = np.zeros_like(series, dtype=np.float64)
    if len(series) == 0:
        final_labels = np.zeros(0, dtype=np.int8) if labels is None else np.asarray(labels, dtype=np.int8)
        return ComparatorTrace(
            detector_name="ewma",
            scenario_name=scenario_name,
            labels=final_labels,
            predictions=np.zeros(0, dtype=np.int8),
            scores=scores,
        )

    ewma = float(series[0])
    ewvar = float(cfg.min_std**2)
    for step in range(1, len(series)):
        residual = float(series[step] - ewma)
        std = max(np.sqrt(ewvar), cfg.min_std)
        scores[step] = max(residual / std, 0.0)

        previous_ewma = ewma
        ewma = cfg.alpha * float(series[step]) + (1.0 - cfg.alpha) * ewma
        ewvar = cfg.alpha * (float(series[step]) - previous_ewma) ** 2 + (1.0 - cfg.alpha) * ewvar

    predictions = _apply_hysteresis(
        scores,
        alert_threshold=cfg.alert_threshold,
        clear_threshold=cfg.clear_threshold,
        alert_windows=cfg.alert_windows,
        clear_windows=cfg.clear_windows,
    )
    final_labels = np.zeros(len(series), dtype=np.int8) if labels is None else np.asarray(labels, dtype=np.int8)
    return ComparatorTrace(
        detector_name="ewma",
        scenario_name=scenario_name,
        labels=final_labels,
        predictions=predictions,
        scores=scores,
    )
