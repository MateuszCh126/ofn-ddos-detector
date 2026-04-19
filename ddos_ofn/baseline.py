"""Baseline and normalization helpers."""

from __future__ import annotations

import numpy as np


def robust_center_scale(
    history: np.ndarray,
    min_scale: float = 1.0,
    eps: float = 1e-9,
) -> tuple[float, float]:
    """Estimate center and scale from traffic history using median and MAD."""

    values = np.asarray(history, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0, max(min_scale, eps)

    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = max(1.4826 * mad, min_scale, eps)
    return center, scale


def normalize_window(
    window: np.ndarray,
    center: float,
    scale: float,
    clip: float,
) -> np.ndarray:
    """Return clipped z-scores for the latest traffic window."""

    values = np.asarray(window, dtype=np.float64)
    normalized = (values - center) / max(scale, 1e-9)
    return np.clip(normalized, -clip, clip)


def split_history_and_window(
    series: np.ndarray,
    step: int,
    window_size: int,
    history_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a fixed-size history segment and the current OFN window."""

    if step < window_size - 1:
        raise ValueError("step is too small for the requested window size")

    window = np.asarray(series[step - window_size + 1 : step + 1], dtype=np.float64)
    history_end = step - window_size + 1
    history_start = max(0, history_end - history_size)
    history = np.asarray(series[history_start:history_end], dtype=np.float64)
    if history.size == 0:
        history = window[:-1]
    return history, window
