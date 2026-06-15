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


def robust_floor_scale(
    series: np.ndarray,
    floor_quantile: float = 0.3,
    min_scale: float = 1.0,
    eps: float = 1e-9,
) -> tuple[float, float]:
    """Estimate an *idle floor* center and a robust scale for a full series.

    Unlike :func:`robust_center_scale` (median + MAD), the center here is a low
    quantile of the series. This is deliberately resistant to a high attack
    fraction: even when most of the timeline is under attack, the idle/benign
    periods still set the low quantile, so "how far above your own floor are
    you" stays meaningful. The scale is the inter-quartile range rescaled to a
    Gaussian sigma (IQR / 1.349), floored by ``min_scale``.
    """

    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0, max(min_scale, eps)

    center = float(np.quantile(values, floor_quantile))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    scale = max(float(q3 - q1) / 1.349, min_scale, eps)
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
    """Extract a fixed-size history segment and the current OFN window.

    Warning: if history is empty, the baseline is calculated from the current window (warmup bias).
    """

    if step < window_size - 1:
        raise ValueError("step is too small for the requested window size")

    window = np.asarray(series[step - window_size + 1 : step + 1], dtype=np.float64)
    history_end = step - window_size + 1
    history_start = max(0, history_end - history_size)
    history = np.asarray(series[history_start:history_end], dtype=np.float64)
    if history.size == 0:
        history = window[:-1]
    return history, window
