"""Convert short packet-count windows into directed OFNs."""

from __future__ import annotations

import numpy as np

from pyofn import OFN, singleton, trapezoidal, trapezoidal_left

from ddos_ofn.baseline import normalize_window, robust_center_scale
from ddos_ofn.config import BuilderConfig
from ddos_ofn.schemas import RouterOFN


def infer_direction(normalized_window: np.ndarray, trend_epsilon: float) -> tuple[int, float]:
    """Infer direction from the first and last normalized samples."""

    values = np.asarray(normalized_window, dtype=np.float64)
    trend = float(values[-1] - values[0])
    if trend > trend_epsilon:
        return 1, trend
    if trend < -trend_epsilon:
        return -1, trend
    return 0, trend


def _ensure_trapezoid_params(
    values: np.ndarray,
    min_spread: float,
    direction: int,
) -> tuple[float, float, float, float]:
    """Convert 4 sorted values into a numerically stable trapezoid."""

    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    a, b, c, d = sorted_values.tolist()
    inner = max(min_spread / 6.0, 1e-6)
    if d - a < min_spread:
        center = float(np.mean(sorted_values))
        half = 0.5 * min_spread
        a = max(0.0, center - half)
        b = max(a, center - inner)
        c = max(b, center + inner)
        d = max(c, center + half)

    # Preserve a strictly directional arm so OFN.direction does not collapse to 0
    if direction >= 0 and (b - a) < inner:
        b = a + inner
        c = max(c, b)
        d = max(d, c + inner)
    if direction < 0 and (d - c) < inner:
        c = d - inner
        b = min(b, c)
        if (c - b) < inner:
            b = max(a, c - inner)
    return float(a), float(b), float(c), float(d)


def build_router_ofn(
    router_id: str,
    window: np.ndarray,
    history: np.ndarray,
    config: BuilderConfig,
) -> RouterOFN:
    """Build a directed trapezoidal OFN from 4 consecutive measurements."""

    center, scale = robust_center_scale(history, min_scale=config.min_baseline_scale)
    normalized = normalize_window(window, center, scale, clip=config.anomaly_clip)
    anomaly = np.maximum(normalized, 0.0)
    direction, trend = infer_direction(normalized, config.trend_epsilon)
    suspicion = float(np.mean(anomaly))

    if direction == 0 and suspicion <= config.min_spread:
        ofn: OFN = singleton(suspicion, n=config.n_points)
    else:
        a, b, c, d = _ensure_trapezoid_params(anomaly, config.min_spread, direction)
        if direction < 0:
            ofn = trapezoidal_left(a, b, c, d, n=config.n_points)
        else:
            ofn = trapezoidal(a, b, c, d, n=config.n_points)

    return RouterOFN(
        router_id=router_id,
        raw_window=np.asarray(window, dtype=np.float64),
        normalized_window=normalized,
        baseline_center=center,
        baseline_scale=scale,
        anomaly_window=anomaly,
        trend=trend,
        direction=direction,
        ofn=ofn,
        suspicion=float(ofn.defuzzify_cog()),
    )
