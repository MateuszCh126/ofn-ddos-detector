"""Convert short packet-count windows into directed OFNs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def _resolve_feature_names(feature_count: int, feature_names: Sequence[str] | None) -> list[str]:
    if feature_names is None:
        if feature_count == 1:
            return ["packet_count"]
        return [f"feature_{idx:02d}" for idx in range(feature_count)]

    resolved = [str(name) for name in feature_names]
    if len(resolved) != feature_count:
        raise ValueError("feature_names length must match the feature dimension")
    return resolved


def _resolve_feature_weights(
    feature_names: Sequence[str],
    feature_weights: Mapping[str, float] | Sequence[float] | None,
) -> np.ndarray:
    if feature_weights is None:
        return np.ones(len(feature_names), dtype=np.float64)

    if isinstance(feature_weights, Mapping):
        return np.asarray([float(feature_weights.get(name, 1.0)) for name in feature_names], dtype=np.float64)

    weights = np.asarray(feature_weights, dtype=np.float64).reshape(-1)
    if weights.size != len(feature_names):
        raise ValueError("feature_weights length must match the feature dimension")
    return weights


def _prepare_feature_windows(
    window: np.ndarray,
    history: np.ndarray,
    config: BuilderConfig,
    *,
    feature_names: Sequence[str] | None = None,
    feature_weights: Mapping[str, float] | Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Normalize 1D/2D router history and collapse it into one composite window."""

    window_values = np.asarray(window, dtype=np.float64)
    history_values = np.asarray(history, dtype=np.float64)

    if window_values.ndim == 1:
        window_values = window_values[:, None]
    if history_values.ndim == 1:
        history_values = history_values[:, None]
    if window_values.ndim != 2 or history_values.ndim != 2:
        raise ValueError("window and history must be 1D or 2D arrays")
    if window_values.shape[1] != history_values.shape[1]:
        raise ValueError("window and history must share the same feature dimension")

    resolved_feature_names = _resolve_feature_names(window_values.shape[1], feature_names)
    weights = _resolve_feature_weights(resolved_feature_names, feature_weights)
    weight_total = float(np.sum(weights))
    if not np.isfinite(weight_total) or weight_total <= 0.0:
        raise ValueError("feature_weights must sum to a positive finite value")

    centers = np.zeros(window_values.shape[1], dtype=np.float64)
    scales = np.zeros(window_values.shape[1], dtype=np.float64)
    normalized_matrix = np.zeros_like(window_values, dtype=np.float64)

    for idx in range(window_values.shape[1]):
        center, scale = robust_center_scale(history_values[:, idx], min_scale=config.min_baseline_scale)
        centers[idx] = center
        scales[idx] = scale
        normalized_matrix[:, idx] = normalize_window(window_values[:, idx], center, scale, clip=config.anomaly_clip)

    if config.feature_aggregation != "weighted_mean":
        raise ValueError(f"unsupported feature_aggregation '{config.feature_aggregation}'")

    composite_normalized = normalized_matrix @ (weights / weight_total)
    positive_anomaly = np.maximum(normalized_matrix, 0.0)
    composite_anomaly = positive_anomaly @ (weights / weight_total)
    return centers, scales, normalized_matrix, positive_anomaly, composite_normalized, composite_anomaly, resolved_feature_names


def build_router_ofn(
    router_id: str,
    window: np.ndarray,
    history: np.ndarray,
    config: BuilderConfig,
    *,
    feature_names: Sequence[str] | None = None,
    feature_weights: Mapping[str, float] | Sequence[float] | None = None,
) -> RouterOFN:
    """Build a directed trapezoidal OFN from 4 consecutive measurements."""

    (
        centers,
        scales,
        normalized_matrix,
        anomaly_matrix,
        composite_normalized,
        composite_window,
        resolved_feature_names,
    ) = _prepare_feature_windows(
        window,
        history,
        config,
        feature_names=feature_names,
        feature_weights=feature_weights,
    )
    direction, trend = infer_direction(composite_normalized, config.trend_epsilon)
    suspicion = float(np.mean(composite_window))

    if direction == 0 and suspicion <= config.min_spread:
        ofn: OFN = singleton(suspicion, n=config.n_points)
    else:
        a, b, c, d = _ensure_trapezoid_params(composite_window, config.min_spread, direction)
        if direction < 0:
            ofn = trapezoidal_left(a, b, c, d, n=config.n_points)
        else:
            ofn = trapezoidal(a, b, c, d, n=config.n_points)

    raw_values = np.asarray(window, dtype=np.float64)
    normalized_values = normalized_matrix[:, 0] if normalized_matrix.shape[1] == 1 else normalized_matrix
    anomaly_values = anomaly_matrix[:, 0] if anomaly_matrix.shape[1] == 1 else anomaly_matrix
    baseline_center: np.ndarray | float = float(centers[0]) if centers.size == 1 else centers
    baseline_scale: np.ndarray | float = float(scales[0]) if scales.size == 1 else scales

    return RouterOFN(
        router_id=router_id,
        raw_window=raw_values,
        normalized_window=normalized_values,
        baseline_center=baseline_center,
        baseline_scale=baseline_scale,
        anomaly_window=anomaly_values,
        trend=trend,
        direction=direction,
        ofn=ofn,
        suspicion=float(ofn.defuzzify_cog()),
        composite_window=composite_window,
        feature_names=resolved_feature_names,
    )
