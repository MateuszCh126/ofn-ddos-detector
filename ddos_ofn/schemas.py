"""Shared data structures for the OFN DDoS detector."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyofn import OFN


@dataclass(slots=True)
class RouterOFN:
    """Per-router OFN representation of the latest traffic window."""

    router_id: str
    raw_window: np.ndarray
    normalized_window: np.ndarray
    baseline_center: float
    baseline_scale: float
    anomaly_window: np.ndarray
    trend: float
    direction: int
    ofn: OFN
    suspicion: float


@dataclass(slots=True)
class AggregatedSignal:
    """Global OFN after weighted fusion of all router signals."""

    global_ofn: OFN
    raw_score: float
    score: float
    positive_routers: int
    negative_routers: int
    neutral_routers: int
    router_signals: list[RouterOFN] = field(default_factory=list)


@dataclass(slots=True)
class DetectionSnapshot:
    """Detector state for one time step."""

    step: int
    score: float
    raw_score: float
    alarm: bool
    positive_routers: int
    negative_routers: int
    neutral_routers: int
    global_direction: int


@dataclass(slots=True)
class DetectionTrace:
    """Detector output for a full scenario."""

    scenario_name: str
    router_ids: list[str]
    labels: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray
    snapshots: list[DetectionSnapshot]


@dataclass(slots=True)
class DetectionMetrics:
    """Classification metrics produced by the evaluator."""

    recall: float
    precision: float
    f1: float
    false_positive_rate: float
    detection_delay: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass(slots=True)
class SimulationResult:
    """Synthetic traffic matrix and labels for one scenario."""

    name: str
    router_ids: list[str]
    traffic: np.ndarray
    labels: np.ndarray
    attack_slice: tuple[int, int] | None = None


@dataclass(slots=True)
class GAResult:
    """Best genome and decoded detector settings after optimization."""

    best_fitness: float
    weights: dict[str, float]
    alert_threshold: float
    clear_threshold: float
    min_positive_routers: int
    alert_windows: int
    clear_windows: int
