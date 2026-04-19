"""Stateful OFN-based DDoS detector."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ddos_ofn.aggregator import aggregate_router_signals
from ddos_ofn.baseline import split_history_and_window
from ddos_ofn.config import BuilderConfig, DetectorConfig
from ddos_ofn.ofn_builder import build_router_ofn
from ddos_ofn.schemas import DetectionSnapshot, DetectionTrace


class DDoSDetector:
    """Evaluate OFN evidence across routers and emit an alarm state."""

    def __init__(
        self,
        builder_config: BuilderConfig | None = None,
        detector_config: DetectorConfig | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.builder_config = builder_config or BuilderConfig()
        self.detector_config = detector_config or DetectorConfig()
        self.weights = dict(weights or {})
        self.reset()

    def reset(self) -> None:
        self.alarm = False
        self.alert_streak = 0
        self.clear_streak = 0

    def _update_alarm(self, score: float, positive_routers: int) -> bool:
        cfg = self.detector_config

        should_alert = (
            score >= cfg.alert_threshold
            and positive_routers >= cfg.min_positive_routers
            and score >= cfg.min_total_score
        )
        should_clear = score <= cfg.clear_threshold

        if should_alert:
            self.alert_streak += 1
            self.clear_streak = 0
        elif should_clear:
            self.clear_streak += 1
            self.alert_streak = 0
        else:
            self.alert_streak = 0
            self.clear_streak = 0

        if not self.alarm and self.alert_streak >= cfg.alert_windows:
            self.alarm = True
        elif self.alarm and self.clear_streak >= cfg.clear_windows:
            self.alarm = False
        return self.alarm

    def run(
        self,
        traffic: np.ndarray,
        router_ids: list[str],
        labels: np.ndarray | None = None,
        scenario_name: str = "custom",
    ) -> DetectionTrace:
        """Run the detector across a router x time traffic matrix."""

        matrix = np.asarray(traffic, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("traffic must have shape (steps, routers)")
        if matrix.shape[1] != len(router_ids):
            raise ValueError("router_ids length must match traffic columns")

        self.reset()
        snapshots: list[DetectionSnapshot] = []
        predictions = np.zeros(matrix.shape[0], dtype=np.int8)
        scores = np.zeros(matrix.shape[0], dtype=np.float64)

        for step in range(self.builder_config.window_size - 1, matrix.shape[0]):
            router_signals = []
            for router_idx, router_id in enumerate(router_ids):
                history, window = split_history_and_window(
                    matrix[:, router_idx],
                    step=step,
                    window_size=self.builder_config.window_size,
                    history_size=self.builder_config.history_size,
                )
                router_signals.append(
                    build_router_ofn(router_id, window=window, history=history, config=self.builder_config)
                )

            aggregated = aggregate_router_signals(router_signals, self.weights, self.builder_config)
            alarm = self._update_alarm(aggregated.score, aggregated.positive_routers)
            predictions[step] = int(alarm)
            scores[step] = aggregated.score
            snapshots.append(
                DetectionSnapshot(
                    step=step,
                    score=aggregated.score,
                    raw_score=aggregated.raw_score,
                    alarm=alarm,
                    positive_routers=aggregated.positive_routers,
                    negative_routers=aggregated.negative_routers,
                    neutral_routers=aggregated.neutral_routers,
                    global_direction=aggregated.global_ofn.direction,
                )
            )

        final_labels = np.zeros(matrix.shape[0], dtype=np.int8) if labels is None else np.asarray(labels, dtype=np.int8)
        return DetectionTrace(
            scenario_name=scenario_name,
            router_ids=list(router_ids),
            labels=final_labels,
            predictions=predictions,
            scores=scores,
            snapshots=snapshots,
        )
