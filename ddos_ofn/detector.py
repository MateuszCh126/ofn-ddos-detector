"""Stateful OFN-based DDoS detector."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ddos_ofn.aggregator import aggregate_router_signals
from ddos_ofn.baseline import robust_floor_scale, split_history_and_window
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
        feature_weights: Mapping[str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> None:
        self.builder_config = builder_config or BuilderConfig()
        self.detector_config = detector_config or DetectorConfig()
        self.weights = dict(weights or {})
        self.feature_weights = feature_weights
        self.reset()

    def reset(self) -> None:
        self.alarm = False
        self.alert_streak = 0
        self.clear_streak = 0

    def _effective_min_positive(self, n_routers: int) -> int:
        """Resolve the minimum positive-router count, scaling with network size."""

        cfg = self.detector_config
        if cfg.min_positive_fraction > 0.0:
            return max(1, min(n_routers, round(cfg.min_positive_fraction * n_routers)))
        return max(1, min(n_routers, cfg.min_positive_routers))

    def _compute_global_floor(self, matrix: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per-router idle floor (center, scale) estimated from the full series.

        Resistant to attacks that dominate the timeline: the low quantile is set
        by whatever quiet periods exist, so elevation above the floor stays a
        meaningful anomaly signal even when most steps are under attack.
        """

        cfg = self.builder_config
        n_routers = matrix.shape[1]
        n_features = matrix.shape[2] if matrix.ndim == 3 else 1
        floors: list[tuple[np.ndarray, np.ndarray]] = []
        for router_idx in range(n_routers):
            centers = np.zeros(n_features, dtype=np.float64)
            scales = np.zeros(n_features, dtype=np.float64)
            for feat_idx in range(n_features):
                series = matrix[:, router_idx, feat_idx] if matrix.ndim == 3 else matrix[:, router_idx]
                center, scale = robust_floor_scale(
                    series,
                    floor_quantile=cfg.floor_quantile,
                    min_scale=cfg.min_baseline_scale,
                )
                centers[feat_idx] = center
                scales[feat_idx] = scale
            floors.append((centers, scales))
        return floors

    def _resolve_thresholds(self, scores: np.ndarray) -> tuple[float, float]:
        """Resolve (alert, clear) thresholds, auto-calibrating to the score series.

        In "auto" mode the thresholds float with the data: the score's own idle
        floor (low quantile) plus a multiple of its robust spread. Because the
        aggregated score is a count-invariant weighted mean of per-router
        elevations, this single rule adapts to any router count and traffic
        scale — broad attacks lift the mean far above its floor, narrow blips do
        not — without hand-tuned magic numbers.
        """

        cfg = self.detector_config
        if cfg.threshold_mode == "absolute":
            return cfg.alert_threshold, cfg.clear_threshold
        if cfg.threshold_mode != "auto":
            raise ValueError(f"unsupported threshold_mode '{cfg.threshold_mode}'")

        evaluated = scores[self.builder_config.window_size - 1 :]
        if evaluated.size == 0:
            return cfg.alert_threshold, cfg.clear_threshold
        floor, scale = robust_floor_scale(
            evaluated, floor_quantile=self.builder_config.floor_quantile, min_scale=0.0
        )
        alert = floor + cfg.auto_alert_sigma * scale
        clear = floor + cfg.auto_clear_sigma * scale
        return alert, clear

    def _update_alarm(
        self,
        score: float,
        positive_routers: int,
        min_positive: int,
        alert_threshold: float,
        clear_threshold: float,
    ) -> bool:
        cfg = self.detector_config

        should_alert = (
            score >= alert_threshold
            and positive_routers >= min_positive
            and score >= cfg.min_total_score
        )
        should_clear = score <= clear_threshold

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
        feature_names: list[str] | None = None,
    ) -> DetectionTrace:
        """Run the detector across a router x time traffic matrix or feature tensor."""

        matrix = np.asarray(traffic, dtype=np.float64)
        if matrix.ndim not in {2, 3}:
            raise ValueError("traffic must have shape (steps, routers) or (steps, routers, features)")
        if matrix.shape[1] != len(router_ids):
            raise ValueError("router_ids length must match traffic columns")
        if matrix.ndim == 3 and feature_names is not None and matrix.shape[2] != len(feature_names):
            raise ValueError("feature_names length must match traffic feature dimension")

        self.reset()
        predictions = np.zeros(matrix.shape[0], dtype=np.int8)
        scores = np.zeros(matrix.shape[0], dtype=np.float64)
        min_positive = self._effective_min_positive(len(router_ids))

        floors = self._compute_global_floor(matrix) if self.builder_config.baseline_mode == "global_floor" else None

        # Pass 1 — score every step (no alarm yet); the decision layer needs the
        # full score distribution to self-calibrate its thresholds.
        evaluated_steps = list(range(self.builder_config.window_size - 1, matrix.shape[0]))
        step_aggregates = []
        for step in evaluated_steps:
            router_signals = []
            for router_idx, router_id in enumerate(router_ids):
                history, window = split_history_and_window(
                    matrix[:, router_idx],
                    step=step,
                    window_size=self.builder_config.window_size,
                    history_size=self.builder_config.history_size,
                )
                router_signals.append(
                    build_router_ofn(
                        router_id,
                        window=window,
                        history=history,
                        config=self.builder_config,
                        feature_names=feature_names,
                        feature_weights=self.feature_weights,
                        baseline_override=floors[router_idx] if floors is not None else None,
                    )
                )

            aggregated = aggregate_router_signals(router_signals, self.weights, self.builder_config)
            scores[step] = aggregated.score
            step_aggregates.append(aggregated)

        # Calibrate the decision thresholds to the observed score distribution.
        alert_threshold, clear_threshold = self._resolve_thresholds(scores)

        # Pass 2 — run the hysteresis automaton with the resolved thresholds.
        snapshots: list[DetectionSnapshot] = []
        for step, aggregated in zip(evaluated_steps, step_aggregates):
            alarm = self._update_alarm(
                aggregated.score,
                aggregated.positive_routers,
                min_positive,
                alert_threshold,
                clear_threshold,
            )
            predictions[step] = int(alarm)
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
