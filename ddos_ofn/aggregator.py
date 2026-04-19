"""Combine per-router OFNs into a global network signal."""

from __future__ import annotations

from collections.abc import Mapping

from pyofn import OFN, singleton

from ddos_ofn.config import BuilderConfig
from ddos_ofn.schemas import AggregatedSignal, RouterOFN


def aggregate_router_signals(
    router_signals: list[RouterOFN],
    weights: Mapping[str, float] | None,
    builder_config: BuilderConfig,
) -> AggregatedSignal:
    """Fuse router OFNs into a single weighted OFN."""

    total: OFN = singleton(0.0, n=builder_config.n_points)
    positive = 0
    negative = 0
    neutral = 0

    for signal in router_signals:
        weight = 1.0 if weights is None else float(weights.get(signal.router_id, 1.0))
        contribution = signal.ofn * weight

        if signal.direction > 0:
            total = total + contribution
            positive += 1
        elif signal.direction < 0:
            total = total - contribution
            negative += 1
        else:
            total = total + contribution * builder_config.neutral_contribution
            neutral += 1

    raw_score = float(total.defuzzify_cog())
    score = max(raw_score, 0.0)
    return AggregatedSignal(
        global_ofn=total,
        raw_score=raw_score,
        score=score,
        positive_routers=positive,
        negative_routers=negative,
        neutral_routers=neutral,
        router_signals=router_signals,
    )
