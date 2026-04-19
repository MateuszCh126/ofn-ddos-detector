"""Dataset helpers for train and evaluation splits."""

from __future__ import annotations

from dataclasses import asdict

from ddos_ofn.config import SimulationConfig
from ddos_ofn.schemas import SimulationResult
from ddos_ofn.simulation import generate_suite


def build_train_validation_sets(
    config: SimulationConfig | None = None,
) -> tuple[list[SimulationResult], list[SimulationResult]]:
    """Generate two synthetic sets with different seeds."""

    base = config or SimulationConfig()
    base_dict = asdict(base)
    train_cfg = SimulationConfig(**{**base_dict, "seed": base.seed})
    valid_cfg = SimulationConfig(**{**base_dict, "seed": (base.seed or 0) + 101})
    return generate_suite(train_cfg), generate_suite(valid_cfg)
