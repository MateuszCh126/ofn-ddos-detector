"""Synthetic traffic generators for DDoS detection experiments."""

from __future__ import annotations

import numpy as np

from ddos_ofn.config import SimulationConfig
from ddos_ofn.schemas import SimulationResult


def _base_matrix(config: SimulationConfig) -> tuple[np.ndarray, list[str], np.random.Generator]:
    rng = np.random.default_rng(config.seed)
    router_ids = [f"router_{idx:02d}" for idx in range(config.routers)]
    baselines = rng.uniform(config.baseline_low, config.baseline_high, size=config.routers)
    noise = rng.normal(0.0, config.noise_std, size=(config.steps, config.routers))
    traffic = baselines + noise
    traffic = np.clip(traffic, 0.0, None)
    return traffic, router_ids, rng


def _attack_router_indices(config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    count = max(1, int(round(config.routers * config.attack_fraction)))
    return np.sort(rng.choice(config.routers, size=count, replace=False))


def generate_scenario(name: str, config: SimulationConfig | None = None) -> SimulationResult:
    """Create one synthetic traffic scenario."""

    cfg = config or SimulationConfig()
    traffic, router_ids, rng = _base_matrix(cfg)
    labels = np.zeros(cfg.steps, dtype=np.int8)
    start = cfg.attack_start
    stop = min(cfg.steps, start + cfg.attack_duration)
    attack_slice: tuple[int, int] | None = None

    if name == "normal":
        attack_slice = None
    elif name == "ddos_ramp":
        attack_slice = (start, stop)
        labels[start:stop] = 1
        routers = _attack_router_indices(cfg, rng)
        ramp = np.linspace(0.0, cfg.attack_scale, stop - start)
        traffic[start:stop, routers] += ramp[:, None] * cfg.noise_std * 3.0
    elif name == "ddos_pulse":
        attack_slice = (start, stop)
        labels[start:stop] = 1
        routers = _attack_router_indices(cfg, rng)
        pulse = np.zeros(stop - start, dtype=np.float64)
        pulse[::3] = cfg.pulse_scale
        pulse[1::3] = cfg.pulse_scale * 0.6
        traffic[start:stop, routers] += pulse[:, None] * cfg.noise_std * 3.2
    elif name == "flash_crowd":
        attack_slice = None
        group = rng.choice(cfg.routers, size=max(1, cfg.routers // 4), replace=False)
        burst = np.sin(np.linspace(0.0, np.pi, cfg.attack_duration)) * cfg.flash_scale * cfg.noise_std * 1.5
        traffic[start:stop, group] += burst[:, None]
    else:
        raise ValueError(f"unknown scenario: {name}")

    traffic = np.clip(traffic, 0.0, None)
    return SimulationResult(
        name=name,
        router_ids=router_ids,
        traffic=traffic,
        labels=labels,
        attack_slice=attack_slice,
    )


def generate_suite(config: SimulationConfig | None = None) -> list[SimulationResult]:
    """Return a small benchmark suite."""

    cfg = config or SimulationConfig()
    return [
        generate_scenario("normal", cfg),
        generate_scenario("ddos_ramp", cfg),
        generate_scenario("ddos_pulse", cfg),
        generate_scenario("flash_crowd", cfg),
    ]
