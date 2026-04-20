import numpy as np

from ddos_ofn.config import SimulationConfig
from ddos_ofn.datasets import build_train_validation_sets
from ddos_ofn.simulation import generate_scenario, generate_suite


def test_generate_scenario_supports_extended_attack_patterns():
    cfg = SimulationConfig(routers=12, steps=80, seed=5, attack_start=30, attack_duration=20)

    low_and_slow = generate_scenario("ddos_low_and_slow", cfg)
    rotating = generate_scenario("ddos_rotating", cfg)
    flash_cascade = generate_scenario("flash_cascade", cfg)

    assert low_and_slow.attack_slice == (30, 50)
    assert rotating.attack_slice == (30, 50)
    assert flash_cascade.attack_slice is None
    assert np.sum(low_and_slow.labels) == 20
    assert np.sum(rotating.labels) == 20
    assert np.sum(flash_cascade.labels) == 0


def test_generate_suite_extended_contains_more_realistic_scenarios():
    cfg = SimulationConfig(routers=8, steps=64, seed=3, attack_start=24, attack_duration=16)

    suite = generate_suite(cfg, suite="extended")
    names = [scenario.name for scenario in suite]

    assert "ddos_low_and_slow" in names
    assert "ddos_rotating" in names
    assert "flash_cascade" in names
    assert len(suite) > 4


def test_build_train_validation_sets_supports_extended_suite():
    train_set, valid_set = build_train_validation_sets(
        SimulationConfig(routers=8, steps=64, seed=9, attack_start=24, attack_duration=16),
        suite="extended",
    )

    assert len(train_set) == len(valid_set)
    assert len(train_set) == 7
