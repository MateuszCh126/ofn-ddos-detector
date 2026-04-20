"""OFN-based DDoS detection package."""

__version__ = "0.3.0"

from ddos_ofn.comparators import (
    EWMAConfig,
    VolumeThresholdConfig,
    run_ewma_detector,
    run_volume_threshold_detector,
)
from ddos_ofn.config import BuilderConfig, DetectorConfig, GAConfig, SimulationConfig
from ddos_ofn.datasets import (
    build_real_train_validation_sets,
    build_train_validation_sets,
    load_csv_scenario,
    load_csv_scenarios,
    split_scenario_train_validation,
)
from ddos_ofn.detector import DDoSDetector
from ddos_ofn.metrics import evaluate_predictions
from ddos_ofn.simulation import generate_scenario, generate_suite

__all__ = [
    "__version__",
    "BuilderConfig",
    "DetectorConfig",
    "EWMAConfig",
    "GAConfig",
    "SimulationConfig",
    "DDoSDetector",
    "VolumeThresholdConfig",
    "evaluate_predictions",
    "build_real_train_validation_sets",
    "build_train_validation_sets",
    "load_csv_scenario",
    "load_csv_scenarios",
    "run_ewma_detector",
    "run_volume_threshold_detector",
    "split_scenario_train_validation",
    "generate_scenario",
    "generate_suite",
]
