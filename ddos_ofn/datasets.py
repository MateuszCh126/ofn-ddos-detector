"""Dataset helpers for synthetic and CSV-backed train/evaluation splits."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ddos_ofn.config import SimulationConfig
from ddos_ofn.schemas import SimulationResult
from ddos_ofn.simulation import generate_suite


def _parse_float(value: str | None, *, column: str, path: Path) -> float:
    if value is None or value.strip() == "":
        raise ValueError(f"missing numeric value in column '{column}' for {path}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value '{value}' in column '{column}' for {path}") from exc


def _parse_label(value: str | None, *, path: Path) -> int:
    if value is None or value.strip() == "":
        raise ValueError(f"missing label value for {path}")

    normalized = value.strip().lower()
    if normalized in {"0", "false", "normal", "no"}:
        return 0
    if normalized in {"1", "true", "attack", "yes"}:
        return 1

    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid label value '{value}' for {path}") from exc

    if numeric in {0.0, 1.0}:
        return int(numeric)
    raise ValueError(f"label value must resolve to 0/1 for {path}, got '{value}'")


def _infer_attack_slice(labels: np.ndarray, *, labels_present: bool) -> tuple[int, int] | None:
    if not labels_present:
        return None

    positive_steps = np.flatnonzero(labels == 1)
    if positive_steps.size == 0:
        return None
    return int(positive_steps[0]), int(positive_steps[-1] + 1)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} is missing a header row")
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"CSV file {path} does not contain any data rows")
    return list(reader.fieldnames), rows


def _sort_rows(rows: list[dict[str, str]], step_column: str) -> list[dict[str, str]]:
    if not rows or step_column not in rows[0]:
        return rows
    if any((row.get(step_column) or "").strip() == "" for row in rows):
        return rows

    try:
        return sorted(rows, key=lambda row: float(row[step_column]))
    except ValueError:
        return rows


def _resolve_existing_column(fieldnames: Sequence[str], preferred: str, fallbacks: Sequence[str]) -> str:
    if preferred in fieldnames:
        return preferred
    for fallback in fallbacks:
        if fallback in fieldnames:
            return fallback
    return preferred


def _load_wide_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    *,
    name: str,
    step_column: str,
    timestamp_column: str,
    label_column: str,
    default_feature_name: str,
    wide_feature_separator: str,
) -> SimulationResult:
    sorted_rows = _sort_rows(rows, step_column)
    metadata_columns = {step_column, timestamp_column, label_column}
    data_columns = [column for column in fieldnames if column not in metadata_columns]
    if not data_columns:
        raise ValueError(f"wide CSV {path} must contain at least one router column")

    router_ids: list[str] = []
    feature_names: list[str] = []
    router_index: dict[str, int] = {}
    feature_index: dict[str, int] = {}
    column_layout: list[tuple[str, int, int]] = []

    for column in data_columns:
        if wide_feature_separator and wide_feature_separator in column:
            router_id, feature_name = column.rsplit(wide_feature_separator, 1)
        else:
            router_id, feature_name = column, default_feature_name

        if not router_id:
            raise ValueError(f"could not infer router id from column '{column}' in {path}")
        if not feature_name:
            raise ValueError(f"could not infer feature name from column '{column}' in {path}")

        if router_id not in router_index:
            router_index[router_id] = len(router_ids)
            router_ids.append(router_id)
        if feature_name not in feature_index:
            feature_index[feature_name] = len(feature_names)
            feature_names.append(feature_name)
        column_layout.append((column, router_index[router_id], feature_index[feature_name]))

    labels_present = label_column in fieldnames
    traffic_tensor = np.full((len(sorted_rows), len(router_ids), len(feature_names)), np.nan, dtype=np.float64)
    labels: list[int] = []

    for row_idx, row in enumerate(sorted_rows):
        for column_name, router_idx, feature_idx in column_layout:
            traffic_tensor[row_idx, router_idx, feature_idx] = _parse_float(
                row.get(column_name),
                column=column_name,
                path=path,
            )
        labels.append(_parse_label(row.get(label_column), path=path) if labels_present else 0)

    missing = np.argwhere(np.isnan(traffic_tensor))
    if missing.size:
        row_idx, router_idx, feature_idx = missing[0].tolist()
        raise ValueError(
            f"wide CSV {path} is missing router '{router_ids[router_idx]}' feature '{feature_names[feature_idx]}'"
            f" at row {row_idx}"
        )

    traffic = traffic_tensor[:, :, 0] if len(feature_names) == 1 else traffic_tensor
    label_array = np.asarray(labels, dtype=np.int8)
    return SimulationResult(
        name=name,
        router_ids=router_ids,
        traffic=traffic,
        labels=label_array,
        attack_slice=_infer_attack_slice(label_array, labels_present=labels_present),
        labels_present=labels_present,
        feature_names=feature_names,
    )


def _load_long_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    *,
    name: str,
    step_column: str,
    timestamp_column: str,
    label_column: str,
    router_column: str,
    value_column: str,
    feature_column: str,
) -> SimulationResult:
    if step_column not in fieldnames and timestamp_column not in fieldnames:
        raise ValueError(
            f"long CSV {path} must contain either '{step_column}' or '{timestamp_column}' to define time steps"
        )

    key_column = step_column if step_column in fieldnames else timestamp_column
    sorted_rows = _sort_rows(rows, key_column if key_column == step_column else "")
    labels_present = label_column in fieldnames
    resolved_value_column = _resolve_existing_column(fieldnames, value_column, ("value",))
    has_feature_column = feature_column in fieldnames

    step_index: dict[str, int] = {}
    router_index: dict[str, int] = {}
    feature_index: dict[str, int] = {}
    step_keys: list[str] = []
    router_ids: list[str] = []
    feature_names: list[str] = []
    values: dict[tuple[int, int, int], float] = {}
    labels_by_step: dict[int, int] = {}

    for row in sorted_rows:
        step_key = (row.get(key_column) or "").strip()
        router_id = (row.get(router_column) or "").strip()
        feature_name = (row.get(feature_column) or "").strip() if has_feature_column else resolved_value_column
        if not step_key:
            raise ValueError(f"missing step value in column '{key_column}' for {path}")
        if not router_id:
            raise ValueError(f"missing router id in column '{router_column}' for {path}")
        if not feature_name:
            raise ValueError(f"missing feature name in column '{feature_column}' for {path}")

        if step_key not in step_index:
            step_index[step_key] = len(step_keys)
            step_keys.append(step_key)
        if router_id not in router_index:
            router_index[router_id] = len(router_ids)
            router_ids.append(router_id)
        if feature_name not in feature_index:
            feature_index[feature_name] = len(feature_names)
            feature_names.append(feature_name)

        row_idx = step_index[step_key]
        col_idx = router_index[router_id]
        feature_idx = feature_index[feature_name]
        coord = (row_idx, col_idx, feature_idx)
        if coord in values:
            raise ValueError(
                f"duplicate observation for step '{step_key}', router '{router_id}', feature '{feature_name}' in {path}"
            )
        values[coord] = _parse_float(row.get(resolved_value_column), column=resolved_value_column, path=path)

        if labels_present:
            label_value = _parse_label(row.get(label_column), path=path)
            previous = labels_by_step.get(row_idx)
            if previous is None:
                labels_by_step[row_idx] = label_value
            elif previous != label_value:
                raise ValueError(f"inconsistent labels for step '{step_key}' in {path}")

    traffic_tensor = np.full((len(step_keys), len(router_ids), len(feature_names)), np.nan, dtype=np.float64)
    for (row_idx, col_idx, feature_idx), value in values.items():
        traffic_tensor[row_idx, col_idx, feature_idx] = value

    missing = np.argwhere(np.isnan(traffic_tensor))
    if missing.size:
        row_idx, col_idx, feature_idx = missing[0].tolist()
        raise ValueError(
            f"long CSV {path} is missing router '{router_ids[col_idx]}' feature '{feature_names[feature_idx]}'"
            f" for step '{step_keys[row_idx]}'"
        )

    label_array = np.zeros(len(step_keys), dtype=np.int8)
    if labels_present:
        for row_idx, label_value in labels_by_step.items():
            label_array[row_idx] = label_value

    traffic = traffic_tensor[:, :, 0] if len(feature_names) == 1 else traffic_tensor
    return SimulationResult(
        name=name,
        router_ids=router_ids,
        traffic=traffic,
        labels=label_array,
        attack_slice=_infer_attack_slice(label_array, labels_present=labels_present),
        labels_present=labels_present,
        feature_names=feature_names,
    )


def load_csv_scenario(
    path: str | Path,
    *,
    name: str | None = None,
    csv_format: str = "auto",
    step_column: str = "step",
    timestamp_column: str = "timestamp",
    label_column: str = "label",
    router_column: str = "router_id",
    value_column: str = "packet_count",
    feature_column: str = "feature_name",
    wide_feature_separator: str = "__",
) -> SimulationResult:
    """Load one real dataset scenario from a CSV file.

    Supported formats:
    - wide: one row per time step, one column per router
    - long: one row per (time step, router) pair
    """

    csv_path = Path(path)
    fieldnames, rows = _read_csv_rows(csv_path)
    scenario_name = name or csv_path.stem
    resolved_format = csv_format.lower()
    resolved_value_column = _resolve_existing_column(fieldnames, value_column, ("value",))

    if resolved_format == "auto":
        if router_column in fieldnames and resolved_value_column in fieldnames:
            resolved_format = "long"
        else:
            resolved_format = "wide"

    if resolved_format == "wide":
        return _load_wide_csv(
            csv_path,
            rows,
            fieldnames,
            name=scenario_name,
            step_column=step_column,
            timestamp_column=timestamp_column,
            label_column=label_column,
            default_feature_name=resolved_value_column,
            wide_feature_separator=wide_feature_separator,
        )
    if resolved_format == "long":
        return _load_long_csv(
            csv_path,
            rows,
            fieldnames,
            name=scenario_name,
            step_column=step_column,
            timestamp_column=timestamp_column,
            label_column=label_column,
            router_column=router_column,
            value_column=resolved_value_column,
            feature_column=feature_column,
        )
    raise ValueError(f"unsupported csv_format '{csv_format}', expected auto/wide/long")


def load_csv_scenarios(
    paths: Sequence[str | Path],
    **kwargs: object,
) -> list[SimulationResult]:
    """Load multiple CSV-backed scenarios."""

    return [load_csv_scenario(path, **kwargs) for path in paths]


def _slice_scenario(
    scenario: SimulationResult,
    *,
    start: int,
    stop: int,
    name: str,
) -> SimulationResult:
    labels = np.asarray(scenario.labels[start:stop], dtype=np.int8).copy()
    return SimulationResult(
        name=name,
        router_ids=list(scenario.router_ids),
        traffic=np.asarray(scenario.traffic[start:stop], dtype=np.float64).copy(),
        labels=labels,
        attack_slice=_infer_attack_slice(labels, labels_present=scenario.labels_present),
        labels_present=scenario.labels_present,
        feature_names=list(scenario.feature_names),
    )


def split_scenario_train_validation(
    scenario: SimulationResult,
    *,
    train_fraction: float = 0.7,
    min_segment_steps: int = 16,
) -> tuple[SimulationResult, SimulationResult]:
    """Split one scenario into time-ordered train and validation segments."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if min_segment_steps < 1:
        raise ValueError("min_segment_steps must be at least 1")

    total_steps = int(scenario.traffic.shape[0])
    if total_steps < min_segment_steps * 2:
        raise ValueError(
            f"scenario '{scenario.name}' is too short for a {min_segment_steps}/{min_segment_steps} train/validation split"
        )

    split_idx = int(round(total_steps * train_fraction))
    split_idx = max(min_segment_steps, min(total_steps - min_segment_steps, split_idx))

    train_scenario = _slice_scenario(
        scenario,
        start=0,
        stop=split_idx,
        name=f"{scenario.name}_train",
    )
    valid_scenario = _slice_scenario(
        scenario,
        start=split_idx,
        stop=total_steps,
        name=f"{scenario.name}_validation",
    )
    return train_scenario, valid_scenario


def _ensure_consistent_router_ids(scenarios: Sequence[SimulationResult]) -> None:
    if not scenarios:
        raise ValueError("at least one scenario is required")

    reference = list(scenarios[0].router_ids)
    for scenario in scenarios[1:]:
        if list(scenario.router_ids) != reference:
            raise ValueError("all scenarios must use the same router_ids in the same order")
        if list(scenario.feature_names) != list(scenarios[0].feature_names):
            raise ValueError("all scenarios must use the same feature_names in the same order")


def build_real_train_validation_sets(
    paths: Sequence[str | Path],
    *,
    train_fraction: float = 0.7,
    min_segment_steps: int = 16,
    **kwargs: object,
) -> tuple[list[SimulationResult], list[SimulationResult]]:
    """Load labeled CSV scenarios and split them into train/validation sets."""

    scenarios = load_csv_scenarios(paths, **kwargs)
    if any(not scenario.labels_present for scenario in scenarios):
        raise ValueError("real training data must include a label column with ground truth")

    _ensure_consistent_router_ids(scenarios)

    train_set: list[SimulationResult] = []
    valid_set: list[SimulationResult] = []
    for scenario in scenarios:
        train_scenario, valid_scenario = split_scenario_train_validation(
            scenario,
            train_fraction=train_fraction,
            min_segment_steps=min_segment_steps,
        )
        train_set.append(train_scenario)
        valid_set.append(valid_scenario)

    return train_set, valid_set


def build_train_validation_sets(
    config: SimulationConfig | None = None,
    *,
    suite: str = "basic",
) -> tuple[list[SimulationResult], list[SimulationResult]]:
    """Generate two synthetic sets with different seeds."""

    base = config or SimulationConfig()
    base_dict = asdict(base)
    train_cfg = SimulationConfig(**{**base_dict, "seed": base.seed})
    valid_cfg = SimulationConfig(**{**base_dict, "seed": (base.seed or 0) + 101})
    return generate_suite(train_cfg, suite=suite), generate_suite(valid_cfg, suite=suite)
