"""Convert raw CIC-DDoS2019 (CICFlowMeter) flow records into the project CSV format.

The CIC-DDoS2019 dataset ships as per-attack CSV files where every row is a single
network flow with a timestamp, source/destination IP, volume counters and an attack
label (``BENIGN`` for normal traffic, otherwise the attack name).

This module streams such a file row by row (the raw files are multi-GB) and folds the
flows into a ``(time step, router)`` grid that matches the long CSV layout already
consumed by :func:`ddos_ofn.datasets.load_csv_scenario`:

``step,router_id,feature_name,value,label``

So the existing detector, baselines and benchmark run on real traffic with no other
changes.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# CICFlowMeter headers carry leading spaces; we strip them on read.
COL_TIMESTAMP = "Timestamp"
COL_SOURCE_IP = "Source IP"
COL_DEST_IP = "Destination IP"
COL_DEST_PORT = "Destination Port"
COL_FWD_PACKETS = "Total Fwd Packets"
COL_BWD_PACKETS = "Total Backward Packets"
COL_FWD_BYTES = "Total Length of Fwd Packets"
COL_BWD_BYTES = "Total Length of Bwd Packets"
COL_LABEL = "Label"

ROUTER_KEYS = ("dst_ip", "src_ip", "src_subnet", "dst_port", "hash")
METRICS = ("packet_count", "byte_count", "flow_count")

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %I:%M:%S %p",
)


@dataclass(slots=True)
class ConversionConfig:
    """Parameters controlling the flow-to-timeseries folding."""

    bin_seconds: float = 1.0
    router_key: str = "dst_ip"
    num_routers: int = 8
    metrics: tuple[str, ...] = ("packet_count",)
    label_threshold: float = 0.5
    max_rows: int | None = None

    def __post_init__(self) -> None:
        if self.bin_seconds <= 0:
            raise ValueError("bin_seconds must be positive")
        if self.router_key not in ROUTER_KEYS:
            raise ValueError(f"router_key must be one of {ROUTER_KEYS}")
        if self.num_routers < 1:
            raise ValueError("num_routers must be at least 1")
        if not self.metrics:
            raise ValueError("at least one metric is required")
        for metric in self.metrics:
            if metric not in METRICS:
                raise ValueError(f"unsupported metric '{metric}', expected one of {METRICS}")
        if not 0.0 < self.label_threshold <= 1.0:
            raise ValueError("label_threshold must be in (0, 1]")


@dataclass(slots=True)
class ConversionReport:
    """Summary statistics describing the produced scenario."""

    rows_read: int
    rows_used: int
    steps: int
    routers: list[str]
    feature_names: list[str]
    attack_steps: int
    benign_steps: int
    span_seconds: float
    output_path: Path


def _parse_timestamp(raw: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def _to_float(raw: str | None) -> float:
    if raw is None:
        return 0.0
    text = raw.strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _router_value(row: dict[str, str], config: ConversionConfig) -> str:
    if config.router_key == "dst_ip":
        return (row.get(COL_DEST_IP) or "unknown").strip()
    if config.router_key == "src_ip":
        return (row.get(COL_SOURCE_IP) or "unknown").strip()
    if config.router_key == "dst_port":
        return (row.get(COL_DEST_PORT) or "unknown").strip()
    if config.router_key == "src_subnet":
        ip = (row.get(COL_SOURCE_IP) or "").strip()
        octets = ip.split(".")
        return ".".join(octets[:3]) + ".0/24" if len(octets) == 4 else (ip or "unknown")
    # hash: spread source IPs across a fixed number of synthetic ingress routers
    ip = (row.get(COL_SOURCE_IP) or "").strip()
    bucket = (hash(ip) % config.num_routers) if ip else 0
    return f"router_{bucket:02d}"


def _metric_value(row: dict[str, str], metric: str) -> float:
    if metric == "packet_count":
        return _to_float(row.get(COL_FWD_PACKETS)) + _to_float(row.get(COL_BWD_PACKETS))
    if metric == "byte_count":
        return _to_float(row.get(COL_FWD_BYTES)) + _to_float(row.get(COL_BWD_BYTES))
    return 1.0  # flow_count


def _is_attack(row: dict[str, str]) -> bool:
    label = (row.get(COL_LABEL) or "").strip().upper()
    return bool(label) and label != "BENIGN"


def convert_cicddos2019(
    input_path: str | Path,
    output_path: str | Path,
    config: ConversionConfig | None = None,
) -> ConversionReport:
    """Stream a raw CIC-DDoS2019 CSV and write a long-format scenario CSV."""

    cfg = config or ConversionConfig()
    src = Path(input_path)
    dst = Path(output_path)

    # (step, router) -> {metric: summed value}
    cell_metrics: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # step -> [attack_flows, total_flows]
    step_label_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    router_totals: dict[str, float] = defaultdict(float)

    rows_read = 0
    rows_used = 0
    t0: float | None = None
    t_last: float = 0.0

    with src.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{src} has no header row")
        index = {name.strip(): pos for pos, name in enumerate(header)}
        for required in (COL_TIMESTAMP, COL_LABEL):
            if required not in index:
                raise ValueError(f"{src} is missing required column '{required}'")

        def field(cols: list[str], name: str) -> str | None:
            pos = index.get(name)
            if pos is None or pos >= len(cols):
                return None
            return cols[pos]

        for cols in reader:
            rows_read += 1
            if cfg.max_rows is not None and rows_used >= cfg.max_rows:
                break
            ts = _parse_timestamp(field(cols, COL_TIMESTAMP) or "")
            if ts is None:
                continue
            if t0 is None:
                t0 = ts
            step = int((ts - t0) // cfg.bin_seconds)
            if step < 0:
                continue  # out-of-order row before the established origin
            t_last = max(t_last, ts)

            row = {
                COL_SOURCE_IP: field(cols, COL_SOURCE_IP) or "",
                COL_DEST_IP: field(cols, COL_DEST_IP) or "",
                COL_DEST_PORT: field(cols, COL_DEST_PORT) or "",
                COL_FWD_PACKETS: field(cols, COL_FWD_PACKETS) or "",
                COL_BWD_PACKETS: field(cols, COL_BWD_PACKETS) or "",
                COL_FWD_BYTES: field(cols, COL_FWD_BYTES) or "",
                COL_BWD_BYTES: field(cols, COL_BWD_BYTES) or "",
                COL_LABEL: field(cols, COL_LABEL) or "",
            }
            router = _router_value(row, cfg)
            bucket = cell_metrics[(step, router)]
            for metric in cfg.metrics:
                value = _metric_value(row, metric)
                bucket[metric] += value
                if metric == cfg.metrics[0]:
                    router_totals[router] += value

            counts = step_label_counts[step]
            counts[1] += 1
            if _is_attack(row):
                counts[0] += 1
            rows_used += 1

    if t0 is None or not cell_metrics:
        raise ValueError(f"no usable rows with a valid timestamp found in {src}")

    # Keep the busiest N routers so every step shares the same router set.
    ranked = sorted(router_totals, key=lambda r: router_totals[r], reverse=True)
    routers = sorted(ranked[: cfg.num_routers])
    router_set = set(routers)

    steps = sorted({step for (step, router) in cell_metrics if router in router_set})
    step_label = {
        step: int((counts[0] / counts[1]) >= cfg.label_threshold) if counts[1] else 0
        for step, counts in step_label_counts.items()
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "router_id", "feature_name", "value", "label"])
        for step in steps:
            label = step_label.get(step, 0)
            for router in routers:
                bucket = cell_metrics.get((step, router), {})
                for metric in cfg.metrics:
                    writer.writerow([step, router, metric, f"{bucket.get(metric, 0.0):.6g}", label])

    attack_steps = sum(1 for step in steps if step_label.get(step, 0) == 1)
    return ConversionReport(
        rows_read=rows_read,
        rows_used=rows_used,
        steps=len(steps),
        routers=routers,
        feature_names=list(cfg.metrics),
        attack_steps=attack_steps,
        benign_steps=len(steps) - attack_steps,
        span_seconds=(t_last - t0),
        output_path=dst,
    )
