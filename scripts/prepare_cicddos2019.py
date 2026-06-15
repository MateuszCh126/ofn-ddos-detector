"""CLI: convert a raw CIC-DDoS2019 CSV into a long-format scenario CSV.

Example
-------
    python scripts/prepare_cicddos2019.py \
        --input data/raw/UDPLag.csv \
        --output data/processed/cicddos_udplag.csv \
        --bin-seconds 1.0 --router-key dst_ip --num-routers 8 \
        --metric packet_count

The output is consumable by every existing entrypoint, e.g.:

    python scripts/benchmark_models.py --csv data/processed/cicddos_udplag.csv --csv-format long \
        --feature-column feature_name --value-column value
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddos_ofn.cicddos2019 import METRICS, ROUTER_KEYS, ConversionConfig, convert_cicddos2019


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw CIC-DDoS2019 flows to scenario CSV")
    parser.add_argument("--input", required=True, help="path to a raw CIC-DDoS2019 CSV file")
    parser.add_argument("--output", required=True, help="destination long-format CSV path")
    parser.add_argument("--bin-seconds", type=float, default=1.0, help="time window width in seconds")
    parser.add_argument("--router-key", choices=ROUTER_KEYS, default="dst_ip")
    parser.add_argument("--num-routers", type=int, default=8, help="keep the N busiest routers")
    parser.add_argument("--metric", action="append", choices=METRICS, default=None, help="repeatable")
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int, default=None, help="cap flows processed (debug)")
    args = parser.parse_args()

    config = ConversionConfig(
        bin_seconds=args.bin_seconds,
        router_key=args.router_key,
        num_routers=args.num_routers,
        metrics=tuple(args.metric) if args.metric else ("packet_count",),
        label_threshold=args.label_threshold,
        max_rows=args.max_rows,
    )
    report = convert_cicddos2019(args.input, args.output, config)

    print(f"wrote {report.output_path}")
    print(f"  rows read / used : {report.rows_read} / {report.rows_used}")
    print(f"  time span        : {report.span_seconds:.1f}s -> {report.steps} steps of {args.bin_seconds}s")
    print(f"  routers ({len(report.routers)})    : {', '.join(report.routers)}")
    print(f"  features         : {', '.join(report.feature_names)}")
    print(f"  steps benign/atk : {report.benign_steps} / {report.attack_steps}")


if __name__ == "__main__":
    main()
