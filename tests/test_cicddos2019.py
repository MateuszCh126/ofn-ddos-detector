"""Tests for the CIC-DDoS2019 flow-to-scenario adapter."""

from __future__ import annotations

import numpy as np
import pytest

from ddos_ofn.cicddos2019 import ConversionConfig, convert_cicddos2019
from ddos_ofn.datasets import load_csv_scenario

RAW_HEADER = (
    "Unnamed: 0,Flow ID, Source IP, Source Port, Destination IP, Destination Port,"
    " Protocol, Timestamp, Total Fwd Packets, Total Backward Packets,"
    "Total Length of Fwd Packets, Total Length of Bwd Packets, Label"
)


def _raw_row(ts: str, src: str, dst: str, fwd: int, bwd: int, label: str) -> str:
    return (
        f"0,flow,{src},1234,{dst},80,17,{ts},{fwd},{bwd},{fwd * 10},{bwd * 10},{label}"
    )


def _write_raw(path, rows: list[str]) -> None:
    path.write_text("\n".join([RAW_HEADER, *rows]) + "\n", encoding="utf-8")


def test_convert_produces_loadable_long_csv(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "out.csv"
    rows = [
        _raw_row("2018-12-01 13:00:00.000000", "10.0.0.1", "192.168.0.1", 2, 1, "BENIGN"),
        _raw_row("2018-12-01 13:00:00.500000", "10.0.0.2", "192.168.0.2", 3, 0, "BENIGN"),
        _raw_row("2018-12-01 13:00:01.200000", "10.0.0.1", "192.168.0.1", 50, 0, "UDP-lag"),
        _raw_row("2018-12-01 13:00:01.800000", "10.0.0.3", "192.168.0.2", 40, 0, "UDP-lag"),
    ]
    _write_raw(raw, rows)

    report = convert_cicddos2019(
        raw, out, ConversionConfig(bin_seconds=1.0, router_key="dst_ip", num_routers=8)
    )

    assert report.steps == 2
    assert sorted(report.routers) == ["192.168.0.1", "192.168.0.2"]
    assert report.attack_steps == 1
    assert report.benign_steps == 1

    # The output must round-trip through the existing loader unchanged.
    scenario = load_csv_scenario(
        out, csv_format="long", feature_column="feature_name", value_column="value"
    )
    assert scenario.labels_present
    assert scenario.traffic.shape[0] == 2  # steps
    assert scenario.traffic.shape[1] == 2  # routers
    np.testing.assert_array_equal(scenario.labels, np.array([0, 1], dtype=np.int8))


def test_label_threshold_controls_attack_marking(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "out.csv"
    # One bin: 1 attack flow, 3 benign -> 25% attack ratio.
    rows = [
        _raw_row("2018-12-01 13:00:00.100000", "10.0.0.1", "192.168.0.1", 1, 0, "UDP-lag"),
        _raw_row("2018-12-01 13:00:00.200000", "10.0.0.2", "192.168.0.1", 1, 0, "BENIGN"),
        _raw_row("2018-12-01 13:00:00.300000", "10.0.0.3", "192.168.0.1", 1, 0, "BENIGN"),
        _raw_row("2018-12-01 13:00:00.400000", "10.0.0.4", "192.168.0.1", 1, 0, "BENIGN"),
    ]
    _write_raw(raw, rows)

    strict = convert_cicddos2019(raw, out, ConversionConfig(bin_seconds=1.0, label_threshold=0.5))
    assert strict.attack_steps == 0

    lenient = convert_cicddos2019(raw, out, ConversionConfig(bin_seconds=1.0, label_threshold=0.2))
    assert lenient.attack_steps == 1


def test_multifeature_conversion(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "out.csv"
    rows = [
        _raw_row("2018-12-01 13:00:00.000000", "10.0.0.1", "192.168.0.1", 2, 1, "BENIGN"),
        _raw_row("2018-12-01 13:00:01.000000", "10.0.0.1", "192.168.0.1", 9, 0, "UDP-lag"),
    ]
    _write_raw(raw, rows)

    report = convert_cicddos2019(
        raw, out, ConversionConfig(bin_seconds=1.0, metrics=("packet_count", "byte_count"))
    )
    assert report.feature_names == ["packet_count", "byte_count"]

    scenario = load_csv_scenario(
        out, csv_format="long", feature_column="feature_name", value_column="value"
    )
    assert scenario.traffic.ndim == 3  # (steps, routers, features)
    assert scenario.traffic.shape[2] == 2


def test_rejects_file_without_timestamp(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Timestamp"):
        convert_cicddos2019(raw, tmp_path / "out.csv")
