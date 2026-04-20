from pathlib import Path

import numpy as np
import pytest

from ddos_ofn.datasets import build_real_train_validation_sets, load_csv_scenario, split_scenario_train_validation


def _write_csv(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_load_csv_scenario_supports_wide_format(tmp_path):
    csv_path = _write_csv(
        tmp_path / "traffic_wide.csv",
        """
step,label,router_a,router_b
0,0,10,20
1,0,12,18
2,1,25,35
3,1,28,40
""",
    )

    scenario = load_csv_scenario(csv_path, csv_format="wide")

    assert scenario.name == "traffic_wide"
    assert scenario.labels_present is True
    assert scenario.router_ids == ["router_a", "router_b"]
    assert scenario.attack_slice == (2, 4)
    assert np.array_equal(scenario.labels, np.array([0, 0, 1, 1], dtype=np.int8))
    assert np.allclose(
        scenario.traffic,
        np.array([[10.0, 20.0], [12.0, 18.0], [25.0, 35.0], [28.0, 40.0]], dtype=np.float64),
    )


def test_load_csv_scenario_supports_long_format(tmp_path):
    csv_path = _write_csv(
        tmp_path / "traffic_long.csv",
        """
step,router_id,packet_count,label
0,router_a,10,0
0,router_b,20,0
1,router_a,12,0
1,router_b,18,0
2,router_a,25,1
2,router_b,35,1
3,router_a,28,1
3,router_b,40,1
""",
    )

    scenario = load_csv_scenario(csv_path, csv_format="long")

    assert scenario.name == "traffic_long"
    assert scenario.labels_present is True
    assert scenario.router_ids == ["router_a", "router_b"]
    assert scenario.attack_slice == (2, 4)
    assert np.array_equal(scenario.labels, np.array([0, 0, 1, 1], dtype=np.int8))
    assert np.allclose(
        scenario.traffic,
        np.array([[10.0, 20.0], [12.0, 18.0], [25.0, 35.0], [28.0, 40.0]], dtype=np.float64),
    )


def test_load_csv_scenario_supports_multifeature_long_format(tmp_path):
    csv_path = _write_csv(
        tmp_path / "traffic_multifeature_long.csv",
        """
step,router_id,feature_name,value,label
0,router_a,packet_count,10,0
0,router_a,byte_count,100,0
0,router_b,packet_count,20,0
0,router_b,byte_count,200,0
1,router_a,packet_count,12,0
1,router_a,byte_count,120,0
1,router_b,packet_count,18,0
1,router_b,byte_count,180,0
2,router_a,packet_count,25,1
2,router_a,byte_count,250,1
2,router_b,packet_count,35,1
2,router_b,byte_count,350,1
""",
    )

    scenario = load_csv_scenario(
        csv_path,
        csv_format="long",
        value_column="packet_count",
        feature_column="feature_name",
    )

    assert scenario.feature_names == ["packet_count", "byte_count"]
    assert scenario.traffic.shape == (3, 2, 2)
    assert np.allclose(
        scenario.traffic[2],
        np.array([[25.0, 250.0], [35.0, 350.0]], dtype=np.float64),
    )


def test_load_csv_scenario_supports_multifeature_wide_format(tmp_path):
    csv_path = _write_csv(
        tmp_path / "traffic_multifeature_wide.csv",
        """
step,label,router_a__packet_count,router_a__byte_count,router_b__packet_count,router_b__byte_count
0,0,10,100,20,200
1,0,12,120,18,180
2,1,25,250,35,350
""",
    )

    scenario = load_csv_scenario(csv_path, csv_format="wide")

    assert scenario.router_ids == ["router_a", "router_b"]
    assert scenario.feature_names == ["packet_count", "byte_count"]
    assert scenario.traffic.shape == (3, 2, 2)
    assert np.allclose(
        scenario.traffic[0],
        np.array([[10.0, 100.0], [20.0, 200.0]], dtype=np.float64),
    )


def test_split_scenario_train_validation_preserves_time_order(tmp_path):
    csv_path = _write_csv(
        tmp_path / "traffic.csv",
        """
step,label,router_a,router_b
0,0,10,20
1,0,12,18
2,1,25,35
3,1,28,40
4,0,14,16
5,0,13,15
""",
    )
    scenario = load_csv_scenario(csv_path, csv_format="wide")

    train_scenario, valid_scenario = split_scenario_train_validation(
        scenario,
        train_fraction=0.5,
        min_segment_steps=2,
    )

    assert train_scenario.name == "traffic_train"
    assert valid_scenario.name == "traffic_validation"
    assert train_scenario.traffic.shape == (3, 2)
    assert valid_scenario.traffic.shape == (3, 2)
    assert np.array_equal(train_scenario.labels, np.array([0, 0, 1], dtype=np.int8))
    assert np.array_equal(valid_scenario.labels, np.array([1, 0, 0], dtype=np.int8))
    assert train_scenario.attack_slice == (2, 3)
    assert valid_scenario.attack_slice == (0, 1)


def test_build_real_train_validation_sets_requires_labels(tmp_path):
    csv_path = _write_csv(
        tmp_path / "traffic_unlabeled.csv",
        """
step,router_a,router_b
0,10,20
1,12,18
2,25,35
3,28,40
""",
    )

    with pytest.raises(ValueError, match="label column"):
        build_real_train_validation_sets(
            [csv_path],
            csv_format="wide",
            train_fraction=0.5,
            min_segment_steps=2,
        )
