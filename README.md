# ofn-ddos-detector

Directed Fuzzy Number (OFN) based DDoS detection framework with multi-router signal fusion and genetic algorithm optimization of weights and alert thresholds.

## License

This project is distributed under **PolyForm Noncommercial License 1.0.0**.

- Allowed: personal, educational, research, and other noncommercial uses.
- Not allowed: commercial use without a separate commercial agreement.

See [LICENSE](./LICENSE) and `COMMERCIAL.md`.

## Project Structure

```text
ofn-ddos-detector/
  ddos_ofn/
    __init__.py
    config.py
    schemas.py
    baseline.py
    ofn_builder.py
    aggregator.py
    detector.py
    ga_optimize.py
    metrics.py
    simulation.py
    datasets.py
  scripts/
    train_ddos_ga.py
    eval_ddos.py
    run_stream_demo.py
    dashboard.py
    benchmark_models.py
  tests/
    test_ofn_builder.py
    test_direction_switch.py
    test_aggregation.py
    test_detector_rules.py
    test_ga_optimize.py
    test_datasets.py
    test_simulation.py
    test_comparators.py
  data/
    raw/
    processed/
  artifacts/
    plots/
  README.md
  LICENSE
  COMMERCIAL.md
  .gitignore
```

## Quick Start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Documentation

- Main technical guide: [DOKUMENTACJA_DDOS_OFN.md](./DOKUMENTACJA_DDOS_OFN.md)

## Core Idea

1. Each router/node provides a short traffic window (default `4` samples, configurable via `BuilderConfig.window_size`).
2. A directed OFN is built per router for each time window.
3. Router OFNs are weighted and aggregated into a global OFN score.
4. Positive direction increases suspicion; negative direction reduces it.
5. Detector triggers alarm using threshold + hysteresis.
6. GA tunes weights and detector parameters on labeled scenarios.

## Current MVP

- Local `pyofn` package copied from the OFN prototype repository.
- Robust baseline normalization with median and MAD.
- OFN builder from short traffic windows per router (default `4` samples).
- Weighted OFN fusion across routers with signed contribution.
- Stateful detector with alert and clear hysteresis.
- CSV loader for real datasets in wide and long format, including multi-feature router data.
- Synthetic scenarios: `normal`, `ddos_ramp`, `ddos_pulse`, `flash_crowd`.
- Extended synthetic validation scenarios: `ddos_low_and_slow`, `ddos_rotating`, `flash_cascade`.
- Basic GA for tuning router weights and detector thresholds.

## Real CSV Input

The project now accepts real traffic data from CSV in two layouts:

- Wide format: one row per time step, one column per router, optional `label`.
- Long format: one row per `(step, router_id)` pair with a metric value and optional `label`.
- Multi-feature wide format: columns named like `router_a__packet_count`, `router_a__byte_count`.
- Multi-feature long format: one row per `(step, router_id, feature_name)` with a generic `value` column.

Wide example:

```csv
step,label,router_a,router_b
0,0,10,20
1,0,12,18
2,1,25,35
```

Long example:

```csv
step,router_id,packet_count,label
0,router_a,10,0
0,router_b,20,0
1,router_a,12,0
1,router_b,18,0
```

Multi-feature long example:

```csv
step,router_id,feature_name,value,label
0,router_a,packet_count,10,0
0,router_a,byte_count,100,0
0,router_b,packet_count,20,0
0,router_b,byte_count,200,0
```

Evaluation on real CSV:

```bash
python scripts/eval_ddos.py --csv path/to/traffic.csv --csv-format wide
python scripts/eval_ddos.py --csv path/to/traffic_long.csv --csv-format long
python scripts/eval_ddos.py --csv path/to/multifeature.csv --csv-format long --feature-column feature_name --value-column value
```

GA training on labeled CSV:

```bash
python scripts/train_ddos_ga.py --csv path/to/traffic.csv --csv-format wide
python scripts/train_ddos_ga.py --csv path/to/traffic_a.csv --csv path/to/traffic_b.csv --csv-format long
python scripts/train_ddos_ga.py --csv path/to/multifeature.csv --csv-format long --feature-column feature_name --value-column value
```

Extended synthetic benchmark:

```bash
python scripts/eval_ddos.py --scenario ddos_low_and_slow --routers 12 --steps 96 --attack-start 48 --attack-duration 24
python scripts/train_ddos_ga.py --suite extended --routers 8 --steps 80 --attack-start 36 --attack-duration 20
python scripts/benchmark_models.py --scenario normal --scenario ddos_ramp --scenario ddos_low_and_slow --scenario flash_crowd
```

## Main Commands

```bash
pytest -q
python run.py
python scripts/dashboard.py
python scripts/dashboard.py --smoke-test
python scripts/eval_ddos.py --scenario ddos_ramp
python scripts/train_ddos_ga.py
python scripts/eval_ddos.py --csv path/to/traffic.csv --csv-format wide
python scripts/train_ddos_ga.py --csv path/to/traffic.csv --csv-format wide
python scripts/run_stream_demo.py
python scripts/benchmark_models.py --scenario normal --scenario ddos_ramp --scenario ddos_low_and_slow --scenario flash_crowd
```

## Desktop App

The main user-facing entrypoint is the local Tkinter dashboard in `scripts/dashboard.py`.

- Generate and inspect synthetic attack scenarios.
- Run the detector and visualize score/alarm behavior.
- Load tuned parameters saved during GA experiments.
- Use the app locally during demos, presentations, and manual debugging.

Start the dashboard:

```bash
python run.py
python scripts/dashboard.py
```

Quick smoke test for the desktop flow:

```bash
python scripts/dashboard.py --smoke-test
```
