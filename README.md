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
  pyofn/                     # local Ordered Fuzzy Number library
    __init__.py
    core.py
    shapes.py
    viz.py
  ddos_ofn/
    __init__.py
    config.py
    schemas.py
    baseline.py
    ofn_builder.py
    aggregator.py
    detector.py
    comparators.py           # EWMA / volume-threshold baselines
    ga_optimize.py
    metrics.py
    simulation.py
    datasets.py
    cicddos2019.py           # CIC-DDoS2019 raw flow -> scenario CSV converter
  scripts/
    train_ddos_ga.py
    eval_ddos.py
    run_stream_demo.py
    dashboard.py             # desktop UI (supports --snap / --label)
    benchmark_models.py
    prepare_cicddos2019.py
  tests/
    test_pyofn_core.py
    test_ofn_builder.py
    test_direction_switch.py
    test_aggregation.py
    test_detector_rules.py
    test_ga_optimize.py
    test_datasets.py
    test_metrics.py
    test_simulation.py
    test_comparators.py
    test_robustness.py       # floor contamination, level direction, count generalization
  data/
    raw/
    processed/
  artifacts/
    plots/
  run.py                     # desktop dashboard entrypoint
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

- Scientific / technical guide (math + algorithm): [DOKUMENTACJA_NAUKOWA.md](./DOKUMENTACJA_NAUKOWA.md)
- Plain-language explanation (Polish): `wytlumaczenie.docx`
- Original technical guide: [DOKUMENTACJA_DDOS_OFN.md](./DOKUMENTACJA_DDOS_OFN.md)
- AI agents: this repo ships a local nodesify-graphify knowledge graph — see [AGENTS.md](./AGENTS.md).

## Core Idea

1. Each router/node provides a short traffic window (default `4` samples, configurable via `BuilderConfig.window_size`).
2. Each router is normalized against a **robust idle floor** (a low quantile over the series), so a sustained attack that occupies most of the timeline cannot poison the baseline.
3. A directed OFN is built per router for each window; **direction comes from the anomaly *level* above the floor** (a sustained-high router stays positive even when its slope is flat), not just the instantaneous trend.
4. Router OFNs are weighted and aggregated into a single global OFN. Positive direction increases suspicion, negative reduces it.
5. The global OFN is defuzzified into a **count-invariant score** (a weighted mean in robust-sigma units), so one threshold generalizes across 8 or 200 routers and across traffic scales.
6. The detector triggers via threshold + hysteresis, plus a **breadth gate** (`min_positive_fraction`) that scales with the router count and separates a narrow flash crowd from a broad attack.
7. GA tunes weights and detector parameters on labeled scenarios.

### Robustness: works across datasets and router counts

The detection model self-adapts so a single default config generalizes:

- **`baseline_mode="global_floor"`** + `floor_quantile` — idle floor resistant to attacks that dominate the timeline.
- **`direction_mode="level"`** + `level_epsilon` — fires on sustained-high attacks, and (set high enough) keeps pure noise out of the positive count so the breadth gate is meaningful.
- **`min_positive_fraction`** — the breadth requirement is a fraction of the router count, so it auto-scales with network size.
- **`threshold_mode`** — `"absolute"` (default; the score is already in robust-sigma units) or `"auto"` (calibrate alert/clear from the score's own distribution).

On the synthetic suite with the default config, across routers ∈ {8, 30, 60, 120}: `normal` + `flash_crowd` + `flash_cascade` false-positive rate is **0.000**, attack recall **0.55–0.975**, stable across counts.

## Current MVP

- Local `pyofn` package with Kosiński Ordered Fuzzy Number arithmetic.
- Robust idle-floor normalization (low-quantile center + IQR scale) plus legacy median/MAD.
- OFN builder from short traffic windows per router (default `4` samples), level-based direction.
- Count-invariant weighted OFN fusion with signed contribution; fractional breadth gate.
- Stateful detector with alert/clear hysteresis and absolute or auto-calibrated thresholds.
- CSV loader for real datasets in wide and long format, including multi-feature router data.
- CIC-DDoS2019 raw-flow converter (`scripts/prepare_cicddos2019.py`).
- Synthetic scenarios: `normal`, `ddos_ramp`, `ddos_pulse`, `flash_crowd`.
- Extended synthetic validation scenarios: `ddos_low_and_slow`, `ddos_rotating`, `flash_cascade`.
- GA for tuning router weights and detector thresholds.
- Desktop dashboard with side-by-side comparison mode (`--snap` / `--label`).

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

### Side-by-side comparison

The dashboard can tile itself to one half of the screen and wear a header badge,
so two instances run comfortably next to each other (e.g. comparing scenarios,
configs, or a legacy build against this one):

```bash
python scripts/dashboard.py --snap left  --label A
python scripts/dashboard.py --snap right --label B
```

## Real CIC-DDoS2019 data

Convert a raw CIC-DDoS2019 (CICFlowMeter) capture into the long-format scenario CSV:

```bash
python scripts/prepare_cicddos2019.py \
  --input data/raw/UDPLag.csv --output data/processed/cicddos_udplag.csv \
  --bin-seconds 5 --router-key dst_ip --num-routers 8 --metric flow_count --metric packet_count
```

Notes from real-data evaluation: per-second binning labels many idle steps inside
the attack window as "attack" (the flood is bursty), which caps recall — coarser
bins (`--bin-seconds 5..10`) recover it. On congested captures, volume-only
features limit benign/attack separation; richer flow features (port/IP entropy,
packet-size distribution, SYN/flow statistics) are the path to higher precision.
