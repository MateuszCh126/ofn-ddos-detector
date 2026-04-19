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
  tests/
    test_ofn_builder.py
    test_direction_switch.py
    test_aggregation.py
    test_detector_rules.py
    test_ga_optimize.py
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

## Core Idea

1. Each router/node provides 4 time-adjacent traffic measurements.
2. A directed OFN is built per router for each time window.
3. Router OFNs are weighted and aggregated into a global OFN score.
4. Positive direction increases suspicion; negative direction reduces it.
5. Detector triggers alarm using threshold + hysteresis.
6. GA tunes weights and detector parameters on labeled scenarios.

## Current MVP

- Local `pyofn` package copied from the OFN prototype repository.
- Robust baseline normalization with median and MAD.
- OFN builder from 4 traffic measurements per router.
- Weighted OFN fusion across routers with signed contribution.
- Stateful detector with alert and clear hysteresis.
- Synthetic scenarios: `normal`, `ddos_ramp`, `ddos_pulse`, `flash_crowd`.
- Basic GA for tuning router weights and detector thresholds.

## Main Commands

```bash
pytest -q
python scripts/eval_ddos.py --scenario ddos_ramp
python scripts/train_ddos_ga.py
python scripts/run_stream_demo.py
python scripts/dashboard.py
```
