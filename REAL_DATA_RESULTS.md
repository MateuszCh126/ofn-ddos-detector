# Battle test on real traffic — CIC-DDoS2019

First evaluation of the detector on **real** network traffic instead of self-generated
synthetic scenarios.

## Setup

- **Dataset:** CIC-DDoS2019, file `01-12/UDPLag.csv` (raw CICFlowMeter flow records),
  public Hugging Face mirror `baalajimaestro/CICDDoS2019`.
- **Adapter:** [`ddos_ofn/cicddos2019.py`](ddos_ofn/cicddos2019.py) streams the raw flows
  and folds them into the existing long CSV format (`step,router_id,feature_name,value,label`).
- **Folding:** 1-second time bins, router = `Destination IP` (8 busiest kept),
  metric = `packet_count`, a step is labelled attack when ≥50% of its flows are non-`BENIGN`.
- **Result:** 370 605 flows → 1095 steps over 1544.8 s. **188 benign / 907 attack steps.**

Reproduce:

```bash
python scripts/prepare_cicddos2019.py \
  --input data/raw/UDPLag.csv \
  --output data/processed/cicddos_udplag.csv \
  --bin-seconds 1.0 --router-key dst_ip --num-routers 8 --metric packet_count

python scripts/benchmark_models.py --csv data/processed/cicddos_udplag.csv \
  --csv-format long --feature-column feature_name --value-column value
```

## Results — default thresholds (tuned on synthetic data)

| Model            | Recall | Precision | F1     | FPR    | Detection delay |
|------------------|--------|-----------|--------|--------|-----------------|
| **OFN**          | 0.0022 | 1.00      | 0.0044 | 0.00   | 556 steps       |
| volume_threshold | 0.082  | 0.52      | 0.141  | 0.36   | 63 steps        |
| ewma             | 0.0055 | 0.29      | 0.011  | 0.064  | 248 steps       |

## Results — OFN after GA tuning on the real train split (60/30)

| Split                 | Recall | Precision | F1   | Detection delay |
|-----------------------|--------|-----------|------|-----------------|
| validation (last 40%) | 0.00   | n/a       | 0.00 | 438 steps       |

GA reached `best_fitness=0.70` on the training segment but generalised to **recall 0** on
validation.

## Diagnosis — why it fails (this is the real finding)

1. **Rolling baseline normalises the attack away.** `robust_center_scale` recomputes a
   median/MAD baseline from the recent history window every step. CIC-DDoS2019 is a
   *sustained* flood: once the attack runs for longer than the history window, the inflated
   volume becomes the new "normal", the per-router trend flattens, the OFN direction
   collapses to neutral, and the score drops below threshold. The detector goes quiet in the
   middle of a live attack.
2. **The synthetic scenarios hid this.** `ddos_ramp`/`ddos_pulse` are *transient* — they
   spike and recover, so a deviation-from-baseline detector always has a benign reference to
   deviate from. Real floods don't recover on the detector's timescale.
3. **Class balance is inverted.** Real capture is 83% attack; the chronological validation
   split is essentially all-attack, so there is no benign reference left at all.
4. **Defaults were calibrated for synthetic magnitudes** — `max_score` on real data was 2.84
   (OFN) vs `alert_threshold=3.0`, i.e. the score barely crosses the line even at its peak.

## So what would actually fix it (future work)

- **Frozen / long-memory baseline:** learn the benign baseline once from a clean warm-up
  window and stop adapting during a suspected attack (freeze the baseline while alarmed).
- **Absolute-volume feature** alongside the relative trend, so a sustained high plateau stays
  suspicious even with zero trend.
- **Evaluate on a transient-attack file** (e.g. `Syn.csv`, or a mixed benign+attack capture)
  to separate "wrong detector" from "wrong scenario shape".
- **Per-flow / windowed F1 with PR-AUC**, not just thresholded alarm, for a fair comparison
  with the published CIC-DDoS2019 baselines.

## Honest takeaway

On self-generated transient scenarios the OFN detector looked strong. On real, sustained
CIC-DDoS2019 traffic, with both default and GA-tuned parameters, it does **not** detect the
attack — and neither do the simple baselines, which confirms the failure is structural
(rolling-baseline adaptation to sustained floods), not a tuning accident. The method needs a
non-adaptive baseline and an absolute-magnitude signal before any real-world claim holds.
