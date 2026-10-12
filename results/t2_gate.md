# T2 with the fallback gate: plan `t2_gate`

Mode realistic (yellow=5), 5 failure draws per scenario, evaluation of FROZEN controllers (`tsc degrade`). All read the same detector layer: MaxPressure computes its decision from the broken state vector, not the engine.

Cell = ATT [s], the mean over draws, then over policy seeds (+- std over seeds); in brackets delta relative to the same controller's clean result. **(!) = worse than FixedTime**, the only place the fallback's lower layer can gain.

## cityflow_jinan3x4_4ph

FixedTime (independent of readings): **507.0 s**.

| scenario | core GAT+PPO+CTDE (n=3) | gate, threshold 0.25 (n=3) | gate, threshold 0.50 (n=3) | gate, threshold 0.75 (n=3) | gate, threshold 1.00 (n=3) |
|---|---:|---:|---:|---:|---:|
| `clean` | 295.9 +- 0.6 | 295.9 +- 0.6 | 295.9 +- 0.6 | 295.9 +- 0.6 | 295.9 +- 0.6 |
| `mask0p1` | 354.2 +- 20.0 (+19.7%) | 365.4 +- 2.0 (+23.5%) | 360.5 +- 18.6 (+21.8%) | 354.2 +- 20.0 (+19.7%) | 354.2 +- 20.0 (+19.7%) |
| `mask0p25` | 424.9 +- 31.1 (+43.6%) | 436.1 +- 2.5 (+47.4%) | 429.1 +- 20.7 (+45.0%) | 424.9 +- 31.1 (+43.6%) | 424.9 +- 31.1 (+43.6%) |
| `mask0p25_intersection` | 471.1 +- 68.8 (+59.2%) | 350.7 +- 1.4 (+18.5%) | 350.7 +- 1.4 (+18.5%) | 350.7 +- 1.4 (+18.5%) | 350.7 +- 1.4 (+18.5%) |
| `mask0p5` | 732.6 +- 28.5 (+147.5%) (!) | 493.8 +- 0.9 (+66.8%) | 491.1 +- 11.7 (+65.9%) | 673.0 +- 30.8 (+127.4%) (!) | 732.6 +- 28.5 (+147.5%) (!) |
| `mask0p75` | 904.0 +- 67.5 (+205.5%) (!) | 501.5 +- 0.2 (+69.5%) | 501.4 +- 0.4 (+69.4%) | 589.3 +- 22.1 (+99.1%) (!) | 880.0 +- 60.2 (+197.4%) (!) |

* **core GAT+PPO+CTDE**: worst scenario `mask0p75`: 904.0 s (+205.5%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p75`.
* **gate, threshold 0.25**: worst scenario `mask0p75`: 501.5 s (+69.5%). Never falls below FixedTime: at these levels a fallback to FixedTime has nothing to gain.
* **gate, threshold 0.50**: worst scenario `mask0p75`: 501.4 s (+69.4%). Never falls below FixedTime: at these levels a fallback to FixedTime has nothing to gain.
* **gate, threshold 0.75**: worst scenario `mask0p5`: 673.0 s (+127.4%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p75`.
* **gate, threshold 1.00**: worst scenario `mask0p75`: 880.0 s (+197.4%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p75`.


![degradation curves cityflow_jinan3x4_4ph](figures/t2_gate_cityflow_jinan3x4_4ph.png)

## Mechanisms x scenario

delta ATT [s] of the ASSESSED COLUMN relative to `core GAT+PPO+CTDE`, paired by (seed, draw). In brackets the share of (decision, node) pairs taken over by the rule. **bold** = |t| >= 2.8. delta < 0 = the assessed column is better than the reference.


### cityflow_jinan3x4_4ph

| mechanism | `clean` | `mask0p1` | `mask0p25` | `mask0p25_intersection` | `mask0p5` | `mask0p75` |
|---|---:|---:|---:|---:|---:|---:|
| gate, threshold 0.25 | +0.0 (0%) | +11.2 (15%) | +11.2 (63%) | **-120.4** (27%) | **-238.8** (93%) | **-402.5** (98%) |
| gate, threshold 0.50 | +0.0 (0%) | +6.3 (2%) | +4.1 (7%) | **-120.4** (27%) | **-241.5** (62%) | **-402.6** (97%) |
| gate, threshold 0.75 | +0.0 (0%) | +0.0 (0%) | +0.0 (0%) | **-120.4** (27%) | **-59.5** (10%) | **-314.7** (65%) |
| gate, threshold 1.00 | +0.0 (0%) | +0.0 (0%) | +0.0 (0%) | **-120.4** (27%) | +0.0 (0%) | -24.0 (5%) |

## Check: the evaluator measures the same policy as T1

The evaluator's clean result vs the last training test, 3 learned runs: largest difference **0.000 s**, consistent to the log's precision.

## What this table does NOT say

* Levels are DETECTOR levels. For `gauss` they transfer to the feature with a factor of ~0.8-0.9, `urand` is ~1.6x weaker at the same `level` (`results/sensor_severity.md`).
* A mask with `zero` substitution changes far fewer feature cells than the fraction of dead lanes (zero is the modal value); the same number does not mean the same thing on networks with different loads.
* `hold` scenarios start at decision 60 (minute 15), so the first quarter of the episode is clean; their delta is not directly comparable with a mask from the start.
