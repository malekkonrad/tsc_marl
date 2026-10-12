# T2: plan `t2_krakow_best`

Mode realistic (yellow=5), 3 failure draws per scenario, evaluation of FROZEN controllers (`tsc degrade`). All read the same detector layer: MaxPressure computes its decision from the broken state vector, not the engine.

Cell = ATT [s], the mean over draws, then over policy seeds (+- std over seeds); in brackets delta relative to the same controller's clean result. **(!) = worse than FixedTime**, the only place the fallback's lower layer can gain.

## sumo_krakow3w_mid_s1

FixedTime (independent of readings): **336.6 s**.

| scenario | core (final) (n=3) | core (best) (n=3) | DR (final) (n=3) | DR (best) (n=3) | DR (best) + rule gate (n=3) | Advanced-MP (n=1) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 339.4 +- 4.4 (!) | 340.5 +- 0.9 (!) | 350.2 +- 4.0 (!) | 352.6 +- 10.6 (!) | 359.8 +- 18.8 (!) | 347.0 (!) |
| `mask0p5` | 554.6 +- 72.6 (+63.4%) (!) | 633.8 +- 127.6 (+86.1%) (!) | 418.6 +- 95.6 (+19.5%) (!) | 396.8 +- 27.4 (+12.6%) (!) | 379.2 +- 23.7 (+5.4%) (!) | 726.3 (+109.3%) (!) |
| `mask0p5_intersection` | 912.7 +- 239.2 (+168.9%) (!) | 737.8 +- 137.5 (+116.7%) (!) | 419.6 +- 34.5 (+19.8%) (!) | 453.4 +- 30.3 (+28.6%) (!) | 353.2 +- 5.2 (-1.9%) (!) | 1167.3 (+236.4%) (!) |

* **core (final)**: worst scenario `mask0p5_intersection`: 912.7 s (+168.9%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **core (best)**: worst scenario `mask0p5_intersection`: 737.8 s (+116.7%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **DR (final)**: worst scenario `mask0p5_intersection`: 419.6 s (+19.8%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **DR (best)**: worst scenario `mask0p5_intersection`: 453.4 s (+28.6%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **DR (best) + rule gate**: worst scenario `mask0p5`: 379.2 s (+5.4%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **Advanced-MP**: worst scenario `mask0p5_intersection`: 1167.3 s (+236.4%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.


![degradation curves sumo_krakow3w_mid_s1](figures/t2_krakow_best_sumo_krakow3w_mid_s1.png)

## sumo_krakow3w_mid

FixedTime (independent of readings): **347.6 s**.

| scenario | core (final) (n=3) | core (best) (n=3) | DR (final) (n=3) | DR (best) (n=3) | DR (best) + rule gate (n=3) | Advanced-MP (n=1) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 376.3 +- 46.1 (!) | 326.8 +- 9.4 | 358.7 +- 17.2 (!) | 323.9 +- 5.6 | 341.9 +- 23.7 | 396.9 (!) |
| `mask0p5` | 529.0 +- 85.4 (+40.6%) (!) | 561.8 +- 32.7 (+71.9%) (!) | 410.8 +- 66.0 (+14.5%) (!) | 452.2 +- 55.6 (+39.6%) (!) | 399.0 +- 15.8 (+16.7%) (!) | 736.5 (+85.6%) (!) |
| `mask0p5_intersection` | 911.4 +- 250.9 (+142.2%) (!) | 771.9 +- 143.4 (+136.2%) (!) | 416.7 +- 16.0 (+16.2%) (!) | 453.3 +- 24.7 (+39.9%) (!) | 365.5 +- 3.7 (+6.9%) (!) | 1164.6 (+193.4%) (!) |

* **core (final)**: worst scenario `mask0p5_intersection`: 911.4 s (+142.2%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **core (best)**: worst scenario `mask0p5_intersection`: 771.9 s (+136.2%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **DR (final)**: worst scenario `mask0p5_intersection`: 416.7 s (+16.2%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **DR (best)**: worst scenario `mask0p5_intersection`: 453.3 s (+39.9%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **DR (best) + rule gate**: worst scenario `mask0p5`: 399.0 s (+16.7%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **Advanced-MP**: worst scenario `mask0p5_intersection`: 1164.6 s (+193.4%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.


![degradation curves sumo_krakow3w_mid](figures/t2_krakow_best_sumo_krakow3w_mid.png)

## Arm: core (best) vs core (final)

The columns have DIFFERENT checkpoints (separate trainings), so pairing is by (network, scenario, seed, draw): traffic and failure pattern stay shared, the spread between seeds enters the difference. The reference column is set in the plan (`baseline: gatppo`), not assumed. `delta` < 0 = the arm is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

The columns differ in HOW THEY WERE TRAINED, not in a mechanism switched on at inference, so there is no "did the mechanism start" diagnostic. The spread between seeds enters the difference in full, and the number of INDEPENDENT replications is the number of seeds, not of pairs: failure draws share a checkpoint, so |t| is optimistic with respect to them.

### sumo_krakow3w_mid_s1

| scenario | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|
| `clean` | +1.10 | 3.6 | +0.53 | 1/3 |
| `mask0p5` | +79.21 | 240.4 | +0.99 | 7/9 |
| `mask0p5_intersection` | -174.90 | 298.3 | -1.76 | 6/9 |

* No scenario crosses the threshold: the arm is **neutral** for ATT.

### sumo_krakow3w_mid

| scenario | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|
| `clean` | -49.56 | 51.3 | -1.67 | 3/3 |
| `mask0p5` | +32.83 | 133.2 | +0.74 | 5/9 |
| `mask0p5_intersection` | -139.50 | 309.9 | -1.35 | 6/9 |

* No scenario crosses the threshold: the arm is **neutral** for ATT.

## Arm: DR (best) vs DR (final)

The columns have DIFFERENT checkpoints (separate trainings), so pairing is by (network, scenario, seed, draw): traffic and failure pattern stay shared, the spread between seeds enters the difference. The reference column is set in the plan (`baseline: gatppo_fail`), not assumed. `delta` < 0 = the arm is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

The columns differ in HOW THEY WERE TRAINED, not in a mechanism switched on at inference, so there is no "did the mechanism start" diagnostic. The spread between seeds enters the difference in full, and the number of INDEPENDENT replications is the number of seeds, not of pairs: failure draws share a checkpoint, so |t| is optimistic with respect to them.

### sumo_krakow3w_mid_s1

| scenario | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|
| `clean` | +2.40 | 12.4 | +0.33 | 2/3 |
| `mask0p5` | -21.75 | 163.9 | -0.40 | 3/9 |
| `mask0p5_intersection` | +33.78 | 70.3 | +1.44 | 3/9 |

* No scenario crosses the threshold: the arm is **neutral** for ATT.

### sumo_krakow3w_mid

| scenario | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|
| `clean` | **-34.74** | 20.6 | -2.92 | 3/3 |
| `mask0p5` | +41.46 | 176.0 | +0.71 | 4/9 |
| `mask0p5_intersection` | +36.60 | 65.1 | +1.69 | 3/9 |

* Above the threshold: `clean` (-34.74 s, t = -2.92, better in 3/3 pairs)

## Mechanism: DR (best) + rule gate vs DR (best)

Both columns are computed from THE SAME checkpoint and the same failure draws (`draw` chooses which detectors failed), so they are compared **in pairs**, not through the +-std of the table above: the shared spread between seeds and draws cancels in the difference. `delta` < 0 = the mechanism is better. `|t| >= 2.8` is the threshold corrected for as many comparisons as this table has.

`gate_share` = share of (decision, node) pairs where the rule took control. `gate_switches` = number of mode CHANGES at nodes per episode; these hurt traffic (a switch resets the phase), not the share itself. `gate_precision` and `gate_recall` are relative to the ORACLE (the truth from the failure layer, diagnostic only) and separate "the detector is weak" from "switching gains little", two things ATT alone merges.

### sumo_krakow3w_mid_s1

| scenario | gate_share | gate_switches | gate_precision | gate_recall | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 1.2% | 13.33 | 0.00 | - | +7.29 | 8.5 | +1.49 | 1/3 |
| `mask0p5` | 21.9% | 47.56 | 0.91 | 0.39 | -17.57 | 39.4 | -1.34 | 6/9 |
| `mask0p5_intersection` | 39.9% | 46.67 | 0.96 | 0.82 | **-100.21** | 61.8 | -4.87 | 9/9 |

* Above the threshold: `mask0p5_intersection` (-100.21 s, t = -4.87, better in 9/9 pairs)

### sumo_krakow3w_mid

| scenario | gate_share | gate_switches | gate_precision | gate_recall | delta ATT [s] | sd | t | better in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `clean` | 0.1% | 2.00 | 0.00 | - | +17.99 | 18.5 | +1.68 | 0/2 |
| `mask0p5` | 15.9% | 43.56 | 0.94 | 0.29 | -53.29 | 116.9 | -1.37 | 5/9 |
| `mask0p5_intersection` | 37.4% | 39.89 | 1.00 | 0.80 | **-87.77** | 69.9 | -3.77 | 9/9 |

* Above the threshold: `mask0p5_intersection` (-87.77 s, t = -3.77, better in 9/9 pairs)

## Check: the evaluator measures the same policy as T1

The evaluator's clean result vs the last training test, 6 learned runs: largest difference **0.000 s**, consistent to the log's precision.

## What this table does NOT say

* Levels are DETECTOR levels. For `gauss` they transfer to the feature with a factor of ~0.8-0.9, `urand` is ~1.6x weaker at the same `level` (`results/sensor_severity.md`).
* A mask with `zero` substitution changes far fewer feature cells than the fraction of dead lanes (zero is the modal value); the same number does not mean the same thing on networks with different loads.
* `hold` scenarios start at decision 60 (minute 15), so the first quarter of the episode is clean; their delta is not directly comparable with a mask from the start.
