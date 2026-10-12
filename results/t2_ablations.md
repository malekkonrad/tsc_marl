# T2 for the aggregation ablations: plan `t2_ablations`

Mode realistic (yellow=5), 5 failure draws per scenario, evaluation of FROZEN controllers (`tsc degrade`). All read the same detector layer: MaxPressure computes its decision from the broken state vector, not the engine.

Cell = ATT [s], the mean over draws, then over policy seeds (+- std over seeds); in brackets delta relative to the same controller's clean result. **(!) = worse than FixedTime**, the only place the fallback's lower layer can gain.

## cityflow_jinan3x4_4ph

FixedTime (independent of readings): **507.0 s**.

| scenario | core GAT+PPO+CTDE (n=5) | actor mean-pool (n=5) | actor without GNN (n=5) |
|---|---:|---:|---:|
| `clean` | 295.8 +- 1.2 | 291.3 +- 0.5 | 291.7 +- 0.8 |
| `gauss1` | 304.1 +- 1.2 (+2.8%) | 301.1 +- 0.6 (+3.4%) | 301.9 +- 1.0 (+3.5%) |
| `gauss2` | 319.0 +- 2.0 (+7.9%) | 318.7 +- 1.8 (+9.4%) | 321.1 +- 3.0 (+10.1%) |
| `gauss4` | 347.4 +- 6.7 (+17.5%) | 352.1 +- 4.5 (+20.9%) | 361.9 +- 11.4 (+24.1%) |
| `mask0p1` | 354.3 +- 16.1 (+19.8%) | 367.3 +- 7.6 (+26.1%) | 366.5 +- 5.5 (+25.6%) |
| `mask0p25` | 426.5 +- 23.9 (+44.2%) | 460.3 +- 12.0 (+58.0%) | 461.8 +- 9.6 (+58.3%) |
| `mask0p5` | 743.4 +- 25.2 (+151.4%) (!) | 811.9 +- 95.3 (+178.7%) (!) | 762.2 +- 103.4 (+161.3%) (!) |
| `mask0p25_intersection` | 510.0 +- 90.2 (+72.4%) (!) | 639.4 +- 199.4 (+119.5%) (!) | 570.4 +- 188.8 (+95.6%) (!) |
| `mask0p5_intersection` | 746.5 +- 132.4 (+152.4%) (!) | 896.0 +- 306.7 (+207.6%) (!) | 776.8 +- 278.0 (+166.3%) (!) |
| `mask0p25_hold_t60` | 344.8 +- 22.2 (+16.6%) | 327.0 +- 9.7 (+12.3%) | 345.7 +- 16.9 (+18.5%) |
| `mask0p5_hold_t60` | 466.9 +- 36.1 (+57.9%) | 480.6 +- 22.0 (+65.0%) | 509.4 +- 47.4 (+74.6%) (!) |
| `drift0p3` | 301.7 +- 0.7 (+2.0%) | 297.9 +- 0.5 (+2.3%) | 298.7 +- 0.6 (+2.4%) |
| `drift0p5` | 331.0 +- 4.6 (+11.9%) | 328.2 +- 2.9 (+12.7%) | 327.9 +- 5.3 (+12.4%) |

* **core GAT+PPO+CTDE**: worst scenario `mask0p5_intersection`: 746.5 s (+152.4%). Worse than FixedTime in 3 scenarios: `mask0p25_intersection`, `mask0p5`, `mask0p5_intersection`.
* **actor mean-pool**: worst scenario `mask0p5_intersection`: 896.0 s (+207.6%). Worse than FixedTime in 3 scenarios: `mask0p25_intersection`, `mask0p5`, `mask0p5_intersection`.
* **actor without GNN**: worst scenario `mask0p5_intersection`: 776.8 s (+166.3%). Worse than FixedTime in 4 scenarios: `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.


![degradation curves cityflow_jinan3x4_4ph](figures/t2_ablations_cityflow_jinan3x4_4ph.png)

## cityflow_hangzhou4x4_4ph

FixedTime (independent of readings): **568.9 s**.

| scenario | core GAT+PPO+CTDE (n=5) | actor mean-pool (n=5) | actor without GNN (n=5) |
|---|---:|---:|---:|
| `clean` | 317.5 +- 0.7 | 316.7 +- 0.3 | 316.2 +- 0.3 |
| `gauss1` | 345.7 +- 1.3 (+8.9%) | 343.6 +- 2.6 (+8.5%) | 345.5 +- 2.7 (+9.3%) |
| `gauss2` | 374.7 +- 4.8 (+18.0%) | 370.0 +- 5.6 (+16.8%) | 377.3 +- 7.5 (+19.3%) |
| `gauss4` | 425.7 +- 13.5 (+34.1%) | 417.5 +- 15.2 (+31.8%) | 437.7 +- 24.4 (+38.4%) |
| `mask0p1` | 361.9 +- 3.6 (+14.0%) | 376.3 +- 13.3 (+18.8%) | 370.1 +- 11.5 (+17.0%) |
| `mask0p25` | 450.1 +- 24.7 (+41.7%) | 506.9 +- 46.8 (+60.0%) | 522.2 +- 41.0 (+65.1%) |
| `mask0p5` | 686.6 +- 102.7 (+116.2%) (!) | 839.9 +- 150.3 (+165.2%) (!) | 868.4 +- 107.1 (+174.7%) (!) |
| `mask0p25_intersection` | 442.1 +- 59.6 (+39.2%) | 553.4 +- 127.3 (+74.7%) | 636.3 +- 50.8 (+101.2%) (!) |
| `mask0p5_intersection` | 682.3 +- 175.7 (+114.9%) (!) | 889.2 +- 263.7 (+180.7%) (!) | 1037.8 +- 75.0 (+228.2%) (!) |
| `mask0p25_hold_t60` | 412.2 +- 13.8 (+29.8%) | 400.3 +- 10.5 (+26.4%) | 441.2 +- 9.9 (+39.5%) |
| `mask0p5_hold_t60` | 548.4 +- 14.3 (+72.7%) | 613.3 +- 17.0 (+93.6%) (!) | 602.3 +- 36.9 (+90.5%) (!) |
| `drift0p3` | 319.5 +- 0.2 (+0.6%) | 318.6 +- 0.3 (+0.6%) | 318.5 +- 0.2 (+0.7%) |
| `drift0p5` | 338.7 +- 2.5 (+6.7%) | 340.8 +- 4.5 (+7.6%) | 337.3 +- 3.1 (+6.7%) |

* **core GAT+PPO+CTDE**: worst scenario `mask0p5`: 686.6 s (+116.2%). Worse than FixedTime in 2 scenarios: `mask0p5`, `mask0p5_intersection`.
* **actor mean-pool**: worst scenario `mask0p5_intersection`: 889.2 s (+180.7%). Worse than FixedTime in 3 scenarios: `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.
* **actor without GNN**: worst scenario `mask0p5_intersection`: 1037.8 s (+228.2%). Worse than FixedTime in 4 scenarios: `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`.


![degradation curves cityflow_hangzhou4x4_4ph](figures/t2_ablations_cityflow_hangzhou4x4_4ph.png)

## Mechanisms x scenario

delta ATT [s] of the ASSESSED COLUMN relative to `core GAT+PPO+CTDE`, paired by (seed, draw). **bold** = |t| >= 2.8. delta < 0 = the assessed column is better than the reference.


### cityflow_jinan3x4_4ph

| mechanism | `clean` | `gauss1` | `gauss2` | `gauss4` | `mask0p1` | `mask0p25` | `mask0p5` | `mask0p25_intersection` | `mask0p5_intersection` | `mask0p25_hold_t60` | `mask0p5_hold_t60` | `drift0p3` | `drift0p5` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| actor mean-pool | **-4.5** | **-2.9** | -0.3 | +4.7 | +13.0 | **+33.8** | **+68.4** | +129.4 | +149.5 | -17.8 | +13.8 | **-3.8** | -2.8 |
| actor without GNN | **-4.1** | **-2.2** | +2.1 | **+14.5** | +12.2 | **+35.3** | +18.7 | +60.4 | +30.3 | +0.9 | +42.5 | **-3.0** | -3.1 |

### cityflow_hangzhou4x4_4ph

| mechanism | `clean` | `gauss1` | `gauss2` | `gauss4` | `mask0p1` | `mask0p25` | `mask0p5` | `mask0p25_intersection` | `mask0p5_intersection` | `mask0p25_hold_t60` | `mask0p5_hold_t60` | `drift0p3` | `drift0p5` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| actor mean-pool | -0.8 | **-2.1** | **-4.7** | **-8.2** | **+14.5** | **+56.8** | **+153.3** | **+111.4** | **+206.9** | -11.9 | **+65.0** | **-0.8** | +2.1 |
| actor without GNN | **-1.3** | -0.2 | +2.6 | +12.0 | +8.2 | **+72.1** | **+181.8** | **+194.2** | **+355.5** | **+29.0** | **+54.0** | **-1.0** | -1.4 |

## Check: the evaluator measures the same policy as T1

The evaluator's clean result vs the last training test, 30 learned runs: largest difference **0.000 s**, consistent to the log's precision.

## What this table does NOT say

* Levels are DETECTOR levels. For `gauss` they transfer to the feature with a factor of ~0.8-0.9, `urand` is ~1.6x weaker at the same `level` (`results/sensor_severity.md`).
* A mask with `zero` substitution changes far fewer feature cells than the fraction of dead lanes (zero is the modal value); the same number does not mean the same thing on networks with different loads.
* `hold` scenarios start at decision 60 (minute 15), so the first quarter of the episode is clean; their delta is not directly comparable with a mask from the start.
