# Core ablations: GAT+PPO+CTDE

Generated: 2026-10-11 16:51. Realistic mode (yellow=5), ATT = mean of the last 10 tests, mean +- std over seeds. Each variant changes ONE thing against the GATE core; the critic stays a GAT in all of them (the ablation asks about the actor's decision). delta > 0 = worse than the core.

## cityflow_jinan3x4_4ph

### Neighbour aggregation in the actor

| variant | ATT | delta vs core | p (Welch) | n |
|---|---:|---:|---:|:-:|
| **GAT (core)** | 295.7 +- 1.0 | - | - | 5 |
| mean-pool (GraphSAGE) | 291.7 +- 0.3 | -1.33% | < 0.001 | 5 |
| no GNN (MLP) | 291.9 +- 0.5 | -1.26% | < 0.001 | 5 |

### State representation

| variant | ATT | delta vs core | p (Welch) | n |
|---|---:|---:|---:|:-:|
| **ATS (core)** | 295.7 +- 1.0 | - | - | 5 |
| EP | 314.2 +- 0.7 | +6.28% | < 0.001 | 3 |
| pressure | 320.4 +- 0.3 | +8.38% | < 0.001 | 3 |
| queue | 313.3 +- 1.4 | +5.95% | < 0.001 | 3 |

Reference (same protocol): advancedcolight: 293.5 +- 0.4, efficientmp: 316.1, advancedmp: 299.2.

## cityflow_hangzhou4x4_4ph

### Neighbour aggregation in the actor

| variant | ATT | delta vs core | p (Welch) | n |
|---|---:|---:|---:|:-:|
| **GAT (core)** | 317.2 +- 0.2 | - | - | 5 |
| mean-pool (GraphSAGE) | 316.5 +- 0.2 | -0.24% | < 0.001 | 5 |
| no GNN (MLP) | 316.3 +- 0.2 | -0.29% | < 0.001 | 5 |

### State representation

| variant | ATT | delta vs core | p (Welch) | n |
|---|---:|---:|---:|:-:|
| **ATS (core)** | 317.2 +- 0.2 | - | - | 5 |
| EP | 329.2 +- 0.2 | +3.77% | < 0.001 | 3 |
| pressure | 341.1 +- 19.7 | +7.55% | 0.170 | 3 |
| queue | 328.9 +- 0.1 | +3.70% | < 0.001 | 3 |

Reference (same protocol): advancedcolight: 317.1 +- 0.2, efficientmp: 330.1, advancedmp: 325.4.

## How to read

* A Welch p at n=3 is an orientation, not proof; the same convention as in the GATE.
* Efficient-MP and Advanced-MP alongside answer "why a GNN if MaxPressure is enough": the variant without a GNN should land between them and the core.
