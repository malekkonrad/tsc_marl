# Ablacje rdzenia (1.6) — GAT+PPO+CTDE

Wygenerowane: 2026-09-12 10:33. Tryb realistyczny (yellow=5), ATT = średnia z 10 ostatnich testów, mean ± std po seedach. Każdy wariant zmienia JEDNĄ rzecz względem rdzenia z GATE; krytyk zostaje GAT-em we wszystkich (ablacja pyta o decyzję actora). Δ > 0 = gorzej niż rdzeń.

## cityflow_jinan3x4_4ph

### Agregacja sąsiadów w actorze

| wariant | ATT | Δ vs rdzeń | p (Welch) | n |
|---|---:|---:|---:|:-:|
| **GAT (rdzeń)** | 295.6 ± 0.8 | — | — | 3 |
| mean-pool (GraphSAGE) | 291.7 ± 0.2 | -1.32% | 0.008 | 3 |
| bez GNN (MLP) | 291.8 ± 0.5 | -1.29% | 0.003 | 3 |

### Reprezentacja stanu

| wariant | ATT | Δ vs rdzeń | p (Welch) | n |
|---|---:|---:|---:|:-:|
| **ATS (rdzeń)** | 295.6 ± 0.8 | — | — | 3 |
| EP | 314.2 ± 0.7 | +6.30% | < 0.001 | 3 |
| pressure | 320.4 ± 0.3 | +8.41% | < 0.001 | 3 |
| queue | 313.3 ± 1.4 | +5.98% | < 0.001 | 3 |

Odniesienie (ten sam protokół): advancedcolight: 293.5 ± 0.6 · efficientmp: 316.1 · advancedmp: 299.2.

## cityflow_hangzhou4x4_4ph

### Agregacja sąsiadów w actorze

| wariant | ATT | Δ vs rdzeń | p (Welch) | n |
|---|---:|---:|---:|:-:|
| **GAT (rdzeń)** | 317.2 ± 0.3 | — | — | 3 |
| mean-pool (GraphSAGE) | 316.4 ± 0.2 | -0.25% | 0.024 | 3 |
| bez GNN (MLP) | 316.3 ± 0.2 | -0.30% | 0.015 | 3 |

### Reprezentacja stanu

| wariant | ATT | Δ vs rdzeń | p (Welch) | n |
|---|---:|---:|---:|:-:|
| **ATS (rdzeń)** | 317.2 ± 0.3 | — | — | 3 |
| EP | 329.2 ± 0.2 | +3.76% | < 0.001 | 3 |
| pressure | 341.1 ± 19.7 | +7.54% | 0.170 | 3 |
| queue | 328.9 ± 0.1 | +3.68% | < 0.001 | 3 |

Odniesienie (ten sam protokół): advancedcolight: 316.9 ± 0.1 · efficientmp: 330.1 · advancedmp: 325.4.

## Jak czytać

* p z testu Welcha przy n=3 to orientacja, nie dowód — ta sama konwencja co w GATE.
* Efficient-MP i Advanced-MP obok: odpowiedź na „po co GNN, skoro MaxPressure wystarcza” — wariant bez GNN powinien lądować między nimi a rdzeniem.
