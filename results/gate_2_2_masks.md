# Gate 2.2: attempt to improve on dead detectors

Criterion fixed BEFORE running: a variant passes if AUROC on wrong readings is >= 0.7 on BOTH unseen masks (mask0p5, mask0p25_intersection) and >= 0.8 on every unseen noise (gauss3p5, gauss4, urand3p5). Trained on clean/gauss1/gauss2/mask0p25, evaluated on the episode tail.

| variant | gauss1 | gauss2 | gauss4 | urand3p5 | mask0p25 | mask0p5 | mask0p25_intersection | drift0p3 | 90% coverage (clean) | passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|
| window 4 (gate 2.2) | 0.797 | 0.866 | 0.906 | 0.872 | 0.700 | 0.630 | 0.603 | 0.534 | 85.3% | NO |
| window 12 | 0.822 | 0.898 | 0.931 | 0.908 | 0.738 | 0.689 | 0.602 | 0.536 | 68.8% | NO |
| window 12 + persistence | 0.856 | 0.903 | 0.922 | 0.913 | 0.788 | 0.739 | 0.651 | 0.536 | 66.7% | NO |

## Verdict

**Gate 2.2 CLOSED AS PARTIALLY PASSED.** No variant reaches 0.7 on both unseen masks. Best on masks: "window 12 + persistence" (min AUROC 0.651). The head estimates uncertainty for NOISE failures and works poorly on SILENT ones (dead detectors).
