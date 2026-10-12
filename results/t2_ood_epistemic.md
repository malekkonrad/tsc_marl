# Epistemic branch test: plan `t2_ood`

Frozen controllers on traffic outside training. Learned policies take weights from the TRAINING network (`{base}` in the parent template), traffic comes from another file. The clean scenario = clean readings; `gauss2` = the same traffic with detector noise, to compare the ensemble's reaction to new traffic with its reaction to sensor failure.

## Policies trained on `cityflow_jinan3x4_4ph`

### ATT on clean readings

| traffic | fixedtime | efficientmp | advancedmp | advancedcolight | gatppo | core vs better rule |
|---|---:|---:|---:|---:|---:|---:|
| training traffic | 507.0 | 313.9 | 301.8 | 294.4 +- 1.8 | 295.8 +- 1.2 | -2.0% (p 1.00) |
| demand x0.25 | 322.3 | 258.8 | 256.8 | 252.9 +- 1.1 | 253.2 +- 1.7 | -1.4% (p 1.00) |
| another traffic file (2000) | 412.1 | 288.4 | 288.1 | 283.1 +- 0.7 | 283.6 +- 0.6 | -1.5% (p 1.00) |

(!) = the core SIGNIFICANTLY worse than the better MaxPressure (one-sided t, p < 0.05, policy seeds vs the deterministic rule); only there would branch (a) have work.

### Disagreement of 5 actors (epistemic in the policy)

| traffic | MI clean | MI noise sigma=2 | disagreeing votes clean | disagreeing votes noise |
|---|---:|---:|---:|---:|
| training traffic | 0.2283 | 0.2872 | 26.5% | 27.8% |
| demand x0.25 | 0.1689 | 0.2781 | 34.4% | 29.0% |
| another traffic file (2000) | 0.1892 | 0.2843 | 24.9% | 28.7% |

## Policies trained on `cityflow_hangzhou4x4_4ph`

### ATT on clean readings

| traffic | fixedtime | efficientmp | advancedmp | advancedcolight | gatppo | core vs better rule |
|---|---:|---:|---:|---:|---:|---:|
| training traffic | 568.9 | 331.2 | 325.5 | 317.4 +- 0.5 | 317.5 +- 0.7 | -2.5% (p 1.00) |
| another traffic file (5816) | 559.5 | 428.0 | 419.0 | 437.7 +- 15.1 | 420.1 +- 1.5 | +0.3% (p 0.10) |

(!) = the core SIGNIFICANTLY worse than the better MaxPressure (one-sided t, p < 0.05, policy seeds vs the deterministic rule); only there would branch (a) have work.

### Disagreement of 5 actors (epistemic in the policy)

| traffic | MI clean | MI noise sigma=2 | disagreeing votes clean | disagreeing votes noise |
|---|---:|---:|---:|---:|
| training traffic | 0.1082 | 0.2125 | 16.6% | 21.2% |
| another traffic file (5816) | 0.1858 | 0.2327 | 22.4% | 22.8% |

## Verdict

1. **The core does not lose significantly to MaxPressure on any traffic outside training.** A tie within the seed spread: "another traffic file (5816)" +0.3% (p 0.10). Switching to MaxPressure has nothing to gain here, so branch (a) of the two-level fallback has no work: a single-level fallback to FixedTime.

Disagreement of 5 actors (MI) on CLEAN input, relative to training traffic:

* `demand x0.25`: MI -0.0594 (-26%); for comparison, noise sigma=2 on training traffic +0.0588 -> does not rise.
* `another traffic file (2000)`: MI -0.0392 (-17%); for comparison, noise sigma=2 on training traffic +0.0588 -> does not rise.
* `another traffic file (5816)`: MI +0.0776 (+72%); for comparison, noise sigma=2 on training traffic +0.1043 -> **rises**: a new-traffic signal exists (weaker than under noise: indistinguishable without the observation channel).

2. **Branch (a) holds only where BOTH conditions hold on the same traffic:** a significant loss to MaxPressure and an MI rise on clean input. They hold nowhere; MI rises on "another traffic file (5816)", but there the policy does not lose, so there is nothing to rescue.
