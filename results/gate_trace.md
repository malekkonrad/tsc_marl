# Gate trace: plan `t2_trace`

An explanatory figure, not a measurement: **one seed, one failure draw**. Numbers for the thesis come from `t2_gate_real` (3 seeds x 5 draws).

Fill = the node's control mode after hysteresis. Marker = the detector fired, but hysteresis held the switch (entry needs `k_on` = 3 consecutive decisions, exit `k_off` = 6, hence the fill is WIDER than the run of markers).

## What the figure showed that no table did

**The rule's false alarms on clean traffic are a COLD START artefact, not a steady state.** The whole 1.5% share falls in decisions 13-24, right after the 12-reading buffer fills, and does not return until the end of the episode. The mechanism is clear: at the start the network is empty, and a lane without traffic reports a constant zero, exactly what a dead detector does (`max - min < 1e-6`). The warm-up guard ends when the buffer is FULL, not when it is full of MEANINGFUL readings.

Three conclusions: (1) in the steady state the rule does not fire on clean at all, i.e. it is BETTER than `gate_share` says; (2) this explains why the clean share was 1.5% against a calibration budget of 0.5%: the difference is not a budget overrun but another phenomenon; (3) the fix is cheap and worth measuring: lengthen the warm-up or require a suspicious movement to have ever had a NONZERO reading. Unmeasured, to be recorded as open.

For comparison, under failure the cold start accounts for 13-14% of cases (`mask0p5`, `mask0p25_intersection`) and 44% under `hold`; there the rest spreads over the whole episode, since the failure persists.

## cityflow_jinan3x4_4ph

![gate trace cityflow_jinan3x4_4ph](figures/gate_trace_cityflow_jinan3x4_4ph.png)

| scenario | share of pairs in rule mode | nodes that took over at least once |
|---|---:|---:|
| `clean` | 1.5% | 5/12 |
| `gauss2` | 0.0% | 0/12 |
| `mask0p25_intersection` | 27.2% | 10/12 |
| `mask0p5` | 43.5% | 11/12 |
| `mask0p25_hold_t60` | 3.3% | 6/12 |
