# Failure severity converter (level in the specification -> deviation on the feature)

Network: `cityflow_jinan3x4_4ph`. State: ATS (EP block + demand block), window `obs_length`.
Column **factor** = `ep_std / level`: how much weaker or stronger noise injected AT THE
DETECTOR is on the feature the policy sees. Without it the RobustLight levels
(3.5-4.0 on the STATE) are not comparable with ours.

| scenario | level | unit | EP std | EP MAD | demand std | demand MAD | changed | factor |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| gauss | 1 | lane | 0.891 | 0.686 | 0.738 | 0.519 | 82.9% | 0.891 |
| gauss | 2 | lane | 1.702 | 1.315 | 1.387 | 0.982 | 82.9% | 0.851 |
| gauss | 3.5 | lane | 2.860 | 2.209 | 2.303 | 1.631 | 82.9% | 0.817 |
| gauss | 4 | lane | 3.236 | 2.498 | 2.601 | 1.842 | 82.9% | 0.809 |
| urand | 3.5 | lane | 1.742 | 1.397 | 1.384 | 1.076 | 82.2% | 0.498 |
| urand | 4 | lane | 1.972 | 1.582 | 1.561 | 1.214 | 82.2% | 0.493 |
| mask | 0.25 | lane | 2.450 | 0.620 | 0.702 | 0.132 | 12.9% | - |
| mask | 0.5 | lane | 3.350 | 1.257 | 1.017 | 0.286 | 24.6% | - |
| mask | 0.25 | intersection | 3.104 | 1.158 | 0.914 | 0.247 | 21.3% | - |
| drift | 0.2 | lane | 1.128 | 0.536 | 0.327 | 0.119 | 53.7% | - |
| drift | 0.3 | lane | 1.692 | 0.804 | 0.491 | 0.179 | 53.7% | - |

## How to read it

On a clean run **28% of EP block cells and 66% of demand block cells are exactly zero**. Two conclusions follow, both relevant for T2:

1. **A mask is largely a no-op.** Zero is the MODAL value, so a dead detector
   reporting 0 looks like a healthy detector on an empty approach; the *changed*
   column is far below the fraction of broken lanes. Not a measurement fault but a
   property of `zero` substitution, which makes the mask a SILENT failure (see
   `domain/sensors.py`: the substitution is not neutral).
2. **Mask severity depends on network LOAD.** The same mask percentage does not mean
   the same thing on JiNan and HangZhou (6295 vs 2983 veh/h); the thesis must give
   the fraction of changed cells next to the mask percentage, or the degradation
   curves of the two networks are not comparable.

For additive noise the factor has a predictable source: e(l,m) is a difference of
MEANS over inbound and outbound lanes, so with M lanes on each side the sigma on the
feature is sigma*sqrt(1/M_in + 1/M_out), 0.816 for M=3. Deviations come from
movements with another M and from clipping at zero (`max(0, ...)`), which eats more
mass at larger sigma, hence the factor DECREASES with the level.
`urand` at the same `level` is markedly WEAKER (width a gives sigma = a/sqrt(3)),
so comparing the two attacks "by the same number" is misleading.
