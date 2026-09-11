# T2 bez mechanizmu — plan `t2_nomech`

Tryb realistic (yellow=5), 5 losowań awarii na scenariusz, ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.

Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. **▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.

## cityflow_jinan3x4_4ph

FixedTime (niezależny od odczytów): **507.0 s**.

| scenariusz | Efficient-MP (n=1) | Advanced-MP (n=1) |
|---|---:|---:|
| `clean` | 313.9 | 301.8 |
| `gauss0p5` | 318.3 (+1.4%) | 303.8 (+0.7%) |
| `gauss1` | 321.5 (+2.4%) | 308.0 (+2.1%) |
| `gauss2` | 329.0 (+4.8%) | 318.6 (+5.6%) |
| `gauss3p5` | 342.1 (+9.0%) | 334.4 (+10.8%) |
| `gauss4` | 344.6 (+9.8%) | 338.0 (+12.0%) |
| `urand2` | 322.5 (+2.7%) | 309.9 (+2.7%) |
| `urand4` | 332.3 (+5.9%) | 324.1 (+7.4%) |
| `mask0p1` | 373.4 (+19.0%) | 362.2 (+20.0%) |
| `mask0p25` | 432.6 (+37.8%) | 459.3 (+52.2%) |
| `mask0p5` | 647.6 (+106.3%) ▲ | 750.0 (+148.5%) ▲ |
| `mask0p75` | 943.8 (+200.7%) ▲ | 956.3 (+216.9%) ▲ |
| `mask0p25_intersection` | 601.5 (+91.6%) ▲ | 663.7 (+119.9%) ▲ |
| `mask0p5_intersection` | 859.9 (+174.0%) ▲ | 864.7 (+186.5%) ▲ |
| `mask0p25_hold_t60` | 348.7 (+11.1%) | 331.7 (+9.9%) |
| `mask0p5_hold_t60` | 584.0 (+86.1%) ▲ | 549.1 (+82.0%) ▲ |
| `drift0p1` | 316.8 (+0.9%) | 301.0 (-0.3%) |
| `drift0p2` | 317.9 (+1.3%) | 302.3 (+0.2%) |
| `drift0p3` | 319.9 (+1.9%) | 305.8 (+1.3%) |
| `drift0p5` | 333.9 (+6.4%) | 325.5 (+7.9%) |

_Niepoliczone na tej maszynie: rdzeń GAT+PPO+CTDE, Advanced-CoLight (brak checkpointów rodzica — patrz `tsc degrade status`)._

* **Efficient-MP** — najgorszy scenariusz `mask0p75`: 943.8 s (+200.7%). Gorzej niż FixedTime w 5 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`, `mask0p75`.
* **Advanced-MP** — najgorszy scenariusz `mask0p75`: 956.3 s (+216.9%). Gorzej niż FixedTime w 5 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`, `mask0p75`.


![krzywe degradacji cityflow_jinan3x4_4ph](figures/t2_nomech_cityflow_jinan3x4_4ph.png)

## cityflow_hangzhou4x4_4ph

FixedTime (niezależny od odczytów): **568.9 s**.

| scenariusz | Efficient-MP (n=1) | Advanced-MP (n=1) |
|---|---:|---:|
| `clean` | 331.2 | 325.5 |
| `gauss0p5` | 336.8 (+1.7%) | 333.8 (+2.5%) |
| `gauss1` | 344.8 (+4.1%) | 344.7 (+5.9%) |
| `gauss2` | 356.8 (+7.7%) | 360.2 (+10.6%) |
| `gauss3p5` | 372.6 (+12.5%) | 379.2 (+16.5%) |
| `gauss4` | 376.6 (+13.7%) | 384.0 (+18.0%) |
| `urand2` | 347.2 (+4.8%) | 347.8 (+6.9%) |
| `urand4` | 362.4 (+9.4%) | 366.3 (+12.5%) |
| `mask0p1` | 358.6 (+8.3%) | 359.0 (+10.3%) |
| `mask0p25` | 459.0 (+38.6%) | 480.9 (+47.7%) |
| `mask0p5` | 601.9 (+81.7%) ▲ | 637.6 (+95.9%) ▲ |
| `mask0p75` | 908.0 (+174.1%) ▲ | 918.2 (+182.1%) ▲ |
| `mask0p25_intersection` | 569.3 (+71.9%) ▲ | 553.0 (+69.9%) |
| `mask0p5_intersection` | 754.8 (+127.9%) ▲ | 752.3 (+131.1%) ▲ |
| `mask0p25_hold_t60` | 421.4 (+27.2%) | 468.6 (+44.0%) |
| `mask0p5_hold_t60` | 623.9 (+88.4%) ▲ | 701.7 (+115.6%) ▲ |
| `drift0p1` | 330.7 (-0.1%) | 325.9 (+0.1%) |
| `drift0p2` | 330.4 (-0.3%) | 326.7 (+0.4%) |
| `drift0p3` | 330.8 (-0.1%) | 327.3 (+0.5%) |
| `drift0p5` | 338.6 (+2.2%) | 338.4 (+3.9%) |

_Niepoliczone na tej maszynie: rdzeń GAT+PPO+CTDE, Advanced-CoLight (brak checkpointów rodzica — patrz `tsc degrade status`)._

* **Efficient-MP** — najgorszy scenariusz `mask0p75`: 908.0 s (+174.1%). Gorzej niż FixedTime w 5 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`, `mask0p75`.
* **Advanced-MP** — najgorszy scenariusz `mask0p75`: 918.2 s (+182.1%). Gorzej niż FixedTime w 4 scenariuszach: `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`, `mask0p75`.


![krzywe degradacji cityflow_hangzhou4x4_4ph](figures/t2_nomech_cityflow_hangzhou4x4_4ph.png)

## Czego ta tabela NIE mówi

* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem ~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` (`results/sensor_severity.md`).
* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego na sieciach o różnym obciążeniu.
* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.
