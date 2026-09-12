# T2 bez mechanizmu — plan `t2_nomech`

Tryb realistic (yellow=5), 5 losowań awarii na scenariusz, ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.

Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. **▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.

## cityflow_jinan3x4_4ph

FixedTime (niezależny od odczytów): **507.0 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | Advanced-CoLight (n=3) | Efficient-MP (n=1) | Advanced-MP (n=1) |
|---|---:|---:|---:|---:|
| `clean` | 295.9 ± 0.6 | 294.7 ± 2.5 | 313.9 | 301.8 |
| `gauss0p5` | 297.6 ± 0.6 (+0.6%) | 297.7 ± 1.3 (+1.0%) | 318.3 (+1.4%) | 303.8 (+0.7%) |
| `gauss1` | 304.5 ± 1.4 (+2.9%) | 305.7 ± 2.2 (+3.7%) | 321.5 (+2.4%) | 308.0 (+2.1%) |
| `gauss2` | 319.9 ± 2.2 (+8.1%) | 329.3 ± 6.3 (+11.7%) | 329.0 (+4.8%) | 318.6 (+5.6%) |
| `gauss3p5` | 342.0 ± 3.4 (+15.6%) | 380.7 ± 34.6 (+29.2%) | 342.1 (+9.0%) | 334.4 (+10.8%) |
| `gauss4` | 350.0 ± 4.5 (+18.3%) | 402.1 ± 51.9 (+36.4%) | 344.6 (+9.8%) | 338.0 (+12.0%) |
| `urand2` | 306.1 ± 0.8 (+3.4%) | 308.6 ± 2.2 (+4.7%) | 322.5 (+2.7%) | 309.9 (+2.7%) |
| `urand4` | 324.3 ± 1.6 (+9.6%) | 339.1 ± 9.1 (+15.0%) | 332.3 (+5.9%) | 324.1 (+7.4%) |
| `mask0p1` | 354.2 ± 20.0 (+19.7%) | 362.6 ± 12.0 (+23.0%) | 373.4 (+19.0%) | 362.2 (+20.0%) |
| `mask0p25` | 424.9 ± 31.1 (+43.6%) | 450.5 ± 23.3 (+52.9%) | 432.6 (+37.8%) | 459.3 (+52.2%) |
| `mask0p5` | 732.6 ± 28.5 (+147.5%) ▲ | 783.7 ± 35.2 (+165.9%) ▲ | 647.6 (+106.3%) ▲ | 750.0 (+148.5%) ▲ |
| `mask0p75` | 904.0 ± 67.5 (+205.5%) ▲ | 934.3 ± 15.2 (+217.0%) ▲ | 943.8 (+200.7%) ▲ | 956.3 (+216.9%) ▲ |
| `mask0p25_intersection` | 471.1 ± 68.8 (+59.2%) | 743.5 ± 90.7 (+152.3%) ▲ | 601.5 (+91.6%) ▲ | 663.7 (+119.9%) ▲ |
| `mask0p5_intersection` | 687.0 ± 110.5 (+132.2%) ▲ | 972.5 ± 85.0 (+230.0%) ▲ | 859.9 (+174.0%) ▲ | 864.7 (+186.5%) ▲ |
| `mask0p25_hold_t60` | 354.7 ± 24.8 (+19.9%) | 343.6 ± 4.4 (+16.6%) | 348.7 (+11.1%) | 331.7 (+9.9%) |
| `mask0p5_hold_t60` | 475.0 ± 38.9 (+60.5%) | 483.9 ± 36.8 (+64.2%) | 584.0 (+86.1%) ▲ | 549.1 (+82.0%) ▲ |
| `drift0p1` | 296.1 ± 0.3 (+0.0%) | 295.3 ± 0.9 (+0.2%) | 316.8 (+0.9%) | 301.0 (-0.3%) |
| `drift0p2` | 298.1 ± 0.6 (+0.7%) | 296.7 ± 1.0 (+0.7%) | 317.9 (+1.3%) | 302.3 (+0.2%) |
| `drift0p3` | 301.5 ± 0.5 (+1.9%) | 300.7 ± 1.0 (+2.0%) | 319.9 (+1.9%) | 305.8 (+1.3%) |
| `drift0p5` | 328.6 ± 4.3 (+11.0%) | 341.0 ± 2.5 (+15.7%) | 333.9 (+6.4%) | 325.5 (+7.9%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p75`: 904.0 s (+205.5%). Gorzej niż FixedTime w 3 scenariuszach: `mask0p5`, `mask0p5_intersection`, `mask0p75`.
* **Advanced-CoLight** — najgorszy scenariusz `mask0p5_intersection`: 972.5 s (+230.0%). Gorzej niż FixedTime w 4 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_intersection`, `mask0p75`.
* **Efficient-MP** — najgorszy scenariusz `mask0p75`: 943.8 s (+200.7%). Gorzej niż FixedTime w 5 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`, `mask0p75`.
* **Advanced-MP** — najgorszy scenariusz `mask0p75`: 956.3 s (+216.9%). Gorzej niż FixedTime w 5 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`, `mask0p75`.


![krzywe degradacji cityflow_jinan3x4_4ph](figures/t2_nomech_cityflow_jinan3x4_4ph.png)

## cityflow_hangzhou4x4_4ph

FixedTime (niezależny od odczytów): **568.9 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | Advanced-CoLight (n=3) | Efficient-MP (n=1) | Advanced-MP (n=1) |
|---|---:|---:|---:|---:|
| `clean` | 317.5 ± 1.0 | 317.3 ± 0.5 | 331.2 | 325.5 |
| `gauss0p5` | 328.9 ± 0.4 (+3.6%) | 331.7 ± 2.2 (+4.6%) | 336.8 (+1.7%) | 333.8 (+2.5%) |
| `gauss1` | 344.9 ± 1.1 (+8.6%) | 349.7 ± 4.3 (+10.2%) | 344.8 (+4.1%) | 344.7 (+5.9%) |
| `gauss2` | 372.1 ± 4.2 (+17.2%) | 383.2 ± 9.0 (+20.8%) | 356.8 (+7.7%) | 360.2 (+10.6%) |
| `gauss3p5` | 406.9 ± 11.0 (+28.2%) | 426.5 ± 16.9 (+34.4%) | 372.6 (+12.5%) | 379.2 (+16.5%) |
| `gauss4` | 419.3 ± 13.8 (+32.1%) | 438.8 ± 20.4 (+38.3%) | 376.6 (+13.7%) | 384.0 (+18.0%) |
| `urand2` | 351.4 ± 1.4 (+10.7%) | 358.7 ± 6.0 (+13.0%) | 347.2 (+4.8%) | 347.8 (+6.9%) |
| `urand4` | 384.5 ± 5.7 (+21.1%) | 397.7 ± 11.3 (+25.3%) | 362.4 (+9.4%) | 366.3 (+12.5%) |
| `mask0p1` | 363.4 ± 4.2 (+14.4%) | 396.2 ± 13.2 (+24.9%) | 358.6 (+8.3%) | 359.0 (+10.3%) |
| `mask0p25` | 460.1 ± 21.2 (+44.9%) | 530.5 ± 59.9 (+67.2%) | 459.0 (+38.6%) | 480.9 (+47.7%) |
| `mask0p5` | 739.2 ± 98.5 (+132.8%) ▲ | 888.6 ± 234.4 (+180.1%) ▲ | 601.9 (+81.7%) ▲ | 637.6 (+95.9%) ▲ |
| `mask0p75` | 1101.5 ± 239.0 (+246.9%) ▲ | 1230.6 ± 323.5 (+287.9%) ▲ | 908.0 (+174.1%) ▲ | 918.2 (+182.1%) ▲ |
| `mask0p25_intersection` | 472.2 ± 58.9 (+48.7%) | 585.5 ± 148.3 (+84.6%) ▲ | 569.3 (+71.9%) ▲ | 553.0 (+69.9%) |
| `mask0p5_intersection` | 765.9 ± 188.2 (+141.2%) ▲ | 950.9 ± 332.6 (+199.7%) ▲ | 754.8 (+127.9%) ▲ | 752.3 (+131.1%) ▲ |
| `mask0p25_hold_t60` | 411.1 ± 16.8 (+29.5%) | 415.4 ± 11.8 (+30.9%) | 421.4 (+27.2%) | 468.6 (+44.0%) |
| `mask0p5_hold_t60` | 544.4 ± 11.8 (+71.5%) | 565.9 ± 38.5 (+78.4%) | 623.9 (+88.4%) ▲ | 701.7 (+115.6%) ▲ |
| `drift0p1` | 317.3 ± 0.3 (-0.1%) | 317.3 ± 0.4 (+0.0%) | 330.7 (-0.1%) | 325.9 (+0.1%) |
| `drift0p2` | 318.2 ± 0.0 (+0.2%) | 318.2 ± 0.6 (+0.3%) | 330.4 (-0.3%) | 326.7 (+0.4%) |
| `drift0p3` | 319.5 ± 0.3 (+0.6%) | 319.6 ± 0.5 (+0.7%) | 330.8 (-0.1%) | 327.3 (+0.5%) |
| `drift0p5` | 339.9 ± 1.1 (+7.0%) | 342.6 ± 5.6 (+8.0%) | 338.6 (+2.2%) | 338.4 (+3.9%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p75`: 1101.5 s (+246.9%). Gorzej niż FixedTime w 3 scenariuszach: `mask0p5`, `mask0p5_intersection`, `mask0p75`.
* **Advanced-CoLight** — najgorszy scenariusz `mask0p75`: 1230.6 s (+287.9%). Gorzej niż FixedTime w 4 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_intersection`, `mask0p75`.
* **Efficient-MP** — najgorszy scenariusz `mask0p75`: 908.0 s (+174.1%). Gorzej niż FixedTime w 5 scenariuszach: `mask0p25_intersection`, `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`, `mask0p75`.
* **Advanced-MP** — najgorszy scenariusz `mask0p75`: 918.2 s (+182.1%). Gorzej niż FixedTime w 4 scenariuszach: `mask0p5`, `mask0p5_hold_t60`, `mask0p5_intersection`, `mask0p75`.


![krzywe degradacji cityflow_hangzhou4x4_4ph](figures/t2_nomech_cityflow_hangzhou4x4_4ph.png)

## Kontrola: ewaluator mierzy tę samą politykę co T1

Czysty wynik ewaluatora vs ostatni test treningu, 12 biegów uczonych: największa różnica **0.022 s** — zgodne co do precyzji logu.

## Czego ta tabela NIE mówi

* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem ~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` (`results/sensor_severity.md`).
* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego na sieciach o różnym obciążeniu.
* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.
