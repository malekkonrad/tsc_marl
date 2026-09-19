# T2 z bramką fallbacku — plan `t2_gate`

Tryb realistic (yellow=5), 5 losowań awarii na scenariusz, ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.

Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. **▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.

## cityflow_jinan3x4_4ph

FixedTime (niezależny od odczytów): **507.0 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | bramka, próg 0,25 (n=3) | bramka, próg 0,50 (n=3) | bramka, próg 0,75 (n=3) | bramka, próg 1,00 (n=3) |
|---|---:|---:|---:|---:|---:|
| `clean` | 295.9 ± 0.6 | 295.9 ± 0.6 | 295.9 ± 0.6 | 295.9 ± 0.6 | 295.9 ± 0.6 |
| `mask0p1` | 354.2 ± 20.0 (+19.7%) | 365.4 ± 2.0 (+23.5%) | 360.5 ± 18.6 (+21.8%) | 354.2 ± 20.0 (+19.7%) | 354.2 ± 20.0 (+19.7%) |
| `mask0p25` | 424.9 ± 31.1 (+43.6%) | 436.1 ± 2.5 (+47.4%) | 429.1 ± 20.7 (+45.0%) | 424.9 ± 31.1 (+43.6%) | 424.9 ± 31.1 (+43.6%) |
| `mask0p25_intersection` | 471.1 ± 68.8 (+59.2%) | 350.7 ± 1.4 (+18.5%) | 350.7 ± 1.4 (+18.5%) | 350.7 ± 1.4 (+18.5%) | 350.7 ± 1.4 (+18.5%) |
| `mask0p5` | 732.6 ± 28.5 (+147.5%) ▲ | 493.8 ± 0.9 (+66.8%) | 491.1 ± 11.7 (+65.9%) | 673.0 ± 30.8 (+127.4%) ▲ | 732.6 ± 28.5 (+147.5%) ▲ |
| `mask0p75` | 904.0 ± 67.5 (+205.5%) ▲ | 501.5 ± 0.2 (+69.5%) | 501.4 ± 0.4 (+69.4%) | 589.3 ± 22.1 (+99.1%) ▲ | 880.0 ± 60.2 (+197.4%) ▲ |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p75`: 904.0 s (+205.5%). Gorzej niż FixedTime w 2 scenariuszach: `mask0p5`, `mask0p75`.
* **bramka, próg 0,25** — najgorszy scenariusz `mask0p75`: 501.5 s (+69.5%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **bramka, próg 0,50** — najgorszy scenariusz `mask0p75`: 501.4 s (+69.4%). Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do FixedTime nie ma z czego zarobić.
* **bramka, próg 0,75** — najgorszy scenariusz `mask0p5`: 673.0 s (+127.4%). Gorzej niż FixedTime w 2 scenariuszach: `mask0p5`, `mask0p75`.
* **bramka, próg 1,00** — najgorszy scenariusz `mask0p75`: 880.0 s (+197.4%). Gorzej niż FixedTime w 2 scenariuszach: `mask0p5`, `mask0p75`.


![krzywe degradacji cityflow_jinan3x4_4ph](figures/t2_gate_cityflow_jinan3x4_4ph.png)

## Mechanizmy × scenariusz

Δ ATT [s] względem `rdzeń GAT+PPO+CTDE`, sparowana po (seed, losowanie); w nawiasie odsetek par (decyzja, węzeł) przejętych przez regułę. **pogrubienie** = |t| ≥ 2.8. Δ < 0 = bramka lepsza.


### cityflow_jinan3x4_4ph

| mechanizm | `clean` | `mask0p1` | `mask0p25` | `mask0p25_intersection` | `mask0p5` | `mask0p75` |
|---|---:|---:|---:|---:|---:|---:|
| bramka, próg 0,25 | +0.0 (0%) | +11.2 (15%) | +11.2 (63%) | **-120.4** (27%) | **-238.8** (93%) | **-402.5** (98%) |
| bramka, próg 0,50 | +0.0 (0%) | +6.3 (2%) | +4.1 (7%) | **-120.4** (27%) | **-241.5** (62%) | **-402.6** (97%) |
| bramka, próg 0,75 | +0.0 (0%) | +0.0 (0%) | +0.0 (0%) | **-120.4** (27%) | **-59.5** (10%) | **-314.7** (65%) |
| bramka, próg 1,00 | +0.0 (0%) | +0.0 (0%) | +0.0 (0%) | **-120.4** (27%) | +0.0 (0%) | -24.0 (5%) |

## Kontrola: ewaluator mierzy tę samą politykę co T1

Czysty wynik ewaluatora vs ostatni test treningu, 3 biegów uczonych: największa różnica **0.000 s** — zgodne co do precyzji logu.

## Czego ta tabela NIE mówi

* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem ~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` (`results/sensor_severity.md`).
* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego na sieciach o różnym obciążeniu.
* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.
