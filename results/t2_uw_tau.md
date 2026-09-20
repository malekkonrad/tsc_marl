# T2 — sweep τ dla UW-attention — plan `t2_uw_tau`

Tryb realistic (yellow=5), 5 losowań awarii na scenariusz, ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.

Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. **▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.

## cityflow_jinan3x4_4ph

FixedTime (niezależny od odczytów): **507.0 s**.

| scenariusz | rdzeń GAT+PPO+CTDE (n=3) | UW-attention, τ = 0,25 (n=3) | UW-attention, τ = 0,10 (n=3) | UW-attention, τ = 0,05 (n=3) |
|---|---:|---:|---:|---:|
| `clean` | 295.9 ± 0.6 | 295.5 ± 1.1 | 295.6 ± 0.2 | 295.9 ± 0.5 |
| `gauss2` | 319.9 ± 2.2 (+8.1%) | 319.8 ± 2.5 (+8.2%) | 319.6 ± 1.9 (+8.1%) | 320.1 ± 1.1 (+8.2%) |
| `gauss4` | 350.0 ± 4.5 (+18.3%) | 350.1 ± 4.9 (+18.5%) | 350.4 ± 6.1 (+18.5%) | 350.1 ± 4.5 (+18.3%) |
| `mask0p25` | 424.9 ± 31.1 (+43.6%) | 424.6 ± 23.4 (+43.7%) | 425.8 ± 26.1 (+44.0%) | 423.7 ± 25.3 (+43.2%) |
| `mask0p5` | 732.6 ± 28.5 (+147.5%) ▲ | 728.4 ± 36.1 (+146.5%) ▲ | 733.8 ± 34.0 (+148.2%) ▲ | 733.2 ± 31.9 (+147.8%) ▲ |
| `mask0p25_intersection` | 471.1 ± 68.8 (+59.2%) | 466.2 ± 67.2 (+57.7%) | 469.2 ± 70.7 (+58.7%) | 460.8 ± 72.0 (+55.7%) |

* **rdzeń GAT+PPO+CTDE** — najgorszy scenariusz `mask0p5`: 732.6 s (+147.5%). Gorzej niż FixedTime w 1 scenariuszach: `mask0p5`.
* **UW-attention, τ = 0,25** — najgorszy scenariusz `mask0p5`: 728.4 s (+146.5%). Gorzej niż FixedTime w 1 scenariuszach: `mask0p5`.
* **UW-attention, τ = 0,10** — najgorszy scenariusz `mask0p5`: 733.8 s (+148.2%). Gorzej niż FixedTime w 1 scenariuszach: `mask0p5`.
* **UW-attention, τ = 0,05** — najgorszy scenariusz `mask0p5`: 733.2 s (+147.8%). Gorzej niż FixedTime w 1 scenariuszach: `mask0p5`.


![krzywe degradacji cityflow_jinan3x4_4ph](figures/t2_uw_tau_cityflow_jinan3x4_4ph.png)

## Mechanizmy × scenariusz

Δ ATT [s] KOLUMNY OCENIANEJ względem `rdzeń GAT+PPO+CTDE`, sparowana po (seed, losowanie). **pogrubienie** = |t| ≥ 2.8. Δ < 0 = kolumna oceniana lepsza od odniesienia.


### cityflow_jinan3x4_4ph

| mechanizm | `clean` | `gauss2` | `gauss4` | `mask0p25` | `mask0p5` | `mask0p25_intersection` |
|---|---:|---:|---:|---:|---:|---:|
| UW-attention, τ = 0,25 | -0.4 | -0.1 | +0.0 | -0.4 | -4.2 | -4.9 |
| UW-attention, τ = 0,10 | -0.3 | -0.3 | +0.4 | +0.8 | +1.3 | -1.9 |
| UW-attention, τ = 0,05 | -0.0 | +0.3 | +0.1 | -1.3 | +0.6 | -10.2 |

## Kontrola: ewaluator mierzy tę samą politykę co T1

Czysty wynik ewaluatora vs ostatni test treningu, 3 biegów uczonych: największa różnica **0.000 s** — zgodne co do precyzji logu.

## Czego ta tabela NIE mówi

* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem ~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` (`results/sensor_severity.md`).
* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego na sieciach o różnym obciążeniu.
* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.
