# Bramka prototypu 2.2 — głowa OBSERWACYJNA (niepewność cech wejściowych)

Okno 12 odczytów + średnia sąsiadów upstream, 60 epok, Gaussian NLL względem `x_clean`, trening na: clean, gauss1, gauss2, mask0p25. Ewaluacja na ogonie epizodu. u = σ w jednostkach odchylenia cechy na czystym; u ruchu = max po jego cechach (EP, demand).

| scenariusz | tren. | popsute ruchy | błędne ruchy | u średnie | AUROC ruch (popsute) | AUROC ruch (błędne) | AUROC krawędź (max) | AUROC krawędź (mean) | pokrycie 90% | RMSE wejścia | RMSE μ |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | x | 0.0% | 0.0% | 0.492 | — | — | — | — | 66.7% | 0.000 | 0.520 |
| drift0p2 |  | 23.3% | 23.3% | 0.517 | 0.529 | 0.529 | 0.513 | 0.515 | 62.5% | 0.446 | 0.661 |
| drift0p3 |  | 34.5% | 34.5% | 0.542 | 0.536 | 0.536 | 0.521 | 0.519 | 59.9% | 0.668 | 0.798 |
| gauss1 | x | 74.7% | 74.7% | 0.974 | 0.856 | 0.856 | 0.832 | 0.860 | 82.3% | 0.865 | 0.818 |
| gauss2 | x | 89.0% | 89.0% | 1.374 | 0.903 | 0.903 | 0.886 | 0.909 | 82.9% | 1.627 | 1.107 |
| gauss3p5 |  | 94.1% | 94.1% | 1.848 | 0.920 | 0.920 | 0.912 | 0.930 | 80.5% | 2.698 | 1.407 |
| gauss4 |  | 94.9% | 94.9% | 2.000 | 0.922 | 0.922 | 0.917 | 0.934 | 79.6% | 3.047 | 1.493 |
| mask0p25 | x | 68.4% | 21.2% | 0.696 | 0.656 | 0.788 | 0.652 | 0.652 | 72.5% | 0.759 | 0.774 |
| mask0p25_intersection |  | 52.8% | 29.1% | 0.606 | 0.600 | 0.651 | 0.580 | 0.583 | 61.8% | 1.031 | 1.025 |
| mask0p5 |  | 91.7% | 44.3% | 0.749 | 0.671 | 0.739 | 0.698 | 0.701 | 68.4% | 1.207 | 1.123 |
| urand3p5 |  | 91.6% | 91.6% | 1.427 | 0.913 | 0.913 | 0.897 | 0.918 | 84.8% | 1.658 | 1.113 |

### Te same metryki na zbiorach OOD (ruch spoza treningu)

| zbiór | scenariusz | u średnie | AUROC ruch | pokrycie 90% |
|---|---|---:|---:|---:|
| r2000 | clean | 0.380 | — | 78.4% |
| r2000 | gauss2 | 1.083 | 0.918 | 88.4% |
| r2000 | mask0p5 | 0.561 | 0.651 | 79.3% |
| r2000 | drift0p3 | 0.401 | 0.528 | 73.5% |
| hz | clean | 0.479 | — | 86.8% |
| hz | gauss2 | 0.935 | 0.857 | 91.2% |
| hz | mask0p5 | 0.639 | 0.534 | 82.8% |
| hz | drift0p3 | 0.493 | 0.481 | 84.4% |
| d0p25 | clean | 0.475 | — | 86.8% |
| d0p25 | gauss2 | 0.946 | 0.873 | 91.2% |
| d0p25 | mask0p5 | 0.616 | 0.572 | 85.3% |
| d0p25 | drift0p3 | 0.486 | 0.524 | 83.9% |

## Separacja 2×2 — czy dwa kanały mierzą dwie różne rzeczy

`u_obs` = średnie u głowy obserwacyjnej, `u_ep` = epistemic deep ensemble (MI między członkami, prototyp 2.0). Wiersz = stan odczytów, kolumna = ruch.

| odczyty \ ruch | trening | d0p25 | hz | r2000 |
|---|---|---|---|---|
| clean | u_obs 0.492 · u_ep 0.3527 | u_obs 0.475 · u_ep 0.2862 | u_obs 0.479 · u_ep 0.2695 | u_obs 0.380 · u_ep 0.3332 |
| gauss2 | u_obs 1.374 · u_ep 0.4506 | u_obs 0.946 · u_ep 0.5041 | u_obs 0.935 · u_ep 0.4801 | u_obs 1.083 · u_ep 0.4880 |
| mask0p5 | u_obs 0.749 · u_ep 0.3083 | u_obs 0.616 · u_ep 0.2064 | u_obs 0.639 · u_ep 0.1944 | u_obs 0.561 · u_ep 0.2777 |

* `u_obs`: przyrost od awarii (wiersz) **+0.628**, od OOD (kolumna) **-0.048**.
* `u_ep`: przyrost od awarii **+0.1703**, od OOD **-0.0564**.
* `u_obs` SPADA na ruchu spoza treningu: σ głowy skaluje się z obciążeniem (mniej pojazdów = mniejszy możliwy błąd odczytu), a zbiory OOD są lżejsze. Kierunek jest nieszkodliwy dla fallbacku (brak fałszywych alarmów), ale próg na `u_obs` trzeba liczyć względem σ przewidywanego na czystym, nie jako stałą bezwzględną.
* Werdykt separacji: `u_obs` reaguje na awarię, nie na OOD — `u_ep` NIE reaguje na OOD mocniej niż na awarię — dwupoziomowy fallback nie ma tu empirycznej podstawy na poziomie (a).

## Kryteria wyjścia

* **(a) monotoniczność** — u po σ szumu (0, 1, 2, 3.5, 4): 0.492, 0.974, 1.374, 1.848, 2.000 → **TAK** (3,5 i 4,0 NIEWIDZIANE w treningu).
* **(c) kalibracja** — pokrycie przedziału 90% na czystym 66.7%, na szumie widzianym 82.6%.
* **(d) detekcja na widzianym** — min AUROC ruchu (gauss1, gauss2, mask0p25): **0.788** (próg 0,8).
* **(e) detekcja na NIEWIDZIANYM** — min AUROC po gauss3p5, gauss4, urand3p5, mask0p5, mask0p25_intersection: **0.651** (próg 0,7); `drift` (inna rodzina): max AUROC **0.536**.
* **(f) agregacja ruch→krawędź** — AUROC(max) − AUROC(mean), średnio: -0.011 → w attention **mean**.

**Bramka 2.2 (głowa obserwacyjna): NIEZALICZONA.**

## Czego ta tabela NIE mówi

* Prototyp jest NADZOROWANY (symulator zna `x_clean`). W pętli sterowania głowa działa tak samo — trening offline na trajektoriach zamrożonej polityki, inferencja bez prawdy.
* AUROC porównuje komórki popsute z CZYSTEGO scenariusza — nie mierzy lokalizacji w obrębie jednego epizodu (tam, gdzie wszystkie detektory szumią, nie ma czego lokalizować).
* „Popsuty ruch” = martwy detektor albo błąd cechy > 0.5 pojazdu. Dryf małych zliczeń rzadko przekracza próg, więc jego AUROC liczy się na nielicznych komórkach.
