# Bramka prototypu 2.2 — głowa OBSERWACYJNA (niepewność cech wejściowych)

Okno 4 odczytów + średnia sąsiadów upstream, 60 epok, Gaussian NLL względem `x_clean`, trening na: clean, gauss1, gauss2, mask0p25. Ewaluacja na ogonie epizodu. u = σ w jednostkach odchylenia cechy na czystym; u ruchu = max po jego cechach (EP, demand).

| scenariusz | tren. | popsute ruchy | błędne ruchy | u średnie | AUROC ruch (popsute) | AUROC ruch (błędne) | AUROC krawędź (max) | AUROC krawędź (mean) | pokrycie 90% | RMSE wejścia | RMSE μ |
|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | x | 0.0% | 0.0% | 0.627 | — | — | — | — | 85.3% | 0.000 | 0.431 |
| drift0p2 |  | 23.3% | 23.3% | 0.663 | 0.500 | 0.500 | 0.515 | 0.515 | 78.4% | 0.446 | 0.599 |
| drift0p3 |  | 34.5% | 34.5% | 0.702 | 0.534 | 0.534 | 0.541 | 0.540 | 75.5% | 0.668 | 0.750 |
| gauss1 | x | 74.7% | 74.7% | 1.020 | 0.797 | 0.797 | 0.764 | 0.801 | 85.7% | 0.865 | 0.785 |
| gauss2 | x | 89.0% | 89.0% | 1.375 | 0.866 | 0.866 | 0.840 | 0.875 | 85.4% | 1.627 | 1.073 |
| gauss3p5 |  | 94.1% | 94.1% | 1.868 | 0.899 | 0.899 | 0.885 | 0.914 | 85.4% | 2.698 | 1.359 |
| gauss4 |  | 94.9% | 94.9% | 2.042 | 0.906 | 0.906 | 0.895 | 0.922 | 85.4% | 3.047 | 1.440 |
| mask0p25 | x | 68.4% | 21.2% | 0.705 | 0.577 | 0.700 | 0.578 | 0.572 | 82.2% | 0.759 | 0.764 |
| mask0p25_intersection |  | 52.8% | 29.1% | 0.615 | 0.566 | 0.603 | 0.542 | 0.548 | 75.7% | 1.031 | 1.035 |
| mask0p5 |  | 91.7% | 44.3% | 0.676 | 0.569 | 0.630 | 0.593 | 0.590 | 74.0% | 1.207 | 1.156 |
| urand3p5 |  | 91.6% | 91.6% | 1.380 | 0.872 | 0.872 | 0.848 | 0.882 | 86.3% | 1.658 | 1.083 |

### Te same metryki na zbiorach OOD (ruch spoza treningu)

| zbiór | scenariusz | u średnie | AUROC ruch | pokrycie 90% |
|---|---|---:|---:|---:|
| r2000 | clean | 0.433 | — | 93.2% |
| r2000 | gauss2 | 1.027 | 0.899 | 90.6% |
| r2000 | mask0p5 | 0.420 | 0.544 | 85.0% |
| r2000 | drift0p3 | 0.465 | 0.496 | 87.1% |
| hz | clean | 0.326 | — | 93.7% |
| hz | gauss2 | 0.838 | 0.909 | 91.8% |
| hz | mask0p5 | 0.287 | 0.472 | 85.8% |
| hz | drift0p3 | 0.337 | 0.494 | 90.5% |
| d0p25 | clean | 0.336 | — | 95.5% |
| d0p25 | gauss2 | 0.861 | 0.917 | 92.9% |
| d0p25 | mask0p5 | 0.311 | 0.508 | 90.5% |
| d0p25 | drift0p3 | 0.348 | 0.519 | 92.4% |

## Separacja 2×2 — czy dwa kanały mierzą dwie różne rzeczy

`u_obs` = średnie u głowy obserwacyjnej, `u_ep` = epistemic deep ensemble (MI między członkami, prototyp 2.0). Wiersz = stan odczytów, kolumna = ruch.

| odczyty \ ruch | trening | d0p25 | hz | r2000 |
|---|---|---|---|---|
| clean | u_obs 0.627 · u_ep 0.3527 | u_obs 0.336 · u_ep 0.2862 | u_obs 0.326 · u_ep 0.2695 | u_obs 0.433 · u_ep 0.3332 |
| gauss2 | u_obs 1.375 · u_ep 0.4506 | u_obs 0.861 · u_ep 0.5041 | u_obs 0.838 · u_ep 0.4801 | u_obs 1.027 · u_ep 0.4880 |
| mask0p5 | u_obs 0.676 · u_ep 0.3083 | u_obs 0.311 · u_ep 0.2064 | u_obs 0.287 · u_ep 0.1944 | u_obs 0.420 · u_ep 0.2777 |

* `u_obs`: przyrost od awarii (wiersz) **+0.595**, od OOD (kolumna) **-0.262**.
* `u_ep`: przyrost od awarii **+0.1703**, od OOD **-0.0564**.
* `u_obs` SPADA na ruchu spoza treningu: σ głowy skaluje się z obciążeniem (mniej pojazdów = mniejszy możliwy błąd odczytu), a zbiory OOD są lżejsze. Kierunek jest nieszkodliwy dla fallbacku (brak fałszywych alarmów), ale próg na `u_obs` trzeba liczyć względem σ przewidywanego na czystym, nie jako stałą bezwzględną.
* Werdykt separacji: `u_obs` NIE jest czysty — `u_ep` NIE reaguje na OOD mocniej niż na awarię — dwupoziomowy fallback nie ma tu empirycznej podstawy na poziomie (a).

## Kryteria wyjścia

* **(a) monotoniczność** — u po σ szumu (0, 1, 2, 3.5, 4): 0.627, 1.020, 1.375, 1.868, 2.042 → **TAK** (3,5 i 4,0 NIEWIDZIANE w treningu).
* **(c) kalibracja** — pokrycie przedziału 90% na czystym 85.3%, na szumie widzianym 85.5%.
* **(d) detekcja na widzianym** — min AUROC ruchu (gauss1, gauss2, mask0p25): **0.700** (próg 0,8).
* **(e) detekcja na NIEWIDZIANYM** — min AUROC po gauss3p5, gauss4, urand3p5, mask0p5, mask0p25_intersection: **0.603** (próg 0,7); `drift` (inna rodzina): max AUROC **0.534**.
* **(f) agregacja ruch→krawędź** — AUROC(max) − AUROC(mean), średnio: -0.016 → w attention **mean**.

**Bramka 2.2 (głowa obserwacyjna): NIEZALICZONA.**

## Czego ta tabela NIE mówi

* Prototyp jest NADZOROWANY (symulator zna `x_clean`). W pętli sterowania głowa działa tak samo — trening offline na trajektoriach zamrożonej polityki, inferencja bez prawdy.
* AUROC porównuje komórki popsute z CZYSTEGO scenariusza — nie mierzy lokalizacji w obrębie jednego epizodu (tam, gdzie wszystkie detektory szumią, nie ma czego lokalizować).
* „Popsuty ruch” = martwy detektor albo błąd cechy > 0.5 pojazdu. Dryf małych zliczeń rzadko przekracza próg, więc jego AUROC liczy się na nielicznych komórkach.
