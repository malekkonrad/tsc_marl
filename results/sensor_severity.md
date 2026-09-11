# Przelicznik dotkliwości awarii (poziom w specyfikacji → odchylenie na cesze)

Sieć: `cityflow_jinan3x4_4ph`. Stan: ATS (blok EP + blok demand), okno `obs_length`.
Kolumna **przelicznik** = `ep_std / level` — o tyle szum wstrzyknięty NA DETEKTORZE
jest słabszy/mocniejszy na cesze, którą widzi polityka. Bez tego poziomy
z [[li-2025-robustlight]] (3,5–4,0 na STANIE) nie są porównywalne z naszymi.

| scenariusz | poziom | jedn. | EP std | EP MAD | demand std | demand MAD | zmienione | przelicznik |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| gauss | 1 | lane | 0.891 | 0.686 | 0.738 | 0.519 | 82.9% | 0.891 |
| gauss | 2 | lane | 1.702 | 1.315 | 1.387 | 0.982 | 82.9% | 0.851 |
| gauss | 3.5 | lane | 2.860 | 2.209 | 2.303 | 1.631 | 82.9% | 0.817 |
| gauss | 4 | lane | 3.236 | 2.498 | 2.601 | 1.842 | 82.9% | 0.809 |
| urand | 3.5 | lane | 1.742 | 1.397 | 1.384 | 1.076 | 82.2% | 0.498 |
| urand | 4 | lane | 1.972 | 1.582 | 1.561 | 1.214 | 82.2% | 0.493 |
| mask | 0.25 | lane | 2.450 | 0.620 | 0.702 | 0.132 | 12.9% | — |
| mask | 0.5 | lane | 3.350 | 1.257 | 1.017 | 0.286 | 24.6% | — |
| mask | 0.25 | intersection | 3.104 | 1.158 | 0.914 | 0.247 | 21.3% | — |
| drift | 0.2 | lane | 1.128 | 0.536 | 0.327 | 0.119 | 53.7% | — |
| drift | 0.3 | lane | 1.692 | 0.804 | 0.491 | 0.179 | 53.7% | — |

## Jak to czytać

Na czystym biegu **28% komórek bloku EP i 66% bloku demand to dokładnie zero**. Stąd dwa wnioski, oba istotne dla T2:

1. **Maska jest w dużej części no-opem.** Zero jest wartością MODALNĄ, więc martwy
   detektor raportujący 0 wygląda jak zdrowy detektor na pustym wlocie — kolumna
   *zmienione* jest znacznie niższa niż ułamek zepsutych pasów. To nie jest usterka
   pomiaru, tylko własność podstawienia `zero`, i dokładnie ona czyni maskę awarią
   CICHĄ (patrz `domain/sensors.py`: podstawienie nie jest neutralne).
2. **Dotkliwość maski zależy od OBCIĄŻENIA sieci.** Ten sam procent maski nie znaczy
   tego samego na JiNan i na HangZhou (6295 vs 2983 poj./h) — w pracy trzeba podawać
   ułamek zmienionych komórek obok procentu maski, inaczej krzywe degradacji obu
   sieci nie są porównywalne między sobą.

Dla szumu addytywnego przelicznik ma przewidywalne źródło: e(l,m) to różnica ŚREDNICH
po pasach wjazdowych i wyjazdowych, więc przy M pasach z każdej strony sigma na cesze
wynosi sigma·sqrt(1/M_in + 1/M_out) — dla M=3 daje to 0,816. Odchylenia od tej wartości
biorą się z ruchów o innym M i z ucięcia w zerze (`max(0, ...)`), które przy większych
sigma zjada coraz więcej masy — stąd przelicznik MALEJE z poziomem.
`urand` o tym samym `level` jest istotnie SŁABSZY (szerokość a to sigma = a/sqrt(3)),
więc porównywanie obu ataków „po tej samej liczbie" jest mylące.
