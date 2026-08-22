# Advanced-MP — strojenie W1 (odpowiednik Figure 2 z Zhang 2022)

Δt = 15 s, `obs_length` = 167 m, `drop_always_green` = true, seed 0, JiNan 3×4 / HangZhou 4×4,
wariant 4-fazowy. Punkt odniesienia = Efficient-MP w tym samym trybie.

## JiNan 3×4 (4 fazy)

| W1 | ATT — realistyczny (yellow=5) | ATT — benchmark (yellow=0) |
|---|---|---|
| 0,05 | 315,8 | **271,7** |
| 0,25 | 313,3 | 271,9 |
| 0,50 | 311,2 | 271,8 |
| 0,75 | 303,3 | 273,4 |
| **1,00** | **299,2** | 274,3 |
| 1,50 | 302,4 | 279,8 |
| 2,00 | 305,7 | 285,1 |
| 3,00 | — | 297,1 |
| *Efficient-MP* | *316,1* | *272,2* |

## HangZhou 4×4 (4 fazy), tryb realistyczny

| W1 | 0,25 | 0,50 | 0,75 | 1,00 | 1,50 | 2,00 |
|---|---|---|---|---|---|---|
| ATT | 326,2 | **323,9** | 324,0 | 325,4 | 327,9 | 331,4 |

## Wnioski

1. **W1 = 1,0** — wartość z referencyjnej implementacji autorów
   (`run_advanced_maxpressure.py`: `"W": 1`) — jest optimum na JiNan i leży 0,5% od
   optimum na HangZhou. Jedna globalna wartość dla całej pracy jest uzasadniona.
2. **Człon demand działa TYLKO w trybie realistycznym.** Przy yellow=5 daje −5,3%
   względem Efficient-MP (299,2 vs 316,1; papier deklaruje −6,0%). Przy yellow=0
   optimum ucieka do W1→0, czyli metoda degeneruje się do Efficient-MP — bo bez
   kosztu przejścia fazowego „utrzymanie fazy" nic nie oszczędza.
   To jest sedno mechanizmu Advanced-MP i argument, że tryb benchmark gubi zjawiska,
   które w realnym wdrożeniu decydują o przewadze metody.
3. **W1 = 0 NIE degeneruje się do Efficient-MP**: gdy `max p(s) ≤ 0` (24% decyzji na
   JiNan), warunek `d·W1 > max p` jest spełniony trywialnie i faza jest utrzymywana.
   Dlatego pierwszy sweep po siatce {0,1,2,3,4} nie mógł znaleźć optimum.
