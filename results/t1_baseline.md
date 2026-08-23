# T1-baseline — punkt odniesienia pracy

Wygenerowane: 2026-08-23 18:33. Protokół: `configs/tsc/thesis_protocol.yml` (Δt=15 s, epizod 3600 s, `PYTHONHASHSEED=0`), sieci w wariancie 4-fazowym — jak protokół papierów.

ATT = średnia z 10 ostatnich epizodów testowych (metody bez treningu: jeden przebieg).
Δ liczone względem literatury w trybie **benchmark**, bo w nim liczą swoje wyniki papiery.

## cityflow_jinan3x4_4ph

| agent | benchmark (yellow=0) | realistyczny (yellow=5) | literatura | Δ benchmark |
|---|---|---|---|---|
| fixedtime | 424.7 | 507.0 | 428.11 | -0.8% |
| maxpressure | 316.8 | 361.0 | 273.96 | +15.6% |
| sotl | 318.7 | 352.9 | — | — |
| efficientmp | 272.2 | 316.1 | 269.87 | +0.9% |
| advancedmp | 274.3 | 299.2 | 253.61 | +8.2% |

## cityflow_hangzhou4x4_4ph

| agent | benchmark (yellow=0) | realistyczny (yellow=5) | literatura | Δ benchmark |
|---|---|---|---|---|
| fixedtime | 506.2 | 568.9 | 495.57 | +2.2% |
| maxpressure | 374.3 | 392.7 | 288.54 | +29.7% |
| sotl | 344.9 | 367.2 | — | — |
| efficientmp | 314.0 | 330.1 | 284.44 | +10.4% |
| advancedmp | 314.7 | 325.4 | 279.47 | +12.6% |

## Jak czytać Δ

- **`maxpressure` to implementacja z LibSignal**, licząca pressure z `lane_count`
  (wszystkie pojazdy, bez normalizacji per pas), a nie wg definicji z papieru —
  jej odchyłka NIE jest kryterium bramki. Rolę poprawnego MP pełni `efficientmp`.
- **Odtwarzalność zależy od sieci**: JiNan trafia w literaturę co do ~1%
  (FixedTime −0,8%, Efficient-MP +0,9%), HangZhou nie (Efficient-MP +10,4%).
  Ten sam protokół, ten sam kod, dane zweryfikowane jako identyczne z plikami
  autorów (`hangzhou_4x4_advanced/` vs `hangzhou_4x4_gudang_18041610_1h/`:
  wyniki zgodne co do 0,01 s). Do opisania w pracy jako ograniczenie porównań.
- **`advancedmp` raportowany jest przy globalnym W1 = 1,0.** W trybie benchmark
  człon demand nie działa (bez kosztu przejścia utrzymanie fazy nic nie daje),
  więc metoda degeneruje się do Efficient-MP i +8,2% względem literatury jest
  OCZEKIWANE. Liczby z papieru dla Advanced-MP nie da się odtworzyć w trybie
  benchmark przy żadnym W1 — to najmocniejszy argument, że ten tryb gubi
  mechanizm metody, którą opisuje papier.
- Kolumna `realistyczny` to nasz protokół (5 s przejścia fazowego). Różnica
  między trybami jest samodzielnym wynikiem — patrz `t1_w1_sweep.md`.
