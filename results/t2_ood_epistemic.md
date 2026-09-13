# Test gałęzi epistemic — plan `t2_ood`

Zamrożone sterowniki na ruchu spoza treningu. Polityki uczone biorą wagi z sieci TRENINGOWEJ (`{base}` w szablonie rodzica), ruch jest z innego pliku. Czysty scenariusz = czyste odczyty; `gauss2` = ten sam ruch z szumem detektorów, żeby porównać reakcję ensemble na nowy ruch z reakcją na awarię sensora.

## Polityki uczone na `cityflow_jinan3x4_4ph`

### ATT na czystych odczytach

| ruch | fixedtime | efficientmp | advancedmp | advancedcolight | gatppo | rdzeń vs lepsza reguła |
|---|---:|---:|---:|---:|---:|---:|
| ruch treningowy | 507.0 | 313.9 | 301.8 | 294.4 ± 1.8 | 295.8 ± 1.2 | -2.0% (p 1.00) |
| popyt ×0,25 | 322.3 | 258.8 | 256.8 | 252.9 ± 1.1 | 253.2 ± 1.7 | -1.4% (p 1.00) |
| inny plik ruchu (2000) | 412.1 | 288.4 | 288.1 | 283.1 ± 0.7 | 283.6 ± 0.6 | -1.5% (p 1.00) |

▲ = rdzeń ISTOTNIE gorszy od lepszego z MaxPressure (jednostronny t, p < 0.05, seedy polityki vs deterministyczna reguła) — tylko tam gałąź (a) miałaby robotę.

### Rozbieżność 5 actorów (epistemic w polityce)

| ruch | MI czyste | MI szum σ=2 | niezgodne głosy czyste | niezgodne głosy szum |
|---|---:|---:|---:|---:|
| ruch treningowy | 0.2283 | 0.2872 | 26.5% | 27.8% |
| popyt ×0,25 | 0.1689 | 0.2781 | 34.4% | 29.0% |
| inny plik ruchu (2000) | 0.1892 | 0.2843 | 24.9% | 28.7% |

## Polityki uczone na `cityflow_hangzhou4x4_4ph`

### ATT na czystych odczytach

| ruch | fixedtime | efficientmp | advancedmp | advancedcolight | gatppo | rdzeń vs lepsza reguła |
|---|---:|---:|---:|---:|---:|---:|
| ruch treningowy | 568.9 | 331.2 | 325.5 | 317.4 ± 0.5 | 317.5 ± 0.7 | -2.5% (p 1.00) |
| inny plik ruchu (5816) | 559.5 | 428.0 | 419.0 | 437.7 ± 15.1 | 420.1 ± 1.5 | +0.3% (p 0.10) |

▲ = rdzeń ISTOTNIE gorszy od lepszego z MaxPressure (jednostronny t, p < 0.05, seedy polityki vs deterministyczna reguła) — tylko tam gałąź (a) miałaby robotę.

### Rozbieżność 5 actorów (epistemic w polityce)

| ruch | MI czyste | MI szum σ=2 | niezgodne głosy czyste | niezgodne głosy szum |
|---|---:|---:|---:|---:|
| ruch treningowy | 0.1082 | 0.2125 | 16.6% | 21.2% |
| inny plik ruchu (5816) | 0.1858 | 0.2327 | 22.4% | 22.8% |

## Werdykt

1. **Rdzeń nie przegrywa istotnie z MaxPressure na żadnym ruchu spoza treningu.** Remis w granicach rozrzutu seedów: „inny plik ruchu (5816)” +0.3% (p 0.10). Przełączenie na MaxPressure nie ma tu nic do zyskania, więc gałąź (a) dwupoziomowego fallbacku nie ma roboty — fallback jednopoziomowy do FixedTime.

Rozbieżność 5 actorów (MI) na CZYSTYM wejściu, względem ruchu treningowego:

* `popyt ×0,25`: MI -0.0594 (-26%); dla porównania szum σ=2 na ruchu treningowym +0.0588 → nie rośnie.
* `inny plik ruchu (2000)`: MI -0.0392 (-17%); dla porównania szum σ=2 na ruchu treningowym +0.0588 → nie rośnie.
* `inny plik ruchu (5816)`: MI +0.0776 (+72%); dla porównania szum σ=2 na ruchu treningowym +0.1043 → **rośnie** — sygnał nowego ruchu istnieje (słabiej niż pod szumem: bez kanału obserwacyjnego nieodróżnialny).

2. **Gałąź (a) broni się tylko tam, gdzie OBA warunki zachodzą na tym samym ruchu:** istotna przegrana z MaxPressure i wzrost MI na czystym wejściu. Nie zachodzą nigdzie — MI rośnie na „inny plik ruchu (5816)”, ale tam polityka nie przegrywa, więc nie ma czego ratować.
