# runlog

Przechowywanie i odpytywanie wyników eksperymentów. Niezależne od dziedziny —
nie wie nic o symulatorach, uczeniu maszynowym ani o projekcie, w którym stoi.
Zależności: stdlib (zapis) + `pandas` i `pyyaml` (odczyt, importowane leniwie).

## Po co

Typowy pipeline badawczy zapisuje wyniki do jednego CSV i odsyła w nim do plików
konfiguracyjnych, które potem się zmieniają. Skutki są zawsze te same: nie da się
odtworzyć wiersza, dwa eksperymenty dzielą jeden katalog, a przerwany bieg wygląda
jak policzony. `runlog` podnosi te trzy rzeczy do rangi niezmienników.

## Model danych

Jeden bieg = jeden katalog. **System plików jest bazą danych**, indeks jest wyliczany
przez skan — nie ma centralnego pliku, więc równoległe biegi w osobnych procesach
nie walczą o zapis i nie potrzebują blokad.

```
runs/<experiment>/<run_id>/
    meta.json            # params, params_hash, proweniencja, status, czasy
    config.resolved.yml  # snapshot WEJŚCIA po scaleniu i nadpisaniach
    metrics.csv          # step,split,metric,value  (format długi)
    result.json          # skalary końcowe
    engine/              # artefakty obcych narzędzi (checkpointy, logi)
```

## Użycie

```python
from runlog import Experiment

exp = Experiment("t1", root="runs")

if not exp.is_done(run_id, params):
    with exp.start(
        run_id, params, config=cfg, repos={"repo": Path("."), "lib": Path("external/lib")}
    ) as run:
        for e in range(episodes):
            run.log(step=e, split="train", att=..., loss=...)
        run.result(att=..., att_last10=...)

exp.results()  # DataFrame do tabel
exp.metrics()  # DataFrame długi do wykresów
exp.aggregate(by=["agent"], over="seed", metrics=["att"])  # mean/std/n
```

`start()` zwraca context managera: wyjście bez wyjątku daje status `ok`, wyjątek —
`failed` z tracebackiem w `meta.json`, `KeyboardInterrupt` — `timeout`. Wyjątek nie
jest połykany; status to zapis faktu, nie obsługa błędu.

## Decyzje projektowe

**Kolizje są błędem, nie nadpisaniem.** Ten sam `run_id` z innym `params_hash`
rzuca `RunIdCollision`. Status `ok` lub `running` rzuca `RunExists` (`running` też,
bo może to być żywy proces). Status `failed`/`timeout` pozwala wznowić.
`replace=True` obchodzi to świadomie.

**Format metryk długi**, nie szeroki. Różne metody logują różne metryki (DQN ma
`q_loss`, PPO ma `entropy` i `kl`); szeroki CSV wymagałby znajomości kolumn z góry
i psuł sklejanie biegów. Do podglądu: `df.pivot_table(...)`.

**`n` w agregacji liczy unikalne wartości `over`**, nie wiersze — powtórzony bieg
tego samego seeda nie udaje większej próby.

**Zapis atomowy** (`tempfile` + `os.replace`) i metryki flushowane po każdym kroku:
proces zabity w trakcie nie zostawia obciętego JSON-a, a historia sprzed awarii
przeżywa i da się z niej zdiagnozować, gdzie bieg się wywalił.

**Odczyt jest tolerancyjny.** Uszkodzony `meta.json` daje bieg o statusie `unknown`,
a nie wyjątek w środku skanu po stu katalogach.

**Proweniencja jest obowiązkowa, nie opcjonalna**: SHA + gałąź + flaga `dirty` dla
wielu repozytoriów naraz (repo, submoduły), plus `argv`, wersja Pythona, host
i whitelista zmiennych środowiskowych. Whitelista, nie całe `os.environ` — to
byłby wyciek sekretów i szum.

## Czego tu nie ma

Uruchamiania biegów, kolejkowania, wykresów, serwera. `runlog` przechowuje
i odpytuje; odpalanie siatki eksperymentów jest zadaniem wołającego.
