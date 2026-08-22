# data/

Wejściowe scenariusze symulacji. Wyniki uruchomień **nie** trafiają tutaj, tylko do
`runs/` (patrz `logger.root_dir` w `configs/tsc/base.yml`).

## Co jest wersjonowane

Tylko sieci faktycznie używane w pracy (9,4 MB) — reszta zbiorów LibSignala
(~300 MB: Manhattan, arterial, cologne, …) zostaje w upstreamie i nie jest tu kopiowana:

| katalog | sieć | źródło |
|---|---|---|
| `raw_data/jinan_3x4/` | JiNan 3×4, warianty ruchu real / 2000 / 2500 | dane z linii CoLight / Advanced-XLight |
| `raw_data/hangzhou_4x4_advanced/` | HangZhou 4×4, warianty real / 5816 | jw. |

Każda sieć ma wariant 4-fazowy (`roadnet_*_4phase.json`) — to jego używa protokół
pracy, bo w nim liczą swoje wyniki papiery referencyjne. Sieci podpina się przez
`configs/sim/*.cfg` (pola `roadnetFile` / `flowFile`, ścieżki względem `data/`).

## Formaty

- **CityFlow**: `roadnet.json` + `flow.json` + config wskazujący na oba.
- **SUMO**: `*.net.xml` + `*.rou.xml` (+ `*.sumocfg`). Konwersja z CityFlow:
  `python external/LibSignal/../CityFlow/tools/Converter/converter.py`.

Duże sieci generowane lokalnie są ignorowane przez `.gitignore` — commituj wyłącznie
scenariusze, których wyniki trafiają do pracy, żeby dało się je odtworzyć.
