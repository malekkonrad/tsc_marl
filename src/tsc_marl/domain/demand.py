"""Warianty POPYTU — scenariusz OOD dla epistemic (bramka 2.0 b).

Bramka (b) wymaga stanu ruchu NIEWIDZIANEGO w treningu: epistemic ma rosnąć na
OOD, aleatoric na szumie wejścia. Bez osobnego scenariusza popytu nie ma czego
zmierzyć — czysty benchmark to jedna, ustalona trajektoria (`flow.json` jest
deterministyczny, `--seed` go nie rusza).

DLACZEGO SKALOWANIE POPYTU, A NIE TRANSFER JiNan→HangZhou: transfer miesza trzy
przesunięcia naraz (popyt, topologia, rozmiar sieci), więc wzrost epistemic
niczego nie lokalizuje. Ten sam roadnet i inny popyt izoluje JEDNĄ przyczynę.

JAK: `flow.json` w tych datasetach to lista pojedynczych POJAZDÓW (`startTime ==
endTime`, `interval` nieużywane), nie strumieni — więc skalowanie natężenia to
przerzedzanie albo zagęszczanie listy, a nie mnożenie parametru.
  * f < 1 — deterministyczny co n-ty wpis. Przerzedza, ZACHOWUJĄC mieszankę
    przestrzenną i czasową (losowy podzbiór zmieniłby też rozkład tras).
  * f > 1 — wszystkie wpisy plus co n-ty zduplikowany, z przesunięciem czasu.
    Duplikat o TYM SAMYM `startTime` na tej samej trasie to dwa pojazdy w jednym
    punkcie w tej samej sekundzie — CityFlow wstawia je jeden za drugim albo
    gubi, więc przesunięcie jest warunkiem, żeby zagęszczenie było prawdziwe.
Wybór po indeksie, nie losowy: wariant ma być odtwarzalny bez zapisywania ziarna.

ZMIERZONE OGRANICZENIE (JiNan 3×4, 4 fazy, Advanced-MP): **da się tylko w dół.**
Przy f = 1,0 sieć stoi tuż pod granicą przepustowości — 585 pojazdów w sieci
w stanie ustalonym, ATT 299,2. Już f = 1,1 przewraca ją w korek: 4032 pojazdy
w sieci po 600 s (7× więcej z 10% większego popytu — podręcznikowy wybuch kolejki
przy ρ→1), symulacja zwalnia ~50× i epizod nie kończy się w rozsądnym czasie.
To nie jest artefakt duplikowania (plik ma dokładnie 1,1× wpisów, bez wpisów
identycznych), tylko własność benchmarku — wariant 4-fazowy ma za mało
przepustowości. Dlatego OOD budujemy z popytu OBNIŻONEGO (0,25 / 0,5 / 0,75),
co daje stopniowaną skalę oddalenia od rozkładu treningowego, a wersje f > 1
zostały skasowane, żeby nikt ich przez pomyłkę nie policzył.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tsc_marl.paths import CONFIGS, DATA


def scale_flow(flow: list[dict[str, Any]], factor: float, offset: float = 2.0) -> list[dict]:
    """Lista pojazdów przeskalowana `factor` razy. `offset` = przesunięcie czasu
    duplikatu w sekundach (tylko dla `factor > 1`).

    Wybór indeksów przez `int(i·n/k)` rozkłada usunięcia i duplikaty RÓWNOMIERNIE
    po całej liście, więc wariant nie zmienia ani rozkładu tras, ani profilu
    czasowego — tylko natężenie. Losowy podzbiór zmieniałby oba.
    """
    if factor <= 0:
        raise ValueError(f"factor={factor}, musi być > 0")
    n = len(flow)
    if factor == 1 or n == 0:
        return [dict(v) for v in flow]

    if factor < 1:
        keep = round(n * factor)
        picks = {int(i * n / keep) for i in range(keep)} if keep else set()
        return [dict(v) for i, v in enumerate(flow) if i in picks]

    extra = round(n * (factor - 1))
    counts = Counter(int(i * n / extra) for i in range(extra))
    out: list[dict] = []
    for i, v in enumerate(flow):
        out.append(dict(v))
        for c in range(counts.get(i, 0)):
            dup = dict(v)
            dup["startTime"] = v["startTime"] + offset * (c + 1)
            dup["endTime"] = v["endTime"] + offset * (c + 1)
            out.append(dup)
    out.sort(key=lambda v: v["startTime"])  # stabilny — kolejność wpisów zachowana
    return out


def tag(factor: float) -> str:
    """`d1p5`, `d0p5` — ten sam styl co sufiksy scenariuszy awarii."""
    return f"d{factor:g}".replace(".", "p")


def write_variant(network: str, factor: float, *, offset: float = 2.0) -> tuple[Path, Path]:
    """Zapisz wariant popytu obok oryginału. -> (plik flow, plik cfg).

    Nowy `.cfg` zamiast nadpisania pola w locie, bo `-n <sieć>` jest jedynym
    wejściem do wyboru scenariusza w całym repo (LibSignal czyta cfg po nazwie),
    a wariant popytu ma być tak samo cytowalny w tabeli jak każda inna sieć.
    """
    cfg_path = CONFIGS / "sim" / f"{network}.cfg"
    cfg = json.loads(cfg_path.read_text())
    src = DATA / cfg["flowFile"]

    suffix = tag(factor)
    flow = scale_flow(json.loads(src.read_text()), factor, offset)
    dst = src.with_name(f"{src.stem}_{suffix}{src.suffix}")
    dst.write_text(json.dumps(flow))

    out_cfg = cfg_path.with_name(f"{network}_{suffix}.cfg")
    cfg["flowFile"] = str(dst.relative_to(DATA))
    cfg["network"] = f"{cfg.get('network', network)}_{suffix}"
    # ścieżki logów replaya LibSignal i tak nadpisze przy pierwszym biegu
    out_cfg.write_text(json.dumps(cfg, indent=2))
    return dst, out_cfg
