"""Scenariusz SUMO korytarza al. Trzech Wieszczów (Kraków) budowany z OpenStreetMap.

Case study z gałęzi `cracow-case-study` — rozpoznanie krakowskiego ITS i plan
w `tsc_marl_research/krakow-its.md`. Kroki (każdy idempotentny, domyślnie wszystkie):

    osm      drogi + torowiska z Overpass API → `3w.osm`. Migawka jest WERSJONOWANA:
             OSM zmienia się z tygodnia na tydzień, a scenariusz ma się odtwarzać.
             Nowa migawka tylko na żądanie (`--refresh`).
    net      netconvert → `3w.net.xml`: sam ruch samochodowy (tramwaje dopiero, gdy
             priorytet wejdzie do scenariusza), sygnalizacje z tagów OSM, klastry
             skrzyżowań złączone, programy stałoczasowe z heurystyki netconverta.
    demand   randomTrips → `3w_{poziom}.rou.xml`: popyt SYNTETYCZNY (poziom 0 z notatki —
             bez liczników ZDMK), wagi wg liczby pasów i prędkości, 3600 s jak w protokole.
    webster  tlsCycleAdaptation (Webster) osobno dla KAŻDEGO poziomu popytu →
             `3w_{poziom}.webster.json` {sygnalizacja: {stan sygnałów fazy: zielone [s]}}.
             Odpowiednik krakowskiej praktyki „≥4 programy na pory dnia": każdy poziom
             popytu to inna pora dnia, a FixedTime dostaje program pod TEN popyt, który
             widzą wszystkie sterowniki (bez tego porównanie byłoby chochołem).
    profile  tabela sygnalizacji (wloty, ruchy, fazy zielone) + bieg SUMO na programach
             stałoczasowych dla każdego poziomu popytu. Nic nie zapisuje.

Uruchomienie z korzenia repo:  .venv/bin/python data/raw_data/krakow_3w/build.py [kroki]
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OSM = HERE / "3w.osm"
NET = HERE / "3w.net.xml"

# (S, W, N, E): al. Krasińskiego / Mickiewicza / Słowackiego od Mostu Dębnickiego
# po ul. Warszawską — Obszar 1 krakowskiego UTCS (MOTION, preferencja ruchu kołowego).
BBOX = (50.050, 19.912, 50.077, 19.948)
HIGHWAYS = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service"
)
OVERPASS = "https://overpass-api.de/api/interpreter"

# Okres między wyjazdami w randomTrips [s] — SKALIBROWANE 2026-10-09 sweepem popytu
# (programy netconverta, 1 h ruchu, strata czasu / teleporty / kolejka na wjazd):
#   1800 poj/h  100 s / 0           → low:  wyraźnie niedosycony
#   4500 poj/h  141 s / 2           → mid:  tuż przed kolanem nasycenia
#   5143 poj/h  230 s / 31 (0,6%)   → high: za kolanem, bez masowych teleportów
#   5538 poj/h  302 s / 107 (1,9%), kolejka na wjazd 18 — teleporty zaczynają psuć ATT
#   6000+ poj/h zakleszczenie: 239 → 5878 teleportów, ATT bez znaczenia
# Warunek z Planu B notatki: poziomy od niedosyconego do przesyconego.
DEMAND_PERIOD = {"low": 2.0, "mid": 0.8, "high": 0.7}
DEMAND_END = 3600
SEED = 0

NETCONVERT_OPTS = [
    "--keep-edges.by-vclass",
    "passenger",
    "--remove-edges.isolated",
    "--geometry.remove",
    "--ramps.guess",
    "--junctions.join",
    "--junctions.corner-detail",
    "5",
    "--roundabouts.guess",
    "--tls.guess-signals",
    "--tls.discard-simple",
    "--tls.join",
    "--tls.default-type",
    "static",
    "--output.street-names",
    "--no-turnarounds",
]


def sumo_home() -> Path:
    if env := os.environ.get("SUMO_HOME"):
        return Path(env)
    import sumo  # noqa: PLC0415 - SUMO z pipa, jak w reszcie repo

    return Path(sumo.__file__).parent


def binary(name: str) -> str:
    os.environ.setdefault("SUMO_HOME", str(sumo_home()))
    import sumolib  # noqa: PLC0415

    return sumolib.checkBinary(name)


def routes(level: str) -> Path:
    return HERE / f"3w_{level}.rou.xml"


def step_osm(refresh: bool) -> None:
    if OSM.exists() and not refresh:
        print(
            f"osm: jest {OSM.name} ({OSM.stat().st_size // 1024} KiB), pomijam (--refresh wymusza)"
        )
        return
    s, w, n, e = BBOX
    box = f"({s},{w},{n},{e})"
    query = (
        f'[out:xml][timeout:120];(way["highway"~"^({HIGHWAYS})(_link)?$"]{box};'
        f'way["railway"="tram"]{box};);(._;>;);out body;'
    )
    data = urllib.parse.urlencode({"data": query}).encode()
    with urllib.request.urlopen(OVERPASS, data=data, timeout=180) as resp:
        OSM.write_bytes(resp.read())
    print(f"osm: zapisano {OSM.name} ({OSM.stat().st_size // 1024} KiB)")


def step_net() -> None:
    typemap = sumo_home() / "data" / "typemap" / "osmNetconvert.typ.xml"
    cmd = [binary("netconvert"), "--osm-files", str(OSM), "--type-files", str(typemap)]
    subprocess.run([*cmd, *NETCONVERT_OPTS, "-o", str(NET)], check=True)
    print(f"net: zapisano {NET.name}")


def step_demand() -> None:
    random_trips = sumo_home() / "tools" / "randomTrips.py"
    for level, period in DEMAND_PERIOD.items():
        trips = HERE / f"3w_{level}.trips.xml"
        subprocess.run(
            [
                sys.executable,
                str(random_trips),
                "-n",
                str(NET),
                "-o",
                str(trips),
                "-r",
                str(routes(level)),
                "-e",
                str(DEMAND_END),
                "-p",
                str(period),
                "--seed",
                str(SEED),
                "--fringe-factor",
                "10",
                "--lanes",
                "--speed-exponent",
                "2",
                "--min-distance",
                "300",
                "--validate",
            ],
            check=True,
            stderr=subprocess.DEVNULL,  # proj.db: sumolib nie potrzebuje projekcji
        )
        trips.unlink(missing_ok=True)  # pośredni plik — routes wystarczą do odtworzenia
        print(f"demand: {level:5s} okres {period} s → {routes(level).name}")


# Webster: strata czasu na fazę = przejście z protokołu (3 s żółtego + 2 s all-red,
# `sumo_world` / `trainer.yellow_length`), nasycenie 1800 poj/h/pas (`-H 2`, wartość
# podręcznikowa). Cykl w granicach domyślnych narzędzia.
WEBSTER_LOST_TIME = 5
WEBSTER_SAT_HEADWAY = 2.0


def webster(level: str) -> Path:
    return HERE / f"3w_{level}.webster.json"


def step_webster() -> None:
    import json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import sumolib  # noqa: PLC0415

    tool = sumo_home() / "tools" / "tlsCycleAdaptation.py"
    for level in DEMAND_PERIOD:
        with tempfile.TemporaryDirectory() as tmp:
            add = Path(tmp) / "webster.add.xml"
            subprocess.run(
                [
                    sys.executable,
                    str(tool),
                    "-n",
                    str(NET),
                    "-r",
                    str(routes(level)),
                    "-b",
                    "0",
                    "-y",
                    "3",
                    "-l",
                    str(WEBSTER_LOST_TIME),
                    "-H",
                    str(WEBSTER_SAT_HEADWAY),
                    "-o",
                    str(add),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Klucz = STAN sygnałów, nie indeks: adapter odkrywa fazy zielone symulacją
            # (`world_sumo.generate_valid_phase`: unikalne stany bez `y` i bez samych
            # `r`/`s`), więc indeks fazy w programie nie musi być indeksem akcji.
            # Ten sam filtr tu i tam — dopasowanie robi `FixedTimeController`.
            programs = {
                tl.id: {
                    ph.state: float(ph.duration)
                    for ph in tl.phase
                    if "y" not in ph.state
                    and ph.state.count("r") + ph.state.count("s") != len(ph.state)
                }
                for tl in sumolib.xml.parse(str(add), "tlLogic")
            }
        webster(level).write_text(json.dumps(programs, indent=1, sort_keys=True))
        cykle = sorted(
            int(sum(g.values()) + WEBSTER_LOST_TIME * len(g)) for g in programs.values() if g
        )
        print(
            f"webster: {level:5s} {len(programs)} sygnalizacji, cykl (zielone + przejścia) "
            f"min {cykle[0]} / mediana {cykle[len(cykle) // 2]} / max {cykle[-1]} s "
            f"→ {webster(level).name}"
        )


def step_profile() -> None:
    import sumolib  # noqa: PLC0415

    net = sumolib.net.readNet(str(NET), withPrograms=True)
    rows = []
    for tls in net.getTrafficLights():
        program = next(iter(tls.getPrograms().values()))
        greens = [ph for ph in program.getPhases() if "y" not in ph.state]
        conns = tls.getConnections()  # (pas wjazdowy, pas wyjazdowy, linkIndex)
        incoming = {c[0].getEdge().getID() for c in conns}
        names = sorted({net.getEdge(e).getName() for e in incoming} - {""})
        permissive = any("g" in ph.state for ph in greens)
        rows.append(
            (tls.getID(), len(incoming), len({c[2] for c in conns}), len(greens), permissive, names)
        )

    print(f"sygnalizacje: {len(rows)}")
    print("  wloty:        ", sorted(collections.Counter(r[1] for r in rows).items()))
    print("  fazy zielone: ", sorted(collections.Counter(r[3] for r in rows).items()))
    print(f"  z zielonym warunkowym 'g': {sum(r[4] for r in rows)}")
    for tid, n_in, n_links, n_green, perm, names in sorted(rows, key=lambda r: -r[2]):
        print(
            f"  {tid[:34]:34s} wloty={n_in} ruchy={n_links:3d} fazy={n_green} g={int(perm)}"
            f" | {', '.join(names)[:70]}"
        )

    keys = ("Inserted", "Running", "Waiting", "Teleports", "Duration", "WaitingTime", "TimeLoss")
    for level in DEMAND_PERIOD:
        out = subprocess.run(
            [
                binary("sumo"),
                "-n",
                str(NET),
                "-r",
                str(routes(level)),
                "-e",
                str(DEMAND_END + 1800),
                "--no-step-log",
                "--duration-log.statistics",
                "--time-to-teleport",
                "300",
                "--seed",
                str(SEED),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        # „Duration" występuje dwa razy: czas zegarowy SUMO na górze i średni czas
        # przejazdu w bloku „Statistics" — liczby pojazdów bierzemy z góry, resztę z bloku
        head, _, stats = out.partition("Statistics")
        found = {
            k: m.group(1)
            for k in keys
            if (m := re.search(rf"^\s*{k}: ([\d.]+)", stats if k in keys[4:] else head, re.M))
        }
        print(
            f"SUMO {level:5s} (programy stałoczasowe): "
            + ", ".join(f"{k}={v}" for k, v in found.items())
        )


def main(argv: list[str] | None = None) -> None:
    steps = ("osm", "net", "demand", "webster", "profile")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # bez `choices`: argparse 3.10 przy nargs="*" odrzuca wtedy także pustą listę
    p.add_argument("steps", nargs="*", metavar="{" + ",".join(steps) + "}")
    p.add_argument("--refresh", action="store_true", help="pobierz świeżą migawkę OSM")
    args = p.parse_args(argv)
    if unknown := set(args.steps) - set(steps):
        p.error(f"nieznane kroki: {sorted(unknown)}, dozwolone: {steps}")
    args.steps = args.steps or list(steps)
    if "osm" in args.steps:
        step_osm(args.refresh)
    if "net" in args.steps:
        step_net()
    if "demand" in args.steps:
        step_demand()
    if "webster" in args.steps:
        step_webster()
    if "profile" in args.steps:
        step_profile()


if __name__ == "__main__":
    main()
