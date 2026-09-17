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

# Okres między wyjazdami w randomTrips [s]. Wartości ROBOCZE — do kalibracji tak,
# żeby poziomy szły od niedosyconego do przesyconego (warunek z Planu B notatki).
DEMAND_PERIOD = {"low": 2.0, "mid": 1.2, "high": 0.8}
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
    steps = ("osm", "net", "demand", "profile")
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
    if "profile" in args.steps:
        step_profile()


if __name__ == "__main__":
    main()
