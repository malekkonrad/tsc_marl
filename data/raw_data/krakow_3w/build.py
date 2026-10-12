"""SUMO scenario of the al. Trzech Wieszczow corridor (Krakow), built from OpenStreetMap.

Steps (each idempotent, all by default):

    osm      roads + tram tracks from the Overpass API -> `3w.osm`. The snapshot is
             VERSIONED: OSM changes weekly and the scenario must reproduce. A new
             snapshot only on request (`--refresh`).
    net      netconvert -> `3w.net.xml`: car traffic only, signals from OSM tags,
             junction clusters joined, fixed-time programs from netconvert's heuristic.
    demand   randomTrips -> `3w_{level}.rou.xml`: SYNTHETIC demand, weights by lane
             count and speed, 3600 s as in the protocol.
    webster  tlsCycleAdaptation (Webster) for EACH demand level separately ->
             `3w_{level}.webster.json` {signal: {phase signal state: green [s]}}, like
             Krakow's practice of time-of-day programs: FixedTime gets a program for
             the demand all controllers see.
    profile  a signal table (approaches, movements, green phases) + a SUMO run on the
             fixed-time programs for every demand level. Writes nothing.

Run from the repository root:  .venv/bin/python data/raw_data/krakow_3w/build.py [steps]
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

# (S, W, N, E): al. Krasinskiego / Mickiewicza / Slowackiego from the Debnicki Bridge to
# ul. Warszawska, Area 1 of Krakow's UTCS (MOTION, priority for road traffic).
BBOX = (50.050, 19.912, 50.077, 19.948)
HIGHWAYS = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service"
)
OVERPASS = "https://overpass-api.de/api/interpreter"

# Period between departures in randomTrips [s], CALIBRATED on 2026-10-09 with a demand
# sweep (netconvert programs, 1 h of traffic, time loss / teleports / insertion queue):
#   1800 veh/h  100 s / 0           -> low:  clearly undersaturated
#   4500 veh/h  141 s / 2           -> mid:  just before the saturation knee
#   5143 veh/h  230 s / 31 (0.6%)   -> high: past the knee, without mass teleports
#   5538 veh/h  302 s / 107 (1.9%), insertion queue 18: teleports start to spoil ATT
#   6000+ veh/h gridlock: 239 -> 5878 teleports, ATT meaningless
DEMAND_PERIOD = {"low": 2.0, "mid": 0.8, "high": 0.7}
# HELD-OUT traffic: the same demand level, another route seed. The policy trains on `mid`
# (seed 0) and its test episode is exactly clean `mid` in `tsc degrade`, so choosing the
# best checkpoint on `mid` would select on the measurement set (`t2_krakow_best`).
DEMAND_HELDOUT = {"mid_s1": ("mid", 1)}


def demand_levels() -> dict[str, tuple[float, int]]:
    """{level: (randomTrips period, route seed)}: main levels + held-out ones."""
    out = {level: (period, SEED) for level, period in DEMAND_PERIOD.items()}
    out |= {name: (DEMAND_PERIOD[base], seed) for name, (base, seed) in DEMAND_HELDOUT.items()}
    return out


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
    import sumo  # SUMO from pip, as in the rest of the repository

    return Path(sumo.__file__).parent


def binary(name: str) -> str:
    os.environ.setdefault("SUMO_HOME", str(sumo_home()))
    import sumolib

    return sumolib.checkBinary(name)


def routes(level: str) -> Path:
    return HERE / f"3w_{level}.rou.xml"


def step_osm(refresh: bool) -> None:
    if OSM.exists() and not refresh:
        print(
            f"osm: {OSM.name} exists ({OSM.stat().st_size // 1024} KiB), skipping "
            "(--refresh forces a new one)"
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
    print(f"osm: wrote {OSM.name} ({OSM.stat().st_size // 1024} KiB)")


def step_net() -> None:
    typemap = sumo_home() / "data" / "typemap" / "osmNetconvert.typ.xml"
    cmd = [binary("netconvert"), "--osm-files", str(OSM), "--type-files", str(typemap)]
    subprocess.run([*cmd, *NETCONVERT_OPTS, "-o", str(NET)], check=True)
    print(f"net: wrote {NET.name}")


def step_demand() -> None:
    random_trips = sumo_home() / "tools" / "randomTrips.py"
    for level, (period, seed) in demand_levels().items():
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
                str(seed),
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
            stderr=subprocess.DEVNULL,  # proj.db: sumolib needs no projection
        )
        trips.unlink(missing_ok=True)  # intermediate file; the routes reproduce it
        print(f"demand: {level:6s} period {period} s, seed {seed} -> {routes(level).name}")


# Webster: lost time per phase = the protocol's transition (3 s yellow + 2 s all-red,
# `sumo_world` / `trainer.yellow_length`), saturation 1800 veh/h/lane (`-H 2`, the
# textbook value). Cycle within the tool's default bounds.
WEBSTER_LOST_TIME = 5
WEBSTER_SAT_HEADWAY = 2.0


def webster(level: str) -> Path:
    return HERE / f"3w_{level}.webster.json"


def step_webster() -> None:
    import json
    import tempfile

    import sumolib

    tool = sumo_home() / "tools" / "tlsCycleAdaptation.py"
    for level in demand_levels():
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
            # key = the signal STATE, not the index: the adapter discovers green phases
            # by simulation (`world_sumo.generate_valid_phase`), so a program index need
            # not be an action index; the same filter here and there, matched by
            # `FixedTimeController`
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
        cycles = sorted(
            int(sum(g.values()) + WEBSTER_LOST_TIME * len(g)) for g in programs.values() if g
        )
        print(
            f"webster: {level:5s} {len(programs)} signals, cycle (greens + transitions) "
            f"min {cycles[0]} / median {cycles[len(cycles) // 2]} / max {cycles[-1]} s "
            f"-> {webster(level).name}"
        )


def step_profile() -> None:
    import sumolib

    net = sumolib.net.readNet(str(NET), withPrograms=True)
    rows = []
    for tls in net.getTrafficLights():
        program = next(iter(tls.getPrograms().values()))
        greens = [ph for ph in program.getPhases() if "y" not in ph.state]
        conns = tls.getConnections()  # (inbound lane, outbound lane, linkIndex)
        incoming = {c[0].getEdge().getID() for c in conns}
        names = sorted({net.getEdge(e).getName() for e in incoming} - {""})
        permissive = any("g" in ph.state for ph in greens)
        rows.append(
            (tls.getID(), len(incoming), len({c[2] for c in conns}), len(greens), permissive, names)
        )

    print(f"signals: {len(rows)}")
    print("  approaches:   ", sorted(collections.Counter(r[1] for r in rows).items()))
    print("  green phases: ", sorted(collections.Counter(r[3] for r in rows).items()))
    print(f"  with permissive green 'g': {sum(r[4] for r in rows)}")
    for tid, n_in, n_links, n_green, perm, names in sorted(rows, key=lambda r: -r[2]):
        print(
            f"  {tid[:34]:34s} approaches={n_in} moves={n_links:3d} phases={n_green} "
            f"g={int(perm)}"
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
        # "Duration" appears twice: SUMO's wall time at the top and the mean trip time in
        # the "Statistics" block; vehicle counts come from the top, the rest from the block
        head, _, stats = out.partition("Statistics")
        found = {
            k: m.group(1)
            for k in keys
            if (m := re.search(rf"^\s*{k}: ([\d.]+)", stats if k in keys[4:] else head, re.M))
        }
        print(
            f"SUMO {level:5s} (fixed-time programs): "
            + ", ".join(f"{k}={v}" for k, v in found.items())
        )


def main(argv: list[str] | None = None) -> None:
    steps = ("osm", "net", "demand", "webster", "profile")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # no `choices`: with nargs="*" argparse 3.10 would reject the empty list too
    p.add_argument("steps", nargs="*", metavar="{" + ",".join(steps) + "}")
    p.add_argument("--refresh", action="store_true", help="fetch a fresh OSM snapshot")
    args = p.parse_args(argv)
    if unknown := set(args.steps) - set(steps):
        p.error(f"unknown steps: {sorted(unknown)}, allowed: {steps}")
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
