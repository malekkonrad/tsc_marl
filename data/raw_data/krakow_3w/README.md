# krakow_3w: the al. Trzech Wieszczow corridor (Krakow), SUMO

Case study from the `cracow-case-study` branch. Survey of Krakow's ITS, assessment and
plan: `tsc_marl_research/krakow-its.md`. Area = Area 1 of Krakow's UTCS
(al. Krasinskiego / Mickiewicza / Slowackiego + 29 Listopada; Siemens MOTION, priority
for road traffic), bbox `(50.050, 19.912, 50.077, 19.948)`.

## Build

```bash
.venv/bin/python data/raw_data/krakow_3w/build.py            # osm -> net -> demand -> webster -> profile
.venv/bin/python data/raw_data/krakow_3w/build.py demand webster  # demand + Webster programs only
.venv/bin/python data/raw_data/krakow_3w/build.py profile    # diagnostics only
.venv/bin/python data/raw_data/krakow_3w/build.py --refresh  # a new OSM snapshot
```

| file | what | reproducible from |
|---|---|---|
| `3w.osm` | OSM snapshot (roads + tram tracks), 2026-09-13 | Overpass (`--refresh` changes the result!) |
| `3w.net.xml` | SUMO network, car traffic only | `3w.osm` + `build.py net` |
| `3w_{low,mid,high}.rou.xml` | synthetic demand, 3600 s, seed 0; levels calibrated (below) | `3w.net.xml` + `build.py demand` |
| `3w_{low,mid,high}.webster.json` | Webster fixed-time program per level: {signal: {phase state: green [s]}} | `.rou.xml` + `build.py webster` |

OSM data: (c) OpenStreetMap contributors, ODbL licence.

Configs: `configs/sim/sumo_krakow3w_{low,mid,high}.cfg`, e.g.

```bash
.venv/bin/tsc train -a maxpressure -w sumo -n sumo_krakow3w_mid --seed 0   # baseline
.venv/bin/tsc core -w sumo -n sumo_krakow3w_mid --seed 0                   # GAT+PPO core
```

Rule gate threshold (a network without an uncertainty head), `runs/heads/rule_ref__<network>.json`:

```bash
.venv/bin/tsc rulecal -n sumo_krakow3w_mid      # 99.5% quantile of the frozen-movement share, clean traffic, Advanced-MP
```

| level | threshold | false alarms on clean (budget 0.5%) |
|---|---:|---:|
| low | 1.00 | **2.8%** |
| mid | 0.79 | 0.52% |
| high | 0.79 | 0.63% |

On `low` the budget CANNOT be met: with little traffic whole intersections are sometimes
empty (all real movements stand still), and the rule cannot tell empty from dead, so the
quantile sits at 1.0. The gate will sometimes hand empty nodes to Webster there; the cost
should be small (an empty node), but it must show in the `clean` column rather than be
assumed. On mid/high the slight overrun is the discreteness of shares (a node has 1-7
approaches, so the share takes few values).

## Demand levels (calibrated 2026-10-09)

A sweep of the `randomTrips` period on netconvert programs, 1 h of traffic (`DEMAND_PERIOD`
in `build.py`):

| demand | time loss | teleports | insertion queue | |
|---:|---:|---:|---:|---|
| 1800 veh/h | 100 s | 0 | 0 | **low**: clearly undersaturated |
| 3000 veh/h | 119 s | 21 | 0 | (former mid; teleports from a gridlock on Pradnicka) |
| 4500 veh/h | 141 s | 2 | 0 | **mid**: just before the saturation knee |
| 5143 veh/h | 230 s | 31 (0.6%) | 0 | **high**: past the knee, without mass teleports |
| 5538 veh/h | 302 s | 107 (1.9%) | 18 | teleports start to spoil ATT |
| 6000 veh/h | 322 s | 239 | 26 | gridlock |
| 8000+ veh/h | 492+ s | 1482+ | 370+ | ATT meaningless |

Teleports at the former mid: 10 of 21 on a 7 m stretch of ul. Pradnicka (edge
`117386961`, an unsignalised junction, 1 -> 2 -> 1 lanes), an OSM import artefact outside
the controlled intersections; it does not occur at the current levels.

## FixedTime = Webster per demand level

`build.py webster` -> SUMO `tlsCycleAdaptation` (Webster: C0 = (1.5L + 5)/(1 - Y), greens
proportional to critical flows), lost time 5 s/phase (= the protocol's transition: 3 s
yellow + 2 s all-red), saturation 1800 veh/h/lane. Each demand level is another time of
day, the counterpart of Krakow's practice of ">=4 time-of-day programs", and FixedTime gets
the program for THE demand all controllers see.

The program enters through the `fixedTimeProgram` key in `configs/sim/sumo_krakow3w_*.cfg`:
`tsc degrade` builds both the `fixedtime` column and the gate's switch target from it
(`runner/degrade.fixed_time_program`). Matching is by the phase's signal STATE, not the
index, since the adapter discovers phases by simulation. LibSignal's `fixedtime` agent
(`tsc train -a fixedtime`) does NOT read the program; take FixedTime results for the
thesis from `tsc degrade`.

## Baselines after calibration (2026-10-09, `tsc degrade` path, clean traffic, seed 0)

ATT [s] / throughput (vehicles that finished):

| | low (1800) | mid (4500) | high (5143) |
|---|---:|---:|---:|
| FixedTime `t_fixed: 30` | 331.4 / 1618 | 390.2 / 3941 | 471.8 / 4111 |
| **FixedTime Webster** | 297.9 / 1645 | **347.6 / 3972** | **390.5 / 4374** |
| Efficient-MP | 282.6 / 1651 | 373.5 / 3846 | 477.2 / 3928 |
| Advanced-MP | **279.8 / 1655** | 396.9 / 3676 | 481.5 / 3840 |

At low traffic the pressure methods win (-6% ATT); at `mid`/`high` they LOSE to Webster by
7-23% and lose throughput. The old comparison with `t_fixed: 30` hid this completely. A
hypothesis to check (the README of 2026-09-13 saw the same at the former `high`): short
stretches between intersection clusters + `t_min` -> spillback local pressure cannot see;
Webster coordinates through the green split alone.

## The SUMO world = the adapter, not upstream

Every `-w sumo` run goes through `src/tsc_marl/libsignal/sumo_world.py`, which overrides
upstream `world_sumo.World` in the Registry, for the core and LibSignal's baselines alike.
The module docstring lists nine differences. The most important for the numbers:

- **phase transitions as in CityFlow**: after the first seconds of an episode upstream
  switched phases immediately, without a transition phase (measured: 0 transition steps
  after second 30), so `yellow` and `t_min` meant nothing,
- **ATT as in CityFlow** (finished + still running): upstream counted finished ones only,
- **`lane_waiting_count` = halted now**: this is the core's reward,
- **permissive green `g` in phases**: 32 lanelinks at 13 signals,
- **a road-based neighbour graph** (`build_graph`, reach 600 m): LibSignal's graph had
  1 edge for 25 nodes here, ours has 51 (in-degree 0-4).

**`-w sumo` results from before the adapter are invalid** (including the table in the first
version of this README).

## Network profile

25 signals, strongly heterogeneous, exactly what JiNan/HangZhou lack:

- approaches: 1-7 (1x1, 6x2, 6x3, 6x4, 3x5, 2x6, 1x7),
- green phases: 2-7 (13x2, 7x3, 3x4, 1x5, 1x7) -> the core masks missing phases,
- 13 of 25 have permissive green `g` (left turn yielding),
- several are pedestrian crossings only (1-2 approaches, 4 movements),
- 8 signals control several nodes (`joinedS_...` clusters).

## Baseline results: OLD (2026-09-13, before demand calibration and Webster)

WARNING: the demand levels and FixedTime of this table are out of date; the new table is
above (section "Baselines after calibration").

| demand | vehicles | FixedTime | MaxPressure | Advanced-MP | throughput FT / MP / AMP |
|---|---|---|---|---|---|
| low  | 1800 | 331.4 | 274.4 | 282.0 | 1618 / 1656 / 1660 |
| mid  | 3000 | 336.7 | 276.2 | 283.0 | 2716 / 2756 / 2755 |
| high | 4500 | 390.2 | 394.5 | 466.6 | 3941 / **3662** / **3465** |

ATT in seconds. Reading: at low/mid MaxPressure beats FixedTime by ~18%, Advanced-MP by
~16%, at the same throughput. At high the pressure methods let fewer vehicles through
(MP -7%, Advanced-MP -12%), and Advanced-MP has the worst ATT; to investigate (spillback on
short stretches? oversaturation?). Always report ATT together with throughput.

GAT+PPO core: a smoke test only so far (2 episodes x 600 s, ~24 s), it works, no result.
A full run (200 episodes x 3600 s) is estimated at ~3 h on a laptop CPU.

## Known limitations (to solve before results for the thesis)

1. ~~Demand not calibrated~~: calibrated 2026-10-09 (section "Demand levels").
2. ~~FixedTime is `t_fixed: 30`~~: Webster per level since 2026-10-09 (section "FixedTime").
3. **Signal set = the whole bbox**, including Kijowska, Dietla, Pawia and pedestrian
   crossings. To decide: cut to the corridor (`--tls.unset` for the rest?) or keep.
4. **Graph:** 4 of 25 nodes without neighbours at a 600 m reach (bbox edge, pedestrian
   crossings); the GAT sees only itself there. The reach is an ablation slot.
5. **No trams** (removed in `net`); tram priority out of scope for now.
