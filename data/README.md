# data/

Input simulation scenarios. Run outputs do **not** go here but to `runs/` (see
`logger.root_dir` in `configs/tsc/base.yml`).

## What is versioned

Only the networks used in the thesis (~25 MB). The rest of LibSignal's datasets
(~300 MB: Manhattan, arterial, Cologne, ...) stays upstream and is not copied here.

| directory | network | source |
|---|---|---|
| `raw_data/jinan_3x4/` | JiNan 3x4 (CityFlow); traffic: real, 2000, 2500, demand x0.25/0.5/0.75 | CoLight / Advanced-XLight line of work; demand variants from `tsc demand` |
| `raw_data/hangzhou_4x4_advanced/` | HangZhou 4x4 (CityFlow); traffic: real, 5816 | as above |
| `raw_data/krakow_3w/` | Krakow, al. Trzech Wieszczow (SUMO, 25 signals); demand low / mid / high + held-out `mid_s1` | OSM + `build.py`, see its [README](raw_data/krakow_3w/README.md) |

Each CityFlow network has a 4-phase variant (`roadnet_*_4phase.json`). The thesis
protocol uses it, because the reference papers report their results on it. Networks
are wired in through `configs/sim/*.cfg` (`roadnetFile` / `flowFile` fields, paths
relative to `data/`).

## Formats

- **CityFlow**: `roadnet.json` + `flow.json` + a config pointing to both.
- **SUMO**: `*.net.xml` + `*.rou.xml`; the Krakow network also has a fixed-time
  Webster program per demand level (`*.webster.json`).

Large locally generated networks are ignored by `.gitignore`; commit only the
scenarios whose results go into the thesis, so they can be reproduced.
