"""Demand variants: an out-of-distribution traffic scenario.

The same road network with another demand isolates ONE shift (a transfer to
another city would shift demand, topology and size at once). `flow.json` in
these datasets is a list of single VEHICLES, so scaling demand means thinning
or densifying the list, chosen by index so a variant is reproducible:

* f < 1 keeps every n-th entry, preserving spatial and temporal mix;
* f > 1 keeps all entries and duplicates every n-th one with a time offset
  (a duplicate at the same second on the same route would not be a real vehicle).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tsc_marl.paths import CONFIGS, DATA


def scale_flow(flow: list[dict[str, Any]], factor: float, offset: float = 2.0) -> list[dict]:
    """The vehicle list scaled `factor` times; `offset` is the duplicate time shift
    in seconds (only for `factor > 1`). Indices `int(i*n/k)` spread removals and
    duplicates EVENLY, so only the intensity changes."""
    if factor <= 0:
        raise ValueError(f"factor={factor}, must be > 0")
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
    out.sort(key=lambda v: v["startTime"])  # stable: entry order preserved
    return out


def tag(factor: float) -> str:
    """`d1p5`, `d0p5`: the same style as failure scenario suffixes."""
    return f"d{factor:g}".replace(".", "p")


def write_variant(network: str, factor: float, *, offset: float = 2.0) -> tuple[Path, Path]:
    """Write a demand variant next to the original -> (flow file, cfg file).
    A new `.cfg` because `-n <network>` is the only scenario selector (LibSignal
    reads the cfg by name), so a variant is a network like any other."""
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
    # LibSignal rewrites the replay log paths on the first run anyway
    out_cfg.write_text(json.dumps(cfg, indent=2))
    return dst, out_cfg
