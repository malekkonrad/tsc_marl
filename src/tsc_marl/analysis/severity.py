"""Failure severity converter: level in the SPECIFICATION -> deviation on the FEATURE.

RobustLight uses an EP + running state (our ATS) with a 3.5-4.0 scale for
`gauss`/`urand`, but injects noise into the STATE, while `domain.sensors` injects it
at the DETECTOR. e(l,m) is a DIFFERENCE OF MEANS over inbound and outbound lanes, so
the two numbers differ. The conversion is measured EMPIRICALLY: M differs between
movements, counts are clipped at zero, and masks or drift have no "sigma" at all.

    tsc severity                      # JiNan, default grid -> results/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tsc_marl import paths
from tsc_marl.domain.sensors import FailureSpec

#: Scenario grid = what appears in T2; `gauss` at several levels to see whether the
#: conversion is linear.
DEFAULT_SPECS = (
    FailureSpec(kind="gauss", level=1.0),
    FailureSpec(kind="gauss", level=2.0),
    FailureSpec(kind="gauss", level=3.5),
    FailureSpec(kind="gauss", level=4.0),
    FailureSpec(kind="urand", level=3.5),
    FailureSpec(kind="urand", level=4.0),
    FailureSpec(kind="mask", level=0.25),
    FailureSpec(kind="mask", level=0.5),
    FailureSpec(kind="mask", level=0.25, unit="intersection"),
    FailureSpec(kind="drift", level=0.2),
    FailureSpec(kind="drift", level=0.3),
)


@dataclass(frozen=True)
class Severity:
    """Severity of one scenario, separately for both ATS feature blocks."""

    spec: FailureSpec
    ep_std: float  # standard deviation of (broken - clean) on the EP block
    ep_mad: float  # mean absolute deviation
    demand_std: float
    demand_mad: float
    touched: float  # fraction of cells that changed at all

    @property
    def transfer(self) -> float | None:
        """ep_std / level, the "sigma at the detector -> sigma on the feature" factor;
        meaningful for additive noise only (for masks and drift `level` means a
        fraction of units or a relative miscalibration)."""
        if self.spec.kind not in ("gauss", "urand") or not self.spec.level:
            return None
        return self.ep_std / self.spec.level


def measure(
    network: str = "cityflow_jinan3x4_4ph",
    specs: tuple[FailureSpec, ...] = DEFAULT_SPECS,
    *,
    decisions: int = 60,
    warmup: int = 300,
    action_interval: int = 15,
) -> tuple[list[Severity], tuple[float, float]]:
    """All scenarios on THE SAME engine states: one world, one `StateBuilder` per
    scenario plus a clean one, so rows differ only by the scenario. Control during
    the measurement cycles the phase every decision; it only has to fill the network.
    """
    from tsc_marl.libsignal import session

    with session.standalone_world("advancedcolight", network, n_agents=1) as (world, _, _):
        return _measure(world, specs, decisions, warmup, action_interval)


def _measure(world, specs, decisions, warmup, action_interval):
    from tsc_marl.core.features import StateBuilder

    inters = world.intersections
    n_phases = max(len(i.phases) for i in inters)
    clean_builder = StateBuilder.from_registry(world, inters, failure=None)
    builders = [StateBuilder.from_registry(world, inters, failure=s) for s in specs]
    n_mov = clean_builder.n_mov

    diffs: list[list[np.ndarray]] = [[] for _ in specs]
    clean_rows: list[np.ndarray] = []
    total = warmup + decisions * action_interval
    for step in range(total):
        if step >= warmup and (step - warmup) % action_interval == 0:
            clean = clean_builder.observe()
            clean_rows.append(clean)
            for acc, b in zip(diffs, builders, strict=True):
                acc.append(b.observe() - clean)
        world.step([(step // action_interval) % len(i.phases) for i in inters])

    rows = [
        _summarize(spec, np.concatenate(acc), n_mov, n_phases)
        for spec, acc in zip(specs, diffs, strict=True)
    ]
    return rows, _zero_share(np.concatenate(clean_rows), n_mov)


def _zero_share(clean: np.ndarray, n_mov: int) -> tuple[float, float]:
    """Fraction of EXACT ZEROS in both feature blocks on a clean run. Zero is the
    MODAL value, so a dead detector reporting 0 looks like a healthy one on an empty
    approach, and mask severity depends on network LOAD."""
    ep, demand = clean[:, :n_mov], clean[:, n_mov : 2 * n_mov]
    return float((ep == 0).mean()), float((demand == 0).mean())


def _summarize(spec: FailureSpec, diffs: np.ndarray, n_mov: int, n_phases: int) -> Severity:
    ep, demand = diffs[:, :n_mov], diffs[:, n_mov : 2 * n_mov]
    body = diffs[:, :-n_phases]
    return Severity(
        spec=spec,
        ep_std=float(ep.std()),
        ep_mad=float(np.abs(ep).mean()),
        demand_std=float(demand.std()),
        demand_mad=float(np.abs(demand).mean()),
        touched=float((np.abs(body) > 1e-9).mean()),
    )


def to_markdown(rows: list[Severity], zero_share: tuple[float, float], network: str) -> str:
    out = [
        "# Failure severity converter (level in the specification -> deviation on the feature)",
        "",
        f"Network: `{network}`. State: ATS (EP block + demand block), window `obs_length`.",
        "Column **factor** = `ep_std / level`: how much weaker or stronger noise injected AT THE",
        "DETECTOR is on the feature the policy sees. Without it the RobustLight levels",
        "(3.5-4.0 on the STATE) are not comparable with ours.",
        "",
        "| scenario | level | unit | EP std | EP MAD | demand std | demand MAD |"
        " changed | factor |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        t = f"{r.transfer:.3f}" if r.transfer is not None else "-"
        out.append(
            f"| {r.spec.kind} | {r.spec.level:g} | {r.spec.unit} | {r.ep_std:.3f} | "
            f"{r.ep_mad:.3f} | {r.demand_std:.3f} | {r.demand_mad:.3f} | "
            f"{r.touched:.1%} | {t} |"
        )
    ep_zero, demand_zero = zero_share
    out += [
        "",
        "## How to read it",
        "",
        f"On a clean run **{ep_zero:.0%} of EP block cells and {demand_zero:.0%} of demand "
        "block cells are exactly zero**. Two conclusions follow, both relevant for T2:",
        "",
        "1. **A mask is largely a no-op.** Zero is the MODAL value, so a dead detector",
        "   reporting 0 looks like a healthy detector on an empty approach; the *changed*",
        "   column is far below the fraction of broken lanes. Not a measurement fault but a",
        "   property of `zero` substitution, which makes the mask a SILENT failure (see",
        "   `domain/sensors.py`: the substitution is not neutral).",
        "2. **Mask severity depends on network LOAD.** The same mask percentage does not mean",
        "   the same thing on JiNan and HangZhou (6295 vs 2983 veh/h); the thesis must give",
        "   the fraction of changed cells next to the mask percentage, or the degradation",
        "   curves of the two networks are not comparable.",
        "",
        "For additive noise the factor has a predictable source: e(l,m) is a difference of",
        "MEANS over inbound and outbound lanes, so with M lanes on each side the sigma on the",
        "feature is sigma*sqrt(1/M_in + 1/M_out), 0.816 for M=3. Deviations come from",
        "movements with another M and from clipping at zero (`max(0, ...)`), which eats more",
        "mass at larger sigma, hence the factor DECREASES with the level.",
        "`urand` at the same `level` is markedly WEAKER (width a gives sigma = a/sqrt(3)),",
        'so comparing the two attacks "by the same number" is misleading.',
    ]
    return "\n".join(out) + "\n"


def main(network: str = "cityflow_jinan3x4_4ph") -> Path:
    rows, zero_share = measure(network)
    path = paths.RESULTS / "sensor_severity.md"
    path.write_text(to_markdown(rows, zero_share, network), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(main())
