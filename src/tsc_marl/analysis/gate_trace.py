"""Figure EXPLAINING gate triggers, per (decision, node) rather than an aggregate.

`gate_share` says HOW MANY pairs the rule took; the question an operator asks is
WHICH intersection handed over control and WHEN. The answer lives in `gate_trace.csv`
(`tsc degrade run <plan> --trace`): the raw rule score, the mask before hysteresis
and the mode after it, for every pair. Fill = control mode, a marker = the detector
fired but hysteresis held the switch.

    python -m tsc_marl.analysis.gate_trace t2_trace
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tsc_marl.paths import RESULTS, RUNS
from tsc_marl.robustness.degrade import Plan

# palette as in `analysis/degradation.py`
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GATE = "#7a4fd6"  # the `gatppo_gate` slot of the mechanism palette

#: Scenario labels, short, since they go on the panel axes.
LABELS = {
    "clean": "no failure",
    "gauss2": "noise sigma=2 (detectors ALIVE)",
    "mask0p25": "25% of lanes dead",
    "mask0p5": "50% of lanes dead",
    "mask0p25_intersection": "25% of intersections without sensors",
    "mask0p5_intersection": "50% of intersections without sensors",
    "mask0p25_hold_t60": "25% frozen from decision 60",
    "mask0p5_hold_t60": "50% frozen from decision 60",
}


def load_trace(plan: Plan, network: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """-> ({scenario: [N, T] rule mode}, {scenario: [N, T] trigger held}).

    Reads ALL runs of the plan on this network and takes draw 0: the figure shows ONE
    rollout, since averaging over draws would blur exactly what it should show.
    """
    mode: dict[str, dict[tuple[int, int], bool]] = defaultdict(dict)
    held: dict[str, dict[tuple[int, int], bool]] = defaultdict(dict)
    for path in sorted(RUNS.glob(f"t2/*__{network}__*__{plan.name}/gate_trace.csv")):
        for row in csv.DictReader(path.open()):
            if row["step"] != "0":  # the first draw only
                continue
            key = (int(row["node"]), int(row["decision"]))
            m = row["rule_mode"] == "1"
            mode[row["split"]][key] = m
            held[row["split"]][key] = (row["raw"] == "1") and not m
    if not mode:
        return {}, {}

    def grid(d: dict[tuple[int, int], bool]) -> np.ndarray:
        n = max(k[0] for k in d) + 1
        tt = max(k[1] for k in d) + 1
        out = np.zeros((n, tt), dtype=bool)
        for (node, dec), v in d.items():
            out[node, dec] = v
        return out

    return ({s: grid(d) for s, d in mode.items()}, {s: grid(d) for s, d in held.items()})


def plot(plan: Plan, network: str, path: Path) -> Path | None:
    """Small multiples: one strip per scenario, nodes vertical, decisions horizontal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    mode, held_all = load_trace(plan, network)
    if not mode:
        return None
    # scenario order FROM THE PLAN (by severity), not alphabetical
    order = [s.slug() for s in plan.scenarios if s.slug() in mode]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(
        len(order),
        1,
        figsize=(9.2, 0.95 * len(order) + 1.9),
        sharex=True,
        facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes)
    cmap = ListedColormap([SURFACE, GATE])

    for ax, scen in zip(axes, order, strict=True):
        grid, held = mode[scen], held_all[scen]
        ax.set_facecolor(SURFACE)
        ax.imshow(
            grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
            extent=(0, grid.shape[1], grid.shape[0] - 0.5, -0.5),
        )  # fmt: skip
        ys, xs = np.nonzero(held)
        if len(xs):
            ax.plot(
                xs + 0.5, ys, ls="none", marker="|", ms=4.5, mew=1.1, color=INK, zorder=3
            )  # fmt: skip
        # END OF WARM-UP: the gate is silent until the 12-reading buffer fills. Right after
        # it the rule fires on CLEAN traffic, because an empty lane reports a constant zero
        # exactly like a dead detector: a cold-start artefact, not a steady state.
        ax.axvline(12, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=4)
        share = grid.mean()
        ax.set_ylabel(
            f"{LABELS.get(scen, scen)}\n{share:.0%} of pairs",
            rotation=0, ha="right", va="center", labelpad=8, fontsize=8.5, color=INK2,
        )  # fmt: skip
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        # an empty strip MUST be labelled, or it reads as missing data
        if share == 0:
            ax.text(
                grid.shape[1] / 2, (grid.shape[0] - 1) / 2,
                "the rule did NOT fire ONCE: noise does not freeze a reading",
                ha="center", va="center", fontsize=8.5, color=MUTED, style="italic",
            )  # fmt: skip

    axes[-1].set_xlabel("decision in the episode (dt = 15 s, episode = 240 decisions)", color=INK2)
    axes[-1].tick_params(colors=INK2, labelsize=8)
    fig.legend(
        handles=[
            Patch(facecolor=GATE, edgecolor="none", label="node in rule mode (FixedTime)"),
            Patch(facecolor=SURFACE, edgecolor=AXIS, label="node controlled by the policy"),
            Line2D(
                [], [], ls="none", marker="|", ms=7, mew=1.3, color=INK,
                label="detector fired, hysteresis held",
            ),  # fmt: skip
            Line2D(
                [], [], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                label="end of window warm-up (12 readings)",
            ),  # fmt: skip
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=2, frameon=False, fontsize=8.5,
    )  # fmt: skip
    fig.suptitle(
        f"When and which intersection hands over control: {network}, seed 0, draw 0",
        fontsize=10, color=INK,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main(plan: Plan) -> Path:
    """-> results/gate_trace.md + figures/gate_trace_<network>.{png,pdf}."""
    lines = [
        f"# Gate trace: plan `{plan.name}`",
        "",
        "An explanatory figure, not a measurement: **one seed, one failure draw**. "
        "Numbers for the thesis come from `t2_gate_real` (3 seeds x 5 draws).",
        "",
        "Fill = the node's control mode after hysteresis. Marker = the detector fired, but "
        "hysteresis held the switch (entry needs `k_on` = 3 consecutive decisions, exit "
        "`k_off` = 6, hence the fill is WIDER than the run of markers).",
        "",
        "## What the figure showed that no table did",
        "",
        "**The rule's false alarms on clean traffic are a COLD START artefact, not a steady "
        "state.** The whole 1.5% share falls in decisions 13-24, right after the 12-reading "
        "buffer fills, and does not return until the end of the episode. The mechanism is "
        "clear: at the start the network is empty, and a lane without traffic reports a "
        "constant zero, exactly what a dead detector does (`max - min < 1e-6`). The warm-up "
        "guard ends when the buffer is FULL, not when it is full of MEANINGFUL readings.",
        "",
        "Three conclusions: (1) in the steady state the rule does not fire on clean at all, "
        "i.e. it is BETTER than `gate_share` says; (2) this explains why the clean share was "
        "1.5% against a calibration budget of 0.5%: the difference is not a budget overrun "
        "but another phenomenon; (3) the fix is cheap and worth measuring: lengthen the "
        "warm-up or require a suspicious movement to have ever had a NONZERO reading. "
        "Unmeasured, to be recorded as open.",
        "",
        "For comparison, under failure the cold start accounts for 13-14% of cases "
        "(`mask0p5`, `mask0p25_intersection`) and 44% under `hold`; there the rest spreads "
        "over the whole episode, since the failure persists.",
        "",
    ]
    figures = []
    for net in plan.networks:
        p = plot(plan, net, RESULTS / "figures" / f"gate_trace_{net}.png")
        if p is None:
            continue
        figures.append(p)
        mode, _ = load_trace(plan, net)
        lines += [f"## {net}", "", f"![gate trace {net}](figures/{p.name})", ""]
        lines += [
            "| scenario | share of pairs in rule mode | nodes that took over at least once |",
            "|---|---:|---:|",
        ]

        for scen in (s.slug() for s in plan.scenarios):
            if scen not in mode:
                continue
            g = mode[scen]
            lines.append(
                f"| `{scen}` | {g.mean():.1%} | {int(g.any(-1).sum())}/{g.shape[0]} |"
            )  # fmt: skip
        lines.append("")
    out = RESULTS / "gate_trace.md"
    out.write_text("\n".join(lines))
    print(f"{out}  ({len(figures)} figures)")
    return out


__all__ = ["LABELS", "load_trace", "main", "plot"]


if __name__ == "__main__":
    from tsc_marl.robustness.degrade import load_plan

    main(load_plan(sys.argv[1] if len(sys.argv) > 1 else "t2_trace"))
