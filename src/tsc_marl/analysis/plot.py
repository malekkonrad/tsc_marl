"""Learning curves from stored runs: `make plot`.

Reads metrics through `runlog`, and for runs IN PROGRESS falls back to LibSignal's
`_DTL.log`, written as training goes (`metrics.csv` appears only at the end, when
`experiment.ingest` reads the engine log). Plotting lives here, not in `runlog`,
because the `_DTL.log` fallback is LibSignal knowledge.

    make plot                                   # test ATT of all runs
    python -m tsc_marl.analysis.plot --metric queue --split train
    python -m tsc_marl.analysis.plot -r colight__jinan__s0_y5 colight__jinan__s1_y5 --band
    python -m tsc_marl.analysis.plot -o results/figures/curve.png

The Phase 1 GATE figures (`gate_figures`) are drawn by `tsc table` together with
`results/p1_gate.md`, so the report and the figures come from the same runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI: runs on a remote machine, saves a file
import matplotlib.pyplot as plt  # noqa: E402

from tsc_marl.libsignal import experiment as tx  # noqa: E402
from tsc_marl.paths import RUNS  # noqa: E402

DEFAULT_OUT = RUNS / "plots"


def live_curve(view) -> list[dict]:
    """Metrics of a run that has not stored them yet, straight from the engine log."""
    params = view.params
    logdir = (
        view.path
        / "engine"
        / "tsc"
        / f"{params.get('world')}_{params.get('agent')}"
        / str(params.get("network"))
        / "test"
        / "logger"
    )
    return tx.parse_episodes(logdir)


def gather(exp, run_ids: list[str] | None, metric: str, split: str):
    """{run_id: [(episode, value)]} from `metrics.csv`, or `_DTL.log` for runs in progress."""
    out: dict[str, list[tuple[int, float]]] = {}
    for view in exp.runs():
        if run_ids and view.id not in run_ids:
            continue
        csv_path = view.path / "metrics.csv"
        if csv_path.is_file():
            import pandas as pd

            df = pd.read_csv(csv_path)
            df = df[(df["metric"] == metric) & (df["split"] == split)]
            points = list(zip(df["step"], df["value"], strict=True))
        else:
            points = [
                (r["episode"], r[metric])
                for r in live_curve(view)
                if r["split"] == split and metric in r
            ]
        if points:
            out[view.id] = sorted(points)
    return out


def group_key(view) -> str:
    """Group label for the mean +- std band: everything but the seed."""
    p = view.params
    return f"{p.get('agent')}, {p.get('network')}, y{p.get('yellow_length')}"


def draw(curves: dict[str, list[tuple[int, float]]], metric: str, split: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, points in sorted(curves.items()):
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label=label)
    ax.set_xlabel("episode")
    ax.set_ylabel(f"{metric} ({split})")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def draw_band(
    groups: dict[str, list[list[tuple[int, float]]]], metric: str, split: str, title: str
):
    """mean +- std over seeds, ONLY over episodes every run in the group has;
    otherwise the longest run's tail would pose as the group mean."""
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, runs in sorted(groups.items()):
        common = sorted(set.intersection(*({s for s, _ in r} for r in runs)))
        if not common:
            continue
        values = np.array([[dict(r)[s] for s in common] for r in runs], dtype=float)
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(common, mean, linewidth=1.8, label=f"{label}  (n={len(runs)})")
        ax.fill_between(common, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("episode")
    ax.set_ylabel(f"{metric} ({split})")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


# Phase 1 GATE figures (`report.build_gate_report`). Colour follows the METHOD in all
# three. The pair passed a colour-blindness palette check (protan dE 24.7), both > 3:1
# on white.
GATE_COLORS = {"advancedcolight": "#2a78d6", "gatppo": "#eb6834"}
INK, INK_MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PPO_DIAGNOSTICS = (
    ("entropy", "policy entropy", False),
    ("approx_kl", "approx. KL (update)", False),
    ("clip_frac", "clip fraction", False),
    ("value_loss", "value loss", True),
)


def _style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, labelsize=8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _log_axis(ax):
    """Log scale with readable numbers (1-2-5 per decade) instead of bare 10^k."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _seed_band(curves, run_ids, split: str, metric: str):
    """(episodes, mean, min, max) over seeds, only episodes EVERY run has."""
    sub = curves[
        curves["run_id"].isin(run_ids) & (curves["split"] == split) & (curves["metric"] == metric)
    ]
    wide = sub.pivot(index="step", columns="run_id", values="value").dropna()
    return wide.index.to_numpy(), wide.mean(axis=1), wide.min(axis=1), wide.max(axis=1)


def _plot_band(ax, band, color, label=None):
    steps, mean, lo, hi = band
    ax.fill_between(steps, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(steps, mean, color=color, linewidth=1.6, label=label)


def gate_figures(curves, runs, labels, reference_att, tolerance, out_dir):
    """Phase 1 GATE learning curves -> `out_dir/p1_*.png` (+ `.pdf` for the thesis).

    `runs`: a frame with run_id, network, agent. `reference_att`: {network: reference
    ATT}, which gives the gate threshold line. Returns [(PNG file, caption)]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    networks = list(reference_att)
    ids = {
        (net, agent): list(runs[(runs["network"] == net) & (runs["agent"] == agent)]["run_id"])
        for net in networks
        for agent in labels
    }
    n_seeds = min(len(v) for v in ids.values())
    reference, core = list(labels)
    figures = []

    def save(fig, name, caption):
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append((f"{name}.png", caption))

    # 1. Test ATT: the full range (start ~1300 s) and a zoom on the plateau with the threshold.
    fig, axes = plt.subplots(2, len(networks), figsize=(10, 6.6), sharex=True)
    for col, net in enumerate(networks):
        top, zoom = axes[0][col], axes[1][col]
        for agent, label in labels.items():
            band = _seed_band(curves, ids[(net, agent)], "test", "att")
            _plot_band(top, band, GATE_COLORS[agent], label if col == 0 else None)
            _plot_band(zoom, band, GATE_COLORS[agent])
        ref = reference_att[net]
        gate = ref * (1 + tolerance)
        zoom.axhline(gate, color=INK_MUTED, linewidth=1, linestyle="--")
        zoom.annotate(
            f"gate threshold: +{tolerance * 100:.0f}% ({gate:.0f} s)",
            (0.99, gate),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=INK,
        )
        zoom.set_ylim(ref * 0.97, gate * 1.03)
        top.set_title(net, fontsize=10)
        zoom.set_xlabel("episode")
        for ax in (top, zoom):
            _style(ax)
    axes[0][0].set_ylabel("test ATT [s], full range")
    axes[1][0].set_ylabel("test ATT [s], plateau")
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(
        fig,
        "p1_att_test",
        f"Test ATT per episode. Line = mean over seeds, band = min-max (n = {n_seeds}). "
        "Top row: full range; bottom: zoom on the plateau with the gate threshold "
        f"(ATT of {labels[reference]} + {tolerance * 100:.0f}%).",
    )

    # 2. The core's PPO diagnostics: is the update healthy (entropy falls, KL and clip in range).
    fig, axes = plt.subplots(
        len(networks), len(PPO_DIAGNOSTICS), figsize=(12, 2.9 * len(networks)), sharex=True
    )
    for row, net in enumerate(networks):
        for col, (metric, title, log) in enumerate(PPO_DIAGNOSTICS):
            ax = axes[row][col]
            _plot_band(ax, _seed_band(curves, ids[(net, core)], "train", metric), GATE_COLORS[core])
            if log:
                _log_axis(ax)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if row == len(networks) - 1:
                ax.set_xlabel("episode")
            _style(ax)
        axes[row][0].set_ylabel(net, fontsize=9)
    fig.tight_layout()
    save(
        fig,
        "p1_ppo_diagnostics",
        f"PPO update signals, {labels[core]}, one update per episode. Rows = networks; "
        f"line = mean over seeds, band = min-max (n = {n_seeds}). Value loss on a log scale.",
    )

    # 3. Advanced-CoLight's TD loss. Zeros before `learning_start` are cut: they do not
    # exist on a log scale and carry no information (the buffer is still filling).
    fig, axes = plt.subplots(1, len(networks), figsize=(10, 3.2))
    for col, net in enumerate(networks):
        ax = axes[col]
        steps, mean, lo, hi = _seed_band(curves, ids[(net, reference)], "train", "loss")
        keep = lo > 0
        _plot_band(ax, (steps[keep], mean[keep], lo[keep], hi[keep]), GATE_COLORS[reference])
        _log_axis(ax)
        ax.set_title(net, fontsize=10)
        ax.set_xlabel("episode")
        _style(ax)
    axes[0].set_ylabel("TD loss (train)")
    fig.tight_layout()
    save(
        fig,
        "p1_colight_loss",
        f"TD loss of {labels[reference]} per episode (log scale; episodes before "
        f"`learning_start` left out). Line = mean over seeds, band = min-max "
        f"(n = {n_seeds}).",
    )
    return figures


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Learning curves from runs/")
    ap.add_argument("-r", "--runs", nargs="+", default=None, help="run_id (default: all)")
    ap.add_argument("--metric", default="att", help="att, queue, reward, loss, delay, throughput")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument(
        "--band", action="store_true", help="mean +- std over seeds instead of separate lines"
    )
    ap.add_argument("-o", "--out", default=None, help="output file (default: runs/plots/)")
    args = ap.parse_args(argv)

    exp = tx.experiment()
    curves = gather(exp, args.runs, args.metric, args.split)
    if not curves:
        print(
            f"No data for metric={args.metric} split={args.split}. "
            "Methods without training have no curve (one measurement).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    title = f"{args.metric} ({args.split}), {exp.name}"
    if args.band:
        views = {v.id: v for v in exp.runs()}
        groups: dict[str, list[list[tuple[int, float]]]] = {}
        for rid, points in curves.items():
            groups.setdefault(group_key(views[rid]), []).append(points)
        fig = draw_band(groups, args.metric, args.split, title)
    else:
        fig = draw(curves, args.metric, args.split, title)

    # a band and separate lines are different plots and must not share a file
    name = f"{args.metric}_{args.split}{'_band' if args.band else ''}.png"
    out = Path(args.out) if args.out else DEFAULT_OUT / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"{out}  ({len(curves)} runs)")


if __name__ == "__main__":
    main()
