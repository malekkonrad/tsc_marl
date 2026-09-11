"""Krzywe uczenia z zapisanych biegów — `make plot`.

Czyta metryki przez `runlog`, a dla biegów W TOKU spada na `_DTL.log` LibSignala,
który jest dopisywany na bieżąco. Dzięki temu ta sama komenda działa w trakcie
7-godzinnego treningu i po nim — `metrics.csv` powstaje dopiero na końcu biegu,
bo `experiment.ingest` czyta log silnika po fakcie.

Rysowanie mieszka tutaj, a nie w `runlog`, bo fallback na `_DTL.log` to wiedza
o LibSignalu. Część generyczna (długa ramka → wykres) siedzi w `draw` i da się
ją stąd wyjąć, gdyby kiedyś miała trafić do narzędzia.

    make plot                                   # ATT (test) ze wszystkich biegów
    python -m tsc_marl.analysis.plot --metric queue --split train
    python -m tsc_marl.analysis.plot -r colight__jinan__s0_y5 colight__jinan__s1_y5 --band
    python -m tsc_marl.analysis.plot -o results/figures/krzywa.png

Rysunki GATE-a Fazy 1 (`gate_figures`) nie idą przez CLI — rysuje je `tsc table`
razem z `results/p1_gate.md`, żeby raport i wykresy zawsze pochodziły z tych samych biegów.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # bez GUI: liczymy na maszynie zdalnej, zapisujemy plik
import matplotlib.pyplot as plt  # noqa: E402

from tsc_marl.libsignal import experiment as tx  # noqa: E402
from tsc_marl.paths import RUNS  # noqa: E402

DEFAULT_OUT = RUNS / "plots"


def live_curve(view) -> list[dict]:
    """Metryki biegu, który jeszcze nie zdążył ich zapisać — prosto z logu silnika."""
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
    """{run_id: [(epizod, wartość)]} — z `metrics.csv`, a dla biegów w toku z `_DTL.log`."""
    out: dict[str, list[tuple[int, float]]] = {}
    for view in exp.runs():
        if run_ids and view.id not in run_ids:
            continue
        csv_path = view.path / "metrics.csv"
        if csv_path.is_file():
            import pandas as pd  # noqa: PLC0415

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
    """Etykieta grupy dla wstęgi mean±std: wszystko poza seedem."""
    p = view.params
    return f"{p.get('agent')} · {p.get('network')} · y{p.get('yellow_length')}"


def draw(curves: dict[str, list[tuple[int, float]]], metric: str, split: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, points in sorted(curves.items()):
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label=label)
    ax.set_xlabel("epizod")
    ax.set_ylabel(f"{metric} ({split})")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def draw_band(
    groups: dict[str, list[list[tuple[int, float]]]], metric: str, split: str, title: str
):
    """mean ± std po seedach. Wstęga liczona TYLKO dla epizodów, które ma każdy bieg
    w grupie — inaczej ogon najdłuższego biegu udawałby średnią z całej grupy."""
    import numpy as np  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, runs in sorted(groups.items()):
        common = sorted(set.intersection(*({s for s, _ in r} for r in runs)))
        if not common:
            continue
        values = np.array([[dict(r)[s] for s in common] for r in runs], dtype=float)
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(common, mean, linewidth=1.8, label=f"{label}  (n={len(runs)})")
        ax.fill_between(common, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("epizod")
    ax.set_ylabel(f"{metric} ({split})")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


# Rysunki GATE-a Fazy 1 (`report.build_gate_report`). Kolor idzie za METODĄ we wszystkich
# trzech — ta sama metoda ma ten sam kolor na każdym rysunku. Para sprawdzona walidatorem
# palety pod daltonizm (protan ΔE 24.7), obie > 3:1 na białym tle.
GATE_COLORS = {"advancedcolight": "#2a78d6", "gatppo": "#eb6834"}
INK, INK_MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PPO_DIAGNOSTICS = (
    ("entropy", "entropia polityki", False),
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
    """Skala log z czytelnymi liczbami (1-2-5 w dekadzie) zamiast samego 10^k."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: PLC0415

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _seed_band(curves, run_ids, split: str, metric: str):
    """(epizody, średnia, min, max) po seedach — tylko epizody, które ma KAŻDY bieg."""
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
    """Krzywe uczenia GATE-a Fazy 1 -> `out_dir/p1_*.png` (+ `.pdf` do pracy).

    `runs`: ramka z kolumnami run_id, network, agent. `reference_att`: {sieć: ATT
    referencji} — z niego linia progu bramki. Zwraca [(plik PNG, podpis)]."""
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

    # 1. Test ATT: pełny zakres (start ~1300 s) i przybliżenie na plateau z progiem.
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
            f"próg bramki: +{tolerance * 100:.0f}% ({gate:.0f} s)",
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
        zoom.set_xlabel("epizod")
        for ax in (top, zoom):
            _style(ax)
    axes[0][0].set_ylabel("ATT test [s] — pełny zakres")
    axes[1][0].set_ylabel("ATT test [s] — plateau")
    fig.legend(loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(
        fig,
        "p1_att_test",
        f"Test ATT po epizodach. Linia = średnia po seedach, wstęga = min–max (n = {n_seeds}). "
        "Górny rząd: pełny zakres; dolny: przybliżenie na plateau z progiem bramki "
        f"(ATT {labels[reference]} + {tolerance * 100:.0f}%).",
    )

    # 2. Diagnostyka PPO rdzenia — czy update jest zdrowy (entropia spada, KL i clip w normie).
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
                ax.set_xlabel("epizod")
            _style(ax)
        axes[row][0].set_ylabel(net, fontsize=9)
    fig.tight_layout()
    save(
        fig,
        "p1_ppo_diagnostics",
        f"Sygnały update'u PPO — {labels[core]}, jeden update na epizod. Wiersze = sieci; "
        f"linia = średnia po seedach, wstęga = min–max (n = {n_seeds}). Value loss w skali log.",
    )

    # 3. Strata TD Advanced-CoLighta. Zera przed `learning_start` wycięte — w skali log
    # nie istnieją, a nie niosą informacji (bufor jeszcze się zapełnia).
    fig, axes = plt.subplots(1, len(networks), figsize=(10, 3.2))
    for col, net in enumerate(networks):
        ax = axes[col]
        steps, mean, lo, hi = _seed_band(curves, ids[(net, reference)], "train", "loss")
        keep = lo > 0
        _plot_band(ax, (steps[keep], mean[keep], lo[keep], hi[keep]), GATE_COLORS[reference])
        _log_axis(ax)
        ax.set_title(net, fontsize=10)
        ax.set_xlabel("epizod")
        _style(ax)
    axes[0].set_ylabel("strata TD (train)")
    fig.tight_layout()
    save(
        fig,
        "p1_colight_loss",
        f"Strata TD {labels[reference]} po epizodach (skala log; epizody przed "
        f"`learning_start` pominięte). Linia = średnia po seedach, wstęga = min–max "
        f"(n = {n_seeds}).",
    )
    return figures


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Krzywe uczenia z runs/")
    ap.add_argument("-r", "--runs", nargs="+", default=None, help="run_id (domyślnie wszystkie)")
    ap.add_argument("--metric", default="att", help="att, queue, reward, loss, delay, throughput")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument(
        "--band", action="store_true", help="mean±std po seedach zamiast osobnych linii"
    )
    ap.add_argument("-o", "--out", default=None, help="plik wyjściowy (domyślnie runs/plots/)")
    args = ap.parse_args(argv)

    exp = tx.experiment()
    curves = gather(exp, args.runs, args.metric, args.split)
    if not curves:
        print(
            f"Brak danych dla metric={args.metric} split={args.split}. "
            "Metody bez treningu nie mają krzywej (jeden pomiar).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    title = f"{args.metric} ({args.split}) — {exp.name}"
    if args.band:
        views = {v.id: v for v in exp.runs()}
        groups: dict[str, list[list[tuple[int, float]]]] = {}
        for rid, points in curves.items():
            groups.setdefault(group_key(views[rid]), []).append(points)
        fig = draw_band(groups, args.metric, args.split, title)
    else:
        fig = draw(curves, args.metric, args.split, title)

    # wstęga i osobne linie to różne wykresy — nie mogą nadpisywać jednego pliku
    name = f"{args.metric}_{args.split}{'_band' if args.band else ''}.png"
    out = Path(args.out) if args.out else DEFAULT_OUT / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"{out}  ({len(curves)} biegów)")


if __name__ == "__main__":
    main()
