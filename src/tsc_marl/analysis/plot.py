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
