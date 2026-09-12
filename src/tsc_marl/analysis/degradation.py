"""Raport T2 bez mechanizmu: tabela i krzywe degradacji zamrożonych sterowników.

Czyta biegi `runs/t2/*__<plan>` (nic nie uruchamia). Dwie decyzje, które zmieniają
liczby, więc są tu jawnie:

* UŚREDNIANIE DWUSTOPNIOWE — najpierw po LOSOWANIACH awarii w obrębie seeda, potem
  mean±std po SEEDACH polityki. Std w tabeli to więc rozrzut między politykami, nie
  między losowaniami; płaska średnia po wszystkich wierszach mieszałaby oba źródła
  (patrz `domain/sensors.py`: RNG awarii jest celowo niezależny od `--seed`).
* ODNIESIENIE — degradacja Δ liczona względem CZYSTEGO wyniku TEGO SAMEGO sterownika
  w tym samym ewaluatorze, nie względem T1. Na czystym ewaluator zgadza się z T1
  co do decyzji poza remisami (`robustness/controllers.py`), ale remisy potrafią
  przesunąć ATT o ~1%, a to jest rząd małych degradacji.

Kontrola spójności: czysty wynik uczonego sterownika MUSI równać się ostatniemu
testowi z jego treningu (te same wagi, argmax, deterministyczny CityFlow). Raport
liczy tę różnicę — niezerowa znaczy, że ewaluator nie mierzy tej polityki, którą
mierzy tabela T1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.paths import RESULTS, RUNS
from tsc_marl.robustness.degrade import EXPERIMENT, LEARNED, Plan, slug

# Kolory: sloty 1–4 palety kategorycznej (walidacja: CVD ΔE ≥ 9,1, normal ≥ 22,9).
# Kolor idzie za STEROWNIKIEM, nie za pozycją — brakujący sterownik nie przemalowuje
# pozostałych.
COLORS = {
    "gatppo": "#2a78d6",
    "advancedcolight": "#eb6834",
    "efficientmp": "#1baf7a",
    "advancedmp": "#eda100",
    # warianty actora z ablacji 1.6 (plan t2_ablations): sloty 5 i 7 palety, żeby
    # nie pożyczać koloru Advanced-CoLighta ani reguł z t2_nomech (walidacja
    # z rdzeniem: CVD ΔE ≥ 13,0, normal ≥ 27,5)
    "gatppo_mean": "#e87ba4",
    "gatppo_nognn": "#4a3aa7",
}
LABELS = {
    "gatppo": "rdzeń GAT+PPO+CTDE",
    "advancedcolight": "Advanced-CoLight",
    "efficientmp": "Efficient-MP",
    "advancedmp": "Advanced-MP",
    "fixedtime": "FixedTime",
    "gatppo_mean": "actor mean-pool",
    "gatppo_nognn": "actor bez GNN",
    "gatppo_ens": "ensemble 5 actorów (decyduje seed 0)",
}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

#: Panele wykresu = rodziny awarii, w kolejności skali obserwowalności (2.0a).
FAMILIES = (
    ("szum gaussowski σ", lambda s: s.kind == "gauss"),
    ("martwe detektory, zero", lambda s: s.kind == "mask" and s.unit == "lane"
     and s.substitution == "zero"),
    ("martwe skrzyżowania", lambda s: s.kind == "mask" and s.unit == "intersection"),
    ("martwe bez śladu, hold", lambda s: s.kind == "mask" and s.substitution == "hold"),
    ("rozkalibrowanie", lambda s: s.kind == "drift"),
)  # fmt: skip


def load(plan: Plan, metrics: tuple[str, ...] = ("att", "throughput")):
    """Długa ramka: controller, network, seed, scenario, draw + kolumny `metrics`."""
    import pandas as pd  # noqa: PLC0415

    from runlog import Experiment  # noqa: PLC0415

    exp = Experiment(EXPERIMENT, root=RUNS)
    frames = []
    for run in exp.runs():
        if run.status != "ok" or not run.id.endswith(f"__{plan.name}"):
            continue
        path = run.path / "metrics.csv"
        if not path.is_file():
            continue
        m = pd.read_csv(path)
        m = m[m["metric"].isin(list(metrics))]
        wide = m.pivot_table(index=["split", "step"], columns="metric", values="value")
        wide = wide.reset_index().rename(columns={"split": "scenario", "step": "draw"})
        p = run.params
        wide["controller"], wide["network"], wide["seed"] = p["controller"], p["network"], p["seed"]
        wide["parent_run"] = p.get("parent_run")
        frames.append(wide)
    if not frames:
        return pd.DataFrame(
            columns=["controller", "network", "seed", "scenario", "draw", "att", "throughput"]
        )
    return pd.concat(frames, ignore_index=True)


def aggregate(df):
    """(controller, network, scenario) -> att_mean, att_std (po seedach), n_seeds, n_draws."""
    if df.empty:
        return df
    per_seed = (
        df.groupby(["controller", "network", "scenario", "seed"])
        .agg(att=("att", "mean"), draws=("draw", "nunique"))
        .reset_index()
    )
    out = (
        per_seed.groupby(["controller", "network", "scenario"])
        .agg(
            att_mean=("att", "mean"),
            att_std=("att", "std"),
            n_seeds=("seed", "nunique"),
            n_draws=("draws", "min"),
        )
        .reset_index()
    )
    out["att_std"] = out["att_std"].fillna(0.0)
    clean = out[out["scenario"] == "clean"].set_index(["controller", "network"])["att_mean"]
    out["delta"] = [
        r.att_mean / clean.get((r.controller, r.network), np.nan) - 1.0 for r in out.itertuples()
    ]
    return out


def parent_consistency(df) -> list[dict[str, Any]]:
    """Czysty wynik ewaluatora vs ostatni test treningu — dla każdego uczonego biegu."""
    import pandas as pd  # noqa: PLC0415

    from tsc_marl.libsignal import experiment as t1  # noqa: PLC0415

    out = []
    runs = {r.id: r for r in t1.experiment().runs()}
    clean = df[(df["scenario"] == "clean") & df["parent_run"].notna()]
    for r in clean.itertuples():
        parent = runs.get(r.parent_run)
        path = parent.path / "metrics.csv" if parent else None
        if path is None or not path.is_file():
            continue
        m = pd.read_csv(path)
        test = m[(m["split"] == "test") & (m["metric"] == "att")].sort_values("step")
        if test.empty:
            continue
        last = float(test["value"].iloc[-1])
        out.append({"run": r.parent_run, "eval": float(r.att), "last_test": last,
                    "diff": float(r.att) - last})  # fmt: skip
    return out


# ------------------------------------------------------------------ markdown


def _spec_by_slug(plan: Plan) -> dict[str, FailureSpec]:
    return {slug(s): s for s in plan.scenarios}


def to_markdown(plan: Plan, agg, checks: list[dict[str, Any]], figures: dict[str, Path]) -> str:
    specs = _spec_by_slug(plan)
    order = [slug(s) for s in plan.scenarios]
    ctrls = [c for c in plan.controllers if c != "fixedtime"]
    lines = [
        f"# T2 bez mechanizmu — plan `{plan.name}`",
        "",
        f"Tryb {plan.mode} (yellow={plan.yellow}), {plan.draws} losowań awarii na scenariusz, "
        "ewaluacja ZAMROŻONYCH sterowników (`tsc degrade`). Wszystkie czytają tę samą warstwę "
        "detektorów: MaxPressure liczy decyzję z popsutego wektora stanu, nie z silnika.",
        "",
        "Komórka = ATT [s], średnia po losowaniach, potem po seedach polityki (± std po "
        "seedach); w nawiasie Δ względem czystego wyniku tego samego sterownika. "
        "**▲ = gorzej niż FixedTime** — tylko tam dolna warstwa fallbacku ma pole do zysku.",
        "",
    ]
    for net in plan.networks:
        sub = agg[agg["network"] == net] if not agg.empty else agg
        ft = sub[(sub["controller"] == "fixedtime") & (sub["scenario"] == "clean")]
        ft_att = float(ft["att_mean"].iloc[0]) if len(ft) else np.nan
        lines += [f"## {net}", "", f"FixedTime (niezależny od odczytów): **{ft_att:.1f} s**.", ""]
        present = [c for c in ctrls if len(sub) and (sub["controller"] == c).any()]
        missing = [c for c in ctrls if c not in present]
        if not present:
            lines += ["_Brak policzonych sterowników dla tej sieci._", ""]
            continue
        head = "| scenariusz | " + " | ".join(
            f"{LABELS.get(c, c)} (n={int(sub[sub['controller'] == c]['n_seeds'].max())})"
            for c in present
        )
        lines += [head + " |", "|---|" + "---:|" * len(present)]
        for name in order:
            cells = []
            for c in present:
                row = sub[(sub["controller"] == c) & (sub["scenario"] == name)]
                if row.empty:
                    cells.append("—")
                    continue
                r = row.iloc[0]
                std = f" ± {r.att_std:.1f}" if r.n_seeds > 1 else ""
                mark = " ▲" if r.att_mean > ft_att else ""
                delta = "" if name == "clean" else f" ({r.delta:+.1%})"
                cells.append(f"{r.att_mean:.1f}{std}{delta}{mark}")
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        lines.append("")
        if missing:
            lines += [
                f"_Niepoliczone na tej maszynie: {', '.join(LABELS.get(c, c) for c in missing)} "
                "(brak checkpointów rodzica — patrz `tsc degrade status`)._",
                "",
            ]
        lines += _findings(sub, present, ft_att, specs)
        if net in figures:
            lines += ["", f"![krzywe degradacji {net}](figures/{figures[net].name})", ""]

    if checks:
        worst = max(abs(c["diff"]) for c in checks)
        lines += [
            "## Kontrola: ewaluator mierzy tę samą politykę co T1",
            "",
            f"Czysty wynik ewaluatora vs ostatni test treningu, {len(checks)} biegów uczonych: "
            f"największa różnica **{worst:.3f} s**"
            + (" — zgodne co do precyzji logu." if worst < 0.05 else " — ROZJAZD, sprawdzić!"),
            "",
        ]
    lines += [
        "## Czego ta tabela NIE mówi",
        "",
        "* Poziomy są poziomami DETEKTORA. Na cechę `gauss` przenosi się ze współczynnikiem "
        "~0,8–0,9, `urand` jest ~1,6× słabszy przy tym samym `level` "
        "(`results/sensor_severity.md`).",
        "* Maska z podstawieniem `zero` zmienia znacznie mniej komórek cech, niż wynosi ułamek "
        "martwych pasów (zero jest wartością modalną) — ta sama liczba nie znaczy tego samego "
        "na sieciach o różnym obciążeniu.",
        "* Scenariusze `hold` zaczynają się w decyzji 60 (15. minuta), więc pierwsza ćwiartka "
        "epizodu jest czysta — ich Δ nie jest porównywalne wprost z maską od początku.",
    ]
    return "\n".join(lines) + "\n"


def _findings(sub, present, ft_att, specs) -> list[str]:
    """Wnioski liczone z tabeli, nie pisane z góry."""
    out = []
    for c in present:
        rows = sub[(sub["controller"] == c) & (sub["scenario"] != "clean")]
        if rows.empty:
            continue
        worst = rows.loc[rows["att_mean"].idxmax()]
        above = rows[rows["att_mean"] > ft_att]["scenario"].tolist()
        text = (
            f"* **{LABELS.get(c, c)}** — najgorszy scenariusz `{worst.scenario}`: "
            f"{worst.att_mean:.1f} s ({worst.delta:+.1%}). "
        )
        if above:
            text += (
                f"Gorzej niż FixedTime w {len(above)} scenariuszach: "
                + ", ".join(f"`{a}`" for a in above)
                + "."
            )
        else:
            text += (
                "Nigdzie nie spada poniżej FixedTime — przy tych poziomach fallback do "
                "FixedTime nie ma z czego zarobić."
            )
        out.append(text)
    return out + [""]


# --------------------------------------------------------------------- wykres


def plot(plan: Plan, agg, network: str, path: Path) -> Path | None:
    """Małe wielokrotności: jeden panel na rodzinę awarii, wspólna oś Y (ATT)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sub = agg[agg["network"] == network]
    if sub.empty:
        return None
    specs = _spec_by_slug(plan)
    families = [(t, [s for s in plan.scenarios if f(s)]) for t, f in FAMILIES]
    families = [(t, s) for t, s in families if s]
    if not families:
        return None
    ft = sub[(sub["controller"] == "fixedtime") & (sub["scenario"] == "clean")]
    ft_att = float(ft["att_mean"].iloc[0]) if len(ft) else None

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(
        1, len(families), figsize=(3.1 * len(families), 3.4), sharey=True, facecolor=SURFACE
    )
    axes = np.atleast_1d(axes)
    for ax, (title, fam) in zip(axes, families, strict=True):
        ax.set_facecolor(SURFACE)
        for c in (c for c in COLORS if (sub["controller"] == c).any()):
            xs, ys, sd = [0.0], [], []
            clean = sub[(sub["controller"] == c) & (sub["scenario"] == "clean")]
            if clean.empty:
                continue
            ys.append(float(clean["att_mean"].iloc[0]))
            sd.append(float(clean["att_std"].iloc[0]))
            for s in sorted(fam, key=lambda s: s.level):
                row = sub[(sub["controller"] == c) & (sub["scenario"] == slug(s))]
                if row.empty:
                    continue
                xs.append(s.level)
                ys.append(float(row["att_mean"].iloc[0]))
                sd.append(float(row["att_std"].iloc[0]))
            xs, ys, sd = np.array(xs), np.array(ys), np.array(sd)
            ax.plot(xs, ys, color=COLORS[c], lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.5,
                    solid_capstyle="round", label=LABELS[c], zorder=3)  # fmt: skip
            if plan.kind(c) in LEARNED and (sd > 0).any():
                ax.fill_between(xs, ys - sd, ys + sd, color=COLORS[c], alpha=0.10, lw=0)
        if ft_att is not None:
            ax.axhline(ft_att, color=INK2, lw=1, zorder=2)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("poziom awarii (detektor)", color=INK2)
        ax.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("ATT [s]", color=INK2)
    if ft_att is not None:
        axes[-1].annotate("FixedTime", (1.0, ft_att), xycoords=("axes fraction", "data"),
                          xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                          color=INK2, fontsize=8)  # fmt: skip
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.02))  # fmt: skip
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    _ = specs
    return path


def main(plan: Plan) -> Path:
    df = load(plan)
    agg = aggregate(df)
    figures = {}
    for net in plan.networks:
        p = plot(plan, agg, net, RESULTS / "figures" / f"{plan.name}_{net}.png")
        if p is not None:
            figures[net] = p
    checks = parent_consistency(df) if not df.empty else []
    out = RESULTS / f"{plan.name}.md"
    out.write_text(to_markdown(plan, agg, checks, figures), encoding="utf-8")
    if not agg.empty:
        agg.to_csv(RESULTS / f"{plan.name}.csv", index=False)
    return out
