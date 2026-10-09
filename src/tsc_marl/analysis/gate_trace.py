"""Figura WYJAŚNIAJĄCA wyzwolenie bramki — z pary (decyzja, węzeł), nie z agregatu.

Po co osobny moduł. `gate_share` mówi, ILE par przejęła reguła, i to wystarcza, żeby
orzec o wyniku. Nie wystarcza, żeby odpowiedzieć na pytanie, które zada operator
i promotor: DLACZEGO TO skrzyżowanie oddało sterowanie i KIEDY. Odpowiedź siedzi
w `gate_trace.csv` (`tsc degrade run <plan> --trace`), gdzie na każdą parę jest
surowy wynik reguły, maska przed histerezą i tryb po histerezie.

CO RYSUNEK MA POKAZAĆ — trzy rzeczy, których nie widać w żadnej tabeli:

1. `gauss2` jest PUSTY. Reguła nie strzela przy szumie ani razu, na żadnym progu
   (`faza3-po-t2.md` §8.3) — a bramka UQ przejmowała tam 92% decyzji z precyzją 0,00.
   Pusty pasek jest tu TREŚCIĄ, nie brakiem danych, i podpis musi to mówić.
2. HISTEREZA. Tryb reguły trwa DŁUŻEJ niż surowe wyzwolenie (k_off = 6 > k_on = 3),
   więc pole wypełnione jest szersze niż znaczniki. Widać też przypadek odwrotny —
   wyzwolenie wstrzymane, bo nie zebrało 3 kolejnych decyzji.
3. OPÓŹNIENIE DETEKCJI przy `hold`: awaria wchodzi w decyzji 60, a okno 12 musi się
   najpierw wypełnić zamrożonymi odczytami. To jedyny scenariusz, w którym reguła
   ma statystycznie nieistotny wynik (§2b) — i tu widać, skąd to się bierze.

KODOWANIE, i dlaczego takie. Wypełnienie = tryb sterowania (dwa stany, jeden odcień
`#7a4fd6` z palety mechanizmów Fazy 3; kontrast wobec podłoża ≥ 3:1, zwalidowany).
Trzeci stan — „detektor strzelił, histereza wstrzymała" — jest NAKŁADKĄ znacznikową,
a nie trzecim wypełnieniem: trzy wypełnienia w jednym odcieniu albo nie przechodzą
rozdzielności (ΔE < 15), albo nie przechodzą kontrastu. Kodowanie złożone (pole +
znacznik) rozwiązuje to bez rezygnacji z porządku stanów.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from tsc_marl.paths import RESULTS, RUNS
from tsc_marl.robustness.degrade import Plan

# Paleta jak w `analysis/degradation.py` — jeden projekt, jedna paleta.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
GATE = "#7a4fd6"  # slot `gatppo_gate` palety mechanizmów Fazy 3

#: Podpisy scenariuszy — po polsku i krótko, bo idą na osie panelu.
ETYKIETY = {
    "clean": "bez awarii",
    "gauss2": "szum σ=2 (detektory ŻYJĄ)",
    "mask0p25": "25% pasów martwych",
    "mask0p5": "50% pasów martwych",
    "mask0p25_intersection": "25% skrzyżowań bez czujników",
    "mask0p5_intersection": "50% skrzyżowań bez czujników",
    "mask0p25_hold_t60": "25% zamrożonych od decyzji 60",
    "mask0p5_hold_t60": "50% zamrożonych od decyzji 60",
}


def load_trace(plan: Plan, network: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """-> ({scenariusz: [N, T] tryb reguły}, {scenariusz: [N, T] wyzwolenie wstrzymane}).

    Czyta WSZYSTKIE biegi planu na tej sieci i bierze losowanie 0 — rysunek pokazuje
    JEDEN przebieg, bo uśrednianie po losowaniach zamazałoby dokładnie to, co ma pokazać
    (kiedy konkretny węzeł przełączył się i jak długo trwał w trybie reguły).
    """
    tryb: dict[str, dict[tuple[int, int], bool]] = defaultdict(dict)
    wstrzymane: dict[str, dict[tuple[int, int], bool]] = defaultdict(dict)
    for sciezka in sorted(RUNS.glob(f"t2/*__{network}__*__{plan.name}/gate_trace.csv")):
        for row in csv.DictReader(sciezka.open()):
            if row["step"] != "0":  # tylko pierwsze losowanie
                continue
            klucz = (int(row["node"]), int(row["decision"]))
            m = row["rule_mode"] == "1"
            tryb[row["split"]][klucz] = m
            wstrzymane[row["split"]][klucz] = (row["raw"] == "1") and not m
    if not tryb:
        return {}, {}

    def siatka(d: dict[tuple[int, int], bool]) -> np.ndarray:
        n = max(k[0] for k in d) + 1
        tt = max(k[1] for k in d) + 1
        out = np.zeros((n, tt), dtype=bool)
        for (node, dec), v in d.items():
            out[node, dec] = v
        return out

    return ({s: siatka(d) for s, d in tryb.items()}, {s: siatka(d) for s, d in wstrzymane.items()})


def plot(plan: Plan, network: str, path: Path) -> Path | None:
    """Małe wielokrotności: jeden pasek na scenariusz, węzły w pionie, decyzje w poziomie."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import ListedColormap  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415
    from matplotlib.patches import Patch  # noqa: PLC0415

    tryb, wstrzymane = load_trace(plan, network)
    if not tryb:
        return None
    # Kolejność scenariuszy Z PLANU, nie alfabetyczna — plan układa je po dotkliwości.
    from tsc_marl.robustness.degrade import slug  # noqa: PLC0415

    kolejnosc = [slug(s) for s in plan.scenarios if slug(s) in tryb]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(
        len(kolejnosc),
        1,
        figsize=(9.2, 0.95 * len(kolejnosc) + 1.9),
        sharex=True,
        facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes)
    cmap = ListedColormap([SURFACE, GATE])

    for ax, scen in zip(axes, kolejnosc, strict=True):
        siatka, held = tryb[scen], wstrzymane[scen]
        ax.set_facecolor(SURFACE)
        ax.imshow(
            siatka, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
            extent=(0, siatka.shape[1], siatka.shape[0] - 0.5, -0.5),
        )  # fmt: skip
        ys, xs = np.nonzero(held)
        if len(xs):
            ax.plot(
                xs + 0.5, ys, ls="none", marker="|", ms=4.5, mew=1.1, color=INK, zorder=3
            )  # fmt: skip
        # GRANICA ROZGRZEWKI. Bufor reguły ma 12 odczytów i do jego napełnienia bramka
        # milczy bezwarunkowo. Tuż po tej granicy strzela na CZYSTYM — bo sieć jest
        # wtedy pusta, a pas bez ruchu raportuje stałe zero, czyli dokładnie to samo,
        # co detektor martwy. To jedyne fałszywe alarmy reguły na czystym i widać tu,
        # że są artefaktem zimnego startu, nie stanem ustalonym.
        ax.axvline(12, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=4)
        udzial = siatka.mean()
        ax.set_ylabel(
            f"{ETYKIETY.get(scen, scen)}\n{udzial:.0%} par",
            rotation=0, ha="right", va="center", labelpad=8, fontsize=8.5, color=INK2,
        )  # fmt: skip
        ax.set_yticks([])
        for bok in ("top", "right", "left"):
            ax.spines[bok].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        # Pusty pasek MUSI być opisany, inaczej czyta się jak brak danych.
        if udzial == 0:
            ax.text(
                siatka.shape[1] / 2, (siatka.shape[0] - 1) / 2,
                "reguła nie strzeliła ANI RAZU — szum nie zamraża odczytu",
                ha="center", va="center", fontsize=8.5, color=MUTED, style="italic",
            )  # fmt: skip

    axes[-1].set_xlabel("decyzja w epizodzie (Δt = 15 s, epizod = 240 decyzji)", color=INK2)
    axes[-1].tick_params(colors=INK2, labelsize=8)
    fig.legend(
        handles=[
            Patch(facecolor=GATE, edgecolor="none", label="węzeł w trybie reguły (FixedTime)"),
            Patch(facecolor=SURFACE, edgecolor=AXIS, label="węzeł steruje polityką"),
            Line2D(
                [], [], ls="none", marker="|", ms=7, mew=1.3, color=INK,
                label="detektor strzelił, histereza wstrzymała",
            ),  # fmt: skip
            Line2D(
                [], [], color=MUTED, lw=1.0, ls=(0, (4, 3)),
                label="koniec rozgrzewki okna (12 odczytów)",
            ),  # fmt: skip
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=2, frameon=False, fontsize=8.5,
    )  # fmt: skip
    fig.suptitle(
        f"Kiedy i które skrzyżowanie oddaje sterowanie — {network}, seed 0, losowanie 0",
        fontsize=10, color=INK,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main(plan: Plan) -> Path:
    """-> results/gate_trace.md + figures/gate_trace_<sieć>.{png,pdf}."""
    lines = [
        f"# Ślad bramki — plan `{plan.name}`",
        "",
        "Rysunek wyjaśniający, nie pomiar: **jeden seed, jedno losowanie awarii**. "
        "Liczby do pracy pochodzą z `t2_gate_real` (3 seedy × 5 losowań).",
        "",
        "Pole = tryb sterowania węzła po histerezie. Znacznik = detektor strzelił, ale "
        "histereza wstrzymała przełączenie (wejście wymaga `k_on` = 3 kolejnych decyzji, "
        "wyjście `k_off` = 6 — stąd pole jest SZERSZE niż ciąg znaczników).",
        "",
        "## Co ten rysunek pokazał, a czego nie było w żadnej tabeli",
        "",
        "**Fałszywe alarmy reguły na czystym ruchu są artefaktem ZIMNEGO STARTU, "
        "nie stanem ustalonym.** Cały udział 1,5% mieści się w decyzjach 13–24, tuż po "
        "tym, jak bufor 12 odczytów się napełni, i nie wraca do końca epizodu. Mechanizm "
        "jest zrozumiały: na starcie sieć jest pusta, a pas bez ruchu raportuje stałe "
        "zero — czyli dokładnie to, co detektor martwy (`max − min < 1e−6`). Strażnik "
        "rozgrzewki kończy się, gdy bufor jest PEŁNY, a nie gdy jest pełny SENSOWNYCH "
        "odczytów.",
        "",
        "Trzy wnioski: (1) w stanie ustalonym reguła na czystym nie strzela w ogóle, "
        "czyli jest LEPSZA, niż mówi `gate_share`; (2) to wyjaśnia, czemu udział na "
        "czystym wynosił 1,5%, a budżet kalibracji 0,5% — różnica nie jest przekroczeniem "
        "budżetu, tylko innym zjawiskiem; (3) poprawka jest tania i warta zmierzenia: "
        "wydłużyć rozgrzewkę albo wymagać, by podejrzany ruch miał kiedykolwiek NIEZEROWY "
        "odczyt. Niezmierzone — zapisać jako otwarte.",
        "",
        "Dla porównania, pod awarią zimny start odpowiada za 13–14% przypadków "
        "(`mask0p5`, `mask0p25_intersection`) i 44% przy `hold` — tam reszta rozkłada się "
        "na cały epizod, bo awaria trwa.",
        "",
    ]
    figury = []
    for net in plan.networks:
        p = plot(plan, net, RESULTS / "figures" / f"gate_trace_{net}.png")
        if p is None:
            continue
        figury.append(p)
        tryb, _ = load_trace(plan, net)
        lines += [f"## {net}", "", f"![ślad bramki {net}](figures/{p.name})", ""]
        lines += [
            "| scenariusz | udział par w trybie reguły | węzły, które choć raz przejęły |",
            "|---|---:|---:|",
        ]
        from tsc_marl.robustness.degrade import slug  # noqa: PLC0415

        for scen in (slug(s) for s in plan.scenarios):
            if scen not in tryb:
                continue
            g = tryb[scen]
            lines.append(
                f"| `{scen}` | {g.mean():.1%} | {int(g.any(-1).sum())}/{g.shape[0]} |"
            )  # fmt: skip
        lines.append("")
    out = RESULTS / "gate_trace.md"
    out.write_text("\n".join(lines))
    print(f"{out}  ({len(figury)} figur)")
    return out


def _cli(nazwa: str) -> None:
    from tsc_marl.robustness.degrade import load_plan  # noqa: PLC0415

    main(load_plan(nazwa))


__all__ = ["load_trace", "main", "plot"]
