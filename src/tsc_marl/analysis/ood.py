"""Raport testu gałęzi epistemic (plan `t2_ood`) -> `results/t2_ood_epistemic.md`.

Dwupoziomowy fallback z planu kieruje „epistemic↑ przy czystym wejściu” do MaxPressure.
Bramka 2.0 pokazała, że na naszych zbiorach tej sytuacji nie widać: na lżejszym ruchu
model nie traci trafności, więc nie ma czego ratować. Ten raport odpowiada wprost na
dwa pytania, które rozstrzygają los tej gałęzi:

1. Czy ZAMROŻONA polityka przegrywa z MaxPressure na ruchu spoza treningu —
   szczególnie CIĘŻSZYM (HangZhou ×1,5 i plik 5816)? Jeśli nigdzie nie przegrywa,
   gałąź nie ma roboty, niezależnie od jakości sygnału epistemic.
2. Czy rozbieżność 5 actorów (epistemic w samej polityce RL, sterownik `gatppo_ens`)
   rośnie tam, gdzie polityka przegrywa — i czy rośnie tam MOCNIEJ niż na awarii
   sensora? Tylko wtedy da się ją odróżnić od kanału obserwacyjnego.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsc_marl.paths import RESULTS
from tsc_marl.robustness.degrade import Plan, base_network

RULES = ("efficientmp", "advancedmp")
ALPHA = 0.05
#: Wzrost MI na czystym wejściu (względem ruchu treningowego), od którego uznajemy, że
#: rozbieżność polityk REAGUJE na nowy ruch. Niżej to rząd różnic między seedami.
MI_RISE = 0.25


def _p_worse(vals, reference: float) -> float:
    """p jednostronnego testu t: czy średnia seedów polityki jest WIĘKSZA (gorsza) od reguły."""
    from scipy.stats import ttest_1samp  # noqa: PLC0415

    if len(vals) < 2 or np.std(vals) == 0:
        return float("nan")
    return float(ttest_1samp(vals, reference, alternative="greater").pvalue)


METRICS = ("att", "throughput", "ens_mi", "ens_disagree")


def traffic_label(network: str) -> str:
    """Czytelna nazwa wariantu ruchu z sufiksu sieci."""
    suffix = network[len(base_network(network)) :]
    if not suffix:
        return "ruch treningowy"
    if suffix.startswith("_d"):
        return f"popyt ×{suffix[2:].replace('p', ',')}"
    return f"inny plik ruchu ({suffix[2:]})"


def _per_seed(df):
    num = df.select_dtypes("number").columns.difference(["draw", "seed"])
    return df.groupby(["controller", "network", "scenario", "seed"])[list(num)].mean().reset_index()


def _cell(vals) -> str:
    vals = [v for v in vals if v == v]
    if not vals:
        return "—"
    sd = f" ± {np.std(vals, ddof=1):.1f}" if len(vals) > 1 else ""
    return f"{np.mean(vals):.1f}{sd}"


def main(plan: Plan) -> Path:
    from tsc_marl.analysis import degradation  # noqa: PLC0415

    df = degradation.load(plan, metrics=METRICS)
    out = [
        f"# Test gałęzi epistemic — plan `{plan.name}`",
        "",
        "Zamrożone sterowniki na ruchu spoza treningu. Polityki uczone biorą wagi z sieci "
        "TRENINGOWEJ (`{base}` w szablonie rodzica), ruch jest z innego pliku. Czysty "
        "scenariusz = czyste odczyty; `gauss2` = ten sam ruch z szumem detektorów, żeby "
        "porównać reakcję ensemble na nowy ruch z reakcją na awarię sensora.",
        "",
    ]
    if df.empty:
        out.append("_Brak policzonych biegów — `tsc degrade run " + plan.name + "`._")
        return _write(plan, out)

    per = _per_seed(df)
    lose: list[tuple[str, float, float]] = []  # (sieć, Δ, p) — przegrana ISTOTNA
    ties: list[tuple[str, float, float]] = []  # Δ > 0, ale w granicach rozrzutu seedów
    mi_rows: dict[str, dict[str, float]] = {}
    for base in dict.fromkeys(base_network(n) for n in plan.networks):
        nets = [n for n in plan.networks if base_network(n) == base]
        out += [f"## Polityki uczone na `{base}`", "", "### ATT na czystych odczytach", ""]
        ctrls = [c for c in ("fixedtime", *RULES, "advancedcolight", "gatppo")
                 if (per["controller"] == c).any()]  # fmt: skip
        out += ["| ruch | " + " | ".join(ctrls) + " | rdzeń vs lepsza reguła |",
                "|---|" + "---:|" * (len(ctrls) + 1)]  # fmt: skip
        for net in nets:
            clean = per[(per["network"] == net) & (per["scenario"] == "clean")]
            cells = [_cell(clean[clean["controller"] == c]["att"].tolist()) for c in ctrls]
            core_vals = clean[clean["controller"] == "gatppo"]["att"].dropna().to_numpy()
            rules = [clean[clean["controller"] == r]["att"].mean() for r in RULES]
            rules = [r for r in rules if r == r]
            delta = "—"
            if len(core_vals) and rules:
                best = min(rules)  # reguły są deterministyczne — jedna liczba
                d = core_vals.mean() / best - 1
                p = _p_worse(core_vals, best)
                sig = d > 0 and p < ALPHA
                delta = f"{d:+.1%} (p {p:.2f})" + (" ▲" if sig else "")
                if d > 0 and net != base:
                    (lose if sig else ties).append((net, d, p))
            out.append(f"| {traffic_label(net)} | " + " | ".join(cells) + f" | {delta} |")
        out += [
            "",
            f"▲ = rdzeń ISTOTNIE gorszy od lepszego z MaxPressure (jednostronny t, p < {ALPHA}, "
            "seedy polityki vs deterministyczna reguła) — tylko tam gałąź (a) miałaby robotę.",
            "",
        ]

        ens = per[per["controller"] == "gatppo_ens"]
        if not ens.empty:
            k = max((len(v.get("ensemble", [])) for v in plan.controllers.values()), default=0)
            out += [
                f"### Rozbieżność {k} actorów (epistemic w polityce)",
                "",
                "| ruch | MI czyste | MI szum σ=2 | niezgodne głosy czyste "
                "| niezgodne głosy szum |",
                "|---|---:|---:|---:|---:|",
            ]
            for net in nets:
                row = {}
                for scen in ("clean", "gauss2"):
                    sub = ens[(ens["network"] == net) & (ens["scenario"] == scen)]
                    row[f"mi_{scen}"] = float(sub["ens_mi"].mean()) if len(sub) else np.nan
                    row[f"dis_{scen}"] = float(sub["ens_disagree"].mean()) if len(sub) else np.nan
                mi_rows[net] = row
                out.append(
                    f"| {traffic_label(net)} | {row['mi_clean']:.4f} | {row['mi_gauss2']:.4f} "
                    f"| {row['dis_clean']:.1%} | {row['dis_gauss2']:.1%} |"
                )
            out.append("")
    has_core = bool((per["controller"] == "gatppo").any())
    out += _verdict(plan, lose, ties, mi_rows, has_core)
    return _write(plan, out)


def _verdict(plan: Plan, lose, ties, mi_rows, has_core: bool = True) -> list[str]:
    """Wnioski z liczb. Logika bramki z planu: kanał OBSERWACYJNY rozstrzyga pierwszy
    (awaria sensora → FixedTime), więc gałąź (a) potrzebuje tylko, żeby MI rosło na
    CZYSTYM wejściu przy nowym ruchu — nie musi przebijać reakcji na szum. Musi za to
    istnieć coś do ratowania: istotna przegrana polityki z MaxPressure."""
    out = ["## Werdykt", ""]
    if not has_core:
        out.append("1. **Brak wyników rdzenia (`gatppo`)** — pytanie 1 bez odpowiedzi.")
    elif lose:
        worst = max(lose, key=lambda x: x[1])
        out.append(
            f"1. **Rdzeń przegrywa ISTOTNIE z MaxPressure na {len(lose)} wariantach ruchu**, "
            f"najmocniej na „{traffic_label(worst[0])}” ({worst[1]:+.1%}, p {worst[2]:.3f}). "
            "Gałąź (a) ma tam robotę."
        )
    else:
        text = "1. **Rdzeń nie przegrywa istotnie z MaxPressure na żadnym ruchu spoza treningu.**"
        if ties:
            text += (
                " Remis w granicach rozrzutu seedów: "
                + ", ".join(f"„{traffic_label(n)}” {d:+.1%} (p {p:.2f})" for n, d, p in ties)
                + "."
            )
        text += (
            " Przełączenie na MaxPressure nie ma tu nic do zyskania, więc gałąź (a) "
            "dwupoziomowego fallbacku nie ma roboty — fallback jednopoziomowy do FixedTime."
        )
        out.append(text)
    rising: list[str] = []
    if mi_rows:
        out += [
            "",
            "Rozbieżność 5 actorów (MI) na CZYSTYM wejściu, względem ruchu treningowego:",
            "",
        ]
        for base in dict.fromkeys(base_network(n) for n in mi_rows):
            ref = mi_rows.get(base, {})
            if not ref:
                continue
            noise = ref["mi_gauss2"] - ref["mi_clean"]
            for net in (n for n in mi_rows if base_network(n) == base and n != base):
                shift = mi_rows[net]["mi_clean"] - ref["mi_clean"]
                rel = shift / ref["mi_clean"] if ref["mi_clean"] else float("nan")
                up = rel >= MI_RISE
                if up:
                    rising.append(net)
                out.append(
                    f"* `{traffic_label(net)}`: MI {shift:+.4f} ({rel:+.0%}); dla porównania szum "
                    f"σ=2 na ruchu treningowym {noise:+.4f} → "
                    + ("**rośnie** — sygnał nowego ruchu istnieje" if up else "nie rośnie")
                    + (
                        ""
                        if not up or shift > noise
                        else " (słabiej niż pod szumem: bez kanału obserwacyjnego nieodróżnialny)"
                    )
                    + "."
                )
        both = [n for n, _, _ in lose if n in rising]
        out += [
            "",
            "2. **Gałąź (a) broni się tylko tam, gdzie OBA warunki zachodzą na tym samym ruchu:** "
            "istotna przegrana z MaxPressure i wzrost MI na czystym wejściu. "
            + (
                f"Zachodzą na: {', '.join(traffic_label(n) for n in both)}."
                if both
                else "Nie zachodzą nigdzie"
                + (
                    f" — MI rośnie na „{', '.join(traffic_label(n) for n in rising)}”, "
                    "ale tam polityka nie przegrywa, więc nie ma czego ratować."
                    if rising
                    else "."
                )
            ),
        ]
    return out


def _write(plan: Plan, lines: list[str]) -> Path:
    # osobny plik: `tsc degrade report <plan>` pisze `<plan>.md` (tabela degradacji)
    # i bez sufiksu jeden raport nadpisywałby drugi
    path = RESULTS / f"{plan.name}_epistemic.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
