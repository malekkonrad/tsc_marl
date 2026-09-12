"""Raport testu gałęzi epistemic (plan `t2_ood`) -> `results/t2_ood.md`.

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
    lose: list[tuple[str, float]] = []
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
            core = clean[clean["controller"] == "gatppo"]["att"].mean()
            rules = [clean[clean["controller"] == r]["att"].mean() for r in RULES]
            rules = [r for r in rules if r == r]
            delta = "—"
            if core == core and rules:
                d = core / min(rules) - 1
                delta = f"{d:+.1%}" + (" ▲" if d > 0 else "")
                if d > 0 and net != base:
                    lose.append((net, d))
            out.append(f"| {traffic_label(net)} | " + " | ".join(cells) + f" | {delta} |")
        out += ["", "▲ = rdzeń gorszy od lepszego z MaxPressure — tu gałąź (a) miałaby robotę.", ""]

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
    out += _verdict(plan, lose, mi_rows, has_core)
    return _write(plan, out)


def _verdict(plan: Plan, lose, mi_rows, has_core: bool = True) -> list[str]:
    out = ["## Werdykt", ""]
    if not has_core:
        out.append("1. **Brak wyników rdzenia (`gatppo`)** — pytanie 1 bez odpowiedzi.")
    elif not lose:
        out.append(
            "1. **Rdzeń nie przegrywa z MaxPressure na żadnym ruchu spoza treningu.** Gałąź (a) "
            "dwupoziomowego fallbacku nie ma czego ratować — fallback jednopoziomowy do "
            "FixedTime jest wystarczający i uczciwszy."
        )
    else:
        worst = max(lose, key=lambda x: x[1])
        out.append(
            f"1. **Rdzeń przegrywa z MaxPressure na {len(lose)} wariantach ruchu**, najmocniej "
            f"na „{traffic_label(worst[0])}” ({worst[1]:+.1%}). Gałąź (a) ma tam robotę."
        )
    if mi_rows:
        by_base: dict[str, list[str]] = {}
        for net in mi_rows:
            by_base.setdefault(base_network(net), []).append(net)
        for base, nets in by_base.items():
            ref = mi_rows.get(base, {})
            if not ref:
                continue
            noise = ref["mi_gauss2"] - ref["mi_clean"]
            for net in nets:
                if net == base:
                    continue
                shift = mi_rows[net]["mi_clean"] - ref["mi_clean"]
                losing = any(n == net for n, _ in lose)
                verdict = (
                    "rośnie mocniej niż na szumie — sygnał odróżnia nowy ruch od awarii"
                    if shift > max(noise, 0)
                    else "rośnie słabiej niż na szumie albo spada — nie odróżnia nowego ruchu"
                )
                out.append(
                    f"* `{traffic_label(net)}`{' (rdzeń przegrywa)' if losing else ''}: "
                    f"MI {shift:+.4f} vs szum {noise:+.4f} na ruchu treningowym → {verdict}."
                )
        out += [
            "",
            "2. Gałąź (a) broni się tylko wtedy, gdy OBA warunki są spełnione na tym samym "
            "wariancie ruchu: rdzeń przegrywa z MaxPressure i MI rośnie tam mocniej niż "
            "na szumie. Każdy inny układ = fallback jednopoziomowy.",
        ]
    return out


def _write(plan: Plan, lines: list[str]) -> Path:
    path = RESULTS / f"{plan.name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
