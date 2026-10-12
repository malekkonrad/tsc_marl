"""Report of the epistemic branch test (plan `t2_ood`) -> `results/t2_ood_epistemic.md`.

The two-level fallback sends "epistemic up on clean input" to MaxPressure. Two
questions decide that branch:

1. Does the FROZEN policy lose to MaxPressure on traffic outside training,
   especially HEAVIER traffic? If it never loses, the branch has nothing to do.
2. Does the disagreement of 5 actors (`gatppo_ens`) grow where the policy loses,
   and MORE than under sensor failure? Only then can it be told apart from the
   observation channel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsc_marl.paths import RESULTS
from tsc_marl.robustness.degrade import Plan, base_network

RULES = ("efficientmp", "advancedmp")
ALPHA = 0.05
#: Relative MI rise on clean input (vs training traffic) from which the ensemble is
#: taken to REACT to new traffic; below it is the size of between-seed differences.
MI_RISE = 0.25


def _p_worse(vals, reference: float) -> float:
    """One-sided t-test p: is the policy's seed mean GREATER (worse) than the rule."""
    from scipy.stats import ttest_1samp

    if len(vals) < 2 or np.std(vals) == 0:
        return float("nan")
    return float(ttest_1samp(vals, reference, alternative="greater").pvalue)


METRICS = ("att", "throughput", "ens_mi", "ens_disagree")


def traffic_label(network: str) -> str:
    """Readable name of a traffic variant from the network suffix."""
    suffix = network[len(base_network(network)) :]
    if not suffix:
        return "training traffic"
    if suffix.startswith("_d"):
        return f"demand x{suffix[2:].replace('p', '.')}"
    return f"another traffic file ({suffix[2:]})"


def _per_seed(df):
    num = df.select_dtypes("number").columns.difference(["draw", "seed"])
    return df.groupby(["controller", "network", "scenario", "seed"])[list(num)].mean().reset_index()


def _cell(vals) -> str:
    vals = [v for v in vals if v == v]
    if not vals:
        return "-"
    sd = f" +- {np.std(vals, ddof=1):.1f}" if len(vals) > 1 else ""
    return f"{np.mean(vals):.1f}{sd}"


def main(plan: Plan) -> Path:
    from tsc_marl.analysis import degradation

    df = degradation.load(plan, metrics=METRICS)
    out = [
        f"# Epistemic branch test: plan `{plan.name}`",
        "",
        "Frozen controllers on traffic outside training. Learned policies take weights from "
        "the TRAINING network (`{base}` in the parent template), traffic comes from another "
        "file. The clean scenario = clean readings; `gauss2` = the same traffic with "
        "detector noise, to compare the ensemble's reaction to new traffic with its "
        "reaction to sensor failure.",
        "",
    ]
    if df.empty:
        out.append("_No computed runs; `tsc degrade run " + plan.name + "`._")
        return _write(plan, out)

    per = _per_seed(df)
    lose: list[tuple[str, float, float]] = []  # (network, delta, p): a SIGNIFICANT loss
    ties: list[tuple[str, float, float]] = []  # delta > 0 but within the seed spread
    mi_rows: dict[str, dict[str, float]] = {}
    for base in dict.fromkeys(base_network(n) for n in plan.networks):
        nets = [n for n in plan.networks if base_network(n) == base]
        out += [f"## Policies trained on `{base}`", "", "### ATT on clean readings", ""]
        ctrls = [c for c in ("fixedtime", *RULES, "advancedcolight", "gatppo")
                 if (per["controller"] == c).any()]  # fmt: skip
        out += ["| traffic | " + " | ".join(ctrls) + " | core vs better rule |",
                "|---|" + "---:|" * (len(ctrls) + 1)]  # fmt: skip
        for net in nets:
            clean = per[(per["network"] == net) & (per["scenario"] == "clean")]
            cells = [_cell(clean[clean["controller"] == c]["att"].tolist()) for c in ctrls]
            core_vals = clean[clean["controller"] == "gatppo"]["att"].dropna().to_numpy()
            rules = [clean[clean["controller"] == r]["att"].mean() for r in RULES]
            rules = [r for r in rules if r == r]
            delta = "-"
            if len(core_vals) and rules:
                best = min(rules)  # rules are deterministic: one number
                d = core_vals.mean() / best - 1
                p = _p_worse(core_vals, best)
                sig = d > 0 and p < ALPHA
                delta = f"{d:+.1%} (p {p:.2f})" + (" (!)" if sig else "")
                if d > 0 and net != base:
                    (lose if sig else ties).append((net, d, p))
            out.append(f"| {traffic_label(net)} | " + " | ".join(cells) + f" | {delta} |")
        out += [
            "",
            f"(!) = the core SIGNIFICANTLY worse than the better MaxPressure (one-sided t, "
            f"p < {ALPHA}, policy seeds vs the deterministic rule); only there would branch "
            "(a) have work.",
            "",
        ]

        ens = per[per["controller"] == "gatppo_ens"]
        if not ens.empty:
            k = max((len(v.get("ensemble", [])) for v in plan.controllers.values()), default=0)
            out += [
                f"### Disagreement of {k} actors (epistemic in the policy)",
                "",
                "| traffic | MI clean | MI noise sigma=2 | disagreeing votes clean "
                "| disagreeing votes noise |",
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
    """Conclusions from the numbers. The OBSERVATION channel decides first (sensor
    failure -> FixedTime), so branch (a) only needs MI to rise on CLEAN input under new
    traffic, and something to rescue: a significant loss of the policy to MaxPressure."""
    out = ["## Verdict", ""]
    if not has_core:
        out.append("1. **No core results (`gatppo`)**: question 1 unanswered.")
    elif lose:
        worst = max(lose, key=lambda x: x[1])
        out.append(
            f"1. **The core loses SIGNIFICANTLY to MaxPressure on {len(lose)} traffic "
            f'variants**, most on "{traffic_label(worst[0])}" ({worst[1]:+.1%}, '
            f"p {worst[2]:.3f}). Branch (a) has work there."
        )
    else:
        text = (
            "1. **The core does not lose significantly to MaxPressure on any traffic "
            "outside training.**"
        )
        if ties:
            text += (
                " A tie within the seed spread: "
                + ", ".join(f'"{traffic_label(n)}" {d:+.1%} (p {p:.2f})' for n, d, p in ties)
                + "."
            )
        text += (
            " Switching to MaxPressure has nothing to gain here, so branch (a) of the "
            "two-level fallback has no work: a single-level fallback to FixedTime."
        )
        out.append(text)
    rising: list[str] = []
    if mi_rows:
        out += [
            "",
            "Disagreement of 5 actors (MI) on CLEAN input, relative to training traffic:",
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
                    f"* `{traffic_label(net)}`: MI {shift:+.4f} ({rel:+.0%}); for comparison, "
                    f"noise sigma=2 on training traffic {noise:+.4f} -> "
                    + ("**rises**: a new-traffic signal exists" if up else "does not rise")
                    + (
                        ""
                        if not up or shift > noise
                        else " (weaker than under noise: indistinguishable without the "
                        "observation channel)"
                    )
                    + "."
                )
        both = [n for n, _, _ in lose if n in rising]
        out += [
            "",
            "2. **Branch (a) holds only where BOTH conditions hold on the same traffic:** "
            "a significant loss to MaxPressure and an MI rise on clean input. "
            + (
                f"They hold on: {', '.join(traffic_label(n) for n in both)}."
                if both
                else "They hold nowhere"
                + (
                    f'; MI rises on "{", ".join(traffic_label(n) for n in rising)}", '
                    "but there the policy does not lose, so there is nothing to rescue."
                    if rising
                    else "."
                )
            ),
        ]
    return out


def _write(plan: Plan, lines: list[str]) -> Path:
    # a separate file: `tsc degrade report <plan>` writes `<plan>.md`
    path = RESULTS / f"{plan.name}_epistemic.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
