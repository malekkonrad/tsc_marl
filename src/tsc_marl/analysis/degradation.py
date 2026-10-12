"""Degradation report of frozen controllers: tables and curves for one plan.

Reads runs `runs/t2/*__<plan>` and runs nothing. Two choices change the numbers:

* TWO-STAGE AVERAGING: first over failure DRAWS within a seed, then mean +- std over
  policy SEEDS, so the std is the spread between policies, not between draws.
* REFERENCE: degradation is relative to the CLEAN result of THE SAME controller in the
  same evaluator, not to T1 (ties can move ATT by ~1%, the size of small degradations).

Consistency check: the clean result of a learned controller MUST equal the last test
of its training (same weights, argmax, deterministic CityFlow); the report shows the
difference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.paths import RESULTS, RUNS
from tsc_marl.robustness.degrade import EXPERIMENT, LEARNED, Plan

# Colours follow the CONTROLLER, not the position, so a missing one does not repaint
# the rest. Slots of one categorical palette (checked for colour blindness); ramps of a
# single hue for sweeps of ONE parameter (threshold, tau), where darker = stronger.
COLORS = {
    "gatppo": "#2a78d6",
    "advancedcolight": "#eb6834",
    "efficientmp": "#1baf7a",
    "advancedmp": "#eda100",
    # ablation actor variants (t2_ablations)
    "gatppo_mean": "#e87ba4",
    "gatppo_nognn": "#4a3aa7",
    # mechanisms (t2_uw, t2_gate)
    "gatppo_uw": "#d63f6e",
    "gatppo_gate": "#7a4fd6",
    # gate threshold sweep (t2_gate): a ramp, light to dark
    "gatppo_gate_t0p25": "#b49ae8",
    "gatppo_gate_t0p50": "#8f66dd",
    "gatppo_gate_t0p75": "#6a3fc4",
    "gatppo_gate_t1p00": "#452a8a",
    # real gates (t2_gate_real): different mechanisms, so categorical
    "gatppo_gate_oracle": "#7a4fd6",
    "gatppo_gate_rule": "#1baf7a",
    "gatppo_gate_uq": "#eda100",
    # tau sweep (t2_uw_tau): a ramp; LOWER tau = STRONGER modulation = darker
    "gatppo_uw_t0p25": "#f3a6bd",
    "gatppo_uw_t0p10": "#d63f6e",
    "gatppo_uw_t0p05": "#8c1c40",
    # arms 2 and 3 of `t2_train_uw`
    "gatppo_fail": "#00a3a3",
    "gatppo_fail_uw": "#a8442c",
    # t2_uw_src: the source side is a darker shade of the same mechanism
    "gatppo_uw_src": "#8c1c40",
    # modulation oracle: the gate oracle's violet, the same role (ceiling)
    "gatppo_uw_oracle": "#7a4fd6",
}
LABELS = {
    "gatppo": "core GAT+PPO+CTDE",
    "advancedcolight": "Advanced-CoLight",
    "efficientmp": "Efficient-MP",
    "advancedmp": "Advanced-MP",
    "fixedtime": "FixedTime",
    "gatppo_mean": "actor mean-pool",
    "gatppo_nognn": "actor without GNN",
    "gatppo_ens": "ensemble of 5 actors (seed 0 decides)",
    "gatppo_uw": "core + UW-attention",
    "gatppo_gate": "core + gate (oracle)",
    "gatppo_gate_t0p25": "gate, threshold 0.25",
    "gatppo_gate_t0p50": "gate, threshold 0.50",
    "gatppo_gate_t0p75": "gate, threshold 0.75",
    "gatppo_gate_t1p00": "gate, threshold 1.00",
    "gatppo_gate_oracle": "gate: oracle (ceiling)",
    "gatppo_gate_rule": "gate: rule (persistence)",
    "gatppo_gate_uq": "gate: UQ (head)",
    "gatppo_uw_t0p25": "UW-attention, tau = 0.25",
    "gatppo_uw_t0p10": "UW-attention, tau = 0.10",
    "gatppo_uw_t0p05": "UW-attention, tau = 0.05",
    "gatppo_fail": "core trained with failures",
    "gatppo_fail_uw": "core with failures + UW-attention",
    "gatppo_fail_gate_rule": "core with failures + gate: rule",
    "gatppo_uw_src": "core + UW-attention (source side)",
    "gatppo_uw_oracle": "UW-attention: oracle (ceiling)",
}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

#: Plot panels = failure families, in order of observability.
FAMILIES = (
    ("Gaussian noise sigma", lambda s: s.kind == "gauss"),
    ("dead detectors, zero", lambda s: s.kind == "mask" and s.unit == "lane"
     and s.substitution == "zero"),
    ("dead intersections", lambda s: s.kind == "mask" and s.unit == "intersection"),
    ("dead without a trace, hold", lambda s: s.kind == "mask" and s.substitution == "hold"),
    ("miscalibration", lambda s: s.kind == "drift"),
)  # fmt: skip


def load(plan: Plan, metrics: tuple[str, ...] = ("att", "throughput")):
    """Long frame: controller, network, seed, scenario, draw + the `metrics` columns."""
    import pandas as pd

    from runlog import Experiment

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
        # only columns of THIS plan: a renamed column leaves old runs with a matching
        # suffix in `runs/t2`, which would enter as a ghost column
        if p["controller"] not in plan.controllers:
            continue
        wide["controller"], wide["network"], wide["seed"] = p["controller"], p["network"], p["seed"]
        wide["parent_run"] = p.get("parent_run")
        frames.append(wide)
    if not frames:
        return pd.DataFrame(
            columns=["controller", "network", "seed", "scenario", "draw", "att", "throughput"]
        )
    return pd.concat(frames, ignore_index=True)


def aggregate(df):
    """(controller, network, scenario) -> att_mean, att_std (over seeds), n_seeds, n_draws."""
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


#: Report titles per plan.
TITLES = {
    "t2_nomech": "T2 without a mechanism",
    "t2_ablations": "T2 for the aggregation ablations",
    "t2_uw": "T2 with UW-attention",
    "t2_ood": "T2 on traffic outside training",
    "t2_gate": "T2 with the fallback gate",
    "t2_gate_real": "T2 with a real gate (rule vs UQ)",
    "t2_uw_tau": "T2: tau sweep for UW-attention",
    "t2_train_uw": "T2 for the policy trained with the mechanism",
}

#: |t| threshold for a paired difference: 2.8 ~ Bonferroni over ~26 comparisons
#: (13 scenarios x 2 networks) at alpha = 0.05, since the table is read WHOLE.
T_THRESHOLD = 2.8


#: Column labels from the plan (`label:` in a controller spec), filled in `main`, taking
#: precedence over `LABELS`. A label is not part of the run fingerprint, so it can be
#: fixed without recomputing (in `t2_krakow` `gatppo_gate` is a RULE gate).
_PLAN_LABELS: dict[str, str] = {}


def label_of(column: str) -> str:
    return _PLAN_LABELS.get(column) or LABELS.get(column, column)


def mechanism_pairs(plan: Plan) -> list[tuple[str, str]]:
    """[(assessed column, reference column)]: the pairs for delta tables.

    Two sources: (a) a column with an inference MECHANISM (`policy_uw`, `policy_gate`,
    `policy_mask`), computed from the same checkpoint as its reference; (b) a column with
    an EXPLICIT `baseline:` of any kind, since an explicit reference declares "this
    comparison is a result" (a plain failure-trained arm used to reach no table).
    """
    out = []
    for label in plan.controllers:
        mechanisms = ("policy_uw", "policy_gate", "policy_mask")
        if plan.kind(label) not in mechanisms and not plan.baseline(label):
            continue
        # an explicit `baseline:` wins over the default (the first `policy` column of
        # the same agent), which is wrong when there are two such columns
        explicit = plan.baseline(label)
        if explicit:
            out.append((label, explicit))
            continue
        bases = [
            c
            for c in plan.controllers
            if plan.kind(c) == "policy" and plan.agent(c) == plan.agent(label)
        ]
        if bases:
            out.append((label, bases[0]))
    return out


def shared_checkpoint(plan: Plan, mechanism: str, base: str) -> bool:
    """Does the pair use THE SAME checkpoint? Decided by the parent template, not by
    whether the reference is explicit. Only the pair description depends on it; the
    pairing by (network, scenario, seed, draw) is the same."""
    from tsc_marl.robustness.degrade import Unit

    net = plan.networks[0]
    seed = int(plan.controllers[mechanism].get("seeds", [0])[0])
    same_file = plan.controllers[mechanism].get("checkpoint", "last") == plan.controllers[base].get(
        "checkpoint", "last"
    )  # best vs final of the same run are DIFFERENT weights
    return same_file and plan.parent_id(Unit(mechanism, net, seed)) == plan.parent_id(
        Unit(base, net, seed)
    )


#: mechanism kind -> (metrics from `result.json`, the sentence explaining them). Every
#: mechanism needs a "did it act at all" diagnostic, or an empty result cannot be told
#: from a mechanism that stayed silent.
DIAGNOSTICS = {
    "policy_uw": (
        ("uw_spread", "uw_active"),
        "`uw_spread` = mean log rho spread within a node's SOFTMAX GROUP, the only part of "
        "the signal attention sees (it is invariant to a constant shared by the group). "
        "0 = the layer sees no difference between neighbours, ln 10 ~ 2.3 = they differ "
        "10x. `uw_active` = share of edges with a nonzero log rho.",
    ),
    "policy_gate": (
        ("gate_share", "gate_switches", "gate_precision", "gate_recall"),
        "`gate_share` = share of (decision, node) pairs where the rule took control. "
        "`gate_switches` = number of mode CHANGES at nodes per episode; these hurt traffic "
        "(a switch resets the phase), not the share itself. `gate_precision` and "
        "`gate_recall` are relative to the ORACLE (the truth from the failure layer, "
        'diagnostic only) and separate "the detector is weak" from "switching gains '
        'little", two things ATT alone merges.',
    ),
    "policy_mask": (
        ("mask_share", "mask_forced"),
        "`mask_share` = share of (decision, node) pairs where the mask CHANGED the policy's "
        "action (not those where it was active and agreed). `mask_forced` = number of such "
        "overrides per episode. The mask targets STARVATION, not travel time, so its effect "
        "is read in the `wait_max` / `wait_max_demand` columns, not in `att`.",
    ),
    # a plain `policy` column with an explicit `baseline:`: the columns differ in
    # TRAINING, so there is nothing to diagnose per step (and a missing entry here
    # would be a KeyError mid-report)
    "policy": (
        (),
        "The columns differ in HOW THEY WERE TRAINED, not in a mechanism switched on at "
        'inference, so there is no "did the mechanism start" diagnostic. The spread '
        "between seeds enters the difference in full, and the number of INDEPENDENT "
        "replications is the number of seeds, not of pairs: failure draws share a "
        "checkpoint, so |t| is optimistic with respect to them.",
    ),
}


def mechanism_diagnostics(
    plan: Plan, controller: str, metrics: tuple[str, ...]
) -> dict[tuple[str, str], dict[str, float]]:
    """(network, scenario) -> means of `metrics` over seeds, from `result.json`. Without
    them the report cannot tell "the mechanism did not help" from "it did not start"."""
    from runlog import Experiment

    collected: dict[tuple[str, str], dict[str, list[float]]] = {}
    exp = Experiment(EXPERIMENT, root=RUNS)
    for run in exp.runs():
        if run.status != "ok" or not run.id.endswith(f"__{plan.name}"):
            continue
        if run.params.get("controller") != controller:
            continue
        net = run.params["network"]
        for key, value in run.result.items():
            for metric in metrics:
                if key.startswith(f"{metric}__"):
                    scen = key[len(metric) + 2 :]
                    collected.setdefault((net, scen), {}).setdefault(metric, []).append(
                        float(value)
                    )
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in collected.items()}


def paired_delta(df, mechanism: str, base: str):
    """ATT difference paired by (network, scenario, seed, draw).

    Both columns share the failure draws (and often the checkpoint), so the common
    spread cancels in the difference; an unpaired comparison drowns a 1 s effect in a
    30 s deviation, or shows a 3% difference that is noise once paired.
    """
    import pandas as pd
    from scipy.stats import ttest_1samp

    sub = df[df["controller"].isin([mechanism, base])]
    wide = sub.pivot_table(
        index=["network", "scenario", "seed", "draw"], columns="controller", values="att"
    ).dropna()
    if wide.empty:
        return pd.DataFrame()
    wide["delta"] = wide[mechanism] - wide[base]
    out = []
    for (net, scen), g in wide.groupby(["network", "scenario"]):
        d = g["delta"].to_numpy()
        t, pval = (
            ttest_1samp(d, 0.0) if len(d) > 1 and d.std() > 0 else (float("nan"), float("nan"))
        )
        # TIES counted separately and EXCLUDED from "better in": for a gate a tie (it did
        # not fire, the episode went identically) is the most common outcome
        out.append({
            "network": net, "scenario": scen, "n": len(d), "ties": int((d == 0).sum()),
            "delta": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "t": float(t), "p": float(pval), "better": int((d < 0).sum()),
        })  # fmt: skip
    return pd.DataFrame(out)


def parent_consistency(df, skip: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """The evaluator's clean result vs the last training test, for every learned run.

    `skip` = controllers whose difference is EXPECTED (a mechanism or another checkpoint
    changes the clean result by design); the check concerns controllers without one.
    """
    import pandas as pd

    from tsc_marl.libsignal import experiment as t1

    out = []
    runs = {r.id: r for r in t1.experiment().runs()}
    clean = df[(df["scenario"] == "clean") & df["parent_run"].notna()]
    for r in clean.itertuples():
        if r.controller in skip:
            continue
        parent = runs.get(r.parent_run)
        path = parent.path / "metrics.csv" if parent else None
        if path is None or not path.is_file():
            continue
        # only on the parent's TRAINING traffic; on another traffic file (t2_ood) a
        # difference is expected
        if parent.params.get("network") != r.network:
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
    return {s.slug(): s for s in plan.scenarios}


def _mechanism_section(plan: Plan, df, mechanism: str, base: str) -> list[str]:
    """The two tables that decide a plan with a mechanism: DID it act and DID it change
    ATT. Without the first the second cannot be interpreted."""
    metrics, explanation = DIAGNOSTICS[plan.kind(mechanism)]
    if plan.kind(mechanism) == "policy_gate" and plan.controllers[mechanism].get("uw"):
        # gate + UW composition: both diagnostics
        metrics = metrics + DIAGNOSTICS["policy_uw"][0]
        explanation = explanation + " " + DIAGNOSTICS["policy_uw"][1]
    diag = mechanism_diagnostics(plan, mechanism, metrics)
    pairs = paired_delta(df, mechanism, base) if df is not None and not df.empty else None
    if not diag and (pairs is None or pairs.empty):
        return []

    label = label_of(mechanism)
    # a plain `policy` column is not a MECHANISM; training separates it from the reference
    arm = plan.kind(mechanism) == "policy"
    what = "Arm" if arm else "Mechanism"
    better = "the arm is better" if arm else "the mechanism is better"
    neutral = "the arm is **neutral** for ATT" if arm else "the mechanism is **neutral** for ATT"
    lines = [
        f"## {what}: {label} vs {label_of(base)}",
        "",
        (
            "Both columns are computed from THE SAME checkpoint and the same failure draws "
            "(`draw` chooses which detectors failed), so they are compared **in pairs**, not "
            "through the +-std of the table above: the shared spread between seeds and draws "
            "cancels in the difference."
            if shared_checkpoint(plan, mechanism, base)
            else "The columns have DIFFERENT checkpoints (separate trainings), so pairing is by "
            "(network, scenario, seed, draw): traffic and failure pattern stay shared, the "
            "spread between seeds enters the difference. The reference column is set in the "
            f"plan (`baseline: {plan.baseline(mechanism)}`), not assumed."
        )
        + f" `delta` < 0 = {better}. `|t| >= "
        + f"{T_THRESHOLD}` is the threshold corrected for as many comparisons as this table has.",
        "",
        explanation,
        "",
    ]
    for net in plan.networks:
        rows = pairs[pairs["network"] == net] if pairs is not None and not pairs.empty else None
        if rows is None or rows.empty:
            continue
        order = [s for s in (x.slug() for x in plan.scenarios) if s in set(rows["scenario"])]
        lines += [
            f"### {net}",
            "",
            "| "
            + " | ".join(("scenario", *metrics, "delta ATT [s]", "sd", "t", "better in"))
            + " |",
            "|---|" + "---:|" * (len(metrics) + 4),
        ]
        for scen in order:
            r = rows[rows["scenario"] == scen].iloc[0]
            d = diag.get((net, scen), {})
            # shares as percentages, the rest (log rho spread, switch counts) as numbers
            cells = [
                (
                    "-"
                    if m not in d
                    else f"{d[m]:.1%}"
                    if m.endswith(("_active", "_share"))
                    else f"{d[m]:.2f}"
                )
                for m in metrics
            ]
            bold = "**" if abs(r["t"]) >= T_THRESHOLD else ""
            # sd = 0 (a deterministic scenario, e.g. `clean`) -> t undefined; "-", not nan
            t = "-" if not np.isfinite(r["t"]) else f"{r['t']:+.2f}"
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"`{scen}`",
                        *cells,
                        f"{bold}{r['delta']:+.2f}{bold}",
                        f"{r['sd']:.1f}",
                        t,
                        f"{int(r['better'])}/{int(r['n'] - r['ties'])}",
                    )
                )
                + " |"
            )
        significant = rows[rows["t"].abs() >= T_THRESHOLD]
        lines += [
            "",
            (
                f"* No scenario crosses the threshold: {neutral}."
                if significant.empty
                else "* Above the threshold: "
                + ", ".join(
                    f"`{r.scenario}` ({r.delta:+.2f} s, t = {r.t:+.2f}, better in "
                    f"{int(r.better)}/{int(r.n)} pairs)"
                    for r in significant.itertuples()
                )
            ),
            "",
        ]
    return lines


def _sweep_section(plan: Plan, df, pairs: list[tuple[str, str]]) -> list[str]:
    """HEADROOM CURVE: column x scenario -> delta ATT (and `gate_share`), one table
    instead of a section per column when all share ONE reference. Read by DIRECTION,
    not single `t`s: with a dozen cells one crosses the threshold by chance."""
    # rows per COLUMN, not per threshold: real gates (`rule`, `uq`) have no threshold in
    # the plan and would overwrite each other under one key
    mechanisms = [m for m, _ in pairs]
    bases = {b for _, b in pairs}
    if len(bases) != 1:
        raise ValueError(
            f"_sweep_section: one table = one reference, but the plan has {sorted(bases)}. "
            "Such a plan goes into separate sections (see the condition in `to_markdown`)."
        )
    base = pairs[0][1]

    with_share = any(
        any("gate_share" in v for v in mechanism_diagnostics(plan, m, ("gate_share",)).values())
        for m in mechanisms
    )
    lines = [
        "## Mechanisms x scenario",
        "",
        f"delta ATT [s] of the ASSESSED COLUMN relative to `{label_of(base)}`, paired by "
        "(seed, draw)"
        + (
            ". In brackets the share of (decision, node) pairs taken over by the rule"
            if with_share
            else ""
        )
        + f". **bold** = |t| >= {T_THRESHOLD}. delta < 0 = the assessed column is better than "
        "the reference.",
        "",
        "",
    ]
    for net in plan.networks:
        order, rows = None, []
        for mechanism in mechanisms:
            pairs_df = paired_delta(df, mechanism, base)
            if pairs_df.empty:
                continue
            sub = pairs_df[pairs_df["network"] == net]
            if sub.empty:
                continue
            diag = mechanism_diagnostics(
                plan, mechanism, ("gate_share", "gate_precision", "gate_recall")
            )
            rows.append((mechanism, sub.set_index("scenario"), diag))
            if order is None:
                order = [
                    s for s in (x.slug() for x in plan.scenarios) if s in sub["scenario"].values
                ]
        if not rows:
            continue
        lines += [
            f"### {net}",
            "",
            "| mechanism | " + " | ".join(f"`{s}`" for s in order) + " |",
            "|---|" + "---:|" * len(order),
        ]
        for mechanism, sub, diag in rows:
            cells = []
            for scen in order:
                r = sub.loc[scen]
                bold = "**" if abs(r["t"]) >= T_THRESHOLD else ""
                share = diag.get((net, scen), {}).get("gate_share")
                share_txt = "" if share is None else f" ({share:.0%})"
                cells.append(f"{bold}{r['delta']:+.1f}{bold}{share_txt}")
            lines.append(f"| {label_of(mechanism)} | " + " | ".join(cells) + " |")

        # detection quality, only for gates with an oracle diagnostic
        detection = [(m, d) for m, _, d in rows if any("gate_recall" in v for v in d.values())]
        if detection:
            lines += [
                "",
                'Detection quality relative to the oracle (precision / recall; "-" = nothing '
                "to detect or the gate did not fire):",
                "",
                "| mechanism | " + " | ".join(f"`{s}`" for s in order) + " |",
                "|---|" + "---:|" * len(order),
            ]
            for mechanism, diag in detection:
                cells = []
                for scen in order:
                    d = diag.get((net, scen), {})
                    p_, r_ = d.get("gate_precision"), d.get("gate_recall")
                    cells.append(
                        "-"
                        if p_ is None and r_ is None
                        else f"{'-' if p_ is None else f'{p_:.2f}'} / "
                        f"{'-' if r_ is None else f'{r_:.2f}'}"
                    )
                lines.append(f"| {label_of(mechanism)} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def to_markdown(
    plan: Plan, agg, checks: list[dict[str, Any]], figures: dict[str, Path], df=None
) -> str:
    specs = _spec_by_slug(plan)
    order = [s.slug() for s in plan.scenarios]
    ctrls = [c for c in plan.controllers if c != "fixedtime"]
    lines = [
        f"# {TITLES.get(plan.name, 'T2')}: plan `{plan.name}`",
        "",
        f"Mode {plan.mode} (yellow={plan.yellow}), {plan.draws} failure draws per scenario, "
        "evaluation of FROZEN controllers (`tsc degrade`). All read the same detector "
        "layer: MaxPressure computes its decision from the broken state vector, not the engine.",
        "",
        "Cell = ATT [s], the mean over draws, then over policy seeds (+- std over seeds); in "
        "brackets delta relative to the same controller's clean result. "
        "**(!) = worse than FixedTime**, the only place the fallback's lower layer can gain.",
        "",
    ]
    for net in plan.networks:
        sub = agg[agg["network"] == net] if not agg.empty else agg
        ft = sub[(sub["controller"] == "fixedtime") & (sub["scenario"] == "clean")]
        ft_att = float(ft["att_mean"].iloc[0]) if len(ft) else np.nan
        lines += [f"## {net}", "", f"FixedTime (independent of readings): **{ft_att:.1f} s**.", ""]
        present = [c for c in ctrls if len(sub) and (sub["controller"] == c).any()]
        missing = [c for c in ctrls if c not in present]
        if not present:
            lines += ["_No computed controllers for this network._", ""]
            continue
        head = "| scenario | " + " | ".join(
            f"{label_of(c)} (n={int(sub[sub['controller'] == c]['n_seeds'].max())})"
            for c in present
        )
        lines += [head + " |", "|---|" + "---:|" * len(present)]
        for name in order:
            cells = []
            for c in present:
                row = sub[(sub["controller"] == c) & (sub["scenario"] == name)]
                if row.empty:
                    cells.append("-")
                    continue
                r = row.iloc[0]
                std = f" +- {r.att_std:.1f}" if r.n_seeds > 1 else ""
                mark = " (!)" if r.att_mean > ft_att else ""
                delta = "" if name == "clean" else f" ({r.delta:+.1%})"
                cells.append(f"{r.att_mean:.1f}{std}{delta}{mark}")
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
        lines.append("")
        if missing:
            lines += [
                f"_Not computed on this machine: {', '.join(label_of(c) for c in missing)} "
                "(parent checkpoints missing, see `tsc degrade status`)._",
                "",
            ]
        lines += _findings(sub, present, ft_att, specs)
        if net in figures:
            lines += ["", f"![degradation curves {net}](figures/{figures[net].name})", ""]

    pairs = mechanism_pairs(plan)
    # one combined sweep table ONLY when all columns share ONE reference; a plan with
    # several (`t2_gate_dr`) gets separate sections, or the gate on the DR core would be
    # measured against the CLEAN core
    if len(pairs) > 1 and len({base for _, base in pairs}) == 1:
        lines += _sweep_section(plan, df, pairs)
    else:
        for mechanism, base in pairs:
            lines += _mechanism_section(plan, df, mechanism, base)

    if checks:
        worst = max(abs(c["diff"]) for c in checks)
        lines += [
            "## Check: the evaluator measures the same policy as T1",
            "",
            f"The evaluator's clean result vs the last training test, {len(checks)} learned "
            f"runs: largest difference **{worst:.3f} s**"
            + (
                ", consistent to the log's precision."
                if worst < 0.05
                else ", a MISMATCH, investigate!"
            ),
            "",
        ]
    lines += [
        "## What this table does NOT say",
        "",
        "* Levels are DETECTOR levels. For `gauss` they transfer to the feature with a factor "
        "of ~0.8-0.9, `urand` is ~1.6x weaker at the same `level` "
        "(`results/sensor_severity.md`).",
        "* A mask with `zero` substitution changes far fewer feature cells than the fraction "
        "of dead lanes (zero is the modal value); the same number does not mean the same "
        "thing on networks with different loads.",
        "* `hold` scenarios start at decision 60 (minute 15), so the first quarter of the "
        "episode is clean; their delta is not directly comparable with a mask from the start.",
    ]
    return "\n".join(lines) + "\n"


def _findings(sub, present, ft_att, specs) -> list[str]:
    """Conclusions computed from the table, not written in advance."""
    out = []
    for c in present:
        rows = sub[(sub["controller"] == c) & (sub["scenario"] != "clean")]
        if rows.empty:
            continue
        worst = rows.loc[rows["att_mean"].idxmax()]
        above = rows[rows["att_mean"] > ft_att]["scenario"].tolist()
        text = (
            f"* **{label_of(c)}**: worst scenario `{worst.scenario}`: "
            f"{worst.att_mean:.1f} s ({worst.delta:+.1%}). "
        )
        if above:
            text += (
                f"Worse than FixedTime in {len(above)} scenarios: "
                + ", ".join(f"`{a}`" for a in above)
                + "."
            )
        else:
            text += (
                "Never falls below FixedTime: at these levels a fallback to FixedTime has "
                "nothing to gain."
            )
        out.append(text)
    return out + [""]


# --------------------------------------------------------------------- plot


def plot(plan: Plan, agg, network: str, path: Path) -> Path | None:
    """Small multiples: one panel per failure family, a shared Y axis (ATT)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = agg[agg["network"] == network]
    if sub.empty:
        return None
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
                row = sub[(sub["controller"] == c) & (sub["scenario"] == s.slug())]
                if row.empty:
                    continue
                xs.append(s.level)
                ys.append(float(row["att_mean"].iloc[0]))
                sd.append(float(row["att_std"].iloc[0]))
            xs, ys, sd = np.array(xs), np.array(ys), np.array(sd)
            ax.plot(xs, ys, color=COLORS[c], lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.5,
                    solid_capstyle="round", label=label_of(c), zorder=3)  # fmt: skip
            if plan.kind(c) in LEARNED and (sd > 0).any():
                ax.fill_between(xs, ys - sd, ys + sd, color=COLORS[c], alpha=0.10, lw=0)
        if ft_att is not None:
            ax.axhline(ft_att, color=INK2, lw=1, zorder=2)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("failure level (detector)", color=INK2)
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
    return path


def main(plan: Plan) -> Path:
    _PLAN_LABELS.clear()
    _PLAN_LABELS.update(
        {c: spec["label"] for c, spec in plan.controllers.items() if spec.get("label")}
    )
    df = load(plan)
    agg = aggregate(df)
    figures = {}
    for net in plan.networks:
        p = plot(plan, agg, net, RESULTS / "figures" / f"{plan.name}_{net}.png")
        if p is not None:
            figures[net] = p
    # left out of the check: a mechanism (another policy than in the training test) or
    # the BEST checkpoint (other weights than the LAST training test)
    with_mechanism = frozenset(
        c
        for c, spec in plan.controllers.items()
        if plan.kind(c) in ("policy_uw", "policy_gate") or spec.get("checkpoint") == "best"
    )
    checks = parent_consistency(df, with_mechanism) if not df.empty else []
    out = RESULTS / f"{plan.name}.md"
    out.write_text(to_markdown(plan, agg, checks, figures, df), encoding="utf-8")
    if not agg.empty:
        agg.to_csv(RESULTS / f"{plan.name}.csv", index=False)
    return out
