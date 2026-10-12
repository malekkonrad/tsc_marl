"""Thesis tables and reports from stored runs, never running anything.

Separate from `runner/suite.py` on purpose: this module READS `runlog` and writes
markdown to `results/`, with no path that could start a simulation.

    tsc table          # rebuild the tables from runs/
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime

from tsc_marl.paths import RESULTS

# Literature numbers for THIS protocol (4 phases, dt=15): Zhang 2022 (Advanced-XLight),
# Table 2, columns JiNan-1 (`anon_3_4_jinan_real.json`) and HangZhou-1.
# advancedmp/efficientmp: the same definitions as implemented in advanced_mp.py.
LITERATURE = {
    "cityflow_jinan3x4_4ph": {
        "fixedtime": 428.11,
        "maxpressure": 273.96,
        "efficientmp": 269.87,
        "advancedmp": 253.61,
        "colight": 272.06,
        "mplight": 297.46,
        "advancedcolight": 245.73,
    },
    "cityflow_hangzhou4x4_4ph": {
        "fixedtime": 495.57,
        "maxpressure": 288.54,
        "efficientmp": 284.44,
        "advancedmp": 279.47,
        "colight": 297.02,
        "mplight": 314.60,
        "advancedcolight": 270.45,
    },
}

# Phase 1 GATE (`configs/suites/p1_gate.yml`): the core must reach the ATT of
# Advanced-CoLight "within a few percent" on the same networks and seeds, in realistic
# mode. The tolerance is the UPPER bound of that phrase, fixed here so the verdict does not
# depend on the reader. The report also gives the smallest tolerance at which the verdict
# holds, since the value was set after seeing the results.
GATE_REFERENCE = "advancedcolight"
GATE_CORE = "gatppo"
GATE_YELLOW = 5
GATE_TOLERANCE = 0.05
GATE_NETWORKS = ("cityflow_jinan3x4_4ph", "cityflow_hangzhou4x4_4ph")
# Report window as `REPORT_WINDOW` in `core/loop.py` (importing it would pull in torch).
WINDOW = 10
# Stability: the worst test in the last TAIL episodes and the first episode from which
# a rolling WINDOW of tests stays within CONVERGED of the final result.
TAIL = 50
CONVERGED = 0.02
# Gate learning curves next to the report, so the markdown links them relatively.
FIGURES = RESULTS / "figures"

# Run VARIANT = the `run_id` suffix after `__s<seed>_y<yellow>`. Grids with `overrides`
# (ablations, `p1_abl_*`) share agent, network, seed and yellow with the core and differ
# only in config fingerprint and suffix; without this column they would mix into the core
# row of T1 and the `gatppo` side of the GATE.
RUN_SUFFIX = re.compile(r"__s\d+_y\d+(?:__(?P<variant>.+))?$")

# Core ablations: variant -> label. "" = the GATE core (GAT, ATS state).
AGGREGATION = {"": "GAT (core)", "p1_abl_meanpool": "mean-pool (GraphSAGE)",
               "p1_abl_nognn": "no GNN (MLP)"}  # fmt: skip
STATE = {"": "ATS (core)", "p1_abl_state_ep": "EP", "p1_abl_state_pressure": "pressure",
         "p1_abl_state_queue": "queue"}  # fmt: skip


def variant_of(run_id: str) -> str:
    m = RUN_SUFFIX.search(run_id)
    return (m.group("variant") or "") if m else ""


def build_baseline_report(rows):
    """T1 baseline: our numbers next to the literature, for both reporting modes.
    A cell of a learned method is the MEAN over seeds; variant runs (ablations,
    sanity) are left out, they are not the same method."""
    acc = {}
    for r in rows:
        if r.get("variant"):
            continue
        v = r.get("att_last10")
        v = r.get("att") if v is None or v != v else v
        # yellow_length is a number (run parameters are typed, unlike CSV)
        acc.setdefault((r["network"], r["agent"], r.get("yellow_length")), []).append(float(v))
    idx = {k: {"att_last10": sum(v) / len(v), "att": sum(v) / len(v), "n": len(v)}
           for k, v in acc.items()}  # fmt: skip
    out = [
        "# T1 baseline: the reference point of the thesis",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}. "
        "Protocol: `configs/tsc/thesis_protocol.yml` "
        "(dt=15 s, episode 3600 s, `PYTHONHASHSEED=0`), networks in the 4-phase variant, "
        "as in the papers' protocol.",
        "",
        "ATT = mean of the last 10 test episodes (methods without training: one run).",
        "delta is computed against the literature in **benchmark** mode, the mode the "
        "papers report in.",
        "",
    ]
    for net, lit in LITERATURE.items():
        out += [
            f"## {net}",
            "",
            "| agent | benchmark (yellow=0) | realistic (yellow=5) | literature "
            "| delta benchmark |",
            "|---|---|---|---|---|",
        ]
        for agent in (
            "fixedtime",
            "maxpressure",
            "sotl",
            "efficientmp",
            "advancedmp",
            "presslight",
            "colight",
            "advancedcolight",
        ):
            b = idx.get((net, agent, 0))
            r5 = idx.get((net, agent, 5))
            if not b and not r5:
                continue

            def val(x):
                if not x:
                    return "-"
                n = f" (n={x['n']})" if x["n"] > 1 else ""
                return f"{float(x['att_last10']):.1f}{n}"

            ref = lit.get(agent)
            d = "-"
            if b and ref:
                got = float(b["att_last10"] or b["att"])
                d = f"{(got - ref) / ref * 100:+.1f}%"
            out.append(f"| {agent} | {val(b)} | {val(r5)} | {ref if ref else '-'} | {d} |")
        out.append("")
    out += [
        "## How to read delta",
        "",
        "- **`maxpressure` is the LibSignal implementation**, computing pressure from",
        "  `lane_count` (all vehicles, no per-lane normalisation) rather than the paper's",
        "  definition; its deviation is NOT a gate criterion. `efficientmp` plays the role",
        "  of a correct MP.",
        "- **Reproducibility depends on the network**: JiNan matches the literature to ~1%",
        "  (FixedTime -0.8%, Efficient-MP +0.9%), HangZhou does not (Efficient-MP +10.4%).",
        "  Same protocol, same code, data verified identical to the authors' files",
        "  (`hangzhou_4x4_advanced/` vs `hangzhou_4x4_gudang_18041610_1h/`: results agree",
        "  to 0.01 s). To be described in the thesis as a limitation of the comparison.",
        "- **`advancedmp` is reported at a global W1 = 1.0.** In benchmark mode the demand",
        "  term does nothing (without a transition cost holding a phase gains nothing), so",
        "  the method degenerates to Efficient-MP and +8.2% against the literature is",
        "  EXPECTED. The paper's Advanced-MP number cannot be reproduced in benchmark mode",
        "  at any W1, the strongest argument that this mode loses the mechanism the paper",
        "  describes.",
        "- The `realistic` column is our protocol (5 s phase transition). The difference",
        "  between the modes is a result in itself, see `t1_w1_sweep.md`.",
        "",
    ]
    (RESULTS / "t1_baseline.md").write_text("\n".join(out))
    print("\n".join(out))


def training_results(exp):
    """Successful TRAINING/measurement runs, without evaluations of frozen policies
    (those have `parent_run` and describe a scenario, not a method)."""
    df = exp.results(status="ok")
    if df.empty:
        return df
    if "parent_run" in df.columns:
        df = df[df["parent_run"].isna()]
    df = df.copy()
    df["variant"] = df["run_id"].map(variant_of)
    return df


def _records(exp):
    """Table rows as dicts (params + result)."""
    df = training_results(exp)
    return [] if df.empty else df.to_dict("records")


def build_table(exp):
    """Table of mean +- std over seeds. std=0 for methods without training is EXPECTED:
    the simulation is deterministic, the seed only moves initialisation/exploration."""
    df = training_results(exp)
    if df.empty:
        print("No computed runs in runs/; run something first.", file=sys.stderr)
        return

    # `advancedmp` at different W1 values are different methods, and so is a variant
    # (run_id suffix), see RUN_SUFFIX
    df["label"] = df.apply(
        lambda r: (
            (
                f"{r['agent']} (W1={r['w1']})"
                if r["agent"] == "advancedmp" and r.get("w1")
                else r["agent"]
            )
            + (f" [{r['variant']}]" if r["variant"] else "")
        ),
        axis=1,
    )
    # the mode (yellow_length) MUST be in the key, or realistic and benchmark get averaged
    agg = exp.aggregate(
        by=["network", "label", "yellow_length"],
        over="seed",
        metrics=["att", "att_last10", "throughput", "queue"],
        df=df,  # `label` is derived, not a stored parameter
    ).sort_values(["network", "label", "yellow_length"])

    lines = [
        "| network | agent | yellow | ATT (last 10 tests) | ATT (last) | throughput | queue | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in agg.to_dict("records"):
        sd = f" +- {r['att_std']:.1f}" if r["n"] > 1 and r["att_std"] == r["att_std"] else ""
        main = (
            f"{r['att_last10_mean']:.1f}" if r["att_last10_mean"] == r["att_last10_mean"] else "-"
        )
        lines.append(
            f"| {r['network']} | {r['label']} | {r['yellow_length']} | {main} | "
            f"{r['att_mean']:.1f}{sd} | {r['throughput_mean']:.0f} | {r['queue_mean']:.1f} | "
            f"{int(r['n'])} |"
        )
    build_baseline_report(_records(exp))
    build_gate_report(exp)
    build_ablation_report(exp)
    table = "\n".join(lines)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "t1_table.md").write_text(
        f"# T1: baselines on the clean benchmark\n\n"
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n{table}\n"
    )
    print(table)


def _curve_stats(curves, run_id):
    """Worst tail test and convergence episode from one run's test ATT curve."""
    att = (
        curves[(curves["run_id"] == run_id) & (curves["split"] == "test")]
        .loc[lambda c: c["metric"] == "att"]
        .sort_values("step")["value"]
        .reset_index(drop=True)
    )
    final = att.iloc[-WINDOW:].mean()
    rolling = att.rolling(WINDOW).mean()
    hit = rolling[rolling <= final * (1 + CONVERGED)]
    return {
        "tail_worst": att.iloc[-TAIL:].max(),
        "converged_at": float(hit.index[0]) if len(hit) else float("nan"),
    }


def _device(run_path, provenance):
    """The run's device: from the core log ("| device cuda:0"), else from `--ngpu` in
    argv (which alone is not proof: without CUDA the run falls back to the CPU)."""
    for log in sorted(run_path.glob("engine/**/*.log")):
        hit = re.search(r"\| device (\S+)", log.read_text(errors="ignore"))
        if hit:
            return hit.group(1)
    argv = provenance.get("argv", [])
    ngpu = argv[argv.index("--ngpu") + 1] if "--ngpu" in argv else "-1"
    return "cpu" if ngpu == "-1" else f"ngpu={ngpu} (not confirmed by the log)"


def _p(value):
    if value != value:
        return "-"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def build_gate_report(exp):
    """Phase 1 GATE: the core vs Advanced-CoLight -> `results/p1_gate.md`. Writes
    nothing without BOTH sides on BOTH networks; half a gate would look like a verdict."""
    import pandas as pd
    from scipy.stats import ttest_ind

    df = training_results(exp)
    if df.empty:
        return
    df = df[
        df["agent"].isin([GATE_REFERENCE, GATE_CORE])
        & (df["yellow_length"] == GATE_YELLOW)
        & df["network"].isin(GATE_NETWORKS)
        & (df["variant"] == "")  # ablations share the agent, see RUN_SUFFIX
    ]
    missing = [
        f"{agent}@{net}"
        for net in GATE_NETWORKS
        for agent in (GATE_REFERENCE, GATE_CORE)
        if df[(df["network"] == net) & (df["agent"] == agent)].empty
    ]
    if missing:
        print(f"Phase 1 GATE: missing runs {', '.join(missing)}; no report.", file=sys.stderr)
        return

    curves = exp.metrics(run_ids=list(df["run_id"]))
    stats = pd.DataFrame([{"run_id": rid, **_curve_stats(curves, rid)} for rid in df["run_id"]])
    df = df.merge(stats, on="run_id")

    # provenance from `meta.json`: device, code version, machine; run times compare only
    # on the same hardware, which the report states rather than assumes
    devices, code, hosts = {}, set(), set()
    wanted = set(df["run_id"])
    for run in exp.runs():
        if run.id not in wanted:
            continue
        prov = json.loads((run.path / "meta.json").read_text()).get("provenance") or {}
        devices[run.id] = _device(run.path, prov)
        code.add(
            tuple((name, g.get("sha"), g.get("dirty")) for name, g in sorted(prov["git"].items()))
            if prov.get("git")
            else ()
        )
        hosts.add(prov.get("hostname") or "?")
    df["device"] = df["run_id"].map(devices)

    labels = {GATE_REFERENCE: "Advanced-CoLight", GATE_CORE: "core (gatppo)"}
    att_rows, seed_rows, curve_rows, verdicts, ref_att = [], [], [], [], {}
    for net in GATE_NETWORKS:
        side = {
            agent: df[(df["network"] == net) & (df["agent"] == agent)].sort_values("seed")
            for agent in labels
        }
        ref = side[GATE_REFERENCE]["att_last10"].astype(float)
        core = side[GATE_CORE]["att_last10"].astype(float)
        ref_att[net] = ref.mean()
        delta = (core.mean() - ref.mean()) / ref.mean()
        pvalue = (
            ttest_ind(core, ref, equal_var=False).pvalue
            if min(len(ref), len(core)) > 1
            else float("nan")
        )
        ok = delta <= GATE_TOLERANCE
        verdicts.append((net, delta, pvalue, ok))
        att_rows.append(
            f"| {net} | {ref.mean():.1f} +- {ref.std():.1f} "
            f"| {core.mean():.1f} +- {core.std():.1f} "
            f"| {delta * 100:+.2f}% | {_p(pvalue)} | {len(ref)} / {len(core)} "
            f"| {'YES' if ok else 'NO'} |"
        )
        for agent, label in labels.items():
            s = side[agent]
            per_seed = ", ".join(
                f"s{int(r['seed'])} {float(r['att_last10']):.1f}" for r in s.to_dict("records")
            )
            seed_rows.append(f"| {net} | {label} | {per_seed} |")
            conv = s["converged_at"].dropna()
            conv_txt = f"{int(conv.min())}-{int(conv.max())}" if len(conv) else "-"
            curve_rows.append(
                f"| {net} | {label} | {s['tail_worst'].max():.1f} | {conv_txt} "
                f"| {s['throughput'].astype(float).mean():.0f} "
                f"| {s['queue'].astype(float).mean():.2f} | {s['duration_s'].mean():.0f} "
                f"| {', '.join(sorted(set(s['device'])))} |"
            )

    # matplotlib only here, the T1 tables should not need it
    from tsc_marl.analysis.plot import gate_figures

    figures = gate_figures(
        curves, df[["run_id", "network", "agent"]], labels, ref_att, GATE_TOLERANCE, FIGURES
    )

    device_of = {
        agent: ", ".join(sorted(set(df[df["agent"] == agent]["device"]))) for agent in labels
    }
    same_hw = device_of[GATE_REFERENCE] == device_of[GATE_CORE]
    if len(code) == 1 and next(iter(code)):
        code_line = "The same code for all runs: " + ", ".join(
            f"{name} `{sha}`" + (" (DIRTY)" if dirty else "")
            for name, sha, dirty in next(iter(code))
        )
    else:
        code_line = (
            f"**WARNING: runs from {len(code)} different code versions**, an uncertain comparison"
        )
    code_line += f"; machine: `{', '.join(sorted(hosts))}`."

    passed = all(ok for *_, ok in verdicts)
    worst = max(delta for _, delta, _, _ in verdicts)
    reading = []
    for net, delta, pvalue, _ in verdicts:
        if pvalue == pvalue and pvalue < 0.05:
            better = labels[GATE_REFERENCE] if delta > 0 else labels[GATE_CORE]
            reading.append(
                f"* **{net}**: a statistically distinguishable difference "
                f"(Welch, p = {_p(pvalue)}) "
                f"in favour of {better}; in practice {abs(delta) * 100:.2f}% ATT."
            )
        else:
            reading.append(
                f"* **{net}**: the methods are indistinguishable (Welch, p = {_p(pvalue)}); "
                f"delta {delta * 100:+.2f}% is within the spread over seeds."
            )

    out = [
        "# Phase 1 GATE: the core (GAT + PPO + CTDE) vs Advanced-CoLight",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}. Suites `p1_reference` + `p1_gate`, "
        f"realistic mode (yellow={GATE_YELLOW}), protocol `configs/tsc/thesis_protocol.yml`.",
        f"ATT = mean of a run's last {WINDOW} test episodes; below, mean +- std over seeds.",
        code_line,
        "",
        "| network | Advanced-CoLight | core (gatppo) | delta | p (Welch) | n | within tolerance |",
        "|---|---:|---:|---:|---:|:-:|:-:|",
        *att_rows,
        "",
        "## Per seed",
        "",
        "| network | agent | ATT (last 10 tests) |",
        "|---|---|---|",
        *seed_rows,
        "",
        "## Stability and cost",
        "",
        f"| network | agent | worst test (last {TAIL}) | convergence [episode] "
        "| throughput | queue | run time [s] | device |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        *curve_rows,
        "",
        f"Convergence = the first episode from which a rolling {WINDOW} tests stays within "
        f"{CONVERGED * 100:.0f}% of the final result (range over seeds). The worst tail test "
        "catches one-off policy collapses the mean of 10 does not show.",
        "",
        "## Learning curves",
        "",
        f"Figures in `results/{FIGURES.name}/` (PNG here, PDF for the thesis), rebuilt "
        "together with this report.",
        "",
        *(
            line
            for name, caption in figures
            for line in (f"![{caption}]({FIGURES.name}/{name})", "", f"*{caption}*", "")
        ),
        "## Exit criterion",
        "",
        f"The core within **{GATE_TOLERANCE * 100:.0f}%** of Advanced-CoLight's ATT on every "
        "network (delta > 0 = the core is worse). The tolerance is the upper bound of the "
        'phrase in `p1_gate.yml` ("within a few percent"), stored as `GATE_TOLERANCE` in '
        "`analysis/report.py` AFTER seeing the results, hence the smallest tolerance at which "
        "the verdict holds is given below as well.",
        "",
        *reading,
        "",
        "## Verdict",
        "",
        f"**Phase 1 GATE: {'PASSED' if passed else 'FAILED'}.** Largest delta = "
        f"{worst * 100:+.2f}%"
        + (
            f"; the verdict holds at every tolerance >= {max(worst, 0) * 100:.2f}%."
            if passed
            else "."
        ),
        "",
        "## Where the run time difference comes from",
        "",
        (
            "The simulation is THE SAME on both paths (a training + a test episode of 3600 s, "
            "`test_when_train` on both sides) and so is the hardware, so the difference lies "
            "in learning:"
            if same_hw
            else f"**The times CANNOT be compared directly: {labels[GATE_REFERENCE]} ran on "
            f"`{device_of[GATE_REFERENCE]}`, the core on `{device_of[GATE_CORE]}`.** The "
            "simulation is the same on both paths (a training + a test episode of 3600 s), but "
            "the time difference mixes two causes these runs cannot separate:"
        ),
        "",
        "* **Advanced-CoLight**: in `external/LibSignal/trainer/tsc_trainer.py` the training "
        "block (l. 172) sits at the `while` level, OUTSIDE `if i % action_interval == 0` "
        "(l. 139). With `update_model_rate: 1` (`configs/tsc/base.yml`) the condition holds at "
        "every simulator tick: after `learning_start` that is 3600 optimiser steps per "
        "episode, each with three GAT forward passes (target on b_tp, model twice on b_t).",
        "* **core**: one PPO update per episode, `epochs x n_minibatches` = 10 x 4 = 40 "
        "optimiser steps (`configs/tsc/gatppo.yml`).",
        *(
            []
            if same_hw
            else [
                "* **hardware**: Advanced-CoLight cannot move to a GPU (`p1_reference.yml`: the "
                "submodule has no `.cuda()`), the core ran on a GPU as `ngpu` in `p1_gate.yml` "
                "says. How much of the time difference is the card and how much the step count "
                "is unknown."
            ]
        ),
        "",
        "Conclusion for the thesis: the hardware-independent claim is ~90x fewer optimiser "
        "steps per episode at the same ATT. Do NOT cite run time as an advantage of the method"
        + ("" if same_hw else " without a core run on the CPU (`--ngpu -1`)")
        + ". Part of the difference is also an artefact of LibSignal's loop, deliberately left "
        "alone, since changing the cadence would change the baseline matched to the "
        "literature (`t1_baseline.md`).",
        "",
        "## What this table does NOT say",
        "",
        "* Run time is `duration_s` from `meta.json`, the WHOLE run. Neither path logs a split "
        "into simulation and update; the split above follows from the code, not a measurement.",
        "* HangZhou separates methods poorly in realistic mode: the classic baselines land "
        'close together (`t1_baseline.md`, column "realistic"), and reproducing the '
        "literature on this network is weak. JiNan carries the burden of proof of the gate.",
        "",
    ]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "p1_gate.md").write_text("\n".join(out))
    print("\n".join(out))


# ------------------------------------------------------------------ core ablations


def _ablation_rows(df, variants: dict[str, str], net: str):
    """(label, variant, att_last10 values over seeds) for variants present in the data."""
    out = []
    for variant, label in variants.items():
        vals = df[(df["network"] == net) & (df["variant"] == variant)]["att_last10"]
        vals = vals.astype(float).dropna().tolist()
        if vals:
            out.append((label, variant, vals))
    return out


def build_ablation_report(exp):
    """Core ablations -> `results/p1_ablations.md`. The reference is the GATE runs
    (empty variant), so delta measures exactly one change against the core."""
    import numpy as np
    from scipy.stats import ttest_ind

    df = training_results(exp)
    if df.empty:
        return
    core = df[(df["agent"] == GATE_CORE) & (df["yellow_length"] == GATE_YELLOW)]
    known = set(AGGREGATION) | set(STATE)
    if not (core["variant"].isin(known - {""})).any():
        return
    refs = df[(df["yellow_length"] == GATE_YELLOW) & (df["variant"] == "")
              & df["agent"].isin(["advancedcolight", "efficientmp", "advancedmp"])]  # fmt: skip

    def fmt(vals):
        sd = f" +- {np.std(vals, ddof=1):.1f}" if len(vals) > 1 else ""
        return f"{np.mean(vals):.1f}{sd}"

    out = [
        "# Core ablations: GAT+PPO+CTDE",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}. Realistic mode (yellow=5), "
        "ATT = mean of the last 10 tests, mean +- std over seeds. Each variant changes ONE "
        "thing against the GATE core; the critic stays a GAT in all of them (the ablation "
        "asks about the actor's decision). delta > 0 = worse than the core.",
        "",
    ]
    for net in GATE_NETWORKS:
        sub = core[core["network"] == net]
        base = sub[sub["variant"] == ""]["att_last10"].astype(float).dropna().tolist()
        if not base:
            continue
        out += [f"## {net}", ""]
        for title, variants in (("Neighbour aggregation in the actor", AGGREGATION),
                                ("State representation", STATE)):  # fmt: skip
            rows = _ablation_rows(sub, variants, net)
            out += [f"### {title}", "", "| variant | ATT | delta vs core | p (Welch) | n |",
                    "|---|---:|---:|---:|:-:|"]  # fmt: skip
            for label, variant, vals in rows:
                if variant == "":
                    out.append(f"| **{label}** | {fmt(vals)} | - | - | {len(vals)} |")
                    continue
                d = np.mean(vals) / np.mean(base) - 1
                p = ttest_ind(vals, base, equal_var=False).pvalue if len(vals) > 1 else float("nan")
                out.append(f"| {label} | {fmt(vals)} | {d:+.2%} | {_p(p)} | {len(vals)} |")
            out.append("")
        ref_rows = []
        for a in ("advancedcolight", "efficientmp", "advancedmp"):
            vals = refs[(refs["network"] == net) & (refs["agent"] == a)]["att_last10"]
            if not vals.empty:
                ref_rows.append(f"{a}: {fmt(vals.astype(float).tolist())}")
        if ref_rows:
            out += ["Reference (same protocol): " + ", ".join(ref_rows) + ".", ""]
    out += [
        "## How to read",
        "",
        "* A Welch p at n=3 is an orientation, not proof; the same convention as in the GATE.",
        '* Efficient-MP and Advanced-MP alongside answer "why a GNN if MaxPressure is '
        'enough": the variant without a GNN should land between them and the core.',
    ]
    (RESULTS / "p1_ablations.md").write_text("\n".join(out) + "\n")
