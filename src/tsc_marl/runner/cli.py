"""The single entry point of the project: `tsc <command>`.

Thin on purpose: commands only dispatch, all logic lives in `train.py`,
`train_core.py`, `evaluate.py`, `suite.py`, `degrade.py` and `analysis/`, so tests,
suites and subprocesses call the same functions rather than the CLI.

`train`, `core` and `eval` pass their arguments UNCHANGED to those modules'
argparse parsers (`--help` included): repeating fifteen flags here would be one
more copy to drift apart.
"""

from __future__ import annotations

import os
import sys

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    # Without this rich wraps EVERY exception in a source frame; our domain errors
    # (RunExists, RunIdCollision, SystemExit from a suite) are messages for a human.
    pretty_exceptions_enable=False,
    help="Traffic signal control: training, experiment grids, reports.",
)
suite_app = typer.Typer(no_args_is_help=True, help="Declarative run grids (configs/suites).")
app.add_typer(suite_app, name="suite")

# Arguments go to the target module's argparse, `--help` included, hence the empty
# `help_option_names` (click would otherwise intercept it).
PASSTHROUGH = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}


def ensure_hashseed() -> None:
    """Restart the interpreter with PYTHONHASHSEED=0 if it is not set.

    PYTHONHASHSEED works ONLY when set before the interpreter starts, and without
    it SUMO is nondeterministic (same arguments, another result). `sys.orig_argv`
    reproduces the call exactly, for `tsc ...` and `python -m tsc_marl ...` alike;
    the guard prevents a loop, and suite subprocesses get the variable from
    `libsignal.child_env()`.
    """
    if os.environ.get("PYTHONHASHSEED") == "0" or not getattr(sys, "orig_argv", None):
        return
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, sys.orig_argv)


@app.callback()
def _before_command() -> None:
    ensure_hashseed()


# ------------------------------------------------------------------ single runs


@app.command(context_settings=PASSTHROUGH)
def train(ctx: typer.Context) -> None:
    """Train or measure a LibSignal agent (`tsc train -a colight -n ... --seed 0`)."""
    from tsc_marl.runner import train as train_module

    train_module.main(ctx.args)


@app.command()
def prototype(
    network: str = typer.Option("cityflow_jinan3x4_4ph", "-n", help="base network of the dataset"),
    mode: str = typer.Option("edl", help="'edl' or 'softmax'"),
    epochs: int = typer.Option(150, help="head training epochs"),
    hop: bool = typer.Option(False, help="1-hop instead of 0-hop"),
    ensemble: int = typer.Option(5, help="deep ensemble members (epistemic reference); 0 = none"),
) -> None:
    """Decision-uncertainty prototype gate (training, table, criteria) -> results/gate_2_0.md."""
    from tsc_marl.uncertainty import prototype as module

    cfg = module.TrainConfig(mode=mode, epochs=epochs, hop=hop, ensemble=ensemble)
    print(module.run(cfg, network=network))


@app.command()
def obshead(
    network: str = typer.Option("cityflow_jinan3x4_4ph", "-n", help="base network of the dataset"),
    window: int = typer.Option(4, help="how many recent readings the head sees"),
    neighbors: bool = typer.Option(True, help="add the mean of upstream neighbours' readings"),
    epochs: int = typer.Option(60, help="head training epochs"),
    persistence: bool = typer.Option(False, help="add reading persistence features (stuck)"),
    mask_variants: bool = typer.Option(
        False, help="compare variants on dead detectors -> results/gate_2_2_masks.md"
    ),
    save: bool = typer.Option(
        False, help="train the head ONLY (no gate) and save the artefacts -> runs/heads/"
    ),
    reference: str | None = typer.Option(
        None,
        help="compute ONLY u_ref (target|source) for an EXISTING head, without retraining",
    ),
) -> None:
    """Observation head gate (per movement -> edge) -> results/gate_2_2.md."""
    from tsc_marl.uncertainty import obs_head as module

    if reference is not None:
        print(module.save_reference(network, reference))
        return

    if mask_variants:
        print(module.mask_variants(network=network))
        return
    cfg = module.ObsConfig(
        window=window, neighbors=neighbors, epochs=epochs, persistence=persistence
    )
    if save:
        for path in module.train_head(network, cfg):
            print(path)
        return
    print(module.run(cfg, network=network))


@app.command()
def rulecal(
    network: str = typer.Option(..., "-n", help="network (e.g. sumo_krakow3w_mid)"),
    episodes: int = typer.Option(3, help="clean episodes under the expert"),
    expert: str = typer.Option("advancedmp", help="controller driving the clean traffic"),
) -> None:
    """Rule-gate threshold ALONE (no head) -> runs/heads/rule_ref__<network>.json."""
    from tsc_marl.robustness import rule_ref

    path = rule_ref.calibrate(network, episodes=episodes, expert=expert)
    print(path.read_text())
    print(path)


@app.command()
def demand(
    network: str = typer.Option("cityflow_jinan3x4_4ph", "-n", help="base network"),
    factors: str = typer.Option("0.5,1.5", help="demand factors, comma-separated"),
) -> None:
    """Demand variants (OOD scenario) -> configs/sim/<network>_d*.cfg."""
    from tsc_marl.domain import demand as module

    for f in [float(x) for x in factors.split(",")]:
        flow, cfg = module.write_variant(network, f)
        print(f"x{f:g}: {cfg}  ({flow.name})")


@app.command()
def dataset(
    network: str = typer.Option("cityflow_jinan3x4_4ph", "-n", help="network"),
    expert: str = typer.Option("advancedmp", "-a", help="behaviour cloning expert"),
    episodes: int = typer.Option(3, help="episodes (they differ in the FAILURE DRAW)"),
    replace: bool = typer.Option(False, help="overwrite an existing run"),
) -> None:
    """Dataset (state -> expert action) for the uncertainty heads -> runs/bc/."""
    from tsc_marl.uncertainty import dataset as module

    print(module.build(network, expert=expert, episodes=episodes, replace=replace))


@app.command()
def severity(
    network: str = typer.Option("cityflow_jinan3x4_4ph", help="network to measure on"),
) -> None:
    """Sensor failure severity converter -> results/sensor_severity.md."""
    from tsc_marl.analysis import severity as module

    print(module.main(network))


@app.command(context_settings=PASSTHROUGH)
def core(ctx: typer.Context) -> None:
    """The core: GAT + PPO + CTDE with its own loop (`tsc core -n ... --seed 0`)."""
    from tsc_marl.runner import train_core

    train_core.main(ctx.args)


@app.command("eval", context_settings=PASSTHROUGH)
def evaluate_(ctx: typer.Context) -> None:
    """Evaluate a frozen policy from a checkpoint (`tsc eval --load-run ...`)."""
    from tsc_marl.runner import evaluate

    evaluate.main(ctx.args)


# ------------------------------------------------------------------------- suites


@suite_app.command("run")
def suite_run(
    name: str = typer.Argument(..., help="name or path of a file in configs/suites/"),
    replace: bool = typer.Option(False, "--replace", help="recompute finished points too"),
    jobs: int = typer.Option(1, "--jobs", "-j", help="points computed at once (subprocesses)"),
    stagger: float = typer.Option(15.0, help="seconds between starts of parallel points"),
) -> None:
    """Compute every grid point (skipping those with a current result)."""
    from tsc_marl.runner import suite

    counts = suite.run(suite.load(name), replace=replace, jobs=jobs, stagger=stagger)
    if counts.get("failed") or counts.get("timeout"):
        raise typer.Exit(1)


@suite_app.command("status")
def suite_status(
    name: str = typer.Argument(..., help="name or path of a file in configs/suites/"),
) -> None:
    """State of every grid point, including those not computed yet."""
    from tsc_marl.runner import suite

    rows = suite.status(suite.load(name))
    print(f"{'run_id':52s} {'status':9s} {'current':9s} att_last10")
    for r in rows:
        att = f"{r['att_last10']:.1f}" if r["att_last10"] is not None else "-"
        print(f"{r['run_id']:52s} {r['status']:9s} {str(r['current']):9s} {att}")
    missing = sum(1 for r in rows if not r["current"])
    print(f"\n{len(rows) - missing}/{len(rows)} points have a current result.")


@suite_app.command("list")
def suite_list() -> None:
    """List the available suites."""
    from tsc_marl.runner import suite

    for path in sorted(suite.SUITES.glob("*.yml")):
        s = suite.load(path)
        print(f"  {s.name:16s} {s.module:20s} yellow={s.yellow}  {len(s.points())} points")


# ------------------------------------------------------------------- failures

degrade_app = typer.Typer(
    no_args_is_help=True, help="Frozen controllers under sensor failure (configs/robustness)."
)
app.add_typer(degrade_app, name="degrade")


@degrade_app.command("run")
def degrade_run(
    plan: str = typer.Argument(..., help="name or path of a file in configs/robustness/"),
    jobs: int = typer.Option(1, "--jobs", "-j", help="points computed at once"),
    replace: bool = typer.Option(False, "--replace", help="recompute finished points too"),
    stagger: float = typer.Option(15.0, help="seconds between subprocess starts"),
    trace: bool = typer.Option(
        False,
        "--trace",
        help="write a per (decision, node) gate trace to gate_trace.csv, to explain "
        "single switches; changes neither decisions nor the run fingerprint",
    ),
) -> None:
    """Compute the missing plan points (controller x network x seed, all scenarios)."""
    from tsc_marl.robustness.degrade import load_plan
    from tsc_marl.runner import degrade

    counts = degrade.run_plan(
        load_plan(plan), jobs=jobs, replace=replace, stagger=stagger, trace=trace
    )
    if counts.get("failed") or counts.get("timeout"):
        raise typer.Exit(1)


@degrade_app.command("status")
def degrade_status(plan: str = typer.Argument(...)) -> None:
    """State of the plan points, including those whose parent checkpoint is missing."""
    from tsc_marl.robustness.degrade import load_plan
    from tsc_marl.runner import degrade

    for r in degrade.status(load_plan(plan)):
        att = f"{r['att_clean']:.1f}" if r["att_clean"] is not None else "-"
        # marker ONLY for a run that exists with other parameters; a missing one is
        # computed without `--replace`
        stale = r["status"] != "missing" and not r["current"]
        note = "  STALE (--replace)" if stale else ""
        print(f"{r['run_id']:64s} {r['status']:8s} {att:>7s}  <- {r['parent']}{note}")


@degrade_app.command("ood")
def degrade_ood(plan: str = typer.Argument("t2_ood")) -> None:
    """Epistemic branch test: policy vs max-pressure on new traffic + ensemble disagreement."""
    from tsc_marl import libsignal
    from tsc_marl.analysis import ood
    from tsc_marl.robustness.degrade import load_plan

    libsignal.bootstrap()
    print(ood.main(load_plan(plan)))


@degrade_app.command("report")
def degrade_report(plan: str = typer.Argument("t2_nomech")) -> None:
    """Degradation table and curves from computed runs -> results/<plan>.md."""
    from tsc_marl.analysis import degradation
    from tsc_marl.robustness.degrade import load_plan

    print(degradation.main(load_plan(plan)))


# ------------------------------------------------------------------------ reports


@app.command()
def table() -> None:
    """Rebuild the baseline and core tables from stored runs into `results/` (runs nothing)."""
    from tsc_marl import libsignal
    from tsc_marl.analysis import report
    from tsc_marl.libsignal import experiment

    libsignal.bootstrap()
    report.build_table(experiment.experiment())


@app.command()
def runs() -> None:
    """List computed runs: status, parameters, ATT."""
    import pandas as pd

    from tsc_marl.libsignal import experiment

    pd.set_option("display.width", 200)
    df = experiment.experiment().results()
    print("No runs in runs/." if df.empty else df.to_string(index=False))


@app.command(context_settings=PASSTHROUGH)
def plot(ctx: typer.Context) -> None:
    """Learning curves from runs/ -> runs/plots/ (also for a run in progress)."""
    from tsc_marl.analysis import plot as plot_module

    plot_module.main(ctx.args)


@app.command()
def doctor() -> None:
    """Check the environment: submodule, SUMO_HOME, versions, PYTHONHASHSEED."""
    from tsc_marl import libsignal, paths

    libsignal.check_ready()
    libsignal.bootstrap()
    print(f"repo:            {paths.REPO_ROOT}")
    print(f"LibSignal:       {paths.LIBSIGNAL}")
    print(f"SUMO_HOME:       {os.environ.get('SUMO_HOME')}")
    print(f"PYTHONHASHSEED:  {os.environ.get('PYTHONHASHSEED')}")
    for module in ("torch", "cityflow", "torch_geometric"):
        try:
            m = __import__(module)
            print(f"{module + ':':16s} {getattr(m, '__version__', 'ok')}")
        except ImportError as e:
            print(f"{module + ':':16s} MISSING ({e})")


if __name__ == "__main__":
    app()
