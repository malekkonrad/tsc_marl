"""`tsc degrade`: frozen controllers under sensor failure.

Two roles in one file:

* PARENT (`run_plan`) expands the plan into points, skips computed ones
  (skip-if-done on the fingerprint in `runs/t2`) and runs the rest in
  subprocesses through `suite.schedule`;
* CHILD (`main`) handles one controller x network x seed: builds the world, loads
  the parent checkpoint from `runs/t1`, runs every scenario and stores the run.

The policy architecture comes from the training run's `config.resolved.yml`, not
the current YAML: weights load into the architecture they were trained in.

    tsc degrade run t2_nomech -j 6
    tsc degrade status t2_nomech
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from tsc_marl import libsignal
from tsc_marl.paths import REPO_ROOT, RUNS
from tsc_marl.robustness.degrade import (
    EXPERIMENT,
    KINDS,
    LEARNED,
    MECHANISM_KEYS,
    Plan,
    Unit,
    base_network,
    load_plan,
)


def experiment():
    from runlog import Experiment

    return Experiment(EXPERIMENT, root=RUNS)


def parent_views(plan: Plan, unit: Unit) -> list[tuple[str, Any]]:
    """[(run_id, RunView or None)]: parents in `runs/t1`, empty for rules.
    None = no run or an unfinished one (`status != ok`)."""
    from tsc_marl.libsignal import experiment as t1

    ids = plan.parent_ids(unit)
    if not ids:
        return []
    runs = {r.id: r for r in t1.experiment().runs()}
    return [(pid, runs[pid] if pid in runs and runs[pid].status == "ok" else None) for pid in ids]


def parent_view(plan: Plan, unit: Unit):
    """The first parent (whose policy controls) or None."""
    views = parent_views(plan, unit)
    return views[0][1] if views else None


def resolve_parents(plan: Plan, unit: Unit) -> tuple[list[Any], list[Path], str | None]:
    """-> (parents, checkpoints, reason missing); reason None = all in place."""
    views = parent_views(plan, unit)
    missing = [pid for pid, v in views if v is None]
    if missing:
        return [], [], f"no run {', '.join(missing)}"
    parents = [v for _, v in views]
    ckpts = [checkpoint(plan, unit, p) for p in parents]
    lost = [str(c) for c in ckpts if c is None or not c.is_file()]
    if lost:
        return parents, ckpts, f"no weights {', '.join(lost)}"
    if plan.kind(unit.controller) == "ensemble":
        hashes = {p.params.get("config_hash") for p in parents}
        if len(hashes) > 1:
            return parents, ckpts, "ensemble members have different configs (architectures)"
    return parents, ckpts, None


def checkpoint(plan: Plan, unit: Unit, parent) -> Path | None:
    """The parent's weight file; the final network in both cases (`core.pt` after the
    last episode, LibSignal's `save_model(e=episodes)` after training)."""
    if parent is None:
        return None
    # Branch on the AGENT, not the controller kind: the weight file format depends
    # on WHO trained (`train_core` -> `engine/core.pt`, LibSignal ->
    # `engine/tsc/.../model/*.pt`), and every overlay on the core has its own kind
    # but core weights. Missing weights skip the point silently, so a kind list that
    # forgets an overlay drops a column from the report.
    if KINDS[plan.agent(unit.controller)] == "policy":
        # `checkpoint: best` = best test episode of training (`CoreTrainer.best_path`);
        # final by default
        best = plan.controllers[unit.controller].get("checkpoint") == "best"
        return parent.path / "engine" / ("core_best.pt" if best else "core.pt")

    from tsc_marl.runner.evaluate import available_checkpoints, engine_run_dir

    agent = plan.agent(unit.controller)
    # the PARENT's network: in an OOD plan LibSignal kept the weights under the
    # training network name
    net = parent.params.get("network", unit.network)
    model_dir = engine_run_dir(parent.path, "cityflow", agent, net, "test") / "model"
    saved = available_checkpoints(model_dir)
    if not saved:
        return model_dir / "<missing>.pt"
    return saved[max(saved)][0]


def unit_params(plan: Plan, unit: Unit, parents: list[Any]) -> dict[str, Any]:
    """Identity of a run, computed by the same function in the parent (skip-if-done)
    and the child (registration)."""
    from runlog import params_hash
    from tsc_marl.libsignal import experiment as t1
    from tsc_marl.libsignal.experiment import parse_args

    params: dict[str, Any] = {
        "controller": unit.controller,
        "agent": plan.agent(unit.controller),
        "network": unit.network,
        "seed": unit.seed,
        "yellow_length": plan.yellow,
        "draws": plan.draws,
        "scenarios": [s.as_dict() for s in plan.scenarios_for(unit)],
    }
    # mechanism parameters (tau, gate threshold) CHANGE the result, so they belong
    # to the identity; otherwise a sweep returns one number under several labels
    spec = plan.controllers[unit.controller]
    # other WEIGHTS from the same parent: `best` must not share the final's fingerprint
    if spec.get("checkpoint", "last") != "last":
        params["checkpoint"] = spec["checkpoint"]
    mechanism = {k: spec[k] for k in MECHANISM_KEYS if k in spec}
    if mechanism:
        params["mechanism"] = mechanism
    if parents:
        # learned: the parent carries the identity (its fingerprint covers training)
        params["parent_run"] = parents[0].id
        params["parent_params_hash"] = parents[0].meta.get("params_hash")
        if len(parents) > 1:
            params["ensemble_runs"] = [p.id for p in parents]
            params["ensemble_hashes"] = [p.meta.get("params_hash") for p in parents]
    else:
        # rule: its config carries the identity (e.g. `w1` of Advanced-MP)
        args = parse_args(["-a", params["agent"], "-n", unit.network])
        with libsignal.sim_configs_untouched():
            config = t1.resolve_config(args)
        params["config_hash"] = params_hash(
            {k: config.get(k) for k in ("model",)} | {"t": config["trainer"]["action_interval"]}
        )
    return params


# ------------------------------------------------------------------- child


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controller under failure (one plan point)")
    p.add_argument("--plan", required=True)
    p.add_argument("--controller", required=True)
    p.add_argument("--network", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replace", action="store_true")
    p.add_argument(
        "--trace",
        action="store_true",
        help="write a per (decision, node) gate trace to gate_trace.csv in the run directory",
    )
    return p.parse_args(argv)


def build_uncertainty(plan: Plan, unit: Unit, env):
    """Source of log rho for a `uw` column (or the head for a UQ gate).

    Built by `training.uw_signal`, the same function training calls, so a policy
    trained with the signal is measured with THE SAME signal. Artefacts belong to
    the BASE network, the one the dataset was recorded on.
    """
    from tsc_marl.robustness.training import uw_signal

    return uw_signal(
        plan.controllers[unit.controller].get("uw", {}), base_network(unit.network), env
    )


def fixed_cycle(spec: dict[str, Any]) -> int:
    """Cycle length of the fixed-time rule the gate switches a node to.

    `t_fixed` is NOT in the policy config, only in `configs/tsc/fixedtime.yml`, the
    same file the `fixedtime` column reads, so both are the same fixed time. A value
    in the plan spec overrides it and enters the fingerprint (`MECHANISM_KEYS`).
    """
    if "t_fixed" in spec:
        return int(spec["t_fixed"])
    import yaml

    from tsc_marl.paths import CONFIGS

    return int(yaml.safe_load((CONFIGS / "tsc" / "fixedtime.yml").read_text())["model"]["t_fixed"])


def gate_reference(unit: Unit):
    """Real gate thresholds for the base network, see `training.gate_reference`."""
    from tsc_marl.robustness.training import gate_reference as load

    return load(base_network(unit.network))


def fixed_time_program(network: str) -> dict[str, dict[str, float]] | None:
    """The network's fixed-time program (`fixedTimeProgram` in `configs/sim/<net>.cfg`)
    or None.

    Krakow: Webster per demand level (`data/raw_data/krakow_3w/build.py webster`).
    The key lives in the NETWORK config, so the `fixedtime` column and the gate's
    switch target both get it; JiNan/HangZhou (no key) keep `t_fixed`.
    """
    import json

    from tsc_marl.paths import CONFIGS, DATA

    cfg = CONFIGS / "sim" / f"{network}.cfg"
    if not cfg.is_file():
        return None
    rel = json.loads(cfg.read_text()).get("fixedTimeProgram")
    if rel is None:
        return None
    path = DATA / rel
    if not path.is_file():
        raise SystemExit(
            f"{cfg.name}: no program {rel}; compute it with: "
            ".venv/bin/python data/raw_data/krakow_3w/build.py webster"
        )
    return json.loads(path.read_text())


def build_gate(spec: dict[str, Any], plan: Plan, unit: Unit, env):
    """The plan's gate signal (`kind` validated by `load_plan`).

    REAL gate thresholds come from an artefact calibrated on clean traffic, not
    the plan; a plan may override them, and the override enters the fingerprint.
    """
    from tsc_marl.robustness import controllers as c

    kind = spec.get("kind")
    if kind == "oracle":
        return c.OracleGate(env, threshold=float(spec.get("threshold", 0.5)))

    ref = gate_reference(unit)
    if kind == "rule":
        return c.RuleGate(
            window=ref.window,
            threshold=float(spec.get("threshold", ref.rule)),
            n_mov=env.features.n_mov,
            real=env.features.movement_mask(),
        )
    return c.UQGate(
        build_uncertainty(plan, unit, env),
        u_mov_ref=ref.u_mov_ref,
        threshold=float(spec.get("threshold", ref.uq)),
    )


def build_controller(
    plan: Plan, unit: Unit, env, config, agents, ckpts: list[Path], *, trace: bool = False
):
    import numpy as np
    import torch

    from tsc_marl.core.loop import world_order
    from tsc_marl.robustness import controllers as c

    kind = plan.kind(unit.controller)
    feats = env.features
    model = config["model"]
    # by AGENT, not kind, as in `checkpoint()`: core weights load the same way for
    # every overlay on it
    if KINDS[plan.agent(unit.controller)] == "policy":
        from tsc_marl.core.nets import ActorCritic

        # phase mask as in `CoreTrainer.from_world`: on a heterogeneous network it
        # stops the actor from choosing a phase the intersection lacks; the buffer is
        # not in `state_dict`, so the checkpoint does not bring it
        mask = feats.phase_mask()
        extra = {} if mask is None else {"phase_mask": torch.from_numpy(mask)}
        actors = []
        for path in ckpts:
            ac = ActorCritic.from_registry(feats.ob_length, feats.n_phases, **extra)
            ac.actor.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)["actor"]
            )
            actors.append(ac.actor)
        if kind == "ensemble":
            return c.EnsembleController(actors, env.graph.edge_index, name=unit.controller)
        policy = (
            c.UWPolicyController(
                actors[0],
                env.graph.edge_index,
                build_uncertainty(plan, unit, env),
                name=unit.controller,
            )
            if plan.controllers[unit.controller].get("uw")
            else c.PolicyController(actors[0], env.graph.edge_index)
        )
        if kind == "policy_uw":
            return policy
        if kind == "policy_mask":
            spec = plan.controllers[unit.controller]["mask"]
            # legal phases: `n_phases` is the network MAXIMUM, so a 2-phase node has
            # dead columns the mechanism must never force
            legal = np.zeros((feats.n_nodes, feats.n_phases), dtype=bool)
            for i, inter in enumerate(feats.intersections):
                legal[i, : len(inter.phases)] = True
            return c.MaskedController(
                c.PolicyController(actors[0], env.graph.edge_index),
                legal=legal,
                k_max=int(spec["k_max"]),
                name=unit.controller,
            )
        if kind == "policy_gate":
            spec = plan.controllers[unit.controller]["gate"]
            # the oracle as a DIAGNOSTIC of a real gate (precision, recall); it never
            # touches decisions. 0.5 = half of the inbound movements dead.
            oracle = (
                None
                if spec.get("kind") == "oracle"
                else c.OracleGate(env, threshold=float(spec.get("oracle_threshold", 0.5)))
            )
            # `policy` with UW when the spec also has `uw` (composition, `t2_uw_rule`)
            return c.GatedController(
                policy,
                c.FixedTimeController(
                    feats.intersections,
                    t_fixed=fixed_cycle(spec),
                    greens=fixed_time_program(unit.network),
                    yellow=plan.yellow,
                ),
                build_gate(spec, plan, unit, env),
                k_on=int(spec.get("k_on", 1)),
                k_off=int(spec.get("k_off", 1)),
                oracle=oracle,
                name=unit.controller,
                trace=trace,
            )
        return c.PolicyController(actors[0], env.graph.edge_index, name=unit.controller)
    if kind == "colight":
        (agent,) = agents
        if agent.ob_length != feats.ob_length:
            raise SystemExit(f"agent ob_length {agent.ob_length} != state {feats.ob_length}")
        agent.model.load_state_dict(torch.load(ckpts[0], map_location="cpu"))
        agent.model.eval()
        order = world_order(env.graph, env.world)
        return c.ColightController(
            agent,
            identity_order=bool(np.array_equal(order, np.arange(len(order)))),
            name=unit.controller,
        )
    if kind == "pressure":
        w1 = model.get("w1") if plan.agent(unit.controller) == "advancedmp" else None
        return c.PressureController(
            feats.phase_incidence(),
            feats.intersections,
            feats.n_mov,
            t_min=int(model["t_min"]),
            w1=w1,
            name=unit.controller,
        )
    return c.FixedTimeController(
        feats.intersections,
        t_fixed=int(model["t_fixed"]),
        greens=fixed_time_program(unit.network),
        yellow=plan.yellow,
    )


def main(argv: list[str] | None = None) -> None:
    import yaml

    args = parse_args(argv)
    libsignal.bootstrap()
    from tsc_marl.seeding import set_global_seed

    plan = load_plan(args.plan)
    unit = Unit(args.controller, args.network, args.seed)
    if unit.controller not in plan.controllers:
        raise SystemExit(f"plan '{plan.name}' has no controller '{unit.controller}'")
    set_global_seed(unit.seed)

    kind = plan.kind(unit.controller)
    parents, ckpts, why = resolve_parents(plan, unit)
    if kind in LEARNED and why:
        raise SystemExit(f"{why}; runs from the GPU machine must be synced WITH their weights")

    patch: dict[str, Any] = {"model": {}, "trainer": {"yellow_length": plan.yellow}}
    if parents:
        resolved = yaml.safe_load((parents[0].path / "config.resolved.yml").read_text())
        patch["model"].update(resolved["model"])
    patch["model"]["graphic"] = True  # TrafficEnv needs the graph for rules too
    params = unit_params(plan, unit, parents)

    from tsc_marl.core.features import StateBuilder
    from tsc_marl.core.graph import RoadNetGraph
    from tsc_marl.core.loop import LoopConfig, TrafficEnv
    from tsc_marl.libsignal import experiment as t1
    from tsc_marl.libsignal import session
    from tsc_marl.robustness.degrade import METRICS, evaluate, summarize

    n_agents = 1 if kind == "colight" else 0
    agent = plan.agent(unit.controller)
    with session.standalone_world(agent, unit.network, config_patch=patch, n_agents=n_agents) as (
        world,
        agents,
        config,
    ):
        loop = LoopConfig.from_registry()
        graph = RoadNetGraph.from_registry()
        order = graph.order_intersections(world)

        def make(spec):
            return StateBuilder.from_registry(world, order, failure=spec)

        env = TrafficEnv(world, graph, make(None), loop.action_interval)
        controller = build_controller(plan, unit, env, config, agents, ckpts, trace=args.trace)

        exp = experiment()
        with exp.start(
            plan.run_id(unit), params, config=config, repos=t1.repos(), replace=args.replace
        ) as run:

            def log(row):
                extra = {k: v for k, v in row.items() if k.startswith("ens_")}
                run.log(
                    step=row["draw"],
                    split=row["scenario"],
                    **{m: row[m] for m in METRICS},
                    **extra,
                )
                print(
                    f"  {row['scenario']:28s} draw {row['draw']}  ATT {row['att']:7.1f}", flush=True
                )

            traces: list[str] = []

            def write_trace(scenario, draw, rows):
                # one file per run, columns as in `metrics.csv` (`split`/`step`)
                for decision, node, score, raw, mode in rows:
                    traces.append(
                        f"{scenario},{draw},{decision},{node},{score:.6f},{int(raw)},{int(mode)}"
                    )

            rows = evaluate(
                env, controller, plan.scenarios_for(unit), plan.draws, make,
                loop.test_decisions, on_row=log,
                on_trace=write_trace if args.trace else None,
            )  # fmt: skip
            if traces:
                path = run.path / "gate_trace.csv"
                path.write_text(
                    "split,step,decision,node,score,raw,rule_mode\n" + "\n".join(traces) + "\n"
                )
                print(f"  gate trace: {path} ({len(traces)} rows)", flush=True)
            run.result(
                **summarize(rows),
                checkpoint=str(ckpts[0].relative_to(REPO_ROOT)) if ckpts else None,
                plan=plan.name,
            )
    print(f"runlog: {run.path}")


# ------------------------------------------------------------------- parent


def run_plan(
    plan: Plan,
    jobs: int = 1,
    replace: bool = False,
    stagger: float = 15.0,
    trace: bool = False,
) -> dict:
    """Compute the missing plan points -> status counts."""
    import subprocess

    from tsc_marl.runner import suite

    libsignal.bootstrap()
    exp = experiment()
    counts: dict[str, int] = {"ok": 0, "skipped": 0, "no parent": 0}
    todo: list[Unit] = []
    print(f"plan '{plan.name}' ({len(plan.scenarios)} scenarios, {plan.draws} draws)")
    for unit in plan.units():
        parents, _, why = resolve_parents(plan, unit)
        if plan.kind(unit.controller) in LEARNED and why:
            print(f"  {unit.label()}  SKIPPING: {why}")
            counts["no parent"] += 1
            continue
        if not replace and exp.is_done(plan.run_id(unit), unit_params(plan, unit, parents)):
            print(f"  {unit.label()}  skipping (has a result)")
            counts["skipped"] += 1
            continue
        todo.append(unit)

    env = libsignal.child_env()
    env.setdefault("OMP_NUM_THREADS", "1")  # evaluation: simulator-bound, not algebra
    logs = RUNS / "_suite_logs" / f"degrade_{plan.name}"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(unit: Unit):
        cmd = [
            sys.executable, "-m", "tsc_marl.runner.degrade", "--plan", str(plan.path),
            "--controller", unit.controller, "--network", unit.network, "--seed", str(unit.seed),
        ]  # fmt: skip
        if replace:
            cmd.append("--replace")
        if trace:
            cmd.append("--trace")
        log = open(logs / f"{unit.controller}__{unit.network}__s{unit.seed}.log", "a")  # noqa: SIM115
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return proc, log

    if todo:
        print(f"  {len(todo)} points, {jobs} at once, logs: {logs}")
        with libsignal.sim_configs_untouched():
            done = suite.schedule(todo, launch, max(1, jobs), plan.timeout, stagger)
        for status in done.values():
            counts[status] = counts.get(status, 0) + 1
    print(f"\n{counts}. Runs: {exp.path}")
    return counts


def status(plan: Plan) -> list[dict[str, Any]]:
    """State of the plan points. `current` is what `run` uses (a fingerprint
    comparison), not mere existence: after a threshold change the directory is
    still `ok`, but `run` would recompute it and hit `RunIdCollision` without
    `--replace`."""
    libsignal.bootstrap()
    exp = experiment()
    rows = []
    for unit in plan.units():
        parent = parent_view(plan, unit)
        view = next((r for r in exp.runs() if r.id == plan.run_id(unit)), None)
        parents, _, _ = resolve_parents(plan, unit)
        rows.append(
            {
                "run_id": plan.run_id(unit),
                "status": "missing" if view is None else view.status,
                "current": exp.is_done(plan.run_id(unit), unit_params(plan, unit, parents)),
                "parent": plan.parent_id(unit) if parent is None else parent.id,
                "att_clean": view.result.get("att__clean") if view else None,
            }
        )
    return rows


if __name__ == "__main__":
    main()
