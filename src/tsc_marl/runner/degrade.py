"""`tsc degrade` — zamrożone sterowniki pod awarią sensorów (T2, oś X fallbacku).

Dwie role w jednym pliku, jak `suite.py` + `train.py`:

* RODZIC (`run_plan`) — rozpisuje plan na punkty, pomija policzone (skip-if-done
  po odcisku w `runs/t2`), odpala brakujące w podprocesach przez `suite.schedule`
  (ten sam scheduler z przesunięciem startów);
* DZIECKO (`main`) — jeden sterownik × sieć × seed: stawia świat, ładuje checkpoint
  rodzica z `runs/t1`, przepuszcza przez wszystkie scenariusze i zapisuje bieg.

Checkpoint polityki NIE jest brany z bieżącego YAML-a, tylko z `config.resolved.yml`
biegu treningowego — architektura musi być ta, na której wagi powstały (ablacja
z `encoder: mean` ma się załadować jako mean-pool, nie jako GAT).

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
from tsc_marl.robustness.degrade import EXPERIMENT, Plan, Unit, load_plan


def experiment():
    from runlog import Experiment  # noqa: PLC0415

    return Experiment(EXPERIMENT, root=RUNS)


def parent_view(plan: Plan, unit: Unit):
    """RunView rodzica z `runs/t1` albo None (sterownik bez treningu)."""
    from tsc_marl.libsignal import experiment as t1  # noqa: PLC0415

    pid = plan.parent_id(unit)
    if pid is None:
        return None
    view = next((r for r in t1.experiment().runs() if r.id == pid), None)
    if view is None or view.status != "ok":
        return None
    return view


def checkpoint(plan: Plan, unit: Unit, parent) -> Path | None:
    """Plik wag rodzica. Końcowa sieć w obu przypadkach: `core.pt` rdzenia zapisuje
    `train_core` po ostatnim epizodzie, a `tsc_trainer` po treningu robi
    `save_model(e=episodes)` — więc „ostatni" checkpoint to ta sama chwila uczenia."""
    if parent is None:
        return None
    kind = plan.kind(unit.controller)
    if kind == "policy":
        return parent.path / "engine" / "core.pt"
    from tsc_marl.runner.evaluate import available_checkpoints, engine_run_dir  # noqa: PLC0415

    agent = plan.agent(unit.controller)
    model_dir = engine_run_dir(parent.path, "cityflow", agent, unit.network, "test") / "model"
    saved = available_checkpoints(model_dir)
    if not saved:
        return model_dir / "<brak>.pt"
    return saved[max(saved)][0]


def unit_params(plan: Plan, unit: Unit, parent) -> dict[str, Any]:
    """Tożsamość biegu T2. Liczona tą samą funkcją przez rodzica (skip-if-done)
    i dziecko (rejestracja) — dwie kopie rozjechałyby się przy pierwszej zmianie."""
    from runlog import params_hash  # noqa: PLC0415
    from tsc_marl.libsignal import experiment as t1  # noqa: PLC0415
    from tsc_marl.runner.train import parse_args  # noqa: PLC0415

    params: dict[str, Any] = {
        "controller": unit.controller,
        "agent": plan.agent(unit.controller),
        "network": unit.network,
        "seed": unit.seed,
        "yellow_length": plan.yellow,
        "draws": plan.draws,
        "scenarios": [s.as_dict() for s in plan.scenarios_for(unit)],
    }
    if parent is not None:
        # uczony: tożsamość niesie rodzic (jego odcisk obejmuje config i trening)
        params["parent_run"] = parent.id
        params["parent_params_hash"] = parent.meta.get("params_hash")
    else:
        # reguła: tożsamość niesie jej config (np. `w1` Advanced-MP)
        args = parse_args(["-a", params["agent"], "-n", unit.network])
        with libsignal.sim_configs_untouched():
            config = t1.resolve_config(args)
        params["config_hash"] = params_hash(
            {k: config.get(k) for k in ("model",)} | {"t": config["trainer"]["action_interval"]}
        )
    return params


# ------------------------------------------------------------------- dziecko


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sterownik pod awarią (jeden punkt planu)")
    p.add_argument("--plan", required=True)
    p.add_argument("--controller", required=True)
    p.add_argument("--network", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replace", action="store_true")
    return p.parse_args(argv)


def build_controller(plan: Plan, unit: Unit, env, config, agents, ckpt: Path | None):
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from tsc_marl.core.loop import world_order  # noqa: PLC0415
    from tsc_marl.robustness import controllers as c  # noqa: PLC0415

    kind = plan.kind(unit.controller)
    feats = env.features
    model = config["model"]
    if kind == "policy":
        from tsc_marl.core.nets import ActorCritic  # noqa: PLC0415

        ac = ActorCritic.from_registry(feats.ob_length, feats.n_phases)
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        ac.actor.load_state_dict(payload["actor"])
        return c.PolicyController(ac.actor, env.graph.edge_index, name=unit.controller)
    if kind == "colight":
        (agent,) = agents
        if agent.ob_length != feats.ob_length:
            raise SystemExit(f"ob_length agenta {agent.ob_length} != stanu {feats.ob_length}")
        agent.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
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
    return c.FixedTimeController(feats.intersections, t_fixed=int(model["t_fixed"]))


def main(argv: list[str] | None = None) -> None:
    import yaml  # noqa: PLC0415

    args = parse_args(argv)
    libsignal.bootstrap()
    from tsc_marl.seeding import set_global_seed  # noqa: PLC0415

    plan = load_plan(args.plan)
    unit = Unit(args.controller, args.network, args.seed)
    if unit.controller not in plan.controllers:
        raise SystemExit(f"plan '{plan.name}' nie ma sterownika '{unit.controller}'")
    set_global_seed(unit.seed)

    parent = parent_view(plan, unit)
    kind = plan.kind(unit.controller)
    if kind in ("policy", "colight") and parent is None:
        raise SystemExit(f"brak biegu-rodzica {plan.parent_id(unit)} ze statusem ok w runs/t1")
    ckpt = checkpoint(plan, unit, parent)
    if ckpt is not None and not ckpt.is_file():
        raise SystemExit(
            f"brak checkpointu {ckpt} — biegi z glidera trzeba zsynchronizować RAZEM z wagami"
        )

    patch: dict[str, Any] = {"model": {}, "trainer": {"yellow_length": plan.yellow}}
    if parent is not None:
        resolved = yaml.safe_load((parent.path / "config.resolved.yml").read_text())
        patch["model"].update(resolved["model"])
    patch["model"]["graphic"] = True  # TrafficEnv potrzebuje grafu także dla reguł
    params = unit_params(plan, unit, parent)

    from tsc_marl.core.features import StateBuilder  # noqa: PLC0415
    from tsc_marl.core.graph import RoadNetGraph  # noqa: PLC0415
    from tsc_marl.core.loop import LoopConfig, TrafficEnv  # noqa: PLC0415
    from tsc_marl.libsignal import experiment as t1  # noqa: PLC0415
    from tsc_marl.libsignal import session  # noqa: PLC0415
    from tsc_marl.robustness.degrade import METRICS, evaluate, summarize  # noqa: PLC0415

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
        controller = build_controller(plan, unit, env, config, agents, ckpt)

        exp = experiment()
        with exp.start(
            plan.run_id(unit), params, config=config, repos=t1.repos(), replace=args.replace
        ) as run:

            def log(row):
                run.log(step=row["draw"], split=row["scenario"], **{m: row[m] for m in METRICS})
                print(
                    f"  {row['scenario']:28s} los {row['draw']}  ATT {row['att']:7.1f}", flush=True
                )

            rows = evaluate(
                env, controller, plan.scenarios_for(unit), plan.draws, make,
                loop.test_decisions, on_row=log,
            )  # fmt: skip
            run.result(
                **summarize(rows),
                checkpoint=str(ckpt.relative_to(REPO_ROOT)) if ckpt else None,
                plan=plan.name,
            )
    print(f"runlog: {run.path}")


# ------------------------------------------------------------------- rodzic


def run_plan(plan: Plan, jobs: int = 1, replace: bool = False, stagger: float = 15.0) -> dict:
    """Policz brakujące punkty planu. -> licznik statusów."""
    import subprocess  # noqa: PLC0415

    from tsc_marl.runner import suite  # noqa: PLC0415

    libsignal.bootstrap()
    exp = experiment()
    licznik: dict[str, int] = {"ok": 0, "pominięte": 0, "brak rodzica": 0}
    todo: list[Unit] = []
    print(f"plan '{plan.name}' ({len(plan.scenarios)} scenariuszy, {plan.draws} losowań)")
    for unit in plan.units():
        parent = parent_view(plan, unit)
        if plan.kind(unit.controller) in ("policy", "colight"):
            ckpt = checkpoint(plan, unit, parent)
            if parent is None or ckpt is None or not ckpt.is_file():
                why = "brak biegu" if parent is None else f"brak wag ({ckpt})"
                print(f"  {unit.label()}  POMIJAM — {why}: {plan.parent_id(unit)}")
                licznik["brak rodzica"] += 1
                continue
        if not replace and exp.is_done(plan.run_id(unit), unit_params(plan, unit, parent)):
            print(f"  {unit.label()}  pomijam (jest wynik)")
            licznik["pominięte"] += 1
            continue
        todo.append(unit)

    env = libsignal.child_env()
    env.setdefault("OMP_NUM_THREADS", "1")  # ewaluacja: symulator, nie algebra
    logs = RUNS / "_suite_logs" / f"degrade_{plan.name}"
    logs.mkdir(parents=True, exist_ok=True)

    def launch(unit: Unit):
        cmd = [
            sys.executable, "-m", "tsc_marl.runner.degrade", "--plan", str(plan.path),
            "--controller", unit.controller, "--network", unit.network, "--seed", str(unit.seed),
        ]  # fmt: skip
        if replace:
            cmd.append("--replace")
        log = open(logs / f"{unit.controller}__{unit.network}__s{unit.seed}.log", "a")  # noqa: SIM115
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return proc, log

    if todo:
        print(f"  {len(todo)} punktów, {jobs} naraz, logi: {logs}")
        with libsignal.sim_configs_untouched():
            done = suite.schedule(todo, launch, max(1, jobs), plan.timeout, stagger)
        for status in done.values():
            licznik[status] = licznik.get(status, 0) + 1
    print(f"\n{licznik}. Biegi: {exp.path}")
    return licznik


def status(plan: Plan) -> list[dict[str, Any]]:
    libsignal.bootstrap()
    exp = experiment()
    rows = []
    for unit in plan.units():
        parent = parent_view(plan, unit)
        view = next((r for r in exp.runs() if r.id == plan.run_id(unit)), None)
        rows.append(
            {
                "run_id": plan.run_id(unit),
                "status": "brak" if view is None else view.status,
                "rodzic": plan.parent_id(unit) if parent is None else parent.id,
                "att_clean": view.result.get("att__clean") if view else None,
            }
        )
    return rows


if __name__ == "__main__":
    main()
