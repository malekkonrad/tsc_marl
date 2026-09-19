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
    from runlog import Experiment  # noqa: PLC0415

    return Experiment(EXPERIMENT, root=RUNS)


def parent_views(plan: Plan, unit: Unit) -> list[tuple[str, Any]]:
    """[(run_id, RunView albo None)] — rodzice z `runs/t1`; pusta lista dla reguł.
    None = brak biegu albo bieg nieskończony (`status != ok`)."""
    from tsc_marl.libsignal import experiment as t1  # noqa: PLC0415

    ids = plan.parent_ids(unit)
    if not ids:
        return []
    runs = {r.id: r for r in t1.experiment().runs()}
    return [(pid, runs[pid] if pid in runs and runs[pid].status == "ok" else None) for pid in ids]


def parent_view(plan: Plan, unit: Unit):
    """Pierwszy rodzic (ten, którego polityka steruje) albo None."""
    views = parent_views(plan, unit)
    return views[0][1] if views else None


def resolve_parents(plan: Plan, unit: Unit) -> tuple[list[Any], list[Path], str | None]:
    """-> (rodzice, checkpointy, powód braku). Powód None = wszystko na miejscu."""
    views = parent_views(plan, unit)
    missing = [pid for pid, v in views if v is None]
    if missing:
        return [], [], f"brak biegu {', '.join(missing)}"
    parents = [v for _, v in views]
    ckpts = [checkpoint(plan, unit, p) for p in parents]
    lost = [str(c) for c in ckpts if c is None or not c.is_file()]
    if lost:
        return parents, ckpts, f"brak wag {', '.join(lost)}"
    if plan.kind(unit.controller) == "ensemble":
        hashes = {p.params.get("config_hash") for p in parents}
        if len(hashes) > 1:
            return parents, ckpts, "członkowie ensemble mają różne configi (architektury)"
    return parents, ckpts, None


def checkpoint(plan: Plan, unit: Unit, parent) -> Path | None:
    """Plik wag rodzica. Końcowa sieć w obu przypadkach: `core.pt` rdzenia zapisuje
    `train_core` po ostatnim epizodzie, a `tsc_trainer` po treningu robi
    `save_model(e=episodes)` — więc „ostatni" checkpoint to ta sama chwila uczenia."""
    if parent is None:
        return None
    # Rozgałęzienie po AGENCIE, nie po rodzaju sterownika — i to jest tu cała treść.
    # Format pliku z wagami zależy od tego, KTO trenował: nasz `train_core` pisze
    # `engine/core.pt`, trener LibSignala rozkłada katalogi `engine/tsc/…/model/*.pt`.
    # Rodzaj sterownika mówi co innego: JAK zbudować sterownik z tych wag. Nakładki na
    # rdzeń (`ensemble`, `policy_uw`, bramki z Fazy 3) mają własny rodzaj, ale agenta
    # `gatppo` i wagi w `core.pt`.
    #
    # Warunek po rodzaju był wyliczanką i dwa razy z niej wypadła nowa nakładka. Objaw
    # jest cichy: brak wag NIE jest błędem, tylko pominięciem punktu (żeby jeden
    # nieprzetrenowany seed nie wywalał planu), więc plan liczy się dalej BEZ kolumny,
    # dla której powstał, i raport wychodzi z kompletem tabel o jedną kolumnę uboższy.
    if KINDS[plan.agent(unit.controller)] == "policy":
        return parent.path / "engine" / "core.pt"

    from tsc_marl.runner.evaluate import available_checkpoints, engine_run_dir  # noqa: PLC0415

    agent = plan.agent(unit.controller)
    # sieć RODZICA, nie ewaluacji: w planie OOD polityka z HangZhou jedzie po innym ruchu,
    # a LibSignal trzymał jej wagi pod nazwą sieci treningowej
    net = parent.params.get("network", unit.network)
    model_dir = engine_run_dir(parent.path, "cityflow", agent, net, "test") / "model"
    saved = available_checkpoints(model_dir)
    if not saved:
        return model_dir / "<brak>.pt"
    return saved[max(saved)][0]


def unit_params(plan: Plan, unit: Unit, parents: list[Any]) -> dict[str, Any]:
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
    # Parametry mechanizmu (τ modulacji, próg bramki) ZMIENIAJĄ liczony wynik, więc
    # należą do tożsamości biegu. Bez nich `is_done` widzi ten sam odcisk po zmianie
    # progu i wypisuje „pomijam (jest wynik)" — sweep zwróciłby trzy razy tę samą
    # liczbę pod trzema etykietami.
    spec = plan.controllers[unit.controller]
    mechanizm = {k: spec[k] for k in MECHANISM_KEYS if k in spec}
    if mechanizm:
        params["mechanism"] = mechanizm
    if parents:
        # uczony: tożsamość niesie rodzic (jego odcisk obejmuje config i trening)
        params["parent_run"] = parents[0].id
        params["parent_params_hash"] = parents[0].meta.get("params_hash")
        if len(parents) > 1:
            params["ensemble_runs"] = [p.id for p in parents]
            params["ensemble_hashes"] = [p.meta.get("params_hash") for p in parents]
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


def build_uncertainty(plan: Plan, unit: Unit, env):
    """Głowa obserwacyjna + jej `u_ref` dla TEJ sieci (`tsc obshead --save`).

    Artefakty są związane z siecią: głowa z JiNan na HangZhou ma AUROC 0,53
    (`results/gate_2_2.md`), więc `load_head`/`load_reference` dostają nazwę sieci
    i mają prawo krzyknąć. Wariant popytu (`_d0p25`) czyta głowę sieci BAZOWEJ —
    tej, na której zrzucono BC — bo taka jest też konwencja rodzica w `runs/t1`.
    Uwaga do rozdziału 4: `u_ref` jest skalibrowane na obciążeniu zrzutu, więc przy
    cięższym ruchu niż w zrzucie u rośnie wszędzie i `relu` wygeneruje nadmiar bez
    awarii; przy lżejszym — zje go (brak fałszywych alarmów).
    """
    from tsc_marl.robustness.training import online_uncertainty  # noqa: PLC0415

    # Ładowanie artefaktów przez `robustness.training.online_uncertainty` — wspólne
    # z treningiem, żeby dwie kopie nie rozjechały się przy zmianie nazw plików.
    # `.get("uw", {})`, bo tę samą głowę wywołuje bramka UQ, która τ nie używa:
    # τ skaluje modulację log ρ, a bramce potrzebne jest samo u na ruch.
    return online_uncertainty(
        base_network(unit.network),
        env.features.n_mov,
        env.graph.edge_index,
        tau=plan.controllers[unit.controller].get("uw", {}).get("tau"),
    )


def fixed_cycle(spec: dict[str, Any]) -> int:
    """Długość cyklu reguły, do której bramka przełącza węzeł.

    `t_fixed` NIE MA w configu polityki — mieszka wyłącznie w `configs/tsc/fixedtime.yml`
    (sprawdzone), a jednostka z bramką biegnie jako `agent: gatppo`, więc sięgnięcie po
    `config["model"]["t_fixed"]` poleciałoby `KeyError` w środku biegu.

    Domyślnie czytamy TEN SAM plik, z którego bierze go kolumna `fixedtime` — inaczej
    zdanie „bramka dogania FixedTime" porównywałoby dwa różnie sparametryzowane
    FixedTime. Wartość w spec planu nadpisuje i wchodzi do odcisku biegu
    (`MECHANISM_KEYS`), więc zmiana cyklu unieważnia wynik, tak jak powinna.
    """
    if "t_fixed" in spec:
        return int(spec["t_fixed"])
    import yaml  # noqa: PLC0415

    from tsc_marl.paths import CONFIGS  # noqa: PLC0415

    return int(yaml.safe_load((CONFIGS / "tsc" / "fixedtime.yml").read_text())["model"]["t_fixed"])


def build_gate(spec: dict[str, Any], plan: Plan, unit: Unit, env):
    """Sygnał bramki z planu. `kind` waliduje `load_plan`, więc tu tylko wybór.

    Progi bramek REALNYCH nie pochodzą z planu, tylko z artefaktu skalibrowanego na
    czystym oknie zrzutu BC (`tsc obshead --save`) — tak jak `u_ref` i τ. Plan może
    je nadpisać, ale wtedy jest to jawna decyzja, która wchodzi do odcisku biegu.
    """
    from tsc_marl.robustness import controllers as c  # noqa: PLC0415
    from tsc_marl.uncertainty import obs_head as oh  # noqa: PLC0415

    kind = spec.get("kind")
    if kind == "oracle":
        return c.OracleGate(env, threshold=float(spec.get("threshold", 0.5)))

    net = base_network(unit.network)
    sciezka = oh.gate_path(net)
    if not sciezka.is_file():
        raise SystemExit(
            f"brak progów bramki dla '{net}' ({sciezka.name}) — policz je: "
            f"tsc obshead -n {net} --save --window 12 --persistence"
        )
    progi = oh.load_gates(sciezka, network=net, head=oh.head_path(net))
    if kind == "rule":
        return c.RuleGate(
            window=progi.window,
            threshold=float(spec.get("threshold", progi.rule)),
            n_mov=env.features.n_mov,
        )
    return c.UQGate(
        build_uncertainty(plan, unit, env),
        u_mov_ref=progi.u_mov_ref,
        threshold=float(spec.get("threshold", progi.uq)),
    )


def build_controller(plan: Plan, unit: Unit, env, config, agents, ckpts: list[Path]):
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from tsc_marl.core.loop import world_order  # noqa: PLC0415
    from tsc_marl.robustness import controllers as c  # noqa: PLC0415

    kind = plan.kind(unit.controller)
    feats = env.features
    model = config["model"]
    # Po AGENCIE, nie po rodzaju — ta sama zasada co w `checkpoint()`: wagi rdzenia
    # wczytuje się tak samo dla każdej nakładki na niego, a wyliczanka rodzajów już
    # dwa razy przepuściła nową (objaw: `policy_gate` leciał przez wszystkie gałęzie
    # do FixedTime i wywalał się na `t_fixed`, którego config polityki nie ma).
    if KINDS[plan.agent(unit.controller)] == "policy":
        from tsc_marl.core.nets import ActorCritic  # noqa: PLC0415

        actors = []
        for path in ckpts:
            ac = ActorCritic.from_registry(feats.ob_length, feats.n_phases)
            ac.actor.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)["actor"]
            )
            actors.append(ac.actor)
        if kind == "ensemble":
            return c.EnsembleController(actors, env.graph.edge_index, name=unit.controller)
        if kind == "policy_uw":
            return c.UWPolicyController(
                actors[0],
                env.graph.edge_index,
                build_uncertainty(plan, unit, env),
                name=unit.controller,
            )
        if kind == "policy_gate":
            spec = plan.controllers[unit.controller]["gate"]
            # Wyrocznia jako DIAGNOSTYKA bramki realnej: precyzja i czułość rozdzielają
            # „detektor jest słaby" od „przełączanie mało daje". Nigdy nie dotyka decyzji.
            wyrocznia = (
                None
                if spec.get("kind") == "oracle"
                # 0,5 = „połowa wlotów martwa" jako definicja oślepienia do
                # diagnostyki; 0,75 dawało przy maskach pasowych zbiór pusty.
                else c.OracleGate(env, threshold=float(spec.get("oracle_threshold", 0.5)))
            )
            return c.GatedController(
                c.PolicyController(actors[0], env.graph.edge_index),
                c.FixedTimeController(feats.intersections, t_fixed=fixed_cycle(spec)),
                build_gate(spec, plan, unit, env),
                k_on=int(spec.get("k_on", 1)),
                k_off=int(spec.get("k_off", 1)),
                oracle=wyrocznia,
                name=unit.controller,
            )
        return c.PolicyController(actors[0], env.graph.edge_index, name=unit.controller)
    if kind == "colight":
        (agent,) = agents
        if agent.ob_length != feats.ob_length:
            raise SystemExit(f"ob_length agenta {agent.ob_length} != stanu {feats.ob_length}")
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

    kind = plan.kind(unit.controller)
    parents, ckpts, why = resolve_parents(plan, unit)
    if kind in LEARNED and why:
        raise SystemExit(f"{why} — biegi z glidera trzeba zsynchronizować RAZEM z wagami")

    patch: dict[str, Any] = {"model": {}, "trainer": {"yellow_length": plan.yellow}}
    if parents:
        resolved = yaml.safe_load((parents[0].path / "config.resolved.yml").read_text())
        patch["model"].update(resolved["model"])
    patch["model"]["graphic"] = True  # TrafficEnv potrzebuje grafu także dla reguł
    params = unit_params(plan, unit, parents)

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
        controller = build_controller(plan, unit, env, config, agents, ckpts)

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
                    f"  {row['scenario']:28s} los {row['draw']}  ATT {row['att']:7.1f}", flush=True
                )

            rows = evaluate(
                env, controller, plan.scenarios_for(unit), plan.draws, make,
                loop.test_decisions, on_row=log,
            )  # fmt: skip
            run.result(
                **summarize(rows),
                checkpoint=str(ckpts[0].relative_to(REPO_ROOT)) if ckpts else None,
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
        parents, _, why = resolve_parents(plan, unit)
        if plan.kind(unit.controller) in LEARNED and why:
            print(f"  {unit.label()}  POMIJAM — {why}")
            licznik["brak rodzica"] += 1
            continue
        if not replace and exp.is_done(plan.run_id(unit), unit_params(plan, unit, parents)):
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
    """Stan punktów planu. `aktualny` = to, czym kieruje się `run` (porównanie
    odcisku parametrów), a nie samo istnienie katalogu.

    Rozróżnienie nie jest kosmetyczne: po dołożeniu scenariusza albo zmianie progu
    bramki katalog dalej istnieje ze statusem `ok`, ale `run` policzy ten punkt od
    nowa — i rozbije się o `RunIdCollision`, bo `runlog` nie nadpisuje biegu o innych
    parametrach. Bez tej kolumny widziałeś „ok" i nie wiedziałeś, że potrzebujesz
    `--replace`. Wzór z `suite.status`, gdzie ta kolumna jest od początku.
    """
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
                "status": "brak" if view is None else view.status,
                "aktualny": exp.is_done(plan.run_id(unit), unit_params(plan, unit, parents)),
                "rodzic": plan.parent_id(unit) if parent is None else parent.id,
                "att_clean": view.result.get("att__clean") if view else None,
            }
        )
    return rows


if __name__ == "__main__":
    main()
