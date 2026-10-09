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
    p.add_argument(
        "--trace",
        action="store_true",
        help="zapisz ślad bramki per (decyzja, węzeł) do gate_trace.csv w katalogu biegu",
    )
    return p.parse_args(argv)


def build_uncertainty(plan: Plan, unit: Unit, env):
    """Źródło log ρ dla kolumny z `uw` (albo głowa dla bramki UQ, która `uw` nie ma).

    Budowa przez `robustness.training.uw_signal` — tę samą funkcję, którą woła
    trening z UW (`train_core`), żeby polityka uczona z sygnałem była mierzona TYM
    SAMYM sygnałem. Artefakty są związane z siecią: głowa z JiNan na HangZhou ma
    AUROC 0,53 (`results/gate_2_2.md`), więc idzie nazwa sieci BAZOWEJ — tej, na
    której zrzucono BC (konwencja rodzica w `runs/t1`). Uwaga do rozdziału 4: `u_ref`
    jest skalibrowane na obciążeniu zrzutu, więc przy cięższym ruchu niż w zrzucie
    u rośnie wszędzie i `relu` wygeneruje nadmiar bez awarii; przy lżejszym — zje go.
    """
    from tsc_marl.robustness.training import uw_signal  # noqa: PLC0415

    return uw_signal(
        plan.controllers[unit.controller].get("uw", {}), base_network(unit.network), env
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


def gate_reference(unit: Unit):
    """Progi bramek realnych dla sieci bazowej — patrz `training.gate_reference`."""
    from tsc_marl.robustness.training import gate_reference as wczytaj  # noqa: PLC0415

    return wczytaj(base_network(unit.network))


def fixed_time_program(network: str) -> dict[str, dict[str, float]] | None:
    """Program stałoczasowy sieci (`fixedTimeProgram` w `configs/sim/<sieć>.cfg`) albo None.

    Kraków: Webster osobno dla każdego poziomu popytu (`data/raw_data/krakow_3w/build.py
    webster`) — tak jak w krakowskiej praktyce programów na pory dnia. Klucz siedzi
    w configu SIECI, a nie w planie: program należy do scenariusza, więc kolumna
    `fixedtime` i cel przełączenia bramki dostają go same, a JiNan/HangZhou (bez klucza)
    zostają przy `t_fixed` — z tymi samymi liczbami co dotąd.
    """
    import json  # noqa: PLC0415

    from tsc_marl.paths import CONFIGS, DATA  # noqa: PLC0415

    cfg = CONFIGS / "sim" / f"{network}.cfg"
    if not cfg.is_file():
        return None
    rel = json.loads(cfg.read_text()).get("fixedTimeProgram")
    if rel is None:
        return None
    path = DATA / rel
    if not path.is_file():
        raise SystemExit(
            f"{cfg.name}: brak programu {rel} — policz go: "
            ".venv/bin/python data/raw_data/krakow_3w/build.py webster"
        )
    return json.loads(path.read_text())


def build_gate(spec: dict[str, Any], plan: Plan, unit: Unit, env):
    """Sygnał bramki z planu. `kind` waliduje `load_plan`, więc tu tylko wybór.

    Progi bramek REALNYCH nie pochodzą z planu, tylko z artefaktu skalibrowanego na
    czystym oknie zrzutu BC (`tsc obshead --save`) — tak jak `u_ref` i τ. Plan może
    je nadpisać, ale wtedy jest to jawna decyzja, która wchodzi do odcisku biegu.
    """
    from tsc_marl.robustness import controllers as c  # noqa: PLC0415

    kind = spec.get("kind")
    if kind == "oracle":
        return c.OracleGate(env, threshold=float(spec.get("threshold", 0.5)))

    progi = gate_reference(unit)
    if kind == "rule":
        return c.RuleGate(
            window=progi.window,
            threshold=float(spec.get("threshold", progi.rule)),
            n_mov=env.features.n_mov,
            real=env.features.movement_mask(),
        )
    return c.UQGate(
        build_uncertainty(plan, unit, env),
        u_mov_ref=progi.u_mov_ref,
        threshold=float(spec.get("threshold", progi.uq)),
    )


def build_controller(
    plan: Plan, unit: Unit, env, config, agents, ckpts: list[Path], *, trace: bool = False
):
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

        # Maska faz jak w `CoreTrainer.from_world`: na sieci niejednorodnej (Kraków, 2–7 faz)
        # actor bez niej wybiera fazę, której skrzyżowanie nie ma, i SUMO się wywala
        # (`phase index 2 is not in the allowed range [0,0]`). Bufor nie jest w `state_dict`
        # (`persistent=False`), więc checkpoint go nie przyniesie — trzeba go podać tu.
        # Na JiNan/HangZhou `phase_mask()` to None i sieć buduje się jak dotąd.
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
        polityka = (
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
            return polityka
        if kind == "policy_mask":
            spec = plan.controllers[unit.controller]["mask"]
            # Maska legalnych faz: `n_phases` to MAKSIMUM po sieci, więc węzeł o 2 fazach
            # ma resztę kolumn martwych. Bez tej maski mechanizm wymuszałby fazy, których
            # nie ma — i na sieci niejednorodnej (Kraków: 2–7 faz) sterowałby w pustkę.
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
            # Wyrocznia jako DIAGNOSTYKA bramki realnej: precyzja i czułość rozdzielają
            # „detektor jest słaby" od „przełączanie mało daje". Nigdy nie dotyka decyzji.
            wyrocznia = (
                None
                if spec.get("kind") == "oracle"
                # 0,5 = „połowa wlotów martwa" jako definicja oślepienia do
                # diagnostyki; 0,75 dawało przy maskach pasowych zbiór pusty.
                else c.OracleGate(env, threshold=float(spec.get("oracle_threshold", 0.5)))
            )
            # `polityka` z UW, gdy spec ma też `uw` — złożenie (`t2_uw_rule`)
            return c.GatedController(
                polityka,
                c.FixedTimeController(
                    feats.intersections,
                    t_fixed=fixed_cycle(spec),
                    greens=fixed_time_program(unit.network),
                    yellow=plan.yellow,
                ),
                build_gate(spec, plan, unit, env),
                k_on=int(spec.get("k_on", 1)),
                k_off=int(spec.get("k_off", 1)),
                oracle=wyrocznia,
                name=unit.controller,
                trace=trace,
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
    return c.FixedTimeController(
        feats.intersections,
        t_fixed=int(model["t_fixed"]),
        greens=fixed_time_program(unit.network),
        yellow=plan.yellow,
    )


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
                    f"  {row['scenario']:28s} los {row['draw']}  ATT {row['att']:7.1f}", flush=True
                )

            slady: list[str] = []

            def zapisz_slad(scenariusz, draw, wiersze):
                # Jeden plik na bieg, kolumny jak w `metrics.csv` (`split`/`step`),
                # żeby dało się je złączyć bez tłumaczenia nazw.
                for decyzja, node, wynik, surowa, tryb in wiersze:
                    slady.append(
                        f"{scenariusz},{draw},{decyzja},{node},{wynik:.6f},"
                        f"{int(surowa)},{int(tryb)}"
                    )

            rows = evaluate(
                env, controller, plan.scenarios_for(unit), plan.draws, make,
                loop.test_decisions, on_row=log,
                on_trace=zapisz_slad if args.trace else None,
            )  # fmt: skip
            if slady:
                sciezka = run.path / "gate_trace.csv"
                sciezka.write_text(
                    "split,step,decision,node,score,raw,rule_mode\n" + "\n".join(slady) + "\n"
                )
                print(f"  ślad bramki: {sciezka} ({len(slady)} wierszy)", flush=True)
            run.result(
                **summarize(rows),
                checkpoint=str(ckpts[0].relative_to(REPO_ROOT)) if ckpts else None,
                plan=plan.name,
            )
    print(f"runlog: {run.path}")


# ------------------------------------------------------------------- rodzic


def run_plan(
    plan: Plan,
    jobs: int = 1,
    replace: bool = False,
    stagger: float = 15.0,
    trace: bool = False,
) -> dict:
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
        if trace:
            cmd.append("--trace")
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
