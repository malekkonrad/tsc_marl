"""Postawienie procesu LibSignala pod zarejestrowanym biegiem `runlog`.

Jedyne miejsce w repo, które zna KOLEJNOŚĆ wywołań `common.interface.*`. Ta
kolejność jest wrażliwa i już raz kosztowała cichy błąd: `root_dir` ustawiony po
`Logger_path_Interface` był ignorowany, przez co `att_last10` przyjmowało wartość
`att` (27% różnicy na Advanced-CoLighcie). Dopóki sekwencja stała w trzech
kopiach (`runner/train.py`, `runner/train_core.py`, `runner/evaluate.py`), każda
poprawka musiała trafić w trzy miejsca — a `evaluate.py` zdążył się już rozjechać.

Szew jest tu, gdzie wejścia naprawdę się różnią: sesja kończy się w stanie
„Registry zapełnione, świat jeszcze NIE zbudowany". `runner/train.py` oddaje budowę
świata `BaseTrainerowi`, `runner/train_core.py` stawia go sam, `evaluate.py` dostaje go od
runnera — więc świat nie może powstawać tutaj, bo wróciłby `if` w środku
sekwencji.

    with session(args) as s:
        ...  # s.run, s.config, s.logger — Registry gotowe

Ciało `with` leci POD otwartym biegiem: pad przy stawianiu świata ma dać status
`failed`, a nie zostawić katalog na wieki w `running` (co blokuje ponowne
uruchomienie przez `RunExists`).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from tsc_marl import libsignal
from tsc_marl.seeding import set_global_seed

# Kolejność, w jakiej `run.py` LibSignala zapełnia Registry. Trzymana jako dane,
# żeby dała się sprawdzić testem — komentarz nie chroni przed niczym.
INTERFACE_ORDER = (
    "Command_Setting_Interface",
    "Logger_param_Interface",
    "World_param_Interface",
    "Graph_World_Interface",  # tylko gdy model.graphic — patrz build_interfaces
    "Logger_path_Interface",
    "Trainer_param_Interface",
    "ModelAgent_param_Interface",
)


@dataclass
class Session:
    """Co wejście dostaje od sesji. `args` niesiemy dalej, bo `experiment.ingest`
    i `engine_log_dir` odtwarzają z niego ścieżki logów silnika."""

    run: Any  # runlog.Run — bieg już otwarty
    config: dict[str, Any]
    logger: logging.Logger
    args: Any
    device: Any = "cpu"  # torch.device — rozwiązane z `--ngpu`, patrz `torch_device`


def torch_device(ngpu: str):
    """`--ngpu` → `torch.device`. JEDYNE miejsce, które tłumaczy jedno na drugie.

    Pułapka, dla której ta funkcja istnieje osobno: `session()` ustawia
    `CUDA_VISIBLE_DEVICES=<ngpu>` PRZED pierwszym dotknięciem CUDA przez torcha.
    Po tym remapowaniu wybrana karta jest dla torcha ZAWSZE indeksem 0 — więc
    `--ngpu 2` daje `cuda:0`, a nie `cuda:2`. Zbudowanie `cuda:2` wywaliłoby się
    na „invalid device ordinal", bo torch widzi wtedy dokładnie jedną kartę.

    Brak karty przy jawnie zażądanym GPU NIE jest błędem twardym (bieg na CPU
    policzy się poprawnie, tylko wolniej), ale musi być głośny: cicha degradacja
    30-godzinnej siatki do CPU to najgorszy możliwy wariant. Ostrzeżenie wypisuje
    `session()`, gdy ma już logger biegu.
    """
    import torch  # noqa: PLC0415 - ciężki import, tylko gdy naprawdę potrzebny

    if str(ngpu) == "-1" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")


def register_engine_components() -> None:
    """Importy WYŁĄCZNIE dla efektu ubocznego: dekoratory zapełniają Registry,
    z którego korzystają interfejsy. Bez `task` i `trainer` `Registry.mapping[...]`
    leci KeyError. Musi być PO `libsignal.bootstrap()` — wcześniej submodułu nie
    ma na sys.path."""
    import agent  # noqa: F401
    import dataset  # noqa: F401
    import task  # noqa: F401
    import trainer  # noqa: F401

    import tsc_marl.libsignal.agents  # noqa: F401 - rejestruje NASZYCH agentów
    import tsc_marl.libsignal.sumo_world  # noqa: F401 - nadpisuje 'sumo' w world_mapping


def build_interfaces(config: dict[str, Any]) -> list[str]:
    """Zapełnia Registry w kolejności z `INTERFACE_ORDER`. Zwraca nazwy REALNIE
    wywołanych interfejsów, żeby test mógł przypiąć kolejność.

    `Graph_World_Interface` warunkowo — tak jak w `run.py`. Rdzeń nie potrzebuje
    tu wyjątku: `configs/tsc/gatppo.yml` ma `graphic: True`, a `require_graph=True`
    w `session()` pilnuje, żeby to zostało prawdą.
    """
    from common import interface
    from common.registry import Registry

    called = []

    interface.Command_Setting_Interface(config)
    called.append("Command_Setting_Interface")
    interface.Logger_param_Interface(config)
    called.append("Logger_param_Interface")
    interface.World_param_Interface(config)
    called.append("World_param_Interface")

    if config["model"].get("graphic", False):
        param = Registry.mapping["world_mapping"]["setting"].param
        roadnet = (
            param["dir"] + param["roadnetFile"]
            if config["command"]["world"] in ("cityflow", "sumo")
            else param["road_file_addr"]
        )
        interface.Graph_World_Interface(roadnet)
        if config["command"]["world"] == "sumo":
            # graf LibSignala na sieci z OSM ma 1 krawędź na 25 węzłów (Kraków) —
            # podmieniamy go dla WSZYSTKICH agentów grafowych, patrz `sumo_world.build_graph`
            from tsc_marl.libsignal.sumo_world import build_graph

            interface.Graph_World_Interface.graph = build_graph(roadnet)
        called.append("Graph_World_Interface")

    interface.Logger_path_Interface(config)
    called.append("Logger_path_Interface")
    # katalog musi istnieć, zanim cokolwiek zacznie do niego pisać
    os.makedirs(Registry.mapping["logger_mapping"]["path"].path, exist_ok=True)
    interface.Trainer_param_Interface(config)
    called.append("Trainer_param_Interface")
    interface.ModelAgent_param_Interface(config)
    called.append("ModelAgent_param_Interface")

    return called


def merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Rekurencyjne scalenie nadpisań w config (in place). Płytkie `update`
    skasowałoby całą sekcję `model` przy nadpisaniu jednego pola."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def resolve(args, config_patch: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    """Config po WSZYSTKICH nadpisaniach + sufiks `run_id`.

    Wspólne dla procesu liczącego (`session`) i dla planowania siatki
    (`suite.plan`). Gdyby liczyły config osobno, `skip-if-done` rozjechałby się
    po cichu: runner uznałby punkt za policzony, choć dziecko policzyłoby go
    z innymi hiperparametrami.

    Kolejność nadpisań: config agenta → `overrides` suite'a → `config_patch`
    wołającego. Ostatni wygrywa, bo `evaluate.py` musi wymusić `train_model:
    False` niezależnie od tego, co deklaruje suite.
    """
    from tsc_marl.libsignal import experiment

    config = experiment.resolve_config(args)
    suffix = ""
    if getattr(args, "suite", None):
        from tsc_marl.runner import suite as suite_module

        loaded = suite_module.load(args.suite)
        merge(config, loaded.overrides)
        suffix = loaded.suffix
    if config_patch:
        merge(config, config_patch)
    return config, suffix


@contextmanager
def session(
    args,
    *,
    config_patch: dict[str, Any] | None = None,
    params_extra: dict[str, Any] | None = None,
    run_id: str | None = None,
    replace: bool | None = None,
    require_graph: bool = False,
    log_level: int | None = None,
):
    """Bootstrap → seed → config → rejestracja biegu → interfejsy → logger.

    `config_patch` jest nakładany PRZED `build_params` i to nie jest szczegół:
    `train_model`/`test_model` (ewaluacja) i nadpisania z suite'a siedzą w sekcji
    `model`, która wchodzi do `config_hash`. Nałożone później zmieniałyby
    zachowanie biegu, nie zmieniając jego odcisku — czyli `skip-if-done`
    pomijałby bieg z innymi hiperparametrami. To ta sama klasa błędu co `root_dir`.
    """
    libsignal.bootstrap()
    # jedna ścieżka do GPU dla treningu I ewaluacji: polityka ma być testowana na
    # tym samym urządzeniu, na którym powstała
    os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu
    set_global_seed(args.seed)

    register_engine_components()
    from utils.logger import setup_logging

    from tsc_marl.libsignal import experiment

    config, suffix = resolve(args, config_patch)

    if require_graph and not config["model"].get("graphic", False):
        raise SystemExit(
            f"config agenta '{args.agent}' nie ma `graphic: True`, a to wejście wymaga grafu "
            "(Graph_World_Interface). Dopisz `graphic: True` w configs/tsc/."
        )

    # rejestracja PRZED interfejsami: `root_dir` musi być ustawiony, zanim
    # LibSignal policzy sobie ścieżki wyjściowe w Logger_path_Interface
    exp = experiment.experiment()
    params = experiment.build_params(args, config) | dict(params_extra or {})
    rid = (
        run_id
        or getattr(args, "run_id", None)
        or experiment.run_id(
            args.agent, args.network, args.seed, config["trainer"]["yellow_length"], suffix=suffix
        )
    )
    if replace is None:
        replace = getattr(args, "replace", False)
    run = exp.start(rid, params, config=config, repos=experiment.repos(), replace=replace)
    # artefakty silnika (checkpointy, logi, replaye) lądują WEWNĄTRZ katalogu biegu,
    # dzięki czemu „jeden bieg = jeden katalog" jest prawdą, a data/ zostaje katalogiem
    # wejściowym. Działa dzięki patchowi get_output_file_path w submodule.
    config["logger"]["root_dir"] = str(run.engine_dir)

    if log_level is None:
        log_level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO

    with run, libsignal.sim_configs_untouched():
        build_interfaces(config)
        logger = setup_logging(log_level)
        device = torch_device(args.ngpu)
        if args.ngpu != "-1" and device.type == "cpu":
            logger.warning(
                f"--ngpu {args.ngpu} zażądane, ale torch nie widzi CUDA — bieg leci na CPU. "
                "Na maszynie z kartą sprawdź `make setup-gpu` (torch z indeksu cu121)."
            )
        yield Session(run=run, config=config, logger=logger, args=args, device=device)


@contextmanager
def standalone_world(
    agent: str,
    network: str,
    *,
    config_patch: dict[str, Any] | None = None,
    n_agents: int | None = None,
):
    """Świat LibSignala + agenci POZA pętlą treningową — diagnostyka i zrzuty danych.

    `session()` kończy się w stanie „Registry zapełnione, świat NIE zbudowany",
    bo każde wejście stawia go inaczej; tutaj stawiamy go sami. Bieg `runlog`
    NIE jest tu zakładany: pomiar przelicznika albo zrzut zbioru to nie to samo
    co bieg eksperymentu, a wołający, który chce proweniencji, otwiera bieg sam.

    Wszystko idzie pod `sim_configs_untouched`, bo `build_config` DOPISUJE do
    `configs/sim/*.cfg` ścieżki logów replaya ze znacznikiem czasu — bez tego
    samo policzenie diagnostyki brudzi wersjonowany config.

    -> (world, agents, config). `n_agents` domyślnie po jednym na skrzyżowanie
    (tak zbudowane są metody MP-owe); modele współdzielone, jak Advanced-CoLight,
    biorą `n_agents=1`.
    """
    libsignal.bootstrap()
    register_engine_components()

    from common.registry import Registry

    from tsc_marl import paths
    from tsc_marl.runner.train import parse_args

    args = parse_args(["-a", agent, "-n", network])
    with libsignal.sim_configs_untouched():
        config, _ = resolve(args, config_patch)
        build_interfaces(config)

        from world.world_cityflow import World

        cfg = paths.CONFIGS / "sim" / f"{network}.cfg"
        world = World(str(cfg), thread_num=config["trainer"].get("thread", 4), interface="libsumo")
        n = len(world.intersections) if n_agents is None else n_agents
        agents = [Registry.mapping["model_mapping"][agent](world, rank) for rank in range(n)]
        yield world, agents, config
