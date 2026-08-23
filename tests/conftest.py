"""Wspólna infrastruktura testów.

Testy ATS nie potrzebują silnika: `Intersection` z LibSignala parsuje roadnet
sam, a jedyne, czego dotyka na zewnątrz, to `world.eng.set_tl_phase` w `reset()`.
Budujemy więc PRAWDZIWY obiekt `Intersection` na atrapie świata — dzięki temu
test sprawdza kod, który naprawdę biegnie w symulacji, a nie własną kopię
parsowania roadnetu, i chodzi w ~0,1 s zamiast stawiać CityFlow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tsc_marl import libsignal

libsignal.bootstrap()  # ścieżka do submodułu + SUMO_HOME (world/__init__ ciągnie world_sumo)

ROADNETS = {
    "jinan3x4": libsignal.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4_4phase.json",
    "hangzhou4x4": libsignal.DATA
    / "raw_data"
    / "hangzhou_4x4_advanced"
    / "roadnet_4_4_4phase.json",
}


class _FakeEngine:
    def set_tl_phase(self, *args):
        pass


class _FakeWorld:
    RIGHT = True

    def __init__(self):
        self.eng = _FakeEngine()


def load_intersections(roadnet: Path):
    """(obiekt Intersection, surowy dict z roadnetu) dla każdego niewirtualnego skrzyżowania."""
    from world.world_cityflow import Intersection  # noqa: PLC0415 - dopiero po bootstrapie

    raw = json.loads(roadnet.read_text())
    world = _FakeWorld()
    return [(Intersection(d, world), d) for d in raw["intersections"] if not d["virtual"]]


@pytest.fixture(scope="session", params=sorted(ROADNETS))
def roadnet(request):
    path = ROADNETS[request.param]
    if not path.is_file():
        pytest.skip(f"brak roadnetu {path} (dane pobiera się osobno — patrz data/README.md)")
    return load_intersections(path)


# --------------------------------------------------------------------------
# Poniżej: prawdziwy świat CityFlow + agent, dla testów oznaczonych `slow`.
# Odtwarza sekwencję z `tsc_marl.run.main()` — LibSignal trzyma konfigurację w
# globalnym `Registry`, zapełnianym przez interfejsy w ściśle określonej
# kolejności, więc nie da się zbudować `World` bez przejścia tej samej ścieżki.
# Registry nie ma czystego resetu, dlatego świat stawiamy RAZ na sesję.
# --------------------------------------------------------------------------

NETWORK = "cityflow_jinan3x4_4ph"


def _build(agent_name: str, network: str):
    libsignal.bootstrap()

    import agent  # noqa: F401 - dekoratory zapełniają Registry
    import dataset  # noqa: F401
    import task  # noqa: F401
    import trainer  # noqa: F401
    from common import interface
    from common.registry import Registry
    from utils.logger import build_config

    import tsc_marl.agents  # noqa: F401 - rejestruje naszych agentów
    from tsc_marl.run import parse_args

    args = parse_args(["-a", agent_name, "-n", network])
    config, _ = build_config(args)

    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config)
    param = Registry.mapping["world_mapping"]["setting"].param
    interface.Graph_World_Interface(param["dir"] + param["roadnetFile"])
    interface.Logger_path_Interface(config)
    os.makedirs(Registry.mapping["logger_mapping"]["path"].path, exist_ok=True)
    interface.Trainer_param_Interface(config)
    interface.ModelAgent_param_Interface(config)

    from world.world_cityflow import World

    world = World(f"configs/sim/{network}.cfg", thread_num=4, interface="libsumo")
    model = Registry.mapping["model_mapping"][agent_name](world, 0)
    return world, model


@pytest.fixture(scope="session")
def advanced_colight_world():
    """(world, agent) dla `advancedcolight` na JiNan 3×4. ~15 s — stąd `slow`."""
    cfg = libsignal.CONFIGS / "sim" / f"{NETWORK}.cfg"
    snapshot = cfg.read_bytes()  # LibSignal wpisuje tu ścieżki logów replaya
    try:
        yield _build("advancedcolight", NETWORK)
    finally:
        Path(cfg).write_bytes(snapshot)
