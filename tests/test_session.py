"""Testy wspólnego przygotowania procesu (`tsc_marl.session`).

Sekwencja interfejsów Registry jest wrażliwa na kolejność i już raz kosztowała
cichy błąd: `root_dir` ustawiony PO `Logger_path_Interface` był ignorowany,
przez co `att_last10` przyjmowało wartość `att` (27% różnicy). Dopóki sekwencja
stała w trzech kopiach wejść CLI, chronił ją wyłącznie komentarz. Tutaj jest
przypięta asercjami — to jest właściwy powód, dla którego została wyciągnięta.

Testy nie stawiają silnika: `interface` i `Registry` są podmienione atrapami,
więc sprawdzamy KOLEJNOŚĆ i MOMENT, a nie CityFlow.
"""

from __future__ import annotations

import types

import pytest

from tsc_marl import experiment, libsignal, session
from tsc_marl.run import parse_args

NETWORK = "cityflow_jinan3x4_4ph"


class _Param:
    def __init__(self, **kw):
        self.param = kw


class _FakeInterfaceModule(types.SimpleNamespace):
    """Zapisuje nazwy wywołanych interfejsów zamiast dotykać globalnego Registry."""

    def __init__(self, calls):
        super().__init__()
        self._calls = calls

    def __getattr__(self, name):
        if not name.endswith("_Interface"):
            raise AttributeError(name)

        def record(*_args, **_kw):
            self._calls.append(name)

        return record


@pytest.fixture
def atrapy_interfejsow(monkeypatch, tmp_path):
    """Atrapy `common.interface` i `common.registry` w sys.modules — `build_interfaces`
    da się wtedy przejść bez stawiania CityFlow, a przedmiotem testu jest kolejność."""
    import sys

    libsignal.bootstrap()
    calls: list[str] = []
    logger_path = tmp_path / "engine_logger"

    registry = types.SimpleNamespace(
        mapping={
            "world_mapping": {"setting": _Param(dir="data/", roadnetFile="roadnet.json")},
            "logger_mapping": {"path": types.SimpleNamespace(path=str(logger_path))},
        }
    )
    common = types.ModuleType("common")
    common_interface = _FakeInterfaceModule(calls)
    common_registry = types.ModuleType("common.registry")
    common_registry.Registry = registry

    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.setitem(sys.modules, "common.interface", common_interface)
    monkeypatch.setitem(sys.modules, "common.registry", common_registry)
    monkeypatch.setattr(common, "interface", common_interface, raising=False)
    return types.SimpleNamespace(calls=calls, logger_path=logger_path, registry=registry)


def _config(graphic: bool) -> dict:
    return {
        "command": {"world": "cityflow"},
        "model": {"graphic": graphic},
        "logger": {},
        "trainer": {"yellow_length": 5},
    }


# --------------------------------------------------------------- kolejność


def test_sekwencja_interfejsow_ma_kanoniczna_kolejnosc(atrapy_interfejsow):
    """Kolejność jak w `run.py` LibSignala. Zmiana tej listy = zmiana kontraktu
    z cudzym kodem, więc ma boleć w teście, a nie po 200 epizodach treningu."""
    wywolane = session.build_interfaces(_config(graphic=True))
    assert wywolane == list(session.INTERFACE_ORDER)
    assert wywolane == atrapy_interfejsow.calls


def test_bez_graphic_graf_sie_nie_buduje(atrapy_interfejsow):
    """`Graph_World_Interface` jest warunkowe (`model.graphic`) — baseline'y bez
    grafu (fixedtime, maxpressure) nie mają go budować."""
    wywolane = session.build_interfaces(_config(graphic=False))
    assert "Graph_World_Interface" not in wywolane
    assert wywolane == [i for i in session.INTERFACE_ORDER if i != "Graph_World_Interface"]


def test_katalog_logow_silnika_powstaje_przed_trenerem(atrapy_interfejsow):
    """`makedirs` leci między `Logger_path_Interface` a `Trainer_param_Interface` —
    trener zaczyna pisać od razu, a nieistniejący katalog to wywrotka w połowie biegu."""
    session.build_interfaces(_config(graphic=True))
    assert atrapy_interfejsow.logger_path.is_dir()


# ------------------------------------------------------- moment rejestracji


@pytest.fixture
def sesja_bez_silnika(monkeypatch, tmp_path):
    """`runs/` w tmp_path + podstawiona sekwencja interfejsów.

    Registry zostaje PRAWDZIWE: `register_engine_components()` importuje moduły
    LibSignala, a te rejestrują się w nim dekoratorami. Atrapa Registry (jak wyżej,
    dla `build_interfaces`) wywróciłaby już sam import. Symulatora i tak nie ma —
    stawia go dopiero ciało `with`, którego te testy nie mają.
    """
    from runlog import Experiment

    monkeypatch.setattr(
        experiment, "experiment", lambda *a, **kw: Experiment(experiment.EXPERIMENT, root=tmp_path)
    )
    monkeypatch.setattr(session, "build_interfaces", lambda config: None)
    return tmp_path / experiment.EXPERIMENT


def test_root_dir_jest_ustawiony_zanim_libsignal_policzy_sciezki(sesja_bez_silnika, monkeypatch):
    """TEN test pilnuje błędu, który kosztował 27%: `root_dir` musi wskazywać na
    `engine/` katalogu biegu ZANIM zadziała `Logger_path_Interface`. Ustawiony
    później jest po cichu ignorowany, a logi lądują w `data/output_data`."""
    widziany = {}

    def zapamietaj(config):
        widziany["root_dir"] = config["logger"].get("root_dir")

    monkeypatch.setattr(session, "build_interfaces", zapamietaj)

    args = parse_args(["-n", NETWORK, "--run-id", "kolejnosc"])
    with session.session(args) as s:
        assert widziany["root_dir"] == str(s.run.engine_dir)


def test_pad_w_ciele_konczy_bieg_jako_failed(sesja_bez_silnika):
    """Ciało `with` leci POD otwartym biegiem. Gdyby leciało obok, pad przy budowie
    świata zostawiałby katalog na wieki w `running` — a `running` blokuje ponowne
    uruchomienie (`RunExists`) i trzeba go kasować ręcznie."""
    from runlog import RunView

    args = parse_args(["-n", NETWORK, "--run-id", "pada"])

    with pytest.raises(RuntimeError), session.session(args):
        raise RuntimeError("zły roadnet")

    assert RunView(sesja_bez_silnika / "pada").status == "failed"


def test_config_patch_wchodzi_do_odcisku_biegu(sesja_bez_silnika):
    """Nadpisania muszą być nałożone PRZED `build_params`. Nałożone później
    zmieniałyby zachowanie biegu, nie zmieniając `config_hash` — czyli
    `skip-if-done` uznałby bieg z innymi hiperparametrami za już policzony.
    Na tym mechanizmie stoi ewaluacja (`train_model`) i nadpisania suite'ów.

    Patchujemy `train_model: True`, bo domyślny agent testu (`efficientmp`) niczego
    nie trenuje i ma w configu `False` — nadpisanie tą samą wartością nie zmieniłoby
    odcisku i test przechodziłby także przy zepsutej kolejności."""
    with session.session(parse_args(["-n", NETWORK, "--run-id", "goly"])) as a:
        goly = a.run.meta["params"]["config_hash"]
        assert a.config["model"]["train_model"] is False
    with session.session(
        parse_args(["-n", NETWORK, "--run-id", "zpatchem"]),
        config_patch={"model": {"train_model": True}},
    ) as b:
        z_patchem = b.run.meta["params"]["config_hash"]
        assert b.config["model"]["train_model"] is True

    assert goly != z_patchem, "nadpisanie configu nie zmieniło odcisku biegu"


def test_params_extra_trafia_do_parametrow(sesja_bez_silnika):
    """Ewaluacja dokłada `parent_run`/`checkpoint`/`scenario` — bez nich nie da się
    odtworzyć, KTÓRA polityka dała ten wiersz wyniku."""
    with session.session(
        parse_args(["-n", NETWORK, "--run-id", "eval"]),
        params_extra={"parent_run": "rodzic", "scenario": "drop10"},
    ) as s:
        assert s.run.meta["params"]["parent_run"] == "rodzic"
        assert s.run.meta["params"]["scenario"] == "drop10"


def test_require_graph_odmawia_configu_bez_grafu(sesja_bez_silnika, monkeypatch):
    """`run_core.py` żąda grafu configiem, a nie bezwarunkowym wywołaniem interfejsu.
    Config bez `graphic: True` ma paść OD RAZU, a nie zbudować rdzeń bez sąsiadów."""
    monkeypatch.setattr(
        experiment, "resolve_config", lambda args: _config(graphic=False) | {"logger": {}}
    )
    with pytest.raises(SystemExit, match="graphic"):
        with session.session(parse_args(["-n", NETWORK]), require_graph=True):
            pass


# ---------------------------------------------------------------- scalanie


def test_merge_nie_kasuje_reszty_sekcji():
    """Płytkie `update` podmieniłoby całą sekcję `model` na jednoelementową —
    i bieg poleciałby bez `state`, `reward` i architektury."""
    base = {"model": {"state": "ats", "reward": "queue"}, "trainer": {"episodes": 200}}
    session.merge(base, {"model": {"reward": "delay"}})
    assert base["model"] == {"state": "ats", "reward": "delay"}
    assert base["trainer"] == {"episodes": 200}
