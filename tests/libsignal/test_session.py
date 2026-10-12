"""Shared process setup (`tsc_marl.libsignal.session`).

The Registry interface sequence is order-sensitive and has already caused a
silent bug (a `root_dir` set AFTER `Logger_path_Interface` was ignored, so
`att_last10` took the value of `att`). Here the order is pinned by assertions.
No engine: `interface` and `Registry` are replaced by stubs, so the tests check
ORDER and TIMING, not CityFlow.
"""

from __future__ import annotations

import types

import pytest

from tsc_marl import libsignal
from tsc_marl.libsignal import experiment, session
from tsc_marl.libsignal.experiment import parse_args

NETWORK = "cityflow_jinan3x4_4ph"


class _Param:
    def __init__(self, **kw):
        self.param = kw


class _FakeInterfaceModule(types.SimpleNamespace):
    """Records called interface names instead of touching the global Registry."""

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
def interface_stubs(monkeypatch, tmp_path):
    """Stubs of `common.interface` and `common.registry` in sys.modules, so
    `build_interfaces` runs without CityFlow and the test is about the order."""
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


# --------------------------------------------------------------- order


def test_interface_sequence_has_the_canonical_order(interface_stubs):
    """The order of LibSignal's `run.py`; changing it changes a contract with
    foreign code, so it must hurt in a test."""
    called = session.build_interfaces(_config(graphic=True))
    assert called == list(session.INTERFACE_ORDER)
    assert called == interface_stubs.calls


def test_without_graphic_no_graph_is_built(interface_stubs):
    """`Graph_World_Interface` is conditional (`model.graphic`)."""
    called = session.build_interfaces(_config(graphic=False))
    assert "Graph_World_Interface" not in called
    assert called == [i for i in session.INTERFACE_ORDER if i != "Graph_World_Interface"]


def test_engine_log_directory_exists_before_the_trainer(interface_stubs):
    """`makedirs` runs between `Logger_path_Interface` and `Trainer_param_Interface`;
    the trainer writes at once."""
    session.build_interfaces(_config(graphic=True))
    assert interface_stubs.logger_path.is_dir()


# ------------------------------------------------------- registration timing


@pytest.fixture
def session_without_engine(monkeypatch, tmp_path):
    """`runs/` in tmp_path and a stubbed interface sequence. The Registry stays
    REAL (LibSignal modules register into it at import); there is no simulator,
    since only the `with` body would build one."""
    from runlog import Experiment

    monkeypatch.setattr(
        experiment, "experiment", lambda *a, **kw: Experiment(experiment.EXPERIMENT, root=tmp_path)
    )
    monkeypatch.setattr(session, "build_interfaces", lambda config: None)
    return tmp_path / experiment.EXPERIMENT


def test_root_dir_is_set_before_libsignal_computes_paths(session_without_engine, monkeypatch):
    """`root_dir` must point to the run's `engine/` BEFORE `Logger_path_Interface`;
    set later it is silently ignored and logs land in `data/output_data`."""
    seen = {}

    def remember(config):
        seen["root_dir"] = config["logger"].get("root_dir")

    monkeypatch.setattr(session, "build_interfaces", remember)

    args = parse_args(["-n", NETWORK, "--run-id", "order"])
    with session.session(args) as s:
        assert seen["root_dir"] == str(s.run.engine_dir)


def test_a_failure_in_the_body_ends_the_run_as_failed(session_without_engine):
    """The `with` body runs UNDER the open run; otherwise a failure while building
    the world would leave the directory `running` forever (blocking a rerun)."""
    from runlog import RunView

    args = parse_args(["-n", NETWORK, "--run-id", "fails"])

    with pytest.raises(RuntimeError), session.session(args):
        raise RuntimeError("bad roadnet")

    assert RunView(session_without_engine / "fails").status == "failed"


def test_config_patch_enters_the_run_fingerprint(session_without_engine):
    """Overrides are applied BEFORE `build_params`; later they would change behaviour
    without changing `config_hash`. `train_model: True` is patched because the test
    agent (`efficientmp`) has `False`, and the same value would prove nothing."""
    with session.session(parse_args(["-n", NETWORK, "--run-id", "plain"])) as a:
        plain = a.run.meta["params"]["config_hash"]
        assert a.config["model"]["train_model"] is False
    with session.session(
        parse_args(["-n", NETWORK, "--run-id", "patched"]),
        config_patch={"model": {"train_model": True}},
    ) as b:
        patched = b.run.meta["params"]["config_hash"]
        assert b.config["model"]["train_model"] is True

    assert plain != patched, "the config override did not change the run fingerprint"


def test_params_extra_reaches_the_parameters(session_without_engine):
    """Evaluation adds `parent_run`/`checkpoint`/`scenario`; without them one cannot
    tell WHICH policy produced a result row."""
    with session.session(
        parse_args(["-n", NETWORK, "--run-id", "eval"]),
        params_extra={"parent_run": "parent", "scenario": "drop10"},
    ) as s:
        assert s.run.meta["params"]["parent_run"] == "parent"
        assert s.run.meta["params"]["scenario"] == "drop10"


def test_require_graph_rejects_a_config_without_a_graph(session_without_engine, monkeypatch):
    """The core asks for the graph through the config; a config without
    `graphic: True` fails AT ONCE instead of building a core without neighbours."""
    monkeypatch.setattr(
        experiment, "resolve_config", lambda args: _config(graphic=False) | {"logger": {}}
    )
    with pytest.raises(SystemExit, match="graphic"):
        with session.session(parse_args(["-n", NETWORK]), require_graph=True):
            pass


# ---------------------------------------------------------------- merging


def test_merge_keeps_the_rest_of_a_section():
    """A shallow `update` would replace the whole `model` section with one field."""
    base = {"model": {"state": "ats", "reward": "queue"}, "trainer": {"episodes": 200}}
    session.merge(base, {"model": {"reward": "delay"}})
    assert base["model"] == {"state": "ats", "reward": "delay"}
    assert base["trainer"] == {"episodes": 200}


# ----------------------------------------------------------------- device


def test_ngpu_minus_one_is_the_cpu():
    """`-1` = CPU, whether or not the machine has a card."""
    assert session.torch_device("-1").type == "cpu"


def test_no_cuda_falls_back_to_the_cpu(monkeypatch):
    """A requested GPU without CUDA must not break a run; `session()` logs the
    warning, this checks the device resolution."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert session.torch_device("0").type == "cpu"


def test_the_card_is_always_zero(monkeypatch):
    """`--ngpu 3` gives `cuda:0`, NOT `cuda:3`: `session()` sets
    `CUDA_VISIBLE_DEVICES=3` first, so torch sees exactly one card, numbered 0."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for ngpu in ("0", "1", "3"):
        assert session.torch_device(ngpu) == torch.device("cuda:0")
