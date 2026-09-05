"""Testy wejścia CLI rdzenia.

Wejście jest krótkie, ale kruche: sekwencja interfejsów Registry jest wrażliwa
na kolejność i już raz kosztowała cichy błąd (`root_dir` ignorowany → `att_last10`
przyjmowało wartość `att`, 27% różnicy na Advanced-CoLighcie). Dlatego test
end-to-end przechodzi CAŁĄ sekwencję na prawdziwym CityFlow, tylko na epizodzie
skróconym do 10 decyzji.

Reszta to testy configu — najważniejszy sprawdza, że KAŻDY hiperparametr PPO
siedzi w YAML-u. Parametr żyjący tylko w Pythonie nie wchodzi do odcisku configu,
więc `skip-if-done` uznałby bieg z innym `clip_range` za „już policzony".
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from tsc_marl import libsignal
from tsc_marl.core.ppo import PPOConfig
from tsc_marl.libsignal import experiment
from tsc_marl.runner.train_core import parse_args

NETWORK = "cityflow_jinan3x4_4ph"


def test_domyslny_agent_to_rdzen():
    """Bez `-a` bierzemy `gatppo`; jawny wybór dalej wygrywa, bo obie ścieżki
    zapisują biegi do jednego `runlog` i muszą się dać uruchomić tak samo."""
    assert parse_args([]).agent == "gatppo"
    assert parse_args(["-a", "colight"]).agent == "colight"
    assert parse_args(["--agent", "advancedcolight"]).agent == "advancedcolight"


@pytest.fixture(scope="module")
def core_config():
    libsignal.bootstrap()
    return experiment.resolve_config(parse_args(["-n", NETWORK]))


def test_rdzen_widzi_ten_sam_protokol_i_stan_co_baseline(core_config):
    """GATE Fazy 1 porównuje rdzeń z Advanced-CoLightem, więc różnicą ma być
    WYŁĄCZNIE algorytm uczenia. Protokół (Δt, yellow, długość epizodu),
    reprezentacja stanu (ATS, okno 167, bez zawsze-zielonych) i architektura GAT
    muszą przyjść z tych samych plików."""
    model, trainer = core_config["model"], core_config["trainer"]
    assert (trainer["action_interval"], trainer["yellow_length"], trainer["steps"]) == (15, 5, 3600)
    assert (model["state"], model["reward"]) == ("ats", "queue")
    assert model["obs_length"] == 167 and model["drop_always_green"] is True
    assert model["NUM_HEADS"] == [5, 5] and model["NODE_EMB_DIM"] == [128, 128]


def test_kazdy_hiperparametr_ppo_jest_w_configu(core_config):
    """Odcisk biegu liczy się z sekcji `model` + `trainer` (`experiment.build_params`).
    Hiperparametr trzymany tylko w `PPOConfig` w kodzie nie zmieniałby odcisku, więc
    bieg z innym `clip_range` zostałby po cichu pominięty jako już policzony."""
    brakujace = [f.name for f in fields(PPOConfig) if f.name not in core_config["model"]]
    assert brakujace == [], f"hiperparametry PPO poza configiem: {brakujace}"


def test_steps_nadpisuje_oba_epizody():
    """`--steps` skraca trening I test naraz — krótki trening z pełnym testem
    dawałby liczby, których nie da się z niczym zestawić."""
    libsignal.bootstrap()
    config = experiment.resolve_config(parse_args(["-n", NETWORK, "--steps", "150"]))
    assert config["trainer"]["steps"] == config["trainer"]["test_steps"] == 150


@pytest.mark.slow
def test_bieg_end_to_end_zapisuje_wyniki(tmp_path, monkeypatch):
    """Cała ścieżka: interfejsy Registry → świat → `CoreTrainer` → `runlog`.
    Jeden epizod po 10 decyzji, bo sprawdzamy hydraulikę, nie jakość polityki.

    `runs/` przekierowane do `tmp_path`, żeby smoke-test nie zaśmiecał katalogu
    z prawdziwymi wynikami.
    """
    from runlog import Experiment
    from tsc_marl.runner import train_core

    monkeypatch.setattr(
        experiment, "experiment", lambda *a, **kw: Experiment(experiment.EXPERIMENT, root=tmp_path)
    )
    train_core.main(["-n", NETWORK, "--episodes", "1", "--steps", "150", "--run-id", "smoke"])

    bieg = tmp_path / experiment.EXPERIMENT / "smoke"
    from runlog import RunView

    widok = RunView(bieg)
    assert widok.status == "ok"
    assert widok.result["att"] > 0 and widok.result["episodes_logged"] == 1
    assert (bieg / "engine" / "core.pt").is_file(), "checkpoint rdzenia nie powstał"

    metryki = (bieg / "metrics.csv").read_text()
    assert "train" in metryki and "test" in metryki
    assert "approx_kl" in metryki, "diagnostyka PPO musi trafiać do metryk biegu"
