"""Próg bramki-REGUŁY bez głowy niepewności — dla sieci, na których głowy nie ma (Kraków).

DLACZEGO OSOBNO. Na JiNan/HangZhou próg reguły powstaje razem z głową obserwacyjną
(`tsc obshead --save`, artefakt `runs/heads/gate_ref__<sieć>.npz`), bo tam ablacja
porównywała regułę z bramką UQ i oba progi MUSIAŁY wyjść z jednego kryterium. Ten potok
czyta jednak roadnet w formacie CityFlow (`obs_head.movement_sources`) i uczy głowę,
której reguła nie potrzebuje — a UW-attention, jedyny inny odbiorca głowy, jest
zamknięty (C5). Na SUMO liczymy więc SAM próg reguły, tym samym przepisem:

    próg = kwantyl (1 − GATE_FALSE_ALARM) udziału zamrożonych ruchów węzła,
           na czystym ruchu pod ekspertem Advanced-MP, tylko kroki z PEŁNYM oknem.

Ekspert ten sam co w zrzucie BC (`uncertainty/dataset.py`), budżet fałszywych alarmów
ten sam (0,5%), okno to samo (12) — różnica wobec `gate_ref` jest tylko taka, że nie
ma podziału na część uczącą głowę, bo głowy nie ma. Udział liczony po PRAWDZIWYCH
ruchach (`StateBuilder.movement_mask`), inaczej dopchane zera Krakowa (1–7 wlotów)
wyglądałyby na martwe detektory.

Sygnał reguły ZALEŻY OD POLITYKI (`p1_uw_rule`: nienauczona polityka głodzi wloty
i reguła strzela na 87% krawędzi przy czystym ruchu). Kalibracja pod ekspertem mówi
więc „ile fałszywych alarmów przy rozsądnym sterowaniu", nie przy każdym.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

#: Okno reguły — to samo, z którym policzono `gate_ref` JiNan/HangZhou
#: (`tsc obshead --save --window 12 --persistence`).
RULE_WINDOW = 12
RULE_FORMAT = 1


@dataclass(frozen=True)
class RuleReference:
    """Próg samej reguły. Pola `window` i `rule` jak w `obs_head.GateReference`, więc
    `build_gate` i `uw_signal` czytają oba artefakty tak samo; `uq` i `u_mov_ref`
    są None — bramka UQ na takiej sieci nie istnieje i ma się o tym głośno dowiedzieć."""

    network: str
    window: int
    rule: float
    false_alarm: float
    expert: str
    episodes: int
    decisions: int
    # Odsetek par (decyzja, węzeł) z czystego ruchu, na których bramka z TYM progiem by
    # strzeliła (porównanie `>=`, jak w `RuleGate`). Budżet to cel, a to jest pomiar:
    # na rzadkim ruchu (Kraków `low`) całe węzły bywają puste, kwantyl siada na 1,0
    # i budżetu nie da się dotrzymać — liczba ma to pokazać, a nie ukryć.
    false_alarm_realized: float = float("nan")
    uq: None = None
    u_mov_ref: None = None


def rule_path(network: str) -> Path:
    from tsc_marl.paths import RUNS  # noqa: PLC0415

    return RUNS / "heads" / f"rule_ref__{network}.json"


def load_rule(path: Path, *, network: str) -> RuleReference:
    raw = json.loads(path.read_text())
    if raw.pop("format", None) != RULE_FORMAT:
        raise ValueError(f"{path.name}: format progu reguły inny niż {RULE_FORMAT}")
    ref = RuleReference(**raw)
    if ref.network != network:
        raise SystemExit(f"{path.name} policzony dla '{ref.network}', a sieć to '{network}'")
    return ref


def threshold(scores: np.ndarray, false_alarm: float) -> float:
    """Kwantyl (1 − budżet) udziałów [kroki z pełnym oknem, N] — jak w `obs_head.gate_reference`."""
    return float(np.quantile(scores, 1.0 - false_alarm))


def calibrate(
    network: str, *, episodes: int = 3, expert: str = "advancedmp", false_alarm: float | None = None
) -> Path:
    """Czysty ruch pod ekspertem → udział zamrożonych ruchów per (decyzja, węzeł) → próg."""
    from tsc_marl import libsignal  # noqa: PLC0415

    libsignal.bootstrap()
    from tsc_marl.core.features import StateBuilder  # noqa: PLC0415
    from tsc_marl.core.graph import RoadNetGraph  # noqa: PLC0415
    from tsc_marl.core.loop import LoopConfig, TrafficEnv  # noqa: PLC0415
    from tsc_marl.libsignal import session  # noqa: PLC0415
    from tsc_marl.robustness import controllers as c  # noqa: PLC0415
    from tsc_marl.uncertainty.obs_head import GATE_FALSE_ALARM  # noqa: PLC0415

    budzet = GATE_FALSE_ALARM if false_alarm is None else false_alarm
    patch = {"model": {"graphic": True}}
    with session.standalone_world(expert, network, config_patch=patch) as (world, _a, config):
        loop = LoopConfig.from_registry()
        graph = RoadNetGraph.from_registry()
        order = graph.order_intersections(world)
        env = TrafficEnv(
            world, graph, StateBuilder.from_registry(world, order), loop.action_interval
        )
        feats = env.features
        model = config["model"]
        mp = c.PressureController(
            feats.phase_incidence(),
            feats.intersections,
            feats.n_mov,
            t_min=int(model["t_min"]),
            w1=model.get("w1") if expert == "advancedmp" else None,
        )
        # próg ma być mierzony DOKŁADNIE tą funkcją, którą potem liczy bramka —
        # `threshold=inf`, bo interesuje nas surowy wynik, nie maska
        regula = c.RuleGate(
            window=RULE_WINDOW, threshold=np.inf, n_mov=feats.n_mov, real=feats.movement_mask()
        )
        wyniki = []
        for ep in range(episodes):
            obs = env.reset(ep)
            regula.reset()
            for t in range(loop.decisions):
                regula.mask(obs)
                if t >= RULE_WINDOW - 1:  # tylko pełne okno — rozgrzewka to same zera
                    wyniki.append(regula.last_score.copy())
                obs, _, _ = env.step(mp.act(obs))
    scores = np.stack(wyniki)
    ref = RuleReference(
        network=network,
        window=RULE_WINDOW,
        rule=(prog := threshold(scores, budzet)),
        false_alarm=float(budzet),
        false_alarm_realized=float((scores >= prog).mean()),
        expert=expert,
        episodes=episodes,
        decisions=int(scores.shape[0]),
    )
    path = rule_path(network)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"format": RULE_FORMAT, **{k: v for k, v in asdict(ref).items() if v is not None}}
    path.write_text(json.dumps(blob, indent=1))
    return path
