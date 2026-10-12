"""Rule-gate threshold without an uncertainty head, for networks that have no head (Krakow).

On JiNan/HangZhou the rule threshold is computed together with the observation
head (`tsc obshead --save`, `runs/heads/gate_ref__<net>.npz`), so that the rule
and the UQ gate come from one criterion. That pipeline reads CityFlow roadnets
and trains a head the rule does not need, so on SUMO the rule threshold alone is
computed with the same recipe:

    threshold = (1 - GATE_FALSE_ALARM) quantile of a node's frozen-movement
                fraction, on clean traffic under the Advanced-MP expert,
                full-window steps only.

Same expert, false-alarm budget (0.5%) and window (12) as `gate_ref`; fractions
over REAL movements (`StateBuilder.movement_mask`), otherwise Krakow's padding
would look like dead detectors. The rule's signal depends on the policy, so this
calibrates "false alarms under reasonable control", not under any control.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

#: Rule window, as in JiNan/HangZhou `gate_ref` (`tsc obshead --save --window 12`).
RULE_WINDOW = 12
RULE_FORMAT = 1


@dataclass(frozen=True)
class RuleReference:
    """Rule threshold alone. `window` and `rule` as in `obs_head.GateReference`, so
    both artefacts are read the same way; `uq` and `u_mov_ref` are None, so a UQ
    gate on such a network fails loudly."""

    network: str
    window: int
    rule: float
    false_alarm: float
    expert: str
    episodes: int
    decisions: int
    # Share of clean (decision, node) pairs on which a gate with THIS threshold would
    # fire (`>=`, as in `RuleGate`). The budget is the target, this is the
    # measurement: on sparse traffic the quantile can sit at 1.0 and miss the budget.
    false_alarm_realized: float = float("nan")
    uq: None = None
    u_mov_ref: None = None


def rule_path(network: str) -> Path:
    from tsc_marl.paths import RUNS

    return RUNS / "heads" / f"rule_ref__{network}.json"


def load_rule(path: Path, *, network: str) -> RuleReference:
    raw = json.loads(path.read_text())
    if raw.pop("format", None) != RULE_FORMAT:
        raise ValueError(f"{path.name}: rule threshold format is not {RULE_FORMAT}")
    ref = RuleReference(**raw)
    if ref.network != network:
        raise SystemExit(
            f"{path.name} was computed for '{ref.network}', the network is '{network}'"
        )
    return ref


def threshold(scores: np.ndarray, false_alarm: float) -> float:
    """(1 - budget) quantile of fractions [full-window steps, N], as in `obs_head`."""
    return float(np.quantile(scores, 1.0 - false_alarm))


def calibrate(
    network: str, *, episodes: int = 3, expert: str = "advancedmp", false_alarm: float | None = None
) -> Path:
    """Clean traffic under the expert -> frozen fraction per (decision, node) -> threshold."""
    from tsc_marl import libsignal

    libsignal.bootstrap()
    from tsc_marl.core.features import StateBuilder
    from tsc_marl.core.graph import RoadNetGraph
    from tsc_marl.core.loop import LoopConfig, TrafficEnv
    from tsc_marl.libsignal import session
    from tsc_marl.robustness import controllers as c
    from tsc_marl.uncertainty.obs_head import GATE_FALSE_ALARM

    budget = GATE_FALSE_ALARM if false_alarm is None else false_alarm
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
        # measured with EXACTLY the gate's function; `threshold=inf` gives the raw score
        rule = c.RuleGate(
            window=RULE_WINDOW, threshold=np.inf, n_mov=feats.n_mov, real=feats.movement_mask()
        )
        scores_list = []
        for ep in range(episodes):
            obs = env.reset(ep)
            rule.reset()
            for t in range(loop.decisions):
                rule.mask(obs)
                if t >= RULE_WINDOW - 1:  # full window only; warm-up is all zeros
                    scores_list.append(rule.last_score.copy())
                obs, _, _ = env.step(mp.act(obs))
    scores = np.stack(scores_list)
    ref = RuleReference(
        network=network,
        window=RULE_WINDOW,
        rule=(cut := threshold(scores, budget)),
        false_alarm=float(budget),
        false_alarm_realized=float((scores >= cut).mean()),
        expert=expert,
        episodes=episodes,
        decisions=int(scores.shape[0]),
    )
    path = rule_path(network)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"format": RULE_FORMAT, **{k: v for k, v in asdict(ref).items() if v is not None}}
    path.write_text(json.dumps(blob, indent=1))
    return path
