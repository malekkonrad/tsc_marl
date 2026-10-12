"""Controllers evaluated under sensor failure; all read the same detector layer.

Why not the LibSignal agents: `efficientmp` and `advancedmp` read the engine
directly, while the failure layer lives only in `StateBuilder`, so under failure
they would see the truth and the policy the corrupted data. Here max-pressure
decides from the OBSERVATION built by `StateBuilder`, through the phase-movement
incidence. On clean input it reproduces the agent's decisions on every decidable
sample (`uncertainty/dataset.expert_check`); differences remain only in ties.

Contract: `act(obs)` takes [N, ob_length] in GRAPH order and returns actions [N]
in the same order; `TrafficEnv.step` permutes them for the engine, once for all
controllers. Wrappers are controllers too, so they compose.

`state_independent` = the controller does not read observations (FixedTime); the
evaluator then runs it on the clean scenario only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch


class Controller(Protocol):
    name: str
    state_independent: bool

    def act(self, obs: np.ndarray) -> np.ndarray: ...


class Gate(Protocol):
    """Fallback gate: [N] bool, True = this node hands control to the rule.

    The three implementations (oracle, rule, UQ) differ only in how the mask is
    computed, so hysteresis, diagnostics and the no-op test live in one wrapper,
    `GatedController`. `obs` may be unused (the oracle reads the failure layer).

    `last_score` [N] float is the RAW score of the last `mask()`, before the
    threshold and hysteresis: the only quantity that answers WHY a node switched.
    Zeros before the first call and during warm-up.
    """

    last_score: np.ndarray

    def mask(self, obs: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


class PressureController:
    """Efficient-MP (`w1=None`) or Advanced-MP (Alg 1) computed from the observation.

    p(s) = sum over the phase's movements of the EP block (first `n_mov` columns);
    d(s) = the same over the demand block (next `n_mov`, `ats` state only).
    The minimum-green latch (`current_phase_time < t_min`) is controller state,
    not a detector reading, so sensor failure does not affect it.
    """

    state_independent = False

    def __init__(
        self,
        incidence: np.ndarray,
        intersections: Sequence[Any],
        n_mov: int,
        *,
        t_min: int,
        w1: float | None = None,
        name: str | None = None,
    ) -> None:
        if incidence.shape[0] != len(intersections) or incidence.shape[2] != n_mov:
            raise ValueError(f"incidence {incidence.shape} does not match N={len(intersections)}")
        self.incidence = incidence.astype(np.float64)
        self.intersections = list(intersections)
        self.n_mov = n_mov
        self.t_min = t_min
        self.w1 = w1
        self.name = name or ("efficientmp" if w1 is None else "advancedmp")

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        if self.w1 is not None and x.shape[1] < 2 * self.n_mov:
            raise ValueError("Advanced-MP needs the demand block; the state must be `ats`")
        pressure = np.einsum("npm,nm->np", self.incidence, x[:, : self.n_mov])
        choice = pressure.argmax(-1)
        cur = np.array([inter.current_phase for inter in self.intersections], dtype=np.int64)
        if self.w1 is not None:
            demand = np.einsum("npm,nm->np", self.incidence, x[:, self.n_mov : 2 * self.n_mov])
            keep = demand[np.arange(len(cur)), cur] * self.w1 > pressure.max(-1)
            choice = np.where(keep, cur, choice)
        young = np.array(
            [inter.current_phase_time < self.t_min for inter in self.intersections], dtype=bool
        )
        return np.where(young, cur, choice).astype(np.int64)


class FixedTimeController:
    """As `agent/fixedtime.py`: hold a phase `t_fixed` seconds, then the next one.
    The only controller independent of readings (the fallback rule).

    `greens` (optional): {intersection id: {phase signal state: green [s]}}, a
    Webster programme (Krakow, `data/raw_data/krakow_3w/build.py webster`).
    Matched by the phase STATE, not the index, since the SUMO adapter discovers
    phases by simulation. With `greens` the limit is yellow + green, because the
    phase timer starts with the transition; without it every phase holds `t_fixed`.
    """

    state_independent = True

    def __init__(
        self,
        intersections: Sequence[Any],
        *,
        t_fixed: int,
        greens: dict[str, dict[str, float]] | None = None,
        yellow: int = 0,
        name: str = "fixedtime",
    ):
        self.intersections = list(intersections)
        self.t_fixed = t_fixed
        self.name = name
        self.limits: list[list[float]] | None = None
        if greens is not None:
            missing = [i.id for i in self.intersections if i.id not in greens]
            if missing:
                raise ValueError(f"fixed-time programme without intersections: {missing[:5]}")
            self.limits = []
            for inter in self.intersections:
                programme = greens[inter.id]
                states = [ph.state for ph in inter.green_phases]
                unmatched = [st for st in states if st not in programme]
                if unmatched or len(states) != len(programme):
                    raise ValueError(
                        f"{inter.id}: programme has {len(programme)} green phases, the "
                        f"intersection {len(states)} (unmatched: {unmatched[:2]}); the phase "
                        "structure no longer matches the network"
                    )
                self.limits.append([yellow + programme[st] for st in states])

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.array(
            [
                inter.current_phase
                if inter.current_phase_time < self._limit(i, inter)
                else (inter.current_phase + 1) % len(inter.phases)
                for i, inter in enumerate(self.intersections)
            ],
            dtype=np.int64,
        )

    def _limit(self, i: int, inter: Any) -> float:
        if self.limits is None:
            return self.t_fixed
        return self.limits[i][inter.current_phase]


class PolicyController:
    """The core's actor: argmax over logits, actor only (as in `CoreTrainer.evaluate`)."""

    state_independent = False

    def __init__(self, actor: torch.nn.Module, edge_index: torch.Tensor, name: str = "gatppo"):
        self.actor = actor.eval()
        self.edge_index = edge_index
        self.name = name

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        logits = self.actor(torch.from_numpy(np.asarray(obs, dtype=np.float32)), self.edge_index)
        return logits.argmax(-1).numpy().astype(np.int64)


class UWPolicyController:
    """The core's actor with attention modulated by edge reliability (UW-attention).

    The SAME weights as `PolicyController`: the modulation has no parameters
    (log_rho comes from outside), so "core" and "core + UW" are computed from one
    checkpoint and differ only in the mechanism. log_rho is derived from the same
    corrupted observation the policy gets; the source has no access to the truth.
    `reset_stats` clears the source's window at the start of every episode.
    """

    state_independent = False

    def __init__(
        self,
        actor: torch.nn.Module,
        edge_index: torch.Tensor,
        uncertainty: Any,
        name: str = "gatppo_uw",
    ) -> None:
        self.actor = actor.eval()
        self.edge_index = edge_index
        self.uncertainty = uncertainty
        self.name = name
        self.reset_stats()

    def reset_stats(self) -> None:
        self.uncertainty.reset()
        self._spread: list[float] = []
        self._active: list[float] = []

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        from tsc_marl.uncertainty.obs_head import group_spread

        x = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        log_rho = self.uncertainty.log_rho(obs)
        self._spread.append(group_spread(log_rho, self.edge_index, x.shape[0]))
        self._active.append(float((log_rho < 0).float().mean()))
        return self.actor(x, self.edge_index, log_rho=log_rho).argmax(-1).numpy().astype(np.int64)

    def episode_stats(self) -> dict[str, float]:
        """Did the mechanism act at all; otherwise a null result cannot be told
        apart from a mechanism that stayed silent. `uw_spread` is the spread of
        log_rho within a softmax group, the only part attention can see."""
        if not self._spread:
            return {}
        return {
            "uw_spread": float(np.mean(self._spread)),
            "uw_spread_p90": float(np.quantile(self._spread, 0.9)),
            "uw_active": float(np.mean(self._active)),
        }


class GatedController:
    """Policy and rule under one gate: each node's action comes from one of them.

    Wraps the PLAIN policy by default, so the column differs from the core's by
    exactly one thing; wrapping a UW policy is allowed only as a composition with
    an explicit `baseline:` in the plan. Per node, not global: when a part of the
    network fails, a global gate would have to choose between breaking the
    healthy nodes and leaving the blind ones.

    BOTH controllers are called every decision for every node and the mask only
    selects; a controller with memory (a reading window) would otherwise get holes.
    """

    state_independent = False

    def __init__(
        self,
        policy: Controller,
        rule: Controller,
        gate: Gate,
        *,
        k_on: int = 1,
        k_off: int = 1,
        oracle: Gate | None = None,
        name: str = "gated",
        trace: bool = False,
    ) -> None:
        self.policy = policy
        self.rule = rule
        self.gate = gate
        # Per (decision, node) trace, off by default. Enabled by `tsc degrade run
        # --trace`, not by a plan key: plan keys enter the run fingerprint
        # (`MECHANISM_KEYS`), and logging changes no decision.
        self.trace = trace
        # Hysteresis: switch on after `k_on` consecutive decisions above the
        # threshold, off after `k_off` below. Off by default (1, 1); a mechanism
        # parameter that belongs in the plan and so in the fingerprint.
        self.k_on, self.k_off = k_on, k_off
        # DIAGNOSTICS ONLY: the oracle reads the truth and never touches decisions.
        # It gives the real gate's precision and recall, separating "the detector
        # is weak" from "switching helps little", which ATT alone mixes up.
        self.oracle = oracle
        self.name = name
        self.reset_stats()

    def reset_stats(self) -> None:
        """Also resets the gate (and the oracle and a policy with memory): their
        windows must not carry readings from before `world.reset()`."""
        self.gate.reset()
        if self.oracle is not None:
            self.oracle.reset()
        if hasattr(self.policy, "reset_stats"):
            self.policy.reset_stats()
        self._share: list[float] = []
        self._switches = 0
        self._previous: np.ndarray | None = None
        self._streak: np.ndarray | None = None  # consecutive decisions on the same side
        self._hits = np.zeros(3)  # [both, gate only, oracle only]
        self._trace: list[tuple[int, int, float, bool, bool]] = []

    def act(self, obs: np.ndarray) -> np.ndarray:
        raw = self.gate.mask(obs)
        mask = self._hysteresis(raw)
        if self.oracle is not None:
            truth = self.oracle.mask(obs)
            self._hits += [
                float((mask & truth).sum()),
                float((mask & ~truth).sum()),
                float((~mask & truth).sum()),
            ]
        self._share.append(float(np.mean(mask)))
        if self._previous is not None:
            self._switches += int((mask != self._previous).sum())
        if self.trace:
            # The raw score and both masks: without the raw score one cannot see a
            # node just below the threshold, without both masks "the detector did
            # not fire" looks like "hysteresis has not let it through yet".
            score = getattr(self.gate, "last_score", None)
            decision = len(self._share) - 1
            for node in range(len(mask)):
                self._trace.append(
                    (
                        decision,
                        node,
                        float(score[node]) if score is not None else float("nan"),
                        bool(raw[node]),
                        bool(mask[node]),
                    )
                )
        self._previous = mask
        return np.where(mask, self.rule.act(obs), self.policy.act(obs))

    def _hysteresis(self, raw: np.ndarray) -> np.ndarray:
        """Raw mask -> mask after hysteresis, per node."""
        if self._previous is None:  # first decision: nothing to smooth
            self._streak = np.zeros(len(raw), dtype=int)
            return raw
        differs = raw != self._previous
        self._streak = np.where(differs, self._streak + 1, 0)
        needed = np.where(self._previous, self.k_off, self.k_on)
        switch = differs & (self._streak >= needed)
        self._streak = np.where(switch, 0, self._streak)
        return np.where(switch, raw, self._previous)

    def episode_trace(self) -> list[tuple[int, int, float, bool, bool]]:
        """[(decision, node, raw score, raw mask, rule mode)]; empty without `trace`,
        cleared at the start of each episode."""
        return self._trace

    def episode_stats(self) -> dict[str, float]:
        """Did the gate fire at all. `gate_switches` counts mode changes, which
        (rather than the share itself) disturb traffic by resetting phases."""
        if not self._share:
            return {}
        out = {
            "gate_share": float(np.mean(self._share)),
            "gate_switches": float(self._switches),
        }
        # Precision and recall only with a non-zero denominator: without failures
        # "0.00" would read as bad detection, while it means nothing to detect.
        both, false_alarm, missed = self._hits
        if both + false_alarm > 0:
            out["gate_precision"] = float(both / (both + false_alarm))
        if both + missed > 0:
            out["gate_recall"] = float(both / (both + missed))
        # in a composition with UW, the policy's `uw_spread` next to the gate stats
        if hasattr(self.policy, "episode_stats"):
            out |= self.policy.episode_stats()
        return out


class MaskedController:
    """Action masking: a hard limit on SERVICE AGE. No existing phase may wait more
    than `k_max` decisions; when one does, the oldest overdue phase overrides the
    policy's choice.

    No demand signal, on purpose: per-movement demand is not readable from the
    observation without an arbitrary threshold (EP can be negative), and reading
    the engine would be an oracle. The mask reads NO sensor, only its own actions,
    so it is robust to sensor failure by construction.

    Guarantee, stated exactly: with one overdue phase per node the age stays
    <= `k_max + 1` decisions; with several overdue at once only one can be served,
    so the worst case is `k_max + n_legal_phases - 1`. It is not a hard `k_max`.
    It does not handle the dilemma zone or minimum green (the engine's `t_min`).
    """

    def __init__(
        self,
        policy: Controller,
        *,
        legal: np.ndarray,
        k_max: int,
        name: str = "masked",
    ) -> None:
        if k_max < 1:
            raise ValueError(f"k_max must be >= 1, got {k_max}")
        self.policy = policy
        self.legal = np.asarray(legal, dtype=bool)  # [N, n_phases]
        self.k_max = int(k_max)
        self.name = name
        self.reset_stats()

    def reset_stats(self) -> None:
        """Zero the ages at episode start, or a new episode would force phases
        from the previous one's backlog."""
        self._since = np.zeros(self.legal.shape, dtype=np.int64)
        self._overridden = 0
        self._decisions = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        actions = np.asarray(self.policy.act(obs), dtype=np.int64).copy()
        # overdue = legal AND waiting at least `k_max` decisions
        overdue = self.legal & (self._since >= self.k_max)
        force = overdue.any(-1)
        if force.any():
            # the OLDEST overdue phase, not the first by index: round-robin when
            # several are overdue, where index order would starve the last one
            age = np.where(overdue, self._since, -1)
            choice = age.argmax(-1)
            self._overridden += int((force & (actions != choice)).sum())
            actions = np.where(force, choice, actions)
        self._since += 1
        self._since[np.arange(len(actions)), actions] = 0
        self._decisions += 1
        return actions

    def episode_stats(self) -> dict[str, float]:
        """`mask_share` = share of (decision, node) pairs where the mask CHANGED the
        policy's action; a mask that never overrode anything is a no-op."""
        if not self._decisions:
            return {}
        return {
            "mask_share": float(self._overridden / (self._decisions * len(self.legal))),
            "mask_forced": float(self._overridden),
        }


class OracleGate:
    """Oracle gate: reads the truth from the failure layer, not the observation.

    Not a deployable controller but an UPPER BOUND for the real gates: if
    switching does not help with perfect detection, no detector will.

    A node is blind when the fraction of its IN lanes in `sensors.broken` is >=
    `threshold` (in lanes, because a lane's detector belongs to the intersection
    the lane leads into). Known limit: a node with healthy own sensors still gets
    pressure partly from its downstream neighbours' detectors.
    """

    def __init__(self, env: Any, *, threshold: float = 0.5) -> None:
        # `env`, not `env.features`: the evaluator replaces `env.features` for every
        # scenario while the controller is built once; a stored snapshot would stay
        # on the first (clean) scenario and never fire.
        self.env = env
        self.threshold = threshold
        self._in_lanes: list[list[str]] | None = None
        self._built_for: Any = None

    def reset(self) -> None:
        """Stateless; present for the common gate interface."""

    def mask(self, obs: np.ndarray) -> np.ndarray:
        features = self.env.features
        sensors = features.sensors
        # no failure layer on a clean scenario; `live` skips decisions before onset
        if sensors is None or not sensors.live:
            self.last_score = np.zeros(features.n_nodes, dtype=np.float64)
            return np.zeros(features.n_nodes, dtype=bool)
        if features is not self._built_for:  # the cache survives an episode,
            self._in_lanes = [list(s.in_lanes) for s in features.structs]
            self._built_for = features  # but not a scenario switch

        broken = sensors.broken
        # max(..., 1): on Krakow some signals have no in lanes; such a node cannot be blind
        part = [
            sum(lane in broken for lane in lanes) / max(len(lanes), 1) for lanes in self._in_lanes
        ]
        self.last_score = np.asarray(part, dtype=np.float64)
        return self.last_score >= self.threshold


class OracleUncertainty:
    """Modulation oracle: log_rho for UW-attention from the failure layer's truth.

    The same role as `OracleGate`: an upper bound on whether weighting neighbours
    by reliability can help with a perfect signal. u of a node = the fraction of
    its broken in lanes (`OracleGate.last_score`, one definition of a blind node),
    applied on the SOURCE side: edge j->i gets u(j), self-loop i->i gets u(i).

        log_rho = -u / tau,  u in [0, 1]

    No `u_ref`: the truth is exactly 0 on a healthy network, so the no-op on clean
    traffic is an identity, not a calibration. Same interface as
    `OnlineUncertainty` (`reset`, `log_rho`).
    """

    def __init__(
        self, env: Any, edge_index: torch.Tensor, *, tau: float, channels: str = "all"
    ) -> None:
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        _check_channels(channels)
        self._gate = OracleGate(env)  # `env`, see `OracleGate.__init__`
        self.src = edge_index[0].numpy()
        self.tau = float(tau)
        self.channels = channels

    def reset(self) -> None:
        self._gate.reset()

    def log_rho(self, obs: np.ndarray) -> torch.Tensor:
        self._gate.mask(obs)
        # [N], zeros on clean traffic and before onset
        return source_log_rho(self._gate.last_score, self.src, self.tau, self.channels)


# Source-side modulation channels (`UW_CHANNELS` in `robustness.training`):
# `self` = self-loops only (a blind node stops trusting its OWN observation),
# `edges` = edges between nodes only (neighbours stop trusting its message).
# The two log_rho parts are disjoint and add up to `all`.


def _check_channels(channels: str) -> None:
    from tsc_marl.robustness.training import UW_CHANNELS

    if channels not in UW_CHANNELS:
        raise ValueError(f"channels={channels!r}, allowed: {UW_CHANNELS}")


def source_log_rho(
    u: np.ndarray, src: np.ndarray, tau: float, channels: str = "all"
) -> torch.Tensor:
    """Node u [N] -> log_rho = -u/tau on the SOURCE side: edge j->i gets u(j),
    self-loop i->i gets u(i). Shared by the oracle and the rule, so their columns
    differ only in the definition of u.

    `channels` zeroes one part (zero = "no modulation", exactly what the core
    sees). Output [E + N]: edges in `edge_index` order, then self-loops, the
    contract of `UncertaintyWeightedAttention`."""
    edges = u[src] if channels != "self" else np.zeros(len(src))
    loops = u if channels != "edges" else np.zeros(len(u))
    out = -np.concatenate([edges, loops]) / tau
    return torch.from_numpy(out.astype(np.float32))


class RuleUncertainty:
    """Modulation from the rule: log_rho for UW-attention from the frozen-detector count.

    The deployable counterpart of `OracleUncertainty`, using the signal that
    already works as a gate (`RuleGate`): same detector, same window, same
    threshold. u of a node = its fraction of frozen movements, but only ABOVE the
    gate threshold, zero below: otherwise the modulation would act on clean
    traffic, where an empty low-traffic approach reports zero and looks frozen.
    Above the threshold u stays graded, as in the oracle.

    Differences from the oracle are deployment costs: the window warm-up and
    blindness to a dead detector that happens to report changing values.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        *,
        window: int,
        threshold: float,
        n_mov: int,
        tau: float,
        channels: str = "all",
        real: np.ndarray | None = None,
    ) -> None:
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        _check_channels(channels)
        self._gate = RuleGate(window=window, threshold=threshold, n_mov=n_mov, real=real)
        self.src = edge_index[0].numpy()
        self.tau = float(tau)
        self.channels = channels

    def reset(self) -> None:
        self._gate.reset()

    def log_rho(self, obs: np.ndarray) -> torch.Tensor:
        mask = self._gate.mask(obs)  # all False during warm-up
        u = np.where(mask, self._gate.last_score, 0.0)
        return source_log_rho(u, self.src, self.tau, self.channels)


class RuleGate:
    """Rule gate: the fraction of a node's movements whose readings STAY STILL for
    the whole window. No learning, no artefacts: a frozen-detector counter.

    One criterion catches both forms of a dead detector, `zero` (zero does not
    change either) and `hold` (a frozen last reading). Built-in weakness: an
    empty low-traffic approach reports zero and is HEALTHY (see
    `obs_head.suspicious_movements`). The lower reference for the UQ gate.
    """

    def __init__(
        self, *, window: int, threshold: float, n_mov: int, real: np.ndarray | None = None
    ) -> None:
        self.window, self.threshold, self.n_mov = window, threshold, n_mov
        # [N, n_mov] real movements (`StateBuilder.movement_mask`); None = all
        self.real = real
        self._buf: list[np.ndarray] = []

    def reset(self) -> None:
        self._buf.clear()

    def mask(self, obs: np.ndarray) -> np.ndarray:
        from tsc_marl.uncertainty.obs_head import suspicious_movements

        obs = np.asarray(obs, dtype=np.float32)
        self._buf.insert(0, obs)  # newest first, as in `window_index`
        del self._buf[self.window :]
        # Warm-up guard: with a short buffer EVERY movement looks frozen and the
        # gate would switch the whole network at episode start (as `full_window`).
        if len(self._buf) < self.window:
            self.last_score = np.zeros(obs.shape[0], dtype=np.float64)
            return np.zeros(obs.shape[0], dtype=bool)

        hist = np.stack(self._buf)  # [k, N, F]
        win = np.arange(self.window)[None, :]  # window of the NEWEST step
        suspicious = suspicious_movements(hist, win, self.n_mov)[0]  # [N, n_mov]
        self.last_score = frozen_fraction(suspicious, self.real)
        return self.last_score >= self.threshold


def frozen_fraction(suspicious: np.ndarray, real: np.ndarray | None) -> np.ndarray:
    """A node's fraction of frozen movements, over REAL movements only; `real` None
    (homogeneous network) is the plain mean. Shared by gate, modulation and
    threshold calibration."""
    if real is None:
        return suspicious.mean(-1)
    return (suspicious & real).sum(-1) / np.maximum(real.sum(-1), 1)


class UQGate:
    """UQ gate: the fraction of a node's movements whose u exceeds its own norm from
    a clean window (`GateReference.u_mov_ref`).

    The same signal shape as `RuleGate` (a fraction of suspicious movements), so
    the ablation compares the suspicion criterion, not two statistics. The rule
    asks "is this detector stuck", UQ asks "is this reading unusually uncertain".
    Expected weakness, written before measurement: sigma also grows under NOISE,
    where detectors are alive; uncertainty is not a measure of a MISSING reading.
    """

    def __init__(self, uncertainty: Any, *, u_mov_ref: np.ndarray, threshold: float) -> None:
        self.uncertainty = uncertainty
        self.u_mov_ref = u_mov_ref
        self.threshold = threshold

    def reset(self) -> None:
        self.uncertainty.reset()

    def mask(self, obs: np.ndarray) -> np.ndarray:
        u_mov = self.uncertainty.movements(obs)  # [N, n_mov]
        if not self.uncertainty.ready:  # warm-up guard, as in the rule
            self.last_score = np.zeros(u_mov.shape[0], dtype=np.float64)
            return np.zeros(u_mov.shape[0], dtype=bool)
        self.last_score = (u_mov >= self.u_mov_ref).mean(-1)
        return self.last_score >= self.threshold


class ColightController:
    """Advanced-CoLight from a checkpoint: the LibSignal agent's `get_action(test=True)`.

    The agent expects rows in `world.intersections` order and gets graph order.
    On CityFlow roadnets the two are identical, and that is a CONDITION of this
    controller, checked at construction (`identity_order`), not assumed."""

    state_independent = False

    def __init__(self, agent: Any, *, identity_order: bool, name: str = "advancedcolight"):
        if not identity_order:
            raise ValueError(
                "world.intersections order != graph order; ColightController would feed "
                "observations to the wrong intersections"
            )
        self.agent = agent
        self.name = name

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.agent.get_action(obs, None, test=True), dtype=np.int64)


class EnsembleController:
    """Several core actors (different seeds): the FIRST decides, the others vote.

    Disagreement of their phase distributions is epistemic uncertainty of the RL
    policy itself, without extra training: MI = H(mean distribution) - mean H.
    Next to it, the share of members that would pick another phase. One member
    decides, not the majority, so the trajectory is that of the deployed policy.
    """

    state_independent = False

    def __init__(self, actors, edge_index: torch.Tensor, name: str = "gatppo_ens") -> None:
        if len(actors) < 2:
            raise ValueError("an ensemble needs at least two actors")
        self.actors = [a.eval() for a in actors]
        self.edge_index = edge_index
        self.name = name
        self.reset_stats()

    def reset_stats(self) -> None:
        self._mi: list[np.ndarray] = []
        self._disagree: list[np.ndarray] = []
        self._aleatoric: list[np.ndarray] = []

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        probs = torch.softmax(torch.stack([a(x, self.edge_index) for a in self.actors]), -1)
        mean = probs.mean(0)  # [N, P]
        total = -(mean * torch.log(mean + 1e-10)).sum(-1)
        aleatoric = -(probs * torch.log(probs + 1e-10)).sum(-1).mean(0)
        votes = probs.argmax(-1)  # [K, N]
        driver = votes[0]
        self._mi.append((total - aleatoric).clamp_min(0).numpy())
        self._aleatoric.append(aleatoric.numpy())
        self._disagree.append((votes[1:] != driver).float().mean(0).numpy())
        return driver.numpy().astype(np.int64)

    def episode_stats(self) -> dict[str, float]:
        if not self._mi:
            return {}
        mi = np.concatenate(self._mi)
        return {
            "ens_mi": float(mi.mean()),
            "ens_mi_p90": float(np.quantile(mi, 0.9)),
            "ens_aleatoric": float(np.concatenate(self._aleatoric).mean()),
            "ens_disagree": float(np.concatenate(self._disagree).mean()),
        }
