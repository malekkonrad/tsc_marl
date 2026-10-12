"""Environment loop of the core: couples `ppo.py` with the simulator.

The protocol follows LibSignal's `trainer/tsc_trainer.py`, so the numbers stay
comparable with CoLight and Advanced-CoLight:

1. One decision per `action_interval` (15 s), then 15 engine steps of 1 s with
   the same action; a 3600 s episode is 240 decisions.
2. The reward of a decision is the mean over those steps, not the sum.
3. Queue and delay are read once per decision and averaged with LibSignal's
   divisors: queue and delay by decisions x intersections, reward by decisions.
4. Actions are permuted from graph order to the world's order (`world_order`).

No epsilon-greedy, warm-up or target network: PPO learns from every collected
segment and explores through the policy's entropy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

from tsc_marl.core.features import StateBuilder
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.core.nets import ActorCritic
from tsc_marl.core.ppo import PPO, PPOConfig, RolloutBuffer
from tsc_marl.domain.sensors import FailureMix, FailureSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from world.world_cityflow import World

    from runlog import Run

# Reporting protocol of Advanced-CoLight: a run's result is the MEAN of the last
# 10 test episodes, not a single test after training.
REPORT_WINDOW = 10


class UncertaintySignal(Protocol):
    """What the loop needs from an uncertainty source for UW-attention
    (implemented by `robustness.training.UWStream`)."""

    def reset(self) -> None: ...

    def log_rho(self, obs: np.ndarray) -> torch.Tensor: ...

    def episode_stats(self) -> dict[str, float]: ...


@dataclass(frozen=True)
class LoopConfig:
    """Run length parameters; defaults are the thesis protocol."""

    steps: int = 3600
    test_steps: int = 3600
    action_interval: int = 15
    episodes: int = 200
    test_when_train: bool = True

    @property
    def decisions(self) -> int:
        return self.steps // self.action_interval

    @property
    def test_decisions(self) -> int:
        return self.test_steps // self.action_interval

    @classmethod
    def from_registry(cls, **overrides: Any) -> LoopConfig:
        from common.registry import Registry

        param = Registry.mapping["trainer_mapping"]["setting"].param
        keys = ("steps", "test_steps", "action_interval", "episodes", "test_when_train")
        base = {k: param[k] for k in keys if k in param}
        return cls(**{**base, **overrides})


class TrafficEnv:
    """LibSignal's `World` seen at the level of DECISIONS, not simulator steps.

    Not a `TSCEnv`: that one asks LibSignal agents for observations, and the core
    is not one. It does what `TSCEnv.step` does with the world, with observations
    from `StateBuilder`.
    """

    def __init__(
        self,
        world: World,
        graph: RoadNetGraph,
        features: StateBuilder,
        action_interval: int,
        *,
        make_features: Callable[[FailureSpec], StateBuilder] | None = None,
        failures: FailureMix | None = None,
    ) -> None:
        self.world = world
        self.graph = graph
        self.features = features
        self.action_interval = action_interval

        # Failures in training. A failure scenario is part of the `StateBuilder`,
        # so switching scenarios between episodes means another builder, made by
        # the same factory `robustness/degrade.py` uses. Builders are cached per spec.
        self._make_features = make_features
        self.failures = failures
        self._builders: dict[FailureSpec, StateBuilder] = {}
        if failures is not None and make_features is None:
            raise ValueError("a failure mix needs `make_features` to rebuild the state")

        self.episode = -1
        self._to_world = world_order(graph, world)
        self._delay = _delay_generators(world, graph.order_intersections(world))
        self._reset_stats()

    # ---------------------------------------------------------------- protocol

    def _reset_stats(self) -> None:
        n = self.graph.n_nodes
        self._decisions = 0
        self._reward_sum = np.zeros(n, dtype=np.float64)
        self._queue_sum = np.zeros(n, dtype=np.float64)
        self._delay_sum = np.zeros(n, dtype=np.float64)
        # Service age: decisions since each phase last got green, [node, phase] in
        # graph order. A starvation metric that network-wide averages hide. Only
        # phases that exist count (`n_phases` is the network maximum).
        self._since_green = np.zeros((n, self.features.n_phases), dtype=np.float64)
        self._legal_phase = np.zeros((n, self.features.n_phases), dtype=bool)
        for idx, inter in enumerate(self.features.intersections):
            self._legal_phase[idx, : len(inter.phases)] = True
        self._age_max = 0.0
        # Service age with demand: only phases with someone waiting, since a phase
        # without traffic may legitimately wait as long as it likes.
        self._phase_mov = self.features.phase_incidence()  # [N, n_phases, n_mov]
        self._age_demand_max = 0.0

    def _builder(self, spec: FailureSpec) -> StateBuilder:
        if spec not in self._builders:
            self._builders[spec] = self._make_features(spec)  # type: ignore[misc]
        return self._builders[spec]

    def reset(
        self, episode: int | None = None, *, failure: FailureSpec | None = None
    ) -> np.ndarray:
        """New episode. -> observation [N, ob_length].

        `episode` seeds the failure draw (`features.reset_sensors`), so passing it
        makes an evaluation reproducible regardless of what ran before.
        `failure` forces a scenario instead of sampling the mix; this keeps test
        episodes clean (robustness is measured by `tsc degrade`).
        """
        self.episode = self.episode + 1 if episode is None else episode
        if self._make_features is not None:
            spec = failure
            if spec is None:
                spec = self.failures.sample(self.episode) if self.failures else FailureSpec()
            self.features = self._builder(spec)
        elif failure is not None and failure.active:
            raise ValueError("an env without `make_features` cannot switch failures on")
        self.world.reset()
        self.features.reset_sensors(self.episode)
        self._reset_stats()
        return self.features.observe()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """One decision: `action_interval` engine steps with the same action.

        -> (observation, reward averaged over the interval, done). `done` is always
        False: the episode is cut, not terminated (`ppo.py`, pitfall 1).
        """
        if actions.shape != (self.graph.n_nodes,):
            raise ValueError(
                f"actions must be [{self.graph.n_nodes}], got {tuple(np.shape(actions))}"
            )
        in_world_order = np.asarray(actions)[self._to_world]

        reward_acc = np.zeros(self.graph.n_nodes, dtype=np.float64)
        for _ in range(self.action_interval):
            self.world.step(in_world_order)
            reward_acc += self.features.rewards()
        rewards = reward_acc / self.action_interval

        queue, demand = self._queue_and_demand()
        self._reward_sum += rewards
        self._queue_sum += queue
        self._delay_sum += np.array([g.generate() for g in self._delay], dtype=np.float64).ravel()
        self._decisions += 1
        # The chosen phase got green, the rest waited one decision longer.
        # `actions` is in graph order (`_to_world` is for the engine only).
        self._since_green += 1.0
        self._since_green[np.arange(self.graph.n_nodes), np.asarray(actions, dtype=int)] = 0.0
        self._age_max = max(self._age_max, float(self._since_green[self._legal_phase].max()))
        # A phase "with demand" serves at least one movement where someone waits now.
        with_demand = self._legal_phase & (self._phase_mov & demand[:, None, :]).any(-1)
        if with_demand.any():
            self._age_demand_max = max(
                self._age_demand_max, float(self._since_green[with_demand].max())
            )
        return self.features.observe(), rewards.astype(np.float32), False

    def _queue(self) -> np.ndarray:
        """Waiting vehicles on the incoming lanes per intersection, as LibSignal's
        `Metrics` computes it; independent of the reward variant."""
        return self._queue_and_demand()[0]

    def _queue_and_demand(self) -> tuple[np.ndarray, np.ndarray]:
        """(queue per node [N], someone waiting on the movement [N, n_mov]).

        Read from the engine, not from the observation: under sensor failure the
        observation lies, and the metrics must measure what really waited.
        """
        waiting = self.world.get_info("lane_waiting_count")
        queue = np.zeros(self.graph.n_nodes, dtype=np.float64)
        demand = np.zeros((self.graph.n_nodes, self.features.n_mov), dtype=bool)
        for idx, s in enumerate(self.features.structs):
            queue[idx] = sum(waiting[lane] for lane in s.reward_lanes)
            for m, (ins, _) in enumerate(s.movements):
                demand[idx, m] = sum(waiting[lane] for lane in ins) > 0
        return queue, demand

    # ---------------------------------------------------------------- metrics

    def metrics(self) -> dict[str, float]:
        """Columns as in LibSignal's `Metrics` (point 3 of the module docstring)
        plus tail metrics."""
        if self._decisions == 0:
            raise RuntimeError("the episode has not started; call reset() and step()")
        n = self.graph.n_nodes
        queue = self._queue_sum / self._decisions  # [N], mean over decisions
        return {
            "att": float(self.world.get_average_travel_time()),
            "throughput": float(self.world.get_cur_throughput()),
            "reward": float(self._reward_sum.sum() / self._decisions),
            "queue": float(queue.sum() / n),
            "delay": float(self._delay_sum.sum() / (self._decisions * n)),
            # Tail: the network mean hides a failure on a part of the network, and
            # starvation-targeting mechanisms (masking) move these, not `att`.
            "queue_max": float(queue.max()),
            "queue_p90": float(np.quantile(queue, 0.9)),
            # Longest service age in the episode, in seconds. Measured between
            # decisions, so yellow is not subtracted.
            "wait_max": float(self._age_max * self.action_interval),
            # The same for phases with someone waiting: the starvation metric to
            # report. An upper bound for a single vehicle, which may have arrived
            # after the phase's last green.
            "wait_max_demand": float(self._age_demand_max * self.action_interval),
        }


def world_order(graph: RoadNetGraph, world: World) -> np.ndarray:
    """Permutation `position in world.intersections -> node_idx`.

    Usage: `world_actions = actions[world_order(...)]`. A separate function so the
    most dangerous silent bug of the loop (one intersection's action executed on
    another) is testable without CityFlow. Ids go through `RoadNetGraph.bare_id`,
    the same rule as `order_intersections`.
    """
    return np.array(
        [graph.node_id2idx[graph.bare_id(inter.id)] for inter in world.intersections],
        dtype=np.int64,
    )


def _delay_generators(world: World, intersections) -> list[Any]:
    from generator.lane_vehicle import LaneVehicleGenerator

    return [
        LaneVehicleGenerator(world, inter, ["lane_delay"], in_only=True, average="all")
        for inter in intersections
    ]


class CoreTrainer:
    """Core training: one collection episode, one PPO update, optionally a test.

    The rollout segment is a whole episode (240 decisions), a choice: with 12-16
    agents that is already 2880-3840 samples, and every update sees the full
    traffic profile without bootstrapping mid-episode.
    """

    def __init__(
        self,
        env: TrafficEnv,
        ppo: PPO,
        config: LoopConfig | None = None,
        run: Run | None = None,
        logger: logging.Logger | None = None,
        uncertainty: UncertaintySignal | None = None,
    ) -> None:
        self.env = env
        self.ppo = ppo
        self.config = config or LoopConfig()
        self.run = run
        # None = the plain core, bit-identical (log_rho=None takes the original path)
        self.uncertainty = uncertainty
        self.logger = logger  # one line per episode; `run.log` writes the CSV
        # Device read from the network; the static `edge_index` is moved once.
        self.device = next(ppo.ac.parameters()).device
        self.edge_index = env.graph.edge_index.to(self.device)
        self.history: list[dict[str, float]] = []
        # Path for the BEST checkpoint (min test ATT), or None. Note: the test
        # episode is the same traffic `tsc degrade` uses as `clean` on the training
        # network, so a fair comparison of this checkpoint needs held-out traffic.
        self.best_path: Path | None = None
        self._best_att = float("inf")
        self._episode_failure = "clean"

    # --------------------------------------------------------------- building

    @classmethod
    def from_world(
        cls,
        world: World,
        *,
        ppo_config: PPOConfig | None = None,
        loop_config: LoopConfig | None = None,
        run: Run | None = None,
        logger: logging.Logger | None = None,
        device: torch.device | str = "cpu",
        failures: FailureMix | None = None,
        uncertainty: UncertaintySignal | None = None,
        **feature_overrides: Any,
    ) -> CoreTrainer:
        """Builds graph, state, networks and PPO from a live `world` and Registry.

        Intersection order goes only through `graph.order_intersections`. `device`
        defaults to CPU; `runner/train_core.py` resolves it from `--ngpu`.
        """
        graph = RoadNetGraph.from_registry()
        order = graph.order_intersections(world)

        def make_features(spec: FailureSpec) -> StateBuilder:
            return StateBuilder.from_registry(world, order, failure=spec, **feature_overrides)

        # Without a mix the scenario comes from the config (`model.failure`, default
        # none). With a mix `TrafficEnv.reset` picks it per episode.
        features = (
            make_features(FailureSpec())
            if failures is not None
            else StateBuilder.from_registry(world, order, **feature_overrides)
        )
        cfg = loop_config or LoopConfig.from_registry()
        env = TrafficEnv(
            world,
            graph,
            features,
            cfg.action_interval,
            make_features=make_features if failures is not None else None,
            failures=failures,
        )
        # Order matters: move the network first, then build `PPO`, whose Adam
        # state would otherwise stay on the CPU. The phase mask is passed only on
        # heterogeneous networks, so JiNan/HangZhou build exactly as without it.
        mask = features.phase_mask()
        extra = {} if mask is None else {"phase_mask": torch.from_numpy(mask)}
        ac = ActorCritic.from_registry(features.ob_length, features.n_phases, **extra).to(device)
        return cls(env, PPO(ac, ppo_config or PPOConfig()), cfg, run, logger, uncertainty)

    # ---------------------------------------------------------------- episodes

    def _log_rho(self, obs: np.ndarray) -> torch.Tensor | None:
        """log_rho [E+N] for this reading, or None without the mechanism.

        Called before `act`, on the same (possibly corrupted) observation the
        policy sees: the head has no access to the failure layer or the truth.
        """
        return None if self.uncertainty is None else self.uncertainty.log_rho(obs)

    def collect(self, episode: int | None = None) -> tuple[RolloutBuffer, dict[str, float]]:
        """One training episode; the policy samples (PPO exploration)."""
        cfg = self.config
        buffer = RolloutBuffer(
            cfg.decisions,
            self.env.graph.n_nodes,
            self.env.features.ob_length,
            self.edge_index,
            device=self.device,
            with_log_rho=self.uncertainty is not None,
        )
        raw = self.env.reset(episode)
        if self.uncertainty is not None:
            self.uncertainty.reset()  # window buffer, see `UncertaintySignal.reset`
        obs, log_rho = self._obs(raw), self._log_rho(raw)
        for _ in range(cfg.decisions):
            actions, log_probs, values = self.ppo.act(obs, self.edge_index, log_rho=log_rho)
            next_raw, rewards, _ = self.env.step(self._actions(actions))
            buffer.add(obs, actions, log_probs, values, self._obs(rewards), log_rho=log_rho)
            obs, log_rho = self._obs(next_raw), self._log_rho(next_raw)
        # The critic never gets log_rho, so the bootstrap value is unchanged.
        buffer.compute_returns(self.ppo.value_of(obs, self.edge_index), self.ppo.config)
        return buffer, self.env.metrics()

    @torch.no_grad()
    def evaluate(self, episode: int | None = None) -> dict[str, float]:
        """Test episode: argmax, actor only, clean traffic.

        The critic is not called, as in deployment. Clean even when training runs
        with failures, so `att_last10` stays comparable across runs; robustness is
        measured by `tsc degrade` on the frozen checkpoint. The mechanism stays on,
        because it is part of the policy, not of the conditions.
        """
        raw = self.env.reset(episode, failure=FailureSpec())
        if self.uncertainty is not None:
            self.uncertainty.reset()
        for _ in range(self.config.test_decisions):
            logits = self.ppo.ac.actor(self._obs(raw), self.edge_index, log_rho=self._log_rho(raw))
            raw, _, _ = self.env.step(self._actions(logits.argmax(-1)))
        return self.env.metrics()

    # ------------------------------------------------------ numpy <-> torch boundary

    def _obs(self, array: np.ndarray) -> torch.Tensor:
        """The simulator speaks numpy (CPU); one of the two device crossings."""
        return torch.from_numpy(array).to(self.device)

    def _actions(self, actions: torch.Tensor) -> np.ndarray:
        """The other crossing; `.cpu()` is required for CUDA and a no-op on CPU."""
        return actions.cpu().numpy()

    # ----------------------------------------------------------------- training

    def train(self, episodes: int | None = None) -> dict[str, float]:
        """Full run. -> report scalars (same names as `experiment.ingest`)."""
        n_episodes = episodes if episodes is not None else self.config.episodes
        test_att: list[float] = []

        for e in range(n_episodes):
            buffer, train_metrics = self.collect(e)
            stats = self.ppo.update(buffer)
            # Read BEFORE `evaluate`, which switches `env.features` to the clean
            # builder and resets the uncertainty window.
            mech = self._mechanism_metrics()
            self._log(e, "train", {**train_metrics, **stats, **mech})

            test_metrics = None
            if self.config.test_when_train:
                test_metrics = self.evaluate(e)
                # mechanism statistics on clean traffic, next to the test ATT
                self._log(e, "test", {**test_metrics, **self._mechanism_metrics(test=True)})
                test_att.append(test_metrics["att"])
                self.history.append(test_metrics)
                if self.best_path is not None and test_metrics["att"] < self._best_att:
                    self._best_att = test_metrics["att"]
                    self.save(self.best_path)
            self._say(e, train_metrics, stats, test_metrics, mech)

        return self._report(test_att)

    def _mechanism_metrics(self, *, test: bool = False) -> dict[str, float]:
        """Episode diagnostics: the active failure and what the mechanism saw.

        Without them a null result cannot be told apart from a run in which the
        modulation never fired. Call before `evaluate` replaces `env.features`.
        """
        out: dict[str, float] = {}
        if self.env.failures is not None and not test:
            sensors = self.env.features.sensors
            self._episode_failure = self._failure_slug()
            out["failure_active"] = float(self.env.features.failure.active)
            out["n_broken"] = float(len(sensors.broken)) if sensors is not None else 0.0
        if self.uncertainty is not None:
            out.update(self.uncertainty.episode_stats())
        return out

    def _failure_slug(self) -> str:
        return self.env.features.failure.slug()

    def _log(self, episode: int, split: str, values: dict[str, float]) -> None:
        if self.run is not None:
            self.run.log(step=episode, split=split, **values)

    def _say(self, episode, train: dict, stats: dict, test: dict | None, mech: dict) -> None:
        """One line per episode; `kl` and `clip` reveal a too large learning rate
        before the ATT curve does."""
        if self.logger is None:
            return
        line = (
            f"ep {episode:3d} | train att {train['att']:7.1f} reward {train['reward']:8.1f} "
            f"| kl {stats['approx_kl']:.4f} clip {stats['clip_frac']:.3f} "
            f"entropy {stats['entropy']:.3f}"
        )
        if "failure_active" in mech:
            line += f" | failure {self._episode_failure} ({int(mech['n_broken'])} lanes)"
        if "uw_spread" in mech:
            line += f" spread {mech['uw_spread']:.2f} active {mech['uw_active']:.1%}"
        if test is not None:
            line += f" | TEST att {test['att']:7.1f} throughput {test['throughput']:.0f}"
        self.logger.info(line)

    def _report(self, test_att: list[float]) -> dict[str, float]:
        result: dict[str, float] = {}
        if test_att:
            window = test_att[-REPORT_WINDOW:]
            result = {
                "att": test_att[-1],
                "att_last10": sum(window) / len(window),
                "att_best": min(test_att),
                "episodes_logged": float(len(test_att)),
            }
            if self.history:
                result.update({k: self.history[-1][k] for k in ("throughput", "queue", "delay")})
        if self.run is not None and result:
            self.run.result(**result)
        return result

    # ------------------------------------------------------------- checkpoints

    def save(self, path: Path | str) -> None:
        """Saves the networks that act, the optimisers and `ValueNorm`; without the
        latter a resumed run would compute values on another scale."""
        payload = {
            "actor": self.ppo.ac.actor.state_dict(),
            "critic": self.ppo.ac.critic.state_dict(),
            "actor_opt": self.ppo.actor_opt.state_dict(),
            "critic_opt": self.ppo.critic_opt.state_dict(),
        }
        if self.ppo.value_norm is not None:
            payload["value_norm"] = self.ppo.value_norm.state_dict()
        torch.save(payload, Path(path))

    def load(self, path: Path | str) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.ppo.ac.actor.load_state_dict(payload["actor"])
        self.ppo.ac.critic.load_state_dict(payload["critic"])
        self.ppo.actor_opt.load_state_dict(payload["actor_opt"])
        self.ppo.critic_opt.load_state_dict(payload["critic_opt"])
        if self.ppo.value_norm is not None and "value_norm" in payload:
            self.ppo.value_norm.load_state_dict(payload["value_norm"])
