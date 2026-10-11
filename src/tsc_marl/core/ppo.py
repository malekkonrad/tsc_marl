"""MAPPO learning loop: PPO over the road-network graph.

This module knows nothing about the simulator: it takes tensors and returns
gradients, so the algorithm is tested in milliseconds on a synthetic task.
MAPPO is plain PPO in which every agent is a separate sample; what the critic
and the actor see is decided in `nets.py`.

Pitfalls handled here:

1. Truncation is not termination. An episode ends after 3600 s, not in a
   terminal state, so the last step bootstraps `last_value` instead of using
   `done=True`, which would put V(s_T) = 0.
2. Minibatches are drawn over time steps, not agents: GAT needs the whole graph.
   k sampled steps are stacked into one block-diagonal graph.
3. Return scale. The ATS state is not normalised, so the raw value loss can
   dominate the gradient; `ValueNorm` lets the critic learn normalised values.
4. Rollouts run under `torch.no_grad()` (the caller's job, as in `ActorCritic.act`).
5. log_rho in a minibatch must follow the stacked graph's layout, see `tiled_log_rho`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from torch import Tensor, nn

from tsc_marl.core.nets import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters; the values used in runs live in `configs/tsc/gatppo.yml`.

    `gamma` = 0.95 follows the baselines (`base.yml`) although MAPPO usually takes
    0.99: discounting counts decisions, and 0.95 is a horizon of about 20
    decisions (5 minutes of traffic). `max_grad_norm` = 0.5 is the PPO
    convention; `grad_clip: 5.0` in `base.yml` is a DQN setting.
    """

    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    epochs: int = 10
    n_minibatches: int = 4
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    normalize_values: bool = True
    # Emergency brake: stop the update's epochs once approx KL exceeds this (nats).
    # None = off (default; clipping is enough).
    target_kl: float | None = None

    @classmethod
    def from_registry(cls, **overrides: Any) -> PPOConfig:
        """From the `model` section of the YAML config.

        Through the config, not code: the run fingerprint covers `model` and
        `trainer`, so changing e.g. `clip_range` invalidates skip-if-done. A
        value living only in Python would let a different run pass as done."""
        from common.registry import Registry

        param = Registry.mapping["model_mapping"]["setting"].param
        known = {f.name: param[f.name] for f in fields(cls) if f.name in param}
        return cls(**{**known, **overrides})

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma outside (0, 1]: {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda outside [0, 1]: {self.gae_lambda}")
        if self.clip_range <= 0 or self.epochs < 1 or self.n_minibatches < 1:
            raise ValueError("need clip_range > 0 and epochs, n_minibatches >= 1")


class ValueNorm(nn.Module):
    """Running normalisation of returns (MAPPO `use_valuenorm`).

    The critic predicts normalised values, so the value loss has unit scale
    whatever the reward. For GAE values go back to the raw scale, otherwise
    advantages would be in different units than rewards. Statistics are
    buffers: saved in the checkpoint, never trained.
    """

    def __init__(self, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Batched Welford update (Chan et al.), numerically stabler than a running
        sum of squares."""
        batch_count = x.numel()
        if batch_count == 0:
            return
        batch_mean, batch_var = x.mean(), x.var(unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def denormalize(self, x: Tensor) -> Tensor:
        return x * torch.sqrt(self.var + self.epsilon) + self.mean


def tiled_edge_index(edge_index: Tensor, n_nodes: int, k: int) -> Tensor:
    """`k` copies of one graph as a single block-diagonal [2, k*E] edge index.

    Same result as `torch_geometric.Batch.from_data_list` (tested) without
    building k `Data` objects per minibatch; valid only because the graph is static.
    """
    offsets = (torch.arange(k, device=edge_index.device) * n_nodes).repeat_interleave(
        edge_index.shape[1]
    )
    return edge_index.repeat(1, k) + offsets


def batch_vector(n_nodes: int, k: int, device: torch.device | None = None) -> Tensor:
    """Node -> graph assignment for `k` stacked graphs, as `global_mean_pool` expects."""
    return torch.arange(k, device=device).repeat_interleave(n_nodes)


def tiled_log_rho(log_rho: Tensor, n_edges: int, n_nodes: int) -> Tensor:
    """[k, E+N] (log_rho per time step) -> [k*(E+N)] in stacked-graph order.

    The layer expects the order of the `edge_index` passed to `forward`, which for
    a minibatch comes from `tiled_edge_index`: all E edges of graph 0, then of
    graph 1, ..., and only at the very end k*N self-loops (`add_self_loops`
    appends them after stacking). The buffer keeps each step's [E+N] in one
    piece, so a naive `log_rho.reshape(-1)` has the right length and the wrong
    layout, and nothing reports it.
    """
    if log_rho.dim() != 2 or log_rho.shape[1] != n_edges + n_nodes:
        raise ValueError(
            f"log_rho must be [k, E+N] = [k, {n_edges + n_nodes}], got {tuple(log_rho.shape)}"
        )
    k = log_rho.shape[0]
    edges = log_rho[:, :n_edges].reshape(k * n_edges)  # graph 0, graph 1, ...
    loops = log_rho[:, n_edges:].reshape(k * n_nodes)  # nodes of graph 0, then graph 1, ...
    return torch.cat([edges, loops])


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """GAE(lambda) per agent. -> (advantages [T, N], returns [T, N]).

    Rewards and values have an agent dimension, so the advantage of
    intersection i depends on its own reward (factorised critic).
    `dones[t]` says whether s_{t+1} is TERMINAL, not whether the episode was cut
    (pitfall 1 in the module docstring).
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            f"rewards {tuple(rewards.shape)}, values {tuple(values.shape)} and dones "
            f"{tuple(dones.shape)} must have the same shape [T, N]"
        )
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nonterminal * next_value - values[t]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[t] = running
    return advantages, advantages + values


@dataclass
class Minibatch:
    """One minibatch: `k` time steps stacked into one graph with `k*N` nodes."""

    x: Tensor  # [k*N, ob_length]
    edge_index: Tensor  # [2, k*E]
    batch: Tensor  # [k*N]
    actions: Tensor  # [k*N]
    old_log_probs: Tensor  # [k*N]
    old_values: Tensor  # [k*N], raw scale
    advantages: Tensor  # [k*N]
    returns: Tensor  # [k*N], raw scale
    #: [k*(E+N)] in stacked-graph order (`tiled_log_rho`), or None without the
    #: mechanism. None means "compute as the original layer", not "zeros": only
    #: the former is bit-identical to the plain core.
    log_rho: Tensor | None = None


class RolloutBuffer:
    """One rollout segment: [T decisions, N agents].

    The graph is stored once (it is static). Flattening `[t, i] -> t*N + i` is a
    contract shared with `tiled_edge_index`: row `t*N + i` of `x` is agent `i` at
    step `t`. A mistake would pair one intersection's advantage with another's
    observation, silently.
    """

    def __init__(
        self,
        capacity: int,
        n_nodes: int,
        ob_length: int,
        edge_index: Tensor,
        device: torch.device | str = "cpu",
        *,
        with_log_rho: bool = False,
    ) -> None:
        self.capacity = capacity
        self.n_nodes = n_nodes
        self.ob_length = ob_length
        self.device = torch.device(device)
        self.edge_index = edge_index.to(self.device)
        self.n_edges = int(self.edge_index.shape[1])
        self.pos = 0

        shape = (capacity, n_nodes)
        self.obs = torch.zeros((*shape, ob_length), device=self.device)
        # log_rho is stored, not recomputed: the observation head reads a window of
        # past readings, so it cannot be rebuilt from `obs[t]`, and `evaluate` must
        # see exactly what `act` saw or the first-epoch ratio is not 1.
        self.log_rho = (
            torch.zeros((capacity, self.n_edges + n_nodes), device=self.device)
            if with_log_rho
            else None
        )
        self.actions = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros(shape, device=self.device)
        self.values = torch.zeros(shape, device=self.device)
        self.rewards = torch.zeros(shape, device=self.device)
        self.dones = torch.zeros(shape, device=self.device)
        self.advantages = torch.zeros(shape, device=self.device)
        self.returns = torch.zeros(shape, device=self.device)
        self._ready = False

    def __len__(self) -> int:
        return self.pos

    @property
    def full(self) -> bool:
        return self.pos >= self.capacity

    def reset(self) -> None:
        self.pos = 0
        self._ready = False

    def add(
        self,
        obs: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        values: Tensor,
        rewards: Tensor,
        dones: Tensor | bool = False,
        log_rho: Tensor | None = None,
    ) -> None:
        """One DECISION (15 s of simulation), not one simulator step.

        `values` on the raw scale: only the caller knows whether the critic
        worked in normalised space. `log_rho` [E+N] for this step, present
        exactly when the buffer was built `with_log_rho` (a silent mismatch would
        be a run without the mechanism it was meant to have).
        """
        if self.full:
            raise IndexError(f"buffer full ({self.capacity}); call reset() or update()")
        if obs.shape != (self.n_nodes, self.ob_length):
            raise ValueError(
                f"obs must be [{self.n_nodes}, {self.ob_length}], got {tuple(obs.shape)}"
            )
        if (log_rho is None) != (self.log_rho is None):
            raise ValueError(
                "buffer built with with_log_rho="
                f"{self.log_rho is not None}, but add() got log_rho={log_rho is not None}"
            )
        if log_rho is not None and log_rho.shape != (self.n_edges + self.n_nodes,):
            raise ValueError(
                f"log_rho must be [{self.n_edges + self.n_nodes}] = E + N, "
                f"got {tuple(log_rho.shape)}"
            )
        t = self.pos
        self.obs[t] = obs
        if self.log_rho is not None:
            self.log_rho[t] = log_rho
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        # `dones` as a scalar (all agents at once) or per agent
        self.dones[t] = torch.as_tensor(dones, dtype=torch.float32, device=self.device).expand(
            self.n_nodes
        )
        self.pos += 1
        self._ready = False

    def compute_returns(self, last_value: Tensor, config: PPOConfig) -> None:
        """GAE after collecting a segment. `last_value` = V(s_T) of the state past
        the segment, raw scale (pitfall 1)."""
        if self.pos == 0:
            raise RuntimeError("empty buffer, nothing to compute returns from")
        adv, ret = compute_gae(
            self.rewards[: self.pos],
            self.values[: self.pos],
            self.dones[: self.pos],
            last_value,
            config.gamma,
            config.gae_lambda,
        )
        self.advantages[: self.pos] = adv
        self.returns[: self.pos] = ret
        self._ready = True

    def minibatches(
        self, n_minibatches: int, generator: torch.Generator | None = None
    ) -> Iterator[Minibatch]:
        """Random split of TIME STEPS into minibatches (pitfall 2).

        The permutation is drawn on the CPU and then moved, so the same seed gives
        the same minibatches on CPU and GPU (and a CPU generator works with CUDA).
        """
        if not self._ready:
            raise RuntimeError("call compute_returns() first, advantages are still zeros")
        order = torch.randperm(self.pos, generator=generator).to(self.device)
        size = max(1, math.ceil(self.pos / n_minibatches))
        for start in range(0, self.pos, size):
            idx = order[start : start + size]
            k = idx.numel()
            yield Minibatch(
                # [k, N, ...] -> [k*N, ...]: `[t, i] -> t*N + i`, as in `tiled_edge_index`
                x=self.obs[idx].reshape(k * self.n_nodes, self.ob_length),
                edge_index=tiled_edge_index(self.edge_index, self.n_nodes, k),
                batch=batch_vector(self.n_nodes, k, self.device),
                actions=self.actions[idx].reshape(-1),
                old_log_probs=self.log_probs[idx].reshape(-1),
                old_values=self.values[idx].reshape(-1),
                advantages=self.advantages[idx].reshape(-1),
                returns=self.returns[idx].reshape(-1),
                # not `self.log_rho[idx].reshape(-1)`, see `tiled_log_rho`
                log_rho=(
                    tiled_log_rho(self.log_rho[idx], self.n_edges, self.n_nodes)
                    if self.log_rho is not None
                    else None
                ),
            )


@dataclass
class PPO:
    """MAPPO update: clipped surrogate, value loss and entropy bonus.

    Two optimisers, because actor and critic are disjoint networks with
    different learning rates.
    """

    ac: ActorCritic
    config: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), lr=self.config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), lr=self.config.critic_lr)
        # The device is read from `ac` (one source of truth); `ValueNorm` must live
        # there too. `ac` is moved before PPO is built (`CoreTrainer.from_world`),
        # so the optimiser state is created on the right device.
        self.device = next(self.ac.parameters()).device
        self.value_norm = ValueNorm().to(self.device) if self.config.normalize_values else None

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def value_of(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        """V(s) on the raw scale, the one of GAE and rewards."""
        v = self.ac.value(x, edge_index, batch)
        return self.value_norm.denormalize(v) if self.value_norm else v

    @torch.no_grad()
    def act(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        deterministic: bool = False,
        log_rho: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """(actions, log_prob, raw values). `no_grad` belongs here, the only data
        collection path; never in `evaluate` (pitfall 4)."""
        actions, log_probs, values = self.ac.act(
            x, edge_index, deterministic=deterministic, log_rho=log_rho
        )
        if self.value_norm:
            values = self.value_norm.denormalize(values)
        return actions, log_probs, values

    # ------------------------------------------------------------------- update

    def update(
        self, buffer: RolloutBuffer, generator: torch.Generator | None = None
    ) -> dict[str, float]:
        """One full update on the collected segment. -> diagnostics.

        `approx_kl` and `clip_frac` are the early sign of a too large learning
        rate, `entropy` of a policy that became deterministic too early.
        """
        cfg = self.config
        returns = buffer.returns[: buffer.pos]
        if self.value_norm:
            self.value_norm.update(returns)

        stats = {k: 0.0 for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")}
        n_batches, stopped = 0, False

        for _ in range(cfg.epochs):
            for mb in buffer.minibatches(cfg.n_minibatches, generator):
                log_probs, entropy, values = self.ac.evaluate(
                    mb.x, mb.edge_index, mb.actions, mb.batch, log_rho=mb.log_rho
                )
                ratio = torch.exp(log_probs - mb.old_log_probs)

                advantages = mb.advantages
                if cfg.normalize_advantage and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                unclipped = ratio * advantages
                clipped = ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                target = mb.returns
                old_values = mb.old_values
                if self.value_norm:
                    target = self.value_norm.normalize(target)
                    old_values = self.value_norm.normalize(old_values)
                # Value clipping for the same reason as policy clipping: the target
                # still comes from the old policy.
                v_clipped = old_values + (values - old_values).clamp(
                    -cfg.value_clip_range, cfg.value_clip_range
                )
                value_loss = (
                    0.5 * torch.max((values - target) ** 2, (v_clipped - target) ** 2).mean()
                )

                entropy_mean = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean

                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.ac.critic.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                with torch.no_grad():
                    # Schulman's k3 estimator: non-negative, less noisy than old - new
                    approx_kl = ((ratio - 1) - (log_probs - mb.old_log_probs)).mean()
                    clip_frac = ((ratio - 1).abs() > cfg.clip_range).float().mean()
                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy_mean)
                stats["approx_kl"] += float(approx_kl)
                stats["clip_frac"] += float(clip_frac)
                n_batches += 1

                if cfg.target_kl is not None and float(approx_kl) > cfg.target_kl:
                    stopped = True
                    break
            if stopped:
                break

        out = {k: v / max(n_batches, 1) for k, v in stats.items()}
        out["n_minibatches"] = float(n_batches)
        out["early_stopped"] = float(stopped)
        return out
