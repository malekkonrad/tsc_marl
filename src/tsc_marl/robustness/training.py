"""Training a policy WITH a mechanism: failure randomization plus UW-attention.

UW-attention attached to an already trained policy is not the same as an
architecture trained with it, so the core can also be trained with log_rho in
the loop. Two things must come together:

1. Failures in training (`domain.sensors.FailureMix`): on clean traffic the
   modulation is almost silent, so the policy would have nothing to learn from.
2. A control arm: "trained with failures, without UW" next to "with failures,
   with UW", on the same failure schedule, so their difference is the mechanism.

The uncertainty source is FROZEN during training (computed under `no_grad`, not
in `parameters()`, not in the checkpoint) and calibrated beforehand. Recomputing
its reference during training would couple policy and reference in a way that
cannot be untangled later. `uw_spread` and `uw_active` are logged per episode to
tell "the mechanism gradually matters" from "it was noise all along".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import Tensor

#: Where UW-attention takes u from: the observation head, the failure layer's truth
#: (modulation oracle, an upper bound, `controllers.OracleUncertainty`) or the
#: frozen-detector counter (`controllers.RuleUncertainty`, deployable).
UW_SOURCES = ("head", "oracle", "rule")
#: Source-side modulation channels, see `controllers.source_log_rho`.
UW_CHANNELS = ("all", "self", "edges")


def check_uw(uw: dict[str, Any]) -> str | None:
    """Error in a `uw` spec, or None. ONE validator for evaluation plans and
    training suites. Every typo is an error rather than a default: a column
    labelled "rule" silently computed by the head would repeat an old measurement
    under a new name.
    """
    from tsc_marl.uncertainty.obs_head import EDGE_SIDES  # pulls in torch

    unknown = set(uw) - {"tau", "side", "u", "channels", "threshold"}
    if unknown:
        return f"unknown keys {sorted(unknown)}"
    if "side" in uw and uw["side"] not in EDGE_SIDES:
        return f"side={uw['side']!r}, allowed: {EDGE_SIDES}"
    source = uw.get("u", "head")
    if source not in UW_SOURCES:
        return f"u={source!r}, allowed: {UW_SOURCES}"
    # oracle and rule exist only on the source side (`source_log_rho`); required
    # EXPLICITLY so the plan never states something the controller does not compute
    if source in ("oracle", "rule") and uw.get("side") != "source":
        return f"modulation u={source} requires side: source"
    if "channels" in uw:
        if uw["channels"] not in UW_CHANNELS:
            return f"channels={uw['channels']!r}, allowed: {UW_CHANNELS}"
        if source == "head":
            return "channels only with u: oracle or u: rule (the head has no channels)"
    if "threshold" in uw and source != "rule":
        return "threshold only with u: rule (frozen-counter threshold)"
    return None


# --------------------------------------------------------- uncertainty in the loop


def online_uncertainty(
    network: str,
    n_mov: int,
    edge_index: Tensor,
    *,
    tau: float | None = None,
    side: str = "target",
) -> Any:
    """`OnlineUncertainty` from the FROZEN artefacts in `runs/heads/` for this network.

    Shared by evaluation and training: a "u_ref from another head than the
    weights" mismatch is silent (numbers in other units), so `load_reference`
    checks the head file's hash. `network` is the BASE network (no demand-variant
    suffix): the artefacts belong to a network. `side` picks the reference FILE,
    and the side recorded inside must match it.
    """
    from tsc_marl.uncertainty import obs_head as oh

    head, ref = oh.head_path(network), oh.reference_path(network, side)
    if not head.is_file():
        raise SystemExit(
            f"no observation head for '{network}' ({head.name}); "
            f"compute it: tsc obshead -n {network} --save --window 12 --persistence"
        )
    if not ref.is_file():
        raise SystemExit(
            f"no reference '{ref.name}' for '{network}'; "
            + (
                f"compute it: tsc obshead -n {network} --save --window 12 --persistence"
                if side == "target"
                else f"compute it for the head: tsc obshead -n {network} --reference {side}"
            )
        )
    fitted = oh.load_head(head, network=network)
    reference = oh.load_reference(ref, network=network, head=head)
    if reference.side != side:
        raise SystemExit(f"{ref.name} has side={reference.side!r}, the plan wants {side!r}")
    sources = oh.movement_sources(network, n_mov)
    return oh.OnlineUncertainty(
        fitted, reference, sources, edge_index, tau=oh.DEFAULT_TAU if tau is None else float(tau)
    )


def gate_reference(network: str) -> Any:
    """Real-gate thresholds for the BASE network (`runs/heads/gate_ref__<net>.npz`).

    Shared by the rule gate and rule modulation, in evaluation and in training,
    so the policy never learns with another window or threshold than
    `tsc degrade` measures it with."""
    from tsc_marl.robustness import rule_ref
    from tsc_marl.uncertainty import obs_head as oh

    path = oh.gate_path(network)
    if path.is_file():
        return oh.load_gates(path, network=network, head=oh.head_path(network))
    # A network without a head (Krakow, SUMO): the rule threshold alone, same recipe
    # (`robustness/rule_ref.py`); `uq` is None and no UQ gate can be built.
    rule = rule_ref.rule_path(network)
    if rule.is_file():
        return rule_ref.load_rule(rule, network=network)
    raise SystemExit(
        f"no gate thresholds for '{network}'; compute them: "
        f"tsc obshead -n {network} --save --window 12 --persistence (CityFlow, with a head) "
        f"or tsc rulecal -n {network} (rule only, also SUMO)"
    )


def uw_signal(uw: dict[str, Any], network: str, env: Any) -> Any:
    """log_rho source for UW-attention from a `uw` spec, ONE for training and evaluation.

    `u: head` (default): the frozen observation head on `side`; `u: oracle`: the
    failure layer's truth (`controllers.OracleUncertainty`); `u: rule`: the
    frozen-detector counter with the rule gate's window and threshold
    (`controllers.RuleUncertainty`). `channels` applies to oracle and rule only.
    `network` is the BASE network; `env` is read live, since `env.features`
    changes per scenario and per failure-mix episode."""
    from tsc_marl.robustness import controllers as c

    tau = uw.get("tau")
    source = uw.get("u", "head")
    channels = uw.get("channels", "all")
    edge_index = env.graph.edge_index
    if source == "oracle":
        return c.OracleUncertainty(
            env, edge_index, tau=float(0.25 if tau is None else tau), channels=channels
        )
    if source == "rule":
        thresholds = gate_reference(network)
        return c.RuleUncertainty(
            edge_index,
            window=thresholds.window,
            threshold=float(uw.get("threshold", thresholds.rule)),
            n_mov=env.features.n_mov,
            tau=float(0.25 if tau is None else tau),
            channels=channels,
            real=env.features.movement_mask(),
        )
    return online_uncertainty(
        network, env.features.n_mov, edge_index, tau=tau, side=uw.get("side", "target")
    )


class UWStream:
    """log_rho step by step plus episode diagnostics, as in `UWPolicyController`.

    Training needs EXACTLY the evaluation path (same window buffer, same reset at
    episode start, same `uw_spread`), or the policy would learn one signal and be
    measured with another. The source runs on the CPU; only the final [E+N]
    vector moves to the policy's device, once per decision.
    """

    def __init__(
        self,
        uncertainty: Any,
        edge_index: Tensor,
        device: torch.device | str = "cpu",
        *,
        n_nodes: int | None = None,
    ) -> None:
        import torch

        self.uncertainty = uncertainty
        self.edge_index = edge_index  # CPU: edge order, not computation
        self.device = torch.device(device)
        self.n_nodes = int(n_nodes if n_nodes is not None else int(edge_index.max()) + 1)
        self.reset()

    def reset(self) -> None:
        """Clearing the window is correctness, not hygiene: a reading from before
        `world.reset()` describes other traffic and would start the episode by
        muting random neighbours."""
        self.uncertainty.reset()
        self._spread: list[float] = []
        self._active: list[float] = []

    def log_rho(self, obs: np.ndarray) -> Tensor:
        """[E+N] on the policy's device; also collects episode diagnostics."""
        from tsc_marl.uncertainty.obs_head import group_spread

        lr = self.uncertainty.log_rho(obs)  # CPU
        self._spread.append(group_spread(lr, self.edge_index, self.n_nodes))
        self._active.append(float((lr < 0).float().mean()))
        return lr.to(self.device)

    def episode_stats(self) -> dict[str, float]:
        """`uw_spread` = mean spread of log_rho within a node's softmax group, the
        only part of the signal attention sees. Without it a training curve with
        the mechanism cannot be told apart from one without."""
        if not self._spread:
            return {}
        return {
            "uw_spread": float(np.mean(self._spread)),
            "uw_active": float(np.mean(self._active)),
        }


def uw_config_from_registry() -> dict[str, Any] | None:
    """`model.uw` from YAML -> {'tau': ...} or None (a run without the mechanism).
    `uw: True` means the default tau (`obs_head.DEFAULT_TAU`), chosen once."""
    from common.registry import Registry

    raw = Registry.mapping["model_mapping"]["setting"].param.get("uw")
    if not raw:
        return None
    uw = {} if raw is True else dict(raw)
    error = check_uw(uw)
    if error:
        raise SystemExit(f"model.uw in the suite: {error}")
    return uw
