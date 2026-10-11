"""Actor and critic networks of the core, built from CoLight's own layers.

The encoder composes `Embedding_MLP` and `MultiHeadAttModel` from
`agent/colight.py`, so the comparison with Advanced-CoLight measures the
learning algorithm, not the architecture. Changes against `ColightNet`:

1. No Q head (and no `MaskedOutput`): the encoder ends at the node embedding,
   the actor and critic add their own heads.
2. No `train` flag. LibSignal wraps the forward pass in `torch.no_grad()` when
   it is False; in PPO that silently stops learning. Gradients are switched
   off only by the caller, during rollouts.
3. Explicit arguments instead of the config dict, because the critic needs a
   different number of layers than the actor.

Input is [N, ob_length]: one graph, nodes in `RoadNetGraph` order. Several time
steps are batched as one block-diagonal graph (PyG convention), never as a
third dimension: `MultiHeadAttModel` reads the first dimension as the number of
edges.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from agent.colight import Embedding_MLP  # importable only after `libsignal.bootstrap()`
from torch import Tensor, nn
from torch.distributions import Categorical
from torch_geometric.nn import SAGEConv, global_mean_pool

from tsc_marl.core.uw_attention import UncertaintyWeightedAttention

# Defaults = `configs/tsc/colight.yml`, repeated so tests can build without Registry.
DEFAULT_EMB_DIMS = (128, 128)
DEFAULT_INPUT_DIM = (128, 128)
DEFAULT_OUTPUT_DIM = (128, 128)
DEFAULT_DIMS_EACH_HEAD = (16, 16)
DEFAULT_NUM_HEADS = (5, 5)


class GATEncoder(nn.Module):
    """`Embedding_MLP` + `n_layers` attention layers: [N, ob_length] -> [N, out_dim].

    One set of weights for all intersections (parameter sharing), so the number
    of parameters does not depend on the network size. Each attention layer is
    one hop: the actor uses 1, the critic 2.
    """

    def __init__(
        self,
        ob_length: int,
        *,
        emb_dims: Sequence[int] = DEFAULT_EMB_DIMS,
        n_layers: int = 1,
        input_dim: Sequence[int] = DEFAULT_INPUT_DIM,
        output_dim: Sequence[int] = DEFAULT_OUTPUT_DIM,
        dims_each_head: Sequence[int] = DEFAULT_DIMS_EACH_HEAD,
        num_heads: Sequence[int] = DEFAULT_NUM_HEADS,
    ) -> None:
        super().__init__()
        self._validate(
            ob_length, emb_dims, n_layers, input_dim, output_dim, num_heads, dims_each_head
        )

        self.ob_length = ob_length
        self.n_layers = n_layers
        self.out_dim = int(output_dim[n_layers - 1])

        self.embedding_mlp = Embedding_MLP(ob_length, layers=list(emb_dims))
        self.blocks = nn.ModuleList(
            UncertaintyWeightedAttention(
                d=int(input_dim[i]),
                dv=int(dims_each_head[i]),
                d_out=int(output_dim[i]),
                nv=int(num_heads[i]),
                suffix=i,
            )
            for i in range(n_layers)
        )

    @staticmethod
    def _validate(
        ob_length: int,
        emb_dims: Sequence[int],
        n_layers: int,
        input_dim: Sequence[int],
        output_dim: Sequence[int],
        num_heads: Sequence[int],
        dims_each_head: Sequence[int],
    ) -> None:
        """Checked here, because mismatched per-layer lists otherwise fail deep
        inside the first forward pass."""
        if ob_length <= 0:
            raise ValueError(f"ob_length must be positive, got {ob_length}")
        if not emb_dims:
            raise ValueError("emb_dims must not be empty (embedding MLP layers)")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        per_layer = {
            "INPUT_DIM": input_dim,
            "OUTPUT_DIM": output_dim,
            "NUM_HEADS": num_heads,
            "NODE_LAYER_DIMS_EACH_HEAD": dims_each_head,
        }
        for name, seq in per_layer.items():
            if len(seq) < n_layers:
                raise ValueError(
                    f"{name} has {len(seq)} entries but n_layers={n_layers}; "
                    f"missing a value for layer {n_layers - 1}"
                )

        if int(input_dim[0]) != int(emb_dims[-1]):
            raise ValueError(f"INPUT_DIM[0]={input_dim[0]} != embedding output {emb_dims[-1]}")
        for i in range(1, n_layers):
            if int(input_dim[i]) != int(output_dim[i - 1]):
                raise ValueError(
                    f"GAT layers do not chain: INPUT_DIM[{i}]={input_dim[i]} != "
                    f"OUTPUT_DIM[{i - 1}]={output_dim[i - 1]}"
                )

    @staticmethod
    def registry_kwargs() -> dict[str, Any]:
        """Encoder hyperparameters from LibSignal's YAML; shared by `Actor` and
        `Critic` so both read exactly the same config."""
        from common.registry import Registry

        param = Registry.mapping["model_mapping"]["setting"].param
        return {
            "emb_dims": param.get("NODE_EMB_DIM", DEFAULT_EMB_DIMS),
            "n_layers": param.get("N_LAYERS", 1),
            "input_dim": param.get("INPUT_DIM", DEFAULT_INPUT_DIM),
            "output_dim": param.get("OUTPUT_DIM", DEFAULT_OUTPUT_DIM),
            "dims_each_head": param.get("NODE_LAYER_DIMS_EACH_HEAD", DEFAULT_DIMS_EACH_HEAD),
            "num_heads": param.get("NUM_HEADS", DEFAULT_NUM_HEADS),
        }

    @classmethod
    def from_registry(cls, ob_length: int, **overrides: Any) -> GATEncoder:
        """`ob_length` is not in the config; it comes from `StateBuilder.ob_length`."""
        return cls(ob_length, **{**cls.registry_kwargs(), **overrides})

    def forward(
        self, x: Tensor, edge_index: Tensor, *, log_rho: Tensor | None = None
    ) -> Tensor:  # [N, F] -> [N, out_dim]
        """`log_rho` is keyword-only in every encoder and in `Actor`: `Critic.forward`
        takes `batch` as its third argument, and swapping two optional tensors
        would raise nothing."""
        if x.dim() != 2:
            raise ValueError(
                f"GATEncoder takes [N, ob_length], got {tuple(x.shape)}; "
                "batch time steps with torch_geometric.Batch, not a third dimension"
            )
        if x.shape[1] != self.ob_length:
            raise ValueError(f"observation width {x.shape[1]} != ob_length {self.ob_length}")

        h = self.embedding_mlp(x, True)  # always True, see point 2 in the module docstring
        for block in self.blocks:
            h = block(h, edge_index, True, log_rho=log_rho)
        return h

    def extra_repr(self) -> str:
        return f"ob_length={self.ob_length}, n_layers={self.n_layers}, out_dim={self.out_dim}"


#: Neighbour aggregation in the ACTOR. The critic is always GAT: the ablation asks
#: whether the decision benefits from weighting neighbours, not the critic.
ENCODERS = ("gat", "mean", "none")


class _AblationEncoder(nn.Module):
    """Shared skeleton of the ablation encoders: the same `Embedding_MLP` and layer
    widths as `GATEncoder`; only what a layer does with neighbours differs."""

    def __init__(
        self,
        ob_length: int,
        *,
        emb_dims: Sequence[int] = DEFAULT_EMB_DIMS,
        n_layers: int = 1,
        input_dim: Sequence[int] = DEFAULT_INPUT_DIM,
        output_dim: Sequence[int] = DEFAULT_OUTPUT_DIM,
        dims_each_head: Sequence[int] = DEFAULT_DIMS_EACH_HEAD,
        num_heads: Sequence[int] = DEFAULT_NUM_HEADS,
    ) -> None:
        super().__init__()
        GATEncoder._validate(
            ob_length, emb_dims, n_layers, input_dim, output_dim, num_heads, dims_each_head
        )
        self.ob_length = ob_length
        self.n_layers = n_layers
        self.out_dim = int(output_dim[n_layers - 1])
        self.embedding_mlp = Embedding_MLP(ob_length, layers=list(emb_dims))
        self.blocks = nn.ModuleList(
            self._block(int(input_dim[i]), int(output_dim[i])) for i in range(n_layers)
        )

    def _block(self, d_in: int, d_out: int) -> nn.Module:
        raise NotImplementedError

    # Not `_apply`: that is the internal `nn.Module` method behind `.to(device)`.

    def _aggregate(self, block: nn.Module, h: Tensor, edge_index: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, edge_index: Tensor, *, log_rho: Tensor | None = None) -> Tensor:
        if x.dim() != 2 or x.shape[1] != self.ob_length:
            raise ValueError(f"expected [N, {self.ob_length}], got {tuple(x.shape)}")
        # Raise instead of ignoring: these encoders have no attention weights, and a
        # run "UW-attention + mean" would otherwise finish with plausible numbers.
        if log_rho is not None:
            raise ValueError(
                f"{type(self).__name__} has no attention weights to modulate; "
                "UW-attention needs encoder='gat'"
            )
        h = self.embedding_mlp(x, True)
        for block in self.blocks:
            h = torch.relu(self._aggregate(block, h, edge_index))
        return h

    def extra_repr(self) -> str:
        return f"ob_length={self.ob_length}, n_layers={self.n_layers}, out_dim={self.out_dim}"


class MeanPoolEncoder(_AblationEncoder):
    """GraphSAGE with mean aggregation (Hamilton et al., 2017): every neighbour
    weighs the same. Asks whether learned neighbour weighting helps at all."""

    def _block(self, d_in: int, d_out: int) -> nn.Module:
        return SAGEConv(d_in, d_out, aggr="mean")

    def _aggregate(self, block: nn.Module, h: Tensor, edge_index: Tensor) -> Tensor:
        return block(h, edge_index)


class LocalEncoder(_AblationEncoder):
    """No neighbours: an MLP of the same depth, `edge_index` ignored, parameter
    sharing kept (as in MPLight, Chen et al., 2020). Asks whether a GNN is needed."""

    def _block(self, d_in: int, d_out: int) -> nn.Module:
        return nn.Linear(d_in, d_out)

    def _aggregate(self, block: nn.Module, h: Tensor, edge_index: Tensor) -> Tensor:
        return block(h)


def build_encoder(kind: str, ob_length: int, **kwargs: Any) -> nn.Module:
    """`gat` | `mean` | `none` -> encoder [N, ob_length] -> [N, out_dim]."""
    classes = {"gat": GATEncoder, "mean": MeanPoolEncoder, "none": LocalEncoder}
    if kind not in classes:
        raise ValueError(f"encoder={kind!r}, allowed: {ENCODERS}")
    return classes[kind](ob_length, **kwargs)


def encoder_kind() -> str:
    """Actor encoder from the config (`model.encoder`), default `gat`. Not written
    into `gatppo.yml`: a new key would change the params fingerprint of existing
    runs. Ablations set it through suite overrides."""
    from common.registry import Registry

    return Registry.mapping["model_mapping"]["setting"].param.get("encoder", "gat")


def orthogonal_(layer: nn.Linear, gain: float) -> nn.Linear:
    """Orthogonal weights and zero bias, the PPO/MAPPO convention.

    Applied to the heads only. The encoder keeps PyTorch's default init, as in
    `agent/colight.py`, so the comparison with Advanced-CoLight stays a
    comparison of learning algorithms.
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    """Policy: [N, ob_length] -> phase LOGITS [N, n_phases].

    Returns logits, not probabilities: `Categorical(logits=...)` applies a stable
    log-softmax, while passing probabilities would flatten the distribution.

    Phase mask (heterogeneous networks only): [N, n_phases], True = the phase
    exists. It is applied additively, `masked_fill(~mask, -inf)`. LibSignal's
    `MaskedOutput` multiplies instead, which for logits gives a masked phase
    e^0 in the softmax, often the largest probability. The mask is a
    non-persistent buffer (not in the checkpoint: it belongs to the road network,
    not to the weights) and is tiled over the time steps of a minibatch.
    """

    def __init__(
        self,
        ob_length: int,
        n_phases: int,
        *,
        hidden: int | None = None,
        encoder: str = "gat",
        phase_mask: Tensor | None = None,
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()
        if n_phases < 2:
            raise ValueError(f"n_phases must be >= 2 (a distribution over phases), got {n_phases}")

        self.ob_length = ob_length
        self.n_phases = n_phases
        self.encoder_kind = encoder
        # attribute name `encoder` for every variant: state_dict keys must match checkpoints
        self.encoder = build_encoder(encoder, ob_length, **encoder_kwargs)

        # One linear layer by default, as in CoLight (`OUTPUT_LAYERS: []`).
        head: list[nn.Module] = []
        in_dim = self.encoder.out_dim
        if hidden is not None:
            head += [orthogonal_(nn.Linear(in_dim, hidden), gain=2**0.5), nn.ReLU()]
            in_dim = hidden
        # gain 0.01: logits start near zero, so the initial policy is almost
        # uniform (entropy close to log n_phases) instead of a random strong preference.
        head.append(orthogonal_(nn.Linear(in_dim, n_phases), gain=0.01))
        self.head = nn.Sequential(*head)

        if phase_mask is not None:
            if phase_mask.dim() != 2 or phase_mask.shape[1] != n_phases:
                raise ValueError(
                    f"phase_mask must be [N, {n_phases}], got {tuple(phase_mask.shape)}"
                )
            phase_mask = phase_mask.bool()
            if not bool(phase_mask.any(dim=1).all()):
                raise ValueError("phase_mask: an intersection with no allowed phase")
        self.register_buffer("phase_mask", phase_mask, persistent=False)

    @classmethod
    def from_registry(cls, ob_length: int, n_phases: int, **overrides: Any) -> Actor:
        """Encoder from `colight.yml`; `n_phases` from the state builder, not from
        YAML, where a value from another network is easy to leave behind."""
        kwargs = {"encoder": encoder_kind(), **GATEncoder.registry_kwargs()}
        return cls(ob_length, n_phases, **{**kwargs, **overrides})

    def forward(
        self, x: Tensor, edge_index: Tensor, *, log_rho: Tensor | None = None
    ) -> Tensor:  # [N, F] -> [N, n_phases]
        logits = self.head(self.encoder(x, edge_index, log_rho=log_rho))
        if self.phase_mask is None:
            return logits
        n_nodes = self.phase_mask.shape[0]
        if logits.shape[0] % n_nodes:
            raise ValueError(f"{logits.shape[0]} rows is not a multiple of {n_nodes} nodes")
        mask = self.phase_mask.repeat(logits.shape[0] // n_nodes, 1)
        return logits.masked_fill(~mask, float("-inf"))

    def extra_repr(self) -> str:
        return f"ob_length={self.ob_length}, n_phases={self.n_phases}, encoder={self.encoder_kind}"


class Critic(nn.Module):
    """Factorised centralised critic: [N, ob_length] -> V_i [N], one value per agent.

    Centralisation is in what the critic sees, not in a single output: two
    attention layers (2 hops) plus the mean embedding of the whole network.
    Weights are separate from the actor. No `log_rho`: the critic exists only in
    training, where it must not use information the actor lacks at deployment.
    Return normalisation lives in the PPO loop (`ValueNorm`).
    """

    def __init__(
        self,
        ob_length: int,
        *,
        hidden: int = 128,
        n_layers: int = 2,
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()
        self.ob_length = ob_length
        self.encoder = GATEncoder(ob_length, n_layers=n_layers, **encoder_kwargs)

        # Input is [node embedding, network context]: context alone would give every
        # node the same value, the node embedding alone would not be centralised.
        self.head = nn.Sequential(
            orthogonal_(nn.Linear(2 * self.encoder.out_dim, hidden), gain=2**0.5),
            nn.ReLU(),
            # gain 1.0, unlike the actor: the value should start on a sensible scale
            orthogonal_(nn.Linear(hidden, 1), gain=1.0),
        )

    @classmethod
    def from_registry(cls, ob_length: int, **overrides: Any) -> Critic:
        """Encoder from `colight.yml` except `n_layers`, which there is the actor's."""
        kwargs = GATEncoder.registry_kwargs()
        kwargs.pop("n_layers", None)
        return cls(ob_length, **{**kwargs, **overrides})

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        """[N, ob_length] -> [N]. `batch` maps nodes to time steps (`Batch.batch`).

        Required when time steps are batched: a plain `h.mean(0)` would average
        over all steps of the minibatch, letting the critic see the future.
        """
        h = self.encoder(x, edge_index)
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        elif batch.shape != (h.shape[0],):
            raise ValueError(f"batch must be [{h.shape[0]}], got {tuple(batch.shape)}")
        context = global_mean_pool(h, batch)  # [B, out_dim]
        return self.head(torch.cat([h, context[batch]], dim=-1)).squeeze(-1)

    def extra_repr(self) -> str:
        return f"ob_length={self.ob_length}, n_layers={self.encoder.n_layers}"


class ActorCritic(nn.Module):
    """Actor and critic behind the interface used by the PPO loop.

    Two independent sets of weights (1-hop actor, 2-hop critic). Gradients are
    never disabled inside: the caller wraps `act()` in `torch.no_grad()` during
    rollouts, while `evaluate()` must pass gradients for the policy loss.
    """

    def __init__(
        self,
        ob_length: int,
        n_phases: int,
        *,
        hidden: int | None = None,
        critic_hidden: int = 128,
        critic_n_layers: int = 2,
        encoder: str = "gat",
        phase_mask: Tensor | None = None,
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()
        critic_kwargs = {k: v for k, v in encoder_kwargs.items() if k != "n_layers"}
        # the phase mask is for the policy only; the critic values states
        self.actor = Actor(
            ob_length,
            n_phases,
            hidden=hidden,
            encoder=encoder,
            phase_mask=phase_mask,
            **encoder_kwargs,
        )
        # the critic is always GAT, see ENCODERS
        self.critic = Critic(
            ob_length, hidden=critic_hidden, n_layers=critic_n_layers, **critic_kwargs
        )

    @classmethod
    def from_registry(cls, ob_length: int, n_phases: int, **overrides: Any) -> ActorCritic:
        kwargs = {"encoder": encoder_kind(), **GATEncoder.registry_kwargs()}
        return cls(ob_length, n_phases, **{**kwargs, **overrides})

    def distribution(
        self, x: Tensor, edge_index: Tensor, *, log_rho: Tensor | None = None
    ) -> Categorical:
        """Distribution over phases; every row is one agent.

        Rollout (`act`) and policy loss (`evaluate`) both go through here, so
        `log_rho` must be passed to both or the first-epoch ratio is not 1.
        """
        return Categorical(logits=self.actor(x, edge_index, log_rho=log_rho))

    def act(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        deterministic: bool = False,
        batch: Tensor | None = None,
        log_rho: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Rollout: -> (actions [N], log_prob [N], values [N]).

        `deterministic=True` takes the argmax (evaluation); training always
        samples, which is where PPO's exploration comes from. `log_rho` goes to
        the actor only.
        """
        dist = self.distribution(x, edge_index, log_rho=log_rho)
        actions = dist.logits.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), self.critic(x, edge_index, batch)

    def evaluate(
        self,
        x: Tensor,
        edge_index: Tensor,
        actions: Tensor,
        batch: Tensor | None = None,
        *,
        log_rho: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update: -> (log_prob [N], entropy [N], values [N]) for the given actions.

        `log_rho` must be the vector `act` saw at that step, so the rollout buffer
        stores it: the observation head reads a window of past readings, and
        recomputing it from `x` alone would give a different value.
        """
        dist = self.distribution(x, edge_index, log_rho=log_rho)
        return dist.log_prob(actions), dist.entropy(), self.critic(x, edge_index, batch)

    def value(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        """Value only: GAE needs V(s_T) one step past the rollout."""
        return self.critic(x, edge_index, batch)
