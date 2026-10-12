"""Attention modulated by edge reliability (UW-attention).

CoLight's `MultiHeadAttModel` plus one additive term in the attention logit:

    rho_ij   = exp(-u_ij / tau)                 reliability of edge j->i, in (0, 1]
    alpha_ij = softmax_j(e_ij + log rho_ij)     i.e. alpha_ij ~ rho_ij * exp(e_ij)

Additive in the logit, so the layer's own normalisation still applies. Needed at
all because in the original e_ij >= 0 (both vectors pass a ReLU before the dot
product), so a neighbour's unnormalised weight has a floor of exp(0) = 1 and the
network cannot express "ignore j", only dilute it; log rho is the only channel
below zero.

`log_rho` CONTRACT: a tensor [E + N], i.e. already including self-loops: first E
entries in the order of the `edge_index` PASSED TO `forward`, then N self-loops
in node order. Not a convention: `add_self_loops` appends the loops at the end,
in node order, keeping the edge order. The full [E + N] is taken (not [E] padded
inside) so that a test can shift every logit of a softmax group, self-loop included.

Caller pitfall, batches: the order is that of the `edge_index` passed in, not of
the graph. For a minibatch from `tiled_edge_index` all edges of all k graphs come
first and the k*N self-loops last (`ppo.tiled_log_rho`). A wrongly stacked vector
has the right length and PyG passes it to `message` unchecked.
"""

import torch
import torch.nn.functional as F
import torch_scatter
from agent.colight import MultiHeadAttModel
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing_extensions import override


class UncertaintyWeightedAttention(MultiHeadAttModel):
    @override
    def forward(self, x, edge_index, train=True, *, log_rho=None):
        """`log_rho` is KEYWORD-ONLY on purpose: the third positional argument is
        `train`. Passed positionally, `True` would add 1 to EVERY edge; softmax
        ignores a constant per group, so the output would be bit-identical and the
        modulation would never switch on, silently. `train` keeps its original meaning.
        """
        if train:
            return self._forward(x, edge_index, log_rho)
        with torch.no_grad():
            return self._forward(x, edge_index, log_rho)

    @override
    def _forward(self, x, edge_index, log_rho=None):
        # `num_nodes` pitfall: `add_self_loops` is called WITHOUT `num_nodes` (as in
        # the original, to stay bit-identical), so N is inferred from the largest
        # index. If the highest-numbered node is isolated, it gets no self-loop and
        # outputs a constant, with a correct output shape. Not fixed with
        # `num_nodes=x.size(0)` (that would change results); turned into an error.
        n = maybe_num_nodes(edge_index)
        if n != x.size(0):
            raise ValueError(
                f"edge_index sees {n} nodes, x has {x.size(0)} rows; the highest-numbered "
                "node is isolated and would get no self-loop"
            )

        edge_index, _ = add_self_loops(edge_index=edge_index)

        # Only the length can be checked here; the order is checked by tests.
        if log_rho is not None and log_rho.shape != (edge_index.shape[1],):
            raise ValueError(
                f"log_rho has shape {tuple(log_rho.shape)}, expected "
                f"({edge_index.shape[1]},) = E + N (edges + self-loops)"
            )

        aggregated = self.propagate(x=x, edge_index=edge_index, log_rho=log_rho)
        out = self.out(aggregated)
        out = F.relu(out)
        return out

    @override
    def message(self, x_i, x_j, edge_index, log_rho=None):
        h_target = F.relu(self.W_target(x_i))
        h_target = h_target.view(h_target.shape[:-1][0], self.nv, self.dv)
        agent_repr = h_target.permute(1, 0, 2)

        h_source = F.relu(self.W_source(x_j))
        h_source = h_source.view(h_source.shape[:-1][0], self.nv, self.dv)

        neighbor_repr = h_source.permute(1, 0, 2)  # [nv, E, dv]
        index = edge_index[1]  # which is target

        e_i = torch.mul(agent_repr, neighbor_repr).sum(-1)  # [nv, E+N]

        if log_rho is not None:
            e_i = e_i + log_rho

        max_node = torch_scatter.scatter_max(e_i, index=index)[0]  # [nv, N]
        max_i = max_node.index_select(1, index=index)  # [nv, E+N]
        ec_i = torch.add(e_i, -max_i)
        ecexp_i = torch.exp(ec_i)
        norm_node = torch_scatter.scatter_sum(ecexp_i, index=index)  # [nv, N]
        normst_node = torch.add(norm_node, 1e-12)
        normst_i = normst_node.index_select(1, index)  # [nv, E+N]

        alpha_i = ecexp_i / normst_i  # [nv, E+N]
        alpha_i_expand = alpha_i.repeat(self.dv, 1, 1)
        alpha_i_expand = alpha_i_expand.permute((1, 2, 0))  # [nv, E+N, dv]

        hidden_neighbor = F.relu(self.hidden_embedding(x_j))
        hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv)
        hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2)  # [nv, E+N, dv]
        return torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0)
