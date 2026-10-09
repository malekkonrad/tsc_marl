"""Attention z modulacją wiarygodnością krawędzi (UW-attention).

`MultiHeadAttModel` z `agent/colight.py` + jeden addytywny składnik w logicie:

    ρ_ij = exp(−u_ij / τ)                  wiarygodność krawędzi j→i, ∈ (0, 1]
    α_ij = softmax_j( e_ij + log ρ_ij )    ⟺  α_ij ∝ ρ_ij · exp(e_ij)

Dlaczego addytywnie w logicie, a nie mnożąco po softmaxie: po softmaxie trzeba by
renormalizować ręcznie, a tu normalizacja jest już policzona przez `scatter_sum`.
Dlaczego w ogóle: w oryginale `e_ij ≥ 0` zawsze (oba wektory idą przez `F.relu`
przed iloczynem skalarnym), więc `exp(e_ij) ≥ 1` — nieznormalizowana waga sąsiada
ma twardą podłogę 1 i sieć NIE POTRAFI wyrazić „ignoruj j", może go tylko
rozcieńczyć. log ρ jest jedynym kanałem schodzącym poniżej zera.

KONTRAKT `log_rho`: tensor [E + N], czyli JUŻ PO pętlach własnych.

- pierwsze E pozycji: w kolejności `edge_index` PRZEKAZANEGO DO `forward`
  (nie „grafu" — patrz niżej o batchu),
- ostatnie N pozycji: pętle własne i→i, w kolejności węzłów 0…N−1.

Ten układ nie jest umowny, tylko wymuszony przez `add_self_loops`, które robi
`cat([edge_index, arange(N).repeat(2,1)], dim=1)` — dokłada pętle NA KOŃCU,
w kolejności węzłów, nie ruszając kolejności krawędzi.

Warstwa przyjmuje pełne [E + N], a nie [E] z dopełnianiem w środku, bo test
niezmienniczości softmaxu (Krok 2 pkt 3) musi dodać stałą do WSZYSTKICH krawędzi
wchodzących do i — a pętla i→i jest jedną z nich. Sztywne zero w środku warstwy
odbiera testowi dostęp do tych pozycji. Dopełnianie jest po stronie wołającego.

PUŁAPKA WOŁAJĄCEGO — BATCH. Niezmiennik brzmi „ta sama kolejność co `edge_index`
przekazany do `forward`", a nie „co `graph.edge_index`". `tiled_edge_index`
(`core/ppo.py`) robi `edge_index.repeat(1, k)`, czyli wszystkie krawędzie grafu 0,
potem grafu 1, …, a pętle dokładają się na samym końcu dla wszystkich k·N węzłów
naraz. Zatem:

    cat([log_rho_edges.repeat(k), petle_dla_k_N])     # dobrze
    cat([pojedynczy_wektor_E_plus_N] * k)             # ŹLE — i bez błędu

Drugi wariant ma POPRAWNĄ DŁUGOŚĆ, więc przejdzie asercję niżej i przejdzie
wszystko oprócz testu, który sprawdza wartości. PyG nie ma jak tego złapać:
argumenty `propagate`, których nazwa nie kończy się na `_i`/`_j`, idą do
`message` nietknięte i niesprawdzone (sprawdzone — wektor o złej długości dociera
do `message` bez mrugnięcia okiem; dopiero broadcast bywa łaskawy i się wywala).
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
        """`log_rho` jest KEYWORD-ONLY i gwiazdka nie jest tu kosmetyką.

        `GATEncoder.forward` woła `block(h, edge_index, True)` — trzeci argument
        POZYCYJNY to `train`. Gdyby `log_rho` stanął na tej pozycji, warstwa
        dostałaby `True` zamiast tensora, a `e_i + True` dodaje 1 do WSZYSTKICH
        krawędzi. Softmax jest niezmienniczy na stałą wspólną dla grupy, więc
        wyjście byłoby co do bitu identyczne: test no-op przeszedłby na zielono,
        a modulacja nie włączyłaby się nigdy.

        `train` zostaje tam, gdzie był, i znaczy to samo co w oryginale.
        """
        if train:
            return self._forward(x, edge_index, log_rho)
        with torch.no_grad():
            return self._forward(x, edge_index, log_rho)

    @override
    def _forward(self, x, edge_index, log_rho=None):
        # PUŁAPKA `num_nodes`. `add_self_loops` woła się tu BEZ `num_nodes`
        # (tak jest w oryginale i tak musi zostać, żeby no-op był bit w bit),
        # więc N wnioskuje z `int(edge_index.max()) + 1`. Gdy najwyżej numerowany
        # węzeł jest izolowany, N wychodzi za małe: ten węzeł nie dostaje pętli
        # własnej, jego agregacja to same zera, a wyjście to `relu(bias)` —
        # stała, niezależna od jego własnej obserwacji. Kształt wyjścia pozostaje
        # POPRAWNY (`propagate` bierze `dim_size` z `x`), więc nic nie krzyczy.
        # To ten sam mechanizm co w `uncertainty/prototype.py` przy pustym
        # `edge_index`, tylko w łagodniejszym wariancie: jeden węzeł zamiast
        # całego grafu. Nie naprawiamy tego przez `num_nodes=x.size(0)`, bo to
        # zmieniłoby wynik względem oryginału i złamało no-op; zamieniamy ciszę
        # na wyjątek.
        n = maybe_num_nodes(edge_index)
        if n != x.size(0):
            raise ValueError(
                f"edge_index widzi {n} węzłów, x ma {x.size(0)} wierszy; "
                "najwyżej numerowany węzeł jest izolowany i nie dostanie pętli własnej"
            )

        edge_index, _ = add_self_loops(edge_index=edge_index)

        # Długość to JEDYNE, co da się sprawdzić maszynowo — kolejności nie.
        # Test zgodności kolejności z `edge_uncertainty` jest w Kroku 4 pkt 1.
        if log_rho is not None and log_rho.shape != (edge_index.shape[1],):
            raise ValueError(
                f"log_rho ma kształt {tuple(log_rho.shape)}, oczekiwane "
                f"({edge_index.shape[1]},) = E + N (krawędzie + pętle własne)"
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

        e_i = torch.mul(agent_repr, neighbor_repr).sum(-1)  # [5, 64] [nv, E+N]

        if log_rho is not None:
            e_i = e_i + log_rho

        max_node = torch_scatter.scatter_max(e_i, index=index)[0]  # [5, 16]
        max_i = max_node.index_select(1, index=index)  # [5, 64]
        ec_i = torch.add(e_i, -max_i)
        ecexp_i = torch.exp(ec_i)
        norm_node = torch_scatter.scatter_sum(ecexp_i, index=index)  # [5, 16]
        normst_node = torch.add(norm_node, 1e-12)  # [5, 16]
        normst_i = normst_node.index_select(1, index)  # [5, 64]

        alpha_i = ecexp_i / normst_i  # [5, 64]
        alpha_i_expand = alpha_i.repeat(self.dv, 1, 1)
        alpha_i_expand = alpha_i_expand.permute((1, 2, 0))  # [5, 64, 16]

        hidden_neighbor = F.relu(self.hidden_embedding(x_j))
        hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv)
        hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2)  # [5, 64, 16]
        out = torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0)

        # TODO: attention ouput in the future
        # self.att_list.append(alpha_i)  # [64, 16]
        return out
