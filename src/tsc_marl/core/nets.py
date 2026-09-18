"""Encoder GAT rdzenia Fazy 1 — dokładnie ta sama sieć, którą liczy CoLight.

Cała teza porównuje GAT+PPO+CTDE z Advanced-CoLight (GAT+DQN). Żeby ta różnica
mierzyła to, co ma mierzyć — algorytm uczenia, nie architekturę — encoder musi
być IDENTYCZNY z baseline'em. Dlatego nie piszemy własnych warstw, tylko składamy
`Embedding_MLP` i `MultiHeadAttModel` z `agent/colight.py`: te same wzory, ta sama
inicjalizacja, ta sama kolejność parametrów.

CO ODCINAMY OD `ColightNet` (i dlaczego, bo to jest cała robota w tym pliku):

1. GŁOWA. `ColightNet.forward` kończy `self.output_layer`, czyli głowę Q wraz
   z `MaskedOutput`. U nas głowy będą DWIE (actor: logity faz, krytyk: wartość),
   więc encoder kończy się na reprezentacji `h` i nic o zadaniu nie wie.
   Przy okazji znika `MaskedOutput`, który mnoży wyjście przez maskę — dla Q to
   tylko wygaszenie, dla polityki byłby błędem: log-softmax z wyzerowanego logitu
   nie jest zerowym prawdopodobieństwem. Maskowanie faz (gdyby było potrzebne)
   należy do głowy actora i musi być addytywne (-inf).

2. FLAGA `train`. LibSignal przenosi ją przez cały forward i przy `False` owija
   go w `torch.no_grad()`. W DQN nieszkodliwe, w PPO zabójcze: update liczy
   log-proby tą samą siecią i potrzebuje przez nią gradientu — jedno wywołanie
   z `train=False` w złym miejscu daje politykę, która się nie uczy, BEZ ŻADNEGO
   BŁĘDU. Sterowanie gradientem zostaje u wołającego (`torch.no_grad()` w
   rolloutcie), a podmodułom LibSignala podajemy na sztywno `True`. Nazwa i tak
   kolidowała z `nn.Module.train()`/`self.training`, które znaczą co innego.

3. WOREK KWARGÓW. `ColightNet` czyta hiperparametry z `self.model_dict`
   wypełnianego z YAML-a. Tutaj są jawnymi argumentami: krytyk (1.4) ma mieć
   `n_layers=2` przy tym samym actorze, więc to musi być parametr konstruktora,
   a nie wpis w globalnej konfiguracji. `from_registry()` zostaje jako most do
   `colight.yml` — żeby prawdziwy bieg czytał te same liczby co baseline.

KSZTAŁT WEJŚCIA: `x` to [N, ob_length] — JEDEN graf, węzły w kolejności
`RoadNetGraph.node_idx`. Batch po czasie robi się przez `torch_geometric.Batch`
(blokowo-diagonalny wielki graf, `edge_index` z przesuniętymi indeksami), NIE
przez trzeci wymiar: `MultiHeadAttModel.message` czyta `x_i.shape[:-1][0]` jako
liczbę krawędzi, więc tensor [B, N, F] policzyłby bzdury albo wywalił się głęboko
w `view()`. Stąd jawna asercja w `forward`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Import na górze modułu: `nets` jest un-importowalny bez `libsignal.bootstrap()`.
# Tak samo zachowuje się `features` (przez `tsc_marl.libsignal.agents`), więc to spójne
# z resztą repo; samodzielny zostaje tylko `graph`.
import torch
from agent.colight import Embedding_MLP
from torch import Tensor, nn
from torch.distributions import Categorical
from torch_geometric.nn import SAGEConv, global_mean_pool

from tsc_marl.core.uw_attention import UncertaintyWeightedAttention

# Domyślne = `configs/tsc/colight.yml`, czyli konfiguracja, na której zmierzyliśmy
# baseline. Powtórzone tutaj, żeby encoder dało się zbudować w teście bez Registry.
DEFAULT_EMB_DIMS = (128, 128)
DEFAULT_INPUT_DIM = (128, 128)
DEFAULT_OUTPUT_DIM = (128, 128)
DEFAULT_DIMS_EACH_HEAD = (16, 16)
DEFAULT_NUM_HEADS = (5, 5)


class GATEncoder(nn.Module):
    """`Embedding_MLP` + `n_layers` × `MultiHeadAttModel`, prosto z LibSignala.

    [N, ob_length] → [N, out_dim]. Jeden komplet wag dla wszystkich skrzyżowań
    (index-free parameter sharing z CoLighta) — sieć nie wie, który to węzeł,
    więc rozmiar sieci drogowej nie zmienia liczby parametrów, a wagi przenoszą
    się na inną topologię (Faza 5).

    Każda warstwa attention to jeden skok w grafie: `n_layers=1` widzi sąsiadów
    (1-hop), `n_layers=2` sąsiadów sąsiadów. Krytyk CTDE dostanie 2 — szerszy
    kontekst wolno mu mieć, bo żyje tylko w treningu.
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
        # nn.ModuleList, NIE zwykła lista: zwykła nie rejestruje podmodułów, więc
        # ich parametry nie trafiają do `.parameters()`, optymalizator ich nie
        # widzi i sieć uczy się samym embeddingiem — cicho, bez błędu.
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
        """Wymiary sprawdzamy w konstruktorze, bo niedopasowanie list per-warstwa
        objawia się dopiero jako `IndexError` przy budowie krytyka albo — gorzej —
        jako `RuntimeError` z mnożenia macierzy w środku pierwszego forwardu."""
        if ob_length <= 0:
            raise ValueError(f"ob_length musi być dodatnie, jest {ob_length}")
        if not emb_dims:
            raise ValueError("emb_dims nie może być puste — to warstwy MLP embeddingu")
        if n_layers < 1:
            raise ValueError(f"n_layers musi być >= 1, jest {n_layers}")

        per_layer = {
            "INPUT_DIM": input_dim,
            "OUTPUT_DIM": output_dim,
            "NUM_HEADS": num_heads,
            "NODE_LAYER_DIMS_EACH_HEAD": dims_each_head,
        }
        for name, seq in per_layer.items():
            if len(seq) < n_layers:
                raise ValueError(
                    f"{name} ma {len(seq)} pozycji, a n_layers={n_layers} — "
                    f"brakuje wartości dla warstwy {n_layers - 1}"
                )

        if int(input_dim[0]) != int(emb_dims[-1]):
            raise ValueError(f"INPUT_DIM[0]={input_dim[0]} != wyjście embeddingu {emb_dims[-1]}")
        for i in range(1, n_layers):
            if int(input_dim[i]) != int(output_dim[i - 1]):
                raise ValueError(
                    f"warstwy GAT się nie łączą: INPUT_DIM[{i}]={input_dim[i]} != "
                    f"OUTPUT_DIM[{i - 1}]={output_dim[i - 1]}"
                )

    @staticmethod
    def registry_kwargs() -> dict[str, Any]:
        """Hiperparametry encodera z YAML-a LibSignala — ten sam wzorzec co
        `StateBuilder.from_registry`. Osobno od `from_registry`, bo `Actor`
        i `Critic` też budują encoder i mają czytać DOKŁADNIE ten sam config."""
        from common.registry import Registry  # noqa: PLC0415 - dopiero po bootstrapie

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
        """`ob_length` zostaje argumentem, bo nie ma go w configu: liczy go
        `StateBuilder.ob_length` z definicji stanu (24 przy 8 fazach i ATS)."""
        return cls(ob_length, **{**cls.registry_kwargs(), **overrides})

    def forward(
        self, x: Tensor, edge_index: Tensor, *, log_rho: Tensor | None = None
    ) -> Tensor:  # [N, F] -> [N, out_dim]
        """`log_rho` KEYWORD-ONLY we wszystkich trzech encoderach i w `Actor` —
        ten sam powód co w `UncertaintyWeightedAttention.forward`, plus lokalny:
        `Critic.forward` ma na trzeciej pozycji `batch` (przypisanie węzeł→graf).
        Oba są `Tensor | None`, więc zamiana nie rzuciłaby wyjątkiem, tylko
        policzyła kontekst z wektora wiarygodności."""
        if x.dim() != 2:
            raise ValueError(
                f"GATEncoder przyjmuje [N, ob_length], dostał {tuple(x.shape)}; "
                "batch po czasie sklej przez torch_geometric.Batch, nie trzecim wymiarem"
            )
        if x.shape[1] != self.ob_length:
            raise ValueError(f"szerokość obserwacji {x.shape[1]} != ob_length {self.ob_length}")

        h = self.embedding_mlp(x, True)  # `True` zawsze — patrz decyzja 2 w docstringu
        for block in self.blocks:
            h = block(h, edge_index, True, log_rho=log_rho)
        return h

    def extra_repr(self) -> str:
        return f"ob_length={self.ob_length}, n_layers={self.n_layers}, out_dim={self.out_dim}"


#: Warianty agregacji sąsiadów w ACTORZE — ablacja 1.6. Krytyk zostaje zawsze GAT-em:
#: ablacja pyta, czy DECYZJA korzysta na ważeniu sąsiadów, a nie czy uczy się jej
#: lepszy krytyk. Zmiana krytyka w tym samym biegu pomieszałaby dwie przyczyny.
ENCODERS = ("gat", "mean", "none")


class _AblationEncoder(nn.Module):
    """Wspólny szkielet wariantów ablacyjnych: TEN SAM `Embedding_MLP` i te same
    wymiary co `GATEncoder`, różni się wyłącznie tym, co warstwa robi z sąsiadami.
    Bez tego ablacja porównywałaby też głębokość i szerokość sieci."""

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

    # NIE `_apply`: to nazwa wewnętrznej metody `nn.Module`, przez którą idzie
    # `.to(device)` — nadpisana wywalała budowę rdzenia na pierwszym `.to()`.

    def _aggregate(self, block: nn.Module, h: Tensor, edge_index: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, edge_index: Tensor, *, log_rho: Tensor | None = None) -> Tensor:
        if x.dim() != 2 or x.shape[1] != self.ob_length:
            raise ValueError(f"oczekiwane [N, {self.ob_length}], jest {tuple(x.shape)}")
        # `raise`, nie ciche ignorowanie: `mean` waży każdego sąsiada tak samo,
        # `none` nie widzi sąsiadów w ogóle, więc nie ma czego modulować. Gdyby
        # wariant argument przyjął i wyrzucił, bieg „UW-attention + mean-pool"
        # doszedłby do końca i dał liczby nieodróżnialne od wyniku.
        if log_rho is not None:
            raise ValueError(
                f"{type(self).__name__} nie ma wag attention do modulowania — "
                "UW-attention ma sens tylko przy encoder='gat'"
            )
        h = self.embedding_mlp(x, True)
        for block in self.blocks:
            h = torch.relu(self._aggregate(block, h, edge_index))
        return h

    def extra_repr(self) -> str:
        return f"ob_length={self.ob_length}, n_layers={self.n_layers}, out_dim={self.out_dim}"


class MeanPoolEncoder(_AblationEncoder):
    """GraphSAGE-mean: h_i' = W₁h_i + W₂·mean_{j∈N(i)} h_j. Każdy sąsiad waży TYLE
    SAMO — dokładnie to, czego attention ma nie robić. Ablacja „GAT vs mean-pool"
    ([[zhou-2025-gmslight]]): czy uczone ważenie sąsiadów w ogóle coś daje."""

    def _block(self, d_in: int, d_out: int) -> nn.Module:
        return SAGEConv(d_in, d_out, aggr="mean")

    def _aggregate(self, block: nn.Module, h: Tensor, edge_index: Tensor) -> Tensor:
        return block(h, edge_index)


class LocalEncoder(_AblationEncoder):
    """Bez sąsiadów: MLP tej samej głębokości, `edge_index` ignorowany. Ablacja
    „po co GNN" ([[chen-2020-mplight]]): agent widzi wyłącznie własne skrzyżowanie,
    parameter sharing zostaje (jak w MPLight)."""

    def _block(self, d_in: int, d_out: int) -> nn.Module:
        return nn.Linear(d_in, d_out)

    def _aggregate(self, block: nn.Module, h: Tensor, edge_index: Tensor) -> Tensor:
        return block(h)


def build_encoder(kind: str, ob_length: int, **kwargs: Any) -> nn.Module:
    """`gat` | `mean` | `none` -> encoder [N, ob_length] -> [N, out_dim]."""
    classes = {"gat": GATEncoder, "mean": MeanPoolEncoder, "none": LocalEncoder}
    if kind not in classes:
        raise ValueError(f"encoder={kind!r}, dozwolone: {ENCODERS}")
    return classes[kind](ob_length, **kwargs)


def encoder_kind() -> str:
    """Wariant actora z configu (`model.encoder`). Domyślnie `gat` i BEZ wpisu
    w `gatppo.yml`: dopisanie klucza zmieniłoby odcisk configu i unieważniło
    policzone biegi GATE-a. Ablacje ustawiają go przez `overrides` suite'a."""
    from common.registry import Registry  # noqa: PLC0415 - dopiero po bootstrapie

    return Registry.mapping["model_mapping"]["setting"].param.get("encoder", "gat")


def orthogonal_(layer: nn.Linear, gain: float) -> nn.Linear:
    """Inicjalizacja ortogonalna + zerowy bias — konwencja PPO/MAPPO.

    Dotyczy WYŁĄCZNIE naszych głów. Encoder zostaje na domyślnej inicjalizacji
    PyTorcha, bo taki jest w `agent/colight.py`: gdybyśmy go przeinicjalizowali,
    różnica w GATE przestałaby być czystą różnicą „DQN vs PPO". To jest decyzja,
    nie przeoczenie — konwencja MAPPO każe robić ortogonalną wszędzie.
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    """Polityka: [N, ob_length] → LOGITY faz [N, n_phases].

    Sama nie próbkuje, nie robi argmaxu i nie liczy strat — to należy do
    `ActorCritic` i do pętli PPO. Tutaj jest tylko odwzorowanie obserwacji na
    rozkład, żeby dało się je przetestować bez żadnej maszynerii RL.

    ZWRACA LOGITY, NIE PRAWDOPODOBIEŃSTWA. `Categorical(logits=...)` robi
    log-softmax w środku, stabilnie numerycznie; podanie mu wartości już
    przepuszczonych przez softmax daje rozkład zbyt płaski i zawyżoną entropię —
    polityka uczy się wolniej i nic tego nie sygnalizuje.

    MASKOWANIA FAZ TU NIE MA i na razie nie może być: w obu sieciach protokołu
    każde skrzyżowanie ma dokładnie 8 faz (sprawdzone w `test_graph.py`), więc
    maska byłaby macierzą jedynek. Gdy przyjdzie sieć niejednorodna (Faza 5),
    wchodzi ADDYTYWNIE — `logits.masked_fill(~mask, -inf)`. Nigdy przez mnożenie
    jak `MaskedOutput` z LibSignala: wyzerowany logit to `e⁰/Σ`, czyli zwykle
    NAJWIĘKSZE prawdopodobieństwo w rozkładzie. Dla Q-wartości to wygaszenie,
    dla polityki odwrotność zamierzonego efektu.
    """

    def __init__(
        self,
        ob_length: int,
        n_phases: int,
        *,
        hidden: int | None = None,
        encoder: str = "gat",
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()  # MUSI być pierwsze: bez tego nn.Module nie ma jeszcze
        # `_parameters`/`_modules` i każde przypisanie podmodułu leci wyjątkiem.
        if n_phases < 2:
            raise ValueError(f"n_phases musi być >= 2 (rozkład nad fazami), jest {n_phases}")

        self.ob_length = ob_length
        self.n_phases = n_phases
        self.encoder_kind = encoder
        # nazwa atrybutu `encoder` zostaje dla KAŻDEGO wariantu — klucze state_dict
        # (`encoder.embedding_mlp...`) muszą się zgadzać z checkpointami GATE-a
        self.encoder = build_encoder(encoder, ob_length, **encoder_kwargs)

        # Domyślnie głowa jest jedną warstwą liniową, bo CoLight ma dokładnie tyle
        # (`OUTPUT_LAYERS: []` w colight.yml). Dzięki temu jedyną różnicą wobec
        # baseline'u zostaje to, co ma nią być: sposób uczenia. `hidden` to slot
        # ablacyjny, nie domyślna głębokość.
        head: list[nn.Module] = []
        in_dim = self.encoder.out_dim
        if hidden is not None:
            head += [orthogonal_(nn.Linear(in_dim, hidden), gain=2**0.5), nn.ReLU()]
            in_dim = hidden
        # gain=0.01 na OSTATNIEJ warstwie: przy domyślnej inicjalizacji logity
        # startują z rozrzutem rzędu jedności, czyli polityka od pierwszego kroku
        # ma losową, ale zdecydowaną preferencję faz i PPO traci pierwsze tysiące
        # kroków na jej odkręcanie. Z 0.01 logity startują ≈ 0, rozkład jest
        # prawie jednostajny, a entropia startowa ≈ log(n_phases) — logujmy ją,
        # to najtańszy sanity check, że inicjalizacja zadziałała.
        head.append(orthogonal_(nn.Linear(in_dim, n_phases), gain=0.01))
        self.head = nn.Sequential(*head)

    @classmethod
    def from_registry(cls, ob_length: int, n_phases: int, **overrides: Any) -> Actor:
        """Encoder z `colight.yml`, kształt głowy z argumentów.

        `n_phases` nie idzie z configu celowo: liczbę faz ma pilnować graf
        (`StateBuilder.n_phases`), a nie YAML, w którym łatwo zostawić wartość
        z innej sieci.
        """
        kwargs = {"encoder": encoder_kind(), **GATEncoder.registry_kwargs()}
        return cls(ob_length, n_phases, **{**kwargs, **overrides})

    def forward(
        self, x: Tensor, edge_index: Tensor, *, log_rho: Tensor | None = None
    ) -> Tensor:  # [N, F] -> [N, n_phases]
        return self.head(self.encoder(x, edge_index, log_rho=log_rho))

    def extra_repr(self) -> str:
        return f"ob_length={self.ob_length}, n_phases={self.n_phases}, encoder={self.encoder_kind}"


class Critic(nn.Module):
    """Krytyk CTDE, faktoryzowany: [N, ob_length] -> V_i [N] — wartość NA AGENTA.

    DLACZEGO FAKTORYZOWANY (decyzja §1.3). Wariant globalny zwraca jedną liczbę
    na krok, więc wszyscy agenci dostają tę samą przewagę policzoną ze wspólnej
    nagrody. Przy 12–16 skrzyżowaniach sygnał uczący jednego z nich jest wtedy
    w >90% szumem od pozostałych i credit assignment przestaje działać: agent
    dostaje kredyt za cudzy sukces. Faktoryzowany liczy GAE osobno z r_i, więc
    wariancja gradientu nie rośnie z rozmiarem sieci. Wariant globalny zostaje
    slotem ablacyjnym — różni się kształtem wyjścia i niczym więcej.

    CO ROBI Z NIEGO CTDE. Nie skalarność wyjścia, tylko to, czego krytykowi
    wolno użyć, a actorowi nie: (a) `n_layers=2`, czyli zasięg 2-hop zamiast
    1-hop, (b) globalny kontekst z mean-poolingu po całej sieci. Actor w
    inferencji ma wyłącznie o_i i sąsiadów; krytyk żyje tylko w treningu, więc
    może widzieć więcej. Wagi są OSOBNE od actora — współdzielony encoder w
    MAPPO regularnie destabilizuje, bo gradient straty wartości ma inną skalę
    niż policy gradient i zaczyna dominować wspólną część.

    CZEGO TU NIE MA, RAZ: normalizacji zwrotów. ATS jest nieznormalizowany
    (pressure bywa rzędu dziesiątek), więc V ma dużą skalę i strata wartości
    potrafi zdominować policy loss. Lekarstwo (bieżąca normalizacja zwrotów /
    PopArt) należy do pętli PPO w 1.4 — ten moduł zostaje bezstanowy.

    CZEGO TU NIE MA, DWA: `log_rho`. Brak tego argumentu jest DECYZJĄ, nie
    przeoczeniem — `forward` woła `self.encoder(x, edge_index)` świadomie.
    UW-attention wygasza sąsiadów, których czujniki padły, a to zdarza się we
    WDROŻENIU; krytyk żyje wyłącznie w treningu, gdzie odczyty są czyste, więc
    nie ma czego modulować. Co więcej, podanie mu wiarygodności byłoby wyciekiem
    od tej gorszej strony CTDE: krytyk uczyłby wartość z informacji, której actor
    w inferencji nie ma, a od tego zależy poprawność przewag. Modulacja wchodzi
    w `Actor` i kończy się na nim — patrz `robustness/controllers.py`.
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

        # Wejście głowy to [reprezentacja węzła ‖ kontekst całej sieci]. Sam
        # kontekst nie wystarczy (wszystkie węzły miałyby jedną wartość, czyli
        # krytyk globalny w przebraniu), a sama reprezentacja lokalna nie jest
        # CTDE — V_i ma znać sytuację poza zasięgiem 2-hop.
        self.head = nn.Sequential(
            orthogonal_(nn.Linear(2 * self.encoder.out_dim, hidden), gain=2**0.5),
            nn.ReLU(),
            # gain=1.0, nie 0.01 jak w actorze: tam chodziło o start przy rozkładzie
            # jednostajnym, tu chcemy, żeby V od razu miało sensowną skalę.
            orthogonal_(nn.Linear(hidden, 1), gain=1.0),
        )

    @classmethod
    def from_registry(cls, ob_length: int, **overrides: Any) -> Critic:
        """Encoder z `colight.yml`; `n_layers` NIE stamtąd — config trzyma wartość
        actora (N_LAYERS: 1), a krytyk ma mieć własną."""
        kwargs = GATEncoder.registry_kwargs()
        kwargs.pop("n_layers", None)
        return cls(ob_length, **{**kwargs, **overrides})

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        """[N, ob_length] -> [N]. `batch` to przypisanie węzeł→graf z `Batch.batch`.

        `batch` MUSI tu być, gdy sklejasz kroki czasu. Naiwne `h.mean(0)` liczy
        średnią po WSZYSTKICH węzłach batcha, czyli miesza 64 różne momenty w
        jeden kontekst: krytyk zaczyna widzieć przyszłość, wartości wychodzą
        podejrzanie dobre i nic tego nie zgłasza.
        """
        h = self.encoder(x, edge_index)
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        elif batch.shape != (h.shape[0],):
            raise ValueError(f"batch musi mieć kształt [{h.shape[0]}], ma {tuple(batch.shape)}")
        context = global_mean_pool(h, batch)  # [B, out_dim]
        return self.head(torch.cat([h, context[batch]], dim=-1)).squeeze(-1)

    def extra_repr(self) -> str:
        return f"ob_length={self.ob_length}, n_layers={self.encoder.n_layers}"


class ActorCritic(nn.Module):
    """Actor + krytyk pod jednym interfejsem, którego używa pętla PPO (1.4).

    Trzyma DWA niezależne komplety wag — `Actor` ma własny encoder 1-hop,
    `Critic` własny 2-hop. Sprawdza to `test_nets.py`, bo przypadkowe
    współdzielenie encodera nie wywala się, tylko psuje trening.

    GRADIENTU NIE WYŁĄCZAMY W ŚRODKU. `torch.no_grad()` obudowuje WYWOŁANIE
    `act()` w rolloutcie; `evaluate()` musi przepuszczać gradient, bo przez nią
    idzie cała strata polityki. To ta sama decyzja co wyrzucenie flagi `train`
    z `GATEncoder` — musi być konsekwentna, inaczej PPO cicho przestaje się uczyć.
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
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()
        critic_kwargs = {k: v for k, v in encoder_kwargs.items() if k != "n_layers"}
        self.actor = Actor(ob_length, n_phases, hidden=hidden, encoder=encoder, **encoder_kwargs)
        # krytyk ZAWSZE GAT — patrz komentarz przy ENCODERS
        self.critic = Critic(
            ob_length, hidden=critic_hidden, n_layers=critic_n_layers, **critic_kwargs
        )

    @classmethod
    def from_registry(cls, ob_length: int, n_phases: int, **overrides: Any) -> ActorCritic:
        kwargs = {"encoder": encoder_kind(), **GATEncoder.registry_kwargs()}
        return cls(ob_length, n_phases, **{**kwargs, **overrides})

    def distribution(self, x: Tensor, edge_index: Tensor) -> Categorical:
        """Rozkład nad fazami. Każdy WIERSZ to osobny agent — przy współdzielonych
        wagach 12 skrzyżowań × T kroków to po prostu 12·T próbek w batchu."""
        return Categorical(logits=self.actor(x, edge_index))

    def act(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        deterministic: bool = False,
        batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Rollout: -> (akcje [N], log_prob [N], wartości [N]).

        `deterministic=True` to argmax i tryb EWALUACJI (kolumna T1
        „GAT+PPO+CTDE (deterministic)"); w treningu zawsze próbkujemy, bo
        eksploracja w PPO bierze się właśnie stąd, nie z epsilon-greedy.
        """
        dist = self.distribution(x, edge_index)
        actions = dist.logits.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), self.critic(x, edge_index, batch)

    def evaluate(
        self,
        x: Tensor,
        edge_index: Tensor,
        actions: Tensor,
        batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update: -> (log_prob [N], entropia [N], wartości [N]) dla ZADANYCH akcji.

        W pierwszej epoce PPO `exp(log_prob - log_prob_stary)` musi wyjść dokładnie
        1.0. Jeśli nie wychodzi, to nie jest problem uczenia, tylko rozjazd między
        ścieżką `act` a `evaluate` — pilnuje tego test.
        """
        dist = self.distribution(x, edge_index)
        return dist.log_prob(actions), dist.entropy(), self.critic(x, edge_index, batch)

    def value(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        """Sama wartość — GAE potrzebuje V(s_T) kroku za końcem rolloutu."""
        return self.critic(x, edge_index, batch)
