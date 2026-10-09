"""Testy encodera GAT.

Sieć neuronowa prawie nigdy nie wywala się na błędzie strukturalnym — po prostu
uczy się gorzej. Te testy celują w trzy takie ciche awarie:

1. warstwy attention w zwykłej liście zamiast `nn.ModuleList` → optymalizator ich
   nie widzi i trenuje się sam embedding (test gradientów);
2. encoder zależny od NUMERU węzła zamiast od jego sąsiedztwa → wagi nie
   przeniosą się na inną topologię (test równoważności permutacyjnej);
3. `edge_index` sklejonego batcha mieszający grafy z różnych kroków czasu →
   agent „widzi" skrzyżowanie z innej próbki (test batcha PyG).

Nie potrzebują silnika: graf czytamy z pliku roadnetu, obserwacje losujemy.
"""

from __future__ import annotations

import math

import pytest
import torch
from agent.colight import MultiHeadAttModel
from torch_geometric.data import Batch, Data

from tsc_marl import paths
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.core.nets import Actor, ActorCritic, Critic, GATEncoder

ROADNET = paths.DATA / "raw_data" / "jinan_3x4" / "roadnet_3_4.json"

# Szerokość wiersza obserwacji ATS przy 8 fazach: 8 pressure + 8 demand + 8 one-hot.
OB_LENGTH = 24


@pytest.fixture(scope="module")
def graph():
    if not ROADNET.is_file():
        pytest.skip(f"brak roadnetu {ROADNET} (dane pobiera się osobno — patrz data/README.md)")
    return RoadNetGraph.from_roadnet(ROADNET)


def make(n_layers: int = 1, seed: int = 0) -> GATEncoder:
    torch.manual_seed(seed)
    return GATEncoder(OB_LENGTH, n_layers=n_layers).eval()


def observations(g, seed: int = 1) -> torch.Tensor:
    return torch.randn(g.n_nodes, OB_LENGTH, generator=torch.Generator().manual_seed(seed))


def test_ksztalt_wyjscia(graph):
    """Kontrakt [N, ob_length] → [N, out_dim]. `out_dim` bierze się z OUTPUT_DIM
    OSTATNIEJ użytej warstwy, nie z długości listy w configu."""
    enc = make()
    out = enc(observations(graph), graph.edge_index)
    assert out.shape == (graph.n_nodes, 128)
    assert enc.out_dim == 128


def test_gradient_dochodzi_do_kazdego_parametru(graph):
    """Gdyby warstwy attention siedziały w zwykłej liście, nie byłoby ich w
    `.parameters()`: optymalizator trenowałby sam embedding, a sieć dawała po
    prostu słabszy wynik — bez błędu i bez ostrzeżenia."""
    enc = make(n_layers=2)
    names = [n for n, _ in enc.named_parameters()]
    assert any("blocks.0" in n for n in names) and any("blocks.1" in n for n in names)

    enc(observations(graph), graph.edge_index).sum().backward()
    bez_gradientu = [n for n, p in enc.named_parameters() if p.grad is None or not p.grad.any()]
    assert bez_gradientu == []


def test_rownowaznosc_permutacyjna(graph):
    """Index-free parameter sharing: encoder nie ma prawa wiedzieć, który to
    węzeł. Przenumerowanie skrzyżowań musi przenumerować wyjście i nic więcej —
    inaczej wagi nauczone na JiNan nie przeniosą się na HangZhou (Faza 5),
    a założenie „jeden model dla wszystkich agentów" jest fikcją.
    """
    enc = make()
    x, ei = observations(graph), graph.edge_index
    out = enc(x, ei)

    perm = torch.randperm(graph.n_nodes, generator=torch.Generator().manual_seed(7))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(graph.n_nodes)  # stary indeks -> nowy

    out_perm = enc(x[perm], inverse[ei])
    assert torch.allclose(out_perm, out[perm], atol=1e-5)


def test_zasieg_rosnie_z_liczba_warstw(graph):
    """Jedna warstwa = 1 skok, dwie = 2 skoki. To jest cały powód, dla którego
    krytyk CTDE dostanie `n_layers=2`: ma widzieć dalej niż actor."""
    center = max(range(graph.n_nodes), key=lambda i: len(graph.neighbors[i]))
    one_hop = set(graph.neighbors[center])
    two_hop = {j for i in one_hop for j in graph.neighbors[i]} - one_hop
    assert two_hop, "siatka bez węzła w odległości 2 — test straciłby sens"
    daleki = min(two_hop)

    x, ei = observations(graph), graph.edge_index
    zaburzony = x.clone()
    zaburzony[daleki] += 5.0

    plytki = make(n_layers=1)
    assert torch.allclose(plytki(x, ei)[center], plytki(zaburzony, ei)[center], atol=1e-6)

    gleboki = make(n_layers=2)
    assert not torch.allclose(gleboki(x, ei)[center], gleboki(zaburzony, ei)[center], atol=1e-6)


def test_batch_pyg_nie_miesza_grafow(graph):
    """Batch po czasie sklejamy w jeden blokowo-diagonalny graf. Test pilnuje, że
    przesunięcie indeksów robi `Batch`, a nie my ręcznie: dwie próbki policzone
    razem muszą dać to samo co policzone osobno."""
    enc = make()
    x1, x2 = observations(graph, seed=1), observations(graph, seed=2)
    osobno = torch.cat([enc(x1, graph.edge_index), enc(x2, graph.edge_index)])

    batch = Batch.from_data_list([Data(x=x, edge_index=graph.edge_index) for x in (x1, x2)])
    assert torch.allclose(enc(batch.x, batch.edge_index), osobno, atol=1e-5)


# ------------------------------------------------------------------ UW-attention


def test_bez_log_rho_bit_w_bit_jak_oryginal(graph):
    """Encoder bez log ρ musi liczyć DOKŁADNIE to, co liczył przed Krokiem 1.

    `test_uw_attention.test_no_op` pilnuje tego dla jednej warstwy w izolacji;
    tutaj chodzi o piętro wyżej: `GATEncoder` woła `block(h, edge_index, True,
    log_rho=log_rho)`, więc gdyby `log_rho` trafiło na trzecią pozycję zamiast
    `train`, warstwa dostałaby `True` — czyli +1 do WSZYSTKICH logitów. Softmax
    jest na wspólną stałą niezmienniczy, więc wyjście byłoby identyczne, a
    modulacja nie włączyłaby się nigdy (pułapka nr 2 z Kroku 1). Ten test tego
    sam nie złapie — łapie go `test_log_rho_dociera_do_warstwy_przez_actora`
    niżej — ale razem zamykają obie strony: bez log ρ nic się nie zmieniło,
    z log ρ zmienia się dokładnie to, co ma.

    `n_layers=2`, bo dla jednej warstwy pętla po `self.blocks` nie ma jak się
    pomylić.
    """
    torch.manual_seed(0)
    enc = GATEncoder(OB_LENGTH, n_layers=2).eval()
    x = observations(graph)

    # Referencja: te same wagi, ale w ORYGINALNEJ warstwie LibSignala.
    h = enc.embedding_mlp(x, True)
    for block in enc.blocks:
        ref = MultiHeadAttModel(d=block.d, dv=block.dv, d_out=block.d_out, nv=block.nv, suffix=0)
        ref.load_state_dict(block.state_dict())  # strict=True — klucze muszą się zgadzać
        h = ref.eval()(h, graph.edge_index, True)

    assert torch.equal(enc(x, graph.edge_index), h)


def test_log_rho_dociera_do_warstwy_przez_actora(graph):
    """Wygaszenie z Kroku 2, ale mierzone na wyjściu `Actor`.

    Warstwa ma własny komplet testów, więc ten łapie dokładnie jedno: że log ρ
    przeżyło drogę `Actor.forward` → `GATEncoder.forward` → `block(...)`.
    Zgubienie go po drodze (choćby przez `self.encoder(x, edge_index)` bez
    argumentu) nie wywala niczego — polityka liczy się dalej, tyle że modulacja
    nie działa, a T2 mierzy rdzeń przebrany za UW-attention.
    """
    torch.manual_seed(0)
    actor = Actor(OB_LENGTH, 8).eval()  # 1-hop, jak w rdzeniu
    ei = graph.edge_index

    j, i = ei[:, 3].tolist()
    x = observations(graph)
    x_zmienione = x.clone()
    x_zmienione[j] = observations(graph, seed=2)[j]

    log_rho = torch.zeros(ei.shape[1] + graph.n_nodes)
    log_rho[3] = -1e9
    assert torch.equal(actor(x, ei, log_rho=log_rho)[i], actor(x_zmienione, ei, log_rho=log_rho)[i])

    # Kontrola: bez log ρ ten sam wiersz REAGUJE na sąsiada (inaczej równość wyżej
    # dowodziłaby tylko tego, że i nigdy nie patrzyło na j).
    assert not torch.equal(actor(x, ei)[i], actor(x_zmienione, ei)[i])


def test_trzeci_wymiar_odrzucony(graph):
    """[B, N, F] przeszłoby przez embedding i dopiero w `message()` policzyło
    bzdury (LibSignal czyta stamtąd liczbę KRAWĘDZI). Lepiej głośno i od razu."""
    enc = make()
    with pytest.raises(ValueError, match="Batch"):
        enc(observations(graph).unsqueeze(0), graph.edge_index)
    with pytest.raises(ValueError, match="ob_length"):
        enc(torch.randn(graph.n_nodes, OB_LENGTH + 1), graph.edge_index)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_layers": 3}, "n_layers=3"),  # listy w colight.yml są 2-elementowe
        ({"emb_dims": (64, 64)}, "wyjście embeddingu"),
        ({"n_layers": 2, "output_dim": (64, 128)}, "nie łączą"),
        ({"n_layers": 0}, "n_layers"),
    ],
)
def test_odrzuca_niespojna_konfiguracje(kwargs, match):
    """Rozjechane wymiary per-warstwa dają albo `IndexError` dopiero przy budowie
    krytyka, albo `RuntimeError` z mnożenia macierzy w środku forwardu. Oba
    komunikaty nie mówią, który klucz w YAML-u jest zły."""
    with pytest.raises(ValueError, match=match):
        GATEncoder(OB_LENGTH, **kwargs)


# ------------------------------------------------------- warianty ablacyjne


@pytest.mark.parametrize("kind", ["gat", "mean", "none"])
def test_warianty_encodera_ten_sam_ksztalt_i_gradient(graph, kind):
    """Ablacja ma porównywać AGREGACJĘ, nie rozmiar sieci: ten sam kontrakt
    [N, ob_length] -> [N, 128] i gradient do każdego parametru."""
    from tsc_marl.core.nets import build_encoder

    torch.manual_seed(0)
    enc = build_encoder(kind, OB_LENGTH)
    out = enc(observations(graph), graph.edge_index)
    assert out.shape == (graph.n_nodes, 128)
    out.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())


@pytest.mark.parametrize("kind", ["gat", "mean", "none"])
def test_warianty_encodera_rownowaznosc_permutacyjna(graph, kind):
    from tsc_marl.core.nets import build_encoder

    torch.manual_seed(0)
    enc = build_encoder(kind, OB_LENGTH).eval()
    x = observations(graph)
    perm = torch.randperm(graph.n_nodes, generator=torch.Generator().manual_seed(3))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(graph.n_nodes)
    ei = inv[graph.edge_index]
    assert torch.allclose(enc(x, graph.edge_index)[perm], enc(x[perm], ei), atol=1e-5)


def test_none_nie_widzi_sasiadow_a_mean_widzi(graph):
    """To, co ablacja ma rozdzielić: `none` zależy WYŁĄCZNIE od własnego wiersza,
    `mean` i `gat` zmieniają wyjście, gdy zmieni się sąsiad."""
    from tsc_marl.core.nets import build_encoder

    x = observations(graph)
    src, dst = graph.edge_index[:, 0]
    x2 = x.clone()
    x2[src] += 5.0
    for kind, powinno in (("none", False), ("mean", True), ("gat", True)):
        torch.manual_seed(0)
        enc = build_encoder(kind, OB_LENGTH).eval()
        zmiana = not torch.allclose(enc(x, graph.edge_index)[dst], enc(x2, graph.edge_index)[dst])
        assert zmiana == powinno, kind


@pytest.mark.parametrize("kind", ["gat", "mean", "none"])
def test_actor_dziala_w_kazdym_wariancie_encodera(graph, kind):
    """Regresja: `Actor.forward` przekazuje `log_rho` do encodera BEZWARUNKOWO,
    więc wariant, który tego argumentu nie przyjmuje, wywala się `TypeError` przy
    KAŻDYM wywołaniu — także bez log ρ. Ablacja 1.6 przestaje istnieć.

    Pozostałe testy wariantów wołają `build_encoder` wprost i omijają `Actor`,
    czyli dokładnie tę ścieżkę, na której to pęka."""
    torch.manual_seed(0)
    actor = Actor(OB_LENGTH, 8, encoder=kind).eval()
    assert actor(observations(graph), graph.edge_index).shape == (graph.n_nodes, 8)


@pytest.mark.parametrize("kind", ["mean", "none"])
def test_warianty_bez_attention_odrzucaja_log_rho(graph, kind):
    """Odmowa ma być GŁOŚNA. `mean` waży sąsiadów po równo, `none` ich nie widzi —
    nie ma tam wag attention do modulowania. Gdyby wariant przyjął log ρ i je
    wyrzucił, bieg „UW-attention + mean-pool" policzyłby się do końca i dał
    liczby wyglądające jak wynik ablacji."""
    torch.manual_seed(0)
    actor = Actor(OB_LENGTH, 8, encoder=kind).eval()
    log_rho = torch.zeros(graph.edge_index.shape[1] + graph.n_nodes)
    with pytest.raises(ValueError, match="attention"):
        actor(observations(graph), graph.edge_index, log_rho=log_rho)


@pytest.mark.parametrize("kind", ["gat", "mean", "none"])
def test_warianty_przezywaja_to_device(graph, kind):
    """Regresja: metoda wariantu nazwana `_apply` nadpisała `nn.Module._apply`, przez
    które idzie `.to(device)` — `CoreTrainer.from_world` padał na pierwszej linii,
    a testy wyjścia i gradientu przechodziły, bo nie przenosiły sieci."""
    ac = ActorCritic(OB_LENGTH, 8, encoder=kind).to("cpu").to(torch.float32)
    assert ac.actor(observations(graph), graph.edge_index).shape == (graph.n_nodes, 8)


def test_nieznany_encoder():
    from tsc_marl.core.nets import build_encoder

    with pytest.raises(ValueError, match="encoder"):
        build_encoder("rakieta", OB_LENGTH)


def test_actorcritic_wariant_tylko_w_actorze():
    """Krytyk zostaje GAT-em przy każdym wariancie actora — ablacja pyta
    o decyzję, nie o krytyka. Klucze state_dict actora GAT bez zmian, więc
    checkpointy GATE-a ładują się dalej."""
    from tsc_marl.core.nets import LocalEncoder

    ac = ActorCritic(OB_LENGTH, 8, encoder="none")
    assert isinstance(ac.actor.encoder, LocalEncoder)
    assert isinstance(ac.critic.encoder, GATEncoder)
    gat = ActorCritic(OB_LENGTH, 8)
    stare = Actor(OB_LENGTH, 8)
    assert gat.actor.state_dict().keys() == stare.state_dict().keys()


@pytest.mark.slow
def test_from_registry_czyta_ten_sam_config_co_baseline(advanced_colight_world):
    """Prawdziwy bieg ma brać hiperparametry z `colight.yml`, a nie z domyślnych
    stałych — inaczej porównanie z Advanced-CoLightem przestaje być uczciwe."""
    world, model = advanced_colight_world
    enc = GATEncoder.from_registry(model.ob_length)
    assert enc.n_layers == 1
    assert enc.ob_length == model.ob_length
    assert enc.out_dim == 128


# ---------------------------------------------------------------------- Actor


def make_actor(n_phases: int = 8, seed: int = 0, **kwargs) -> Actor:
    torch.manual_seed(seed)
    return Actor(OB_LENGTH, n_phases, **kwargs).eval()


def test_actor_ksztalt_i_glowa_domyslnie_jednowarstwowa(graph):
    """Głowa = jeden `nn.Linear`, tak jak `OUTPUT_LAYERS: []` w colight.yml —
    różnicą wobec baseline'u ma być sposób uczenia, nie głębszy MLP."""
    actor = make_actor()
    logits = actor(observations(graph), graph.edge_index)
    assert logits.shape == (graph.n_nodes, 8)
    assert len([m for m in actor.head if isinstance(m, torch.nn.Linear)]) == 1
    assert len([m for m in make_actor(hidden=64).head if isinstance(m, torch.nn.Linear)]) == 2


def test_actor_startuje_prawie_jednostajnie(graph):
    """`gain=0.01` na ostatniej warstwie. Bez tego polityka od pierwszego kroku
    ma losową, ale ZDECYDOWANĄ preferencję faz i PPO marnuje początek treningu
    na jej odkręcanie. Entropia startowa ≈ log(n_phases) to najtańszy sygnał,
    że inicjalizacja faktycznie zadziałała — warto go logować też w treningu."""
    logits = make_actor()(observations(graph), graph.edge_index)
    assert logits.std() < 0.1, "logity startowe za szerokie — gain 0.01 nie zadziałał"

    entropia = torch.distributions.Categorical(logits=logits).entropy()
    assert torch.allclose(entropia, torch.full_like(entropia, math.log(8)), atol=0.01)


def test_actor_zwraca_logity_a_nie_prawdopodobienstwa(graph):
    """Gdyby ktoś dopisał `softmax` w `forward`, wiersze sumowałyby się do 1
    i `Categorical(logits=...)` policzyłby log-softmax po raz drugi: rozkład za
    płaski, entropia zawyżona, trening wolniejszy — bez żadnego błędu."""
    logits = make_actor()(observations(graph), graph.edge_index)
    assert not torch.allclose(logits.sum(-1), torch.ones(graph.n_nodes), atol=1e-3)
    assert (logits < 0).any(), "logity muszą móc być ujemne"


def test_actor_gradient_dochodzi_do_encodera_i_glowy(graph):
    """Głowa w zwykłej liście zamiast `nn.Sequential` (albo encoder zbudowany
    przed `super().__init__()`) daje sieć, w której trenuje się tylko część."""
    actor = make_actor(hidden=64)
    actor(observations(graph), graph.edge_index).sum().backward()
    bez_gradientu = [n for n, p in actor.named_parameters() if p.grad is None or not p.grad.any()]
    assert bez_gradientu == []
    assert any(n.startswith("encoder.blocks") for n, _ in actor.named_parameters())


def test_actor_rownowaznosc_permutacyjna(graph):
    """To samo co dla encodera, ale sprawdzone na końcu ścieżki: numer węzła nie
    ma prawa wpływać na jego rozkład nad fazami."""
    actor = make_actor()
    x, ei = observations(graph), graph.edge_index
    perm = torch.randperm(graph.n_nodes, generator=torch.Generator().manual_seed(7))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(graph.n_nodes)
    assert torch.allclose(actor(x[perm], inverse[ei]), actor(x, ei)[perm], atol=1e-5)


def test_actor_odrzuca_jedna_faze():
    """Rozkład nad jedną fazą jest zdegenerowany: entropia 0, gradient polityki
    zerowy. Lepiej wywalić się w konstruktorze niż trenować martwego agenta."""
    with pytest.raises(ValueError, match="n_phases"):
        Actor(OB_LENGTH, 1)


@pytest.mark.slow
def test_actor_from_registry(advanced_colight_world):
    """Encoder actora ma czytać ten sam config co baseline (`GATEncoder.registry_kwargs`)."""
    _, model = advanced_colight_world
    actor = Actor.from_registry(model.ob_length, 8)
    assert actor.encoder.n_layers == 1
    assert actor.encoder.ob_length == model.ob_length
    assert actor.n_phases == 8


# --------------------------------------------------------------- Critic / A2C


def make_critic(seed: int = 0, **kwargs) -> Critic:
    torch.manual_seed(seed)
    return Critic(OB_LENGTH, **kwargs).eval()


def test_critic_zwraca_wartosc_na_agenta(graph):
    """Faktoryzowany krytyk (decyzja §1.3): V_i na każdego agenta, nie jeden
    skalar na krok. Domyślnie 2-hop — to jest to „C" z CTDE, bo actor ma 1-hop."""
    critic = make_critic()
    v = critic(observations(graph), graph.edge_index)
    assert v.shape == (graph.n_nodes,)
    assert critic.encoder.n_layers == 2


def test_critic_batch_nie_miesza_krokow_czasu(graph):
    """NAJWAŻNIEJSZY test krytyka. Kontekst globalny to `global_mean_pool(h, batch)`.
    Gdyby ktoś napisał `h.mean(0)`, przy jednym grafie byłoby dobrze, a po
    sklejeniu 64 kroków czasu uśredniłby węzły z 64 RÓŻNYCH momentów: krytyk
    zaczyna widzieć przyszłość, wartości wychodzą podejrzanie dobre i nic tego
    nie zgłasza."""
    critic = make_critic()
    x1, x2 = observations(graph, seed=1), observations(graph, seed=2)
    osobno = torch.cat([critic(x1, graph.edge_index), critic(x2, graph.edge_index)])

    batch = Batch.from_data_list([Data(x=x, edge_index=graph.edge_index) for x in (x1, x2)])
    razem = critic(batch.x, batch.edge_index, batch.batch)
    assert torch.allclose(razem, osobno, atol=1e-5)


def test_critic_brak_batcha_to_jeden_graf(graph):
    """`batch=None` musi znaczyć dokładnie to samo co same zera — inaczej ścieżka
    rolloutu (jeden graf) i ścieżka update'u (Batch) liczyłyby co innego."""
    critic = make_critic()
    x, ei = observations(graph), graph.edge_index
    zera = torch.zeros(graph.n_nodes, dtype=torch.long)
    assert torch.allclose(critic(x, ei), critic(x, ei, zera))

    with pytest.raises(ValueError, match="batch"):
        critic(x, ei, torch.zeros(graph.n_nodes + 1, dtype=torch.long))


def test_critic_rownowaznosc_permutacyjna(graph):
    """Mean-pooling jest permutacyjnie niezmienniczy, więc cały krytyk musi być
    równoważny: przenumerowanie węzłów przenumerowuje wektor wartości."""
    critic = make_critic()
    x, ei = observations(graph), graph.edge_index
    perm = torch.randperm(graph.n_nodes, generator=torch.Generator().manual_seed(7))
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(graph.n_nodes)
    assert torch.allclose(critic(x[perm], inverse[ei]), critic(x, ei)[perm], atol=1e-5)


def test_critic_gradient_dochodzi_do_wszystkiego(graph):
    critic = make_critic()
    critic(observations(graph), graph.edge_index).sum().backward()
    assert [n for n, p in critic.named_parameters() if p.grad is None or not p.grad.any()] == []


def make_ac(seed: int = 0, **kwargs) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(OB_LENGTH, 8, **kwargs).eval()


def test_actorcritic_ma_rozlaczne_wagi():
    """Współdzielony encoder w MAPPO destabilizuje: gradient straty wartości ma
    inną skalę niż policy gradient i zaczyna dominować wspólną część. Nie wywala
    się — po prostu uczy gorzej, więc rozłączność sprawdzamy testem."""
    ac = make_ac()
    assert not {id(p) for p in ac.actor.parameters()} & {id(p) for p in ac.critic.parameters()}
    assert ac.actor.encoder.n_layers == 1 and ac.critic.encoder.n_layers == 2


def test_evaluate_zgadza_sie_z_act(graph):
    """W PIERWSZEJ epoce PPO ratio = exp(log_prob − log_prob_stary) musi wyjść
    dokładnie 1.0. Jeśli nie wychodzi, to nie jest problem uczenia, tylko rozjazd
    między ścieżką `act` a `evaluate` — a objawia się jako „PPO dziwnie się
    zachowuje od startu", czyli najdroższy rodzaj błędu do zdiagnozowania."""
    ac = make_ac()
    x, ei = observations(graph), graph.edge_index
    torch.manual_seed(3)
    actions, logp, wartosci = ac.act(x, ei)

    logp2, entropia, wartosci2 = ac.evaluate(x, ei, actions)
    assert torch.allclose(torch.exp(logp2 - logp), torch.ones_like(logp), atol=1e-6)
    assert torch.allclose(wartosci, wartosci2)
    assert entropia.shape == (graph.n_nodes,)
    assert torch.allclose(ac.value(x, ei), wartosci)


def test_act_ksztalty_i_tryby(graph):
    """`deterministic=True` to tryb ewaluacji (kolumna T1). W treningu próbkujemy —
    w PPO eksploracja bierze się z entropii polityki, nie z epsilon-greedy."""
    ac = make_ac()
    x, ei = observations(graph), graph.edge_index

    a1, logp, v = ac.act(x, ei, deterministic=True)
    a2, _, _ = ac.act(x, ei, deterministic=True)
    assert a1.shape == logp.shape == v.shape == (graph.n_nodes,)
    assert torch.equal(a1, a2)
    assert a1.max() < 8 and a1.min() >= 0

    torch.manual_seed(11)
    proby = torch.stack([ac.act(x, ei)[0] for _ in range(5)])
    assert not (proby == proby[0]).all(), "polityka nie próbkuje — sprawdź Categorical"


def heterogeneous_mask(n_nodes: int, n_phases: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """(liczby faz, maska) jak na Krakowie: skrzyżowania od 2 faz do pełnych 8."""
    counts = torch.tensor([2 + i % (n_phases - 1) for i in range(n_nodes)])
    return counts, torch.arange(n_phases)[None, :] < counts[:, None]


def test_maska_faz_wycina_nieistniejace_fazy(graph):
    """Sieć niejednorodna: polityka NIE MOŻE wybrać fazy, której skrzyżowanie nie
    ma — ani próbkując, ani argmaxem. A strata PPO musi zostać skończona: -inf
    w logitach nie może przeciec do log_prob, entropii ani gradientu."""
    counts, mask = heterogeneous_mask(graph.n_nodes)
    torch.manual_seed(0)
    ac = ActorCritic(OB_LENGTH, 8, phase_mask=mask)
    x, ei = observations(graph), graph.edge_index

    logits = ac.actor(x, ei)
    assert torch.isneginf(logits[~mask]).all()
    assert torch.isfinite(logits[mask]).all()

    torch.manual_seed(3)
    for _ in range(50):
        actions, log_probs, _ = ac.act(x, ei)
        assert (actions < counts).all()
        assert torch.isfinite(log_probs).all()
    assert (ac.act(x, ei, deterministic=True)[0] < counts).all()

    log_probs, entropy, _ = ac.evaluate(x, ei, actions)
    assert torch.isfinite(log_probs).all() and torch.isfinite(entropy).all()
    (log_probs.sum() + entropy.sum()).backward()
    for name, p in ac.actor.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_maska_faz_w_minibatchu_i_poza_state_dict(graph):
    """Minibatch PPO skleja k kroków czasu w k·N wierszy (`[t, i] → t·N + i`), więc
    maska ma się powtórzyć k razy. I nie może trafić do checkpointu — inaczej
    checkpointy GATE-a (bez maski) przestałyby się ładować."""
    from tsc_marl.core.ppo import tiled_edge_index  # noqa: PLC0415

    _, mask = heterogeneous_mask(graph.n_nodes)
    k = 3
    torch.manual_seed(0)
    ac = ActorCritic(OB_LENGTH, 8, phase_mask=mask)
    x = torch.randn(k * graph.n_nodes, OB_LENGTH)
    logits = ac.actor(x, tiled_edge_index(graph.edge_index, graph.n_nodes, k))
    assert torch.equal(torch.isneginf(logits), ~mask.repeat(k, 1))

    assert not any("phase_mask" in key for key in ac.state_dict())
    ActorCritic(OB_LENGTH, 8).load_state_dict(ac.state_dict())  # i w drugą stronę też


@pytest.mark.slow
def test_actorcritic_from_registry(advanced_colight_world):
    """Krytyk NIE bierze `n_layers` z configu: tam siedzi wartość actora
    (`N_LAYERS: 1`), a krytyk ma mieć własny, szerszy zasięg."""
    _, model = advanced_colight_world
    ac = ActorCritic.from_registry(model.ob_length, 8)
    assert ac.actor.encoder.n_layers == 1
    assert ac.critic.encoder.n_layers == 2
    assert Critic.from_registry(model.ob_length).encoder.n_layers == 2
