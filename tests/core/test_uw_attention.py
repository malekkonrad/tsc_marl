import pytest
import torch
from agent.colight import MultiHeadAttModel

from tests.core.test_ppo import ROADNET
from tsc_marl.core.graph import RoadNetGraph
from tsc_marl.core.uw_attention import UncertaintyWeightedAttention

D, DV, D_OUT, NV = 128, 16, 128, 5


@pytest.fixture(scope="module")
def graph():
    if not ROADNET.is_file():
        pytest.skip()
    return RoadNetGraph.from_roadnet(ROADNET)


def observations(g, seed: int = 1):
    return torch.randn(g.n_nodes, D, generator=torch.Generator().manual_seed(seed))


def make_pair(seed: int = 0):
    torch.manual_seed(seed=seed)
    original = MultiHeadAttModel(d=D, dv=DV, d_out=D_OUT, nv=NV, suffix=0).eval()
    uw = UncertaintyWeightedAttention(d=D, dv=DV, d_out=D_OUT, nv=NV, suffix=0).eval()
    uw.load_state_dict(original.state_dict())

    return original, uw


def test_no_op(graph):
    """Przy log ρ = 0 warstwa musi liczyć DOKŁADNIE to samo co oryginał.

    Bez tego testu każda późniejsza różnica w T2 jest niemożliwa do przypisania:
    nie wiadomo, czy to zasługa modulacji, czy skutek tego, że przepisując
    `message` coś się po drodze zgubiło.
    """
    orig, uw = make_pair()
    x = observations(graph)
    ei = graph.edge_index

    expected = orig(x, ei)
    assert torch.equal(uw(x, ei), expected)  # ścieżka log_rho=None

    n_edges = ei.shape[1] + graph.n_nodes  # E + N, po pętlach
    assert torch.equal(uw(x, ei, log_rho=torch.zeros(n_edges)), expected)


# Dlaczego −1e9, a nie −inf i nie −50. `exp(-1e9 - max_i)` podpływa w float32 do
# DOKŁADNEGO zera, więc α_ij = 0/norm = 0 co do bitu, wkład krawędzi to zera i
# suma agregacji się nie rusza — stąd `torch.equal` zamiast tolerancji.
# −50 dałoby ~1e−22, czyli niezerowy wkład i słabszy test na `allclose`.
# −inf dałoby NaN w dniu, w którym WSZYSTKIE krawędzie wchodzące do i będą
# wygaszone: `max_i` grupy to wtedy −inf, a `-inf - (-inf)` to NaN.
SUPPRESS = -1e9


def test_wygaszenie_odcina_wklad_sasiada(graph):
    """Bardzo ujemne log ρ na j→i → wyjście węzła i przestaje zależeć od x_j.

    To jest właściwość, dla której cała modyfikacja powstała: w oryginale
    `e_ij ≥ 0`, więc `exp(e_ij) ≥ 1` i sąsiada nie da się wyzerować, tylko
    rozcieńczyć. Jeśli ten test przechodzi, log ρ faktycznie schodzi poniżej zera.

    Dwie asercje kontrolne nie są ozdobą: bez nich test jest zielony również
    wtedy, gdy patrzę na wiersz, który na x_j nigdy nie wisiał (pomylony kierunek
    krawędzi — `edge_index[0]` to ŹRÓDŁO), albo gdy podmieniona obserwacja wcale
    się nie różni od oryginalnej.
    """
    _, uw = make_pair()
    ei = graph.edge_index
    n_edges = ei.shape[1] + graph.n_nodes

    # Krawędź wybrana przez POZYCJĘ w edge_index, bo to jej pozycja indeksuje
    # log_rho; parę (źródło, cel) odczytuję z niej, a nie odwrotnie.
    e = 3
    j, i = ei[:, e].tolist()
    assert [tuple(col) for col in ei.t().tolist()].count((j, i)) == 1, (
        "para (j, i) powtarza się w edge_index — wygaszenie jednej kolumny "
        "nie odcina wtedy całego kanału z x_j do i"
    )

    x = observations(graph)
    x_zmienione = x.clone()
    x_zmienione[j] = observations(graph, seed=2)[j]

    log_rho = torch.zeros(n_edges)
    log_rho[e] = SUPPRESS

    wygaszone = uw(x, ei, log_rho=log_rho)
    wygaszone_po = uw(x_zmienione, ei, log_rho=log_rho)
    assert torch.equal(wygaszone[i], wygaszone_po[i])

    # Kontrola 1: x_zmienione naprawdę jest inne — wiersz samego j musi drgnąć.
    assert not torch.equal(wygaszone, wygaszone_po)

    # Kontrola 2: bez wygaszenia ten sam wiersz i REAGUJE na x_j. Inaczej
    # równość wyżej nic nie dowodzi.
    zera = torch.zeros(n_edges)
    assert not torch.equal(uw(x, ei, log_rho=zera)[i], uw(x_zmienione, ei, log_rho=zera)[i])


def test_niezmienniczosc_na_wspolna_stala(graph):
    """Stała dodana do log ρ WSZYSTKICH krawędzi wchodzących do i → wyjście i stoi.

    To jest Załącznik A3b, zapisany jako test, a nie jako zdanie w tekście:
    softmax normalizuje w grupie, więc attention widzi wyłącznie WZGLĘDNĄ
    wiarygodność sąsiadów. „Wszyscy sąsiedzi i są tak samo niewiarygodni" jest
    dla tej warstwy nieodróżnialne od „wszyscy są wiarygodni" — dlatego samo
    UW-attention nie wystarczy i potrzebny jest osobny fallback.

    Stąd też hipoteza z Kroku 5: modulacja pomaga przy JEDNYM zepsutym sąsiedzie,
    a nie pomaga przy martwym całym skrzyżowaniu.

    `allclose`, nie `equal`: `(e + c) - (max + c)` to nie to samo co `e - max`
    co do bitu. Niezmienniczość jest matematyczna, arytmetyka float32 nie.
    """
    _, uw = make_pair()
    ei = graph.edge_index
    n_edges = ei.shape[1] + graph.n_nodes

    i = int(ei[1, 3])
    # Grupa softmaxu węzła i: krawędzie j→i ORAZ pętla własna i→i, która po
    # `add_self_loops` leży na pozycji E + i. Pominięcie pętli to dokładnie ten
    # błąd, przez który kontrakt warstwy brzmi [E + N], a nie [E].
    wchodzace = torch.cat([(ei[1] == i).nonzero().flatten(), torch.tensor([ei.shape[1] + i])])

    x = observations(graph)
    odniesienie = uw(x, ei, log_rho=torch.zeros(n_edges))

    log_rho = torch.zeros(n_edges)
    log_rho[wchodzace] = -3.7
    assert torch.allclose(uw(x, ei, log_rho=log_rho)[i], odniesienie[i], atol=1e-6)

    # Kontrola: gdyby warstwa log ρ po prostu IGNOROWAŁA (np. argument nie dotarł
    # do `message`), asercja wyżej też byłaby zielona. Ta sama stała na WSZYSTKICH
    # krawędziach oprócz pętli własnej łamie grupę i wiersz i musi drgnąć.
    log_rho[ei.shape[1] + i] = 0.0
    assert not torch.allclose(uw(x, ei, log_rho=log_rho)[i], odniesienie[i], atol=1e-6)


def test_klucze_state_dict_jak_w_oryginale():
    """Checkpointy GATE ładują się do nowej warstwy bez tłumaczenia nazw.

    Warstwa nie ma prawa dołożyć ani jednego parametru: log ρ przychodzi z
    zewnątrz, więc modulacja jest bezparametrowa i wytrenowane wagi zostają
    wymienne w obie strony.
    """
    orig, uw = make_pair()
    assert uw.state_dict().keys() == orig.state_dict().keys()
    orig.load_state_dict(uw.state_dict())  # strict=True — w drugą stronę też
