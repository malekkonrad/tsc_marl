"""Testy warstwy awarii sensora (`domain.sensors`).

Ta warstwa jest PRZYRZĄDEM POMIAROWYM bramki 2.0 i protokołu 3.2, więc jej błąd
nie objawia się wyjątkiem, tylko przesuniętą krzywą degradacji — czyli w miejscu,
w którym leży główny wynik pracy. Stąd testy celują w cztery rzeczy, których nie
widać po wynikach: (1) że wyłączona warstwa nie zmienia NICZEGO, (2) że nagroda
zostaje czysta, (3) że jeden fizyczny detektor pokazuje wszystkim czytelnikom tę
samą wartość, (4) że ground truth zgadza się z tym, co faktycznie popsuto.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsc_marl.core.features import StateBuilder
from tsc_marl.domain.sensors import FailureSpec, SensorModel, failure_units

# 4 drogi × 3 pasy — układ jak na skrzyżowaniach JiNan/HangZhou.
LANES = [f"road_{r}_{i}" for r in "NESW" for i in range(3)]
CLEAN = {lane: float(10 + i) for i, lane in enumerate(LANES)}


def model(**spec) -> SensorModel:
    s = FailureSpec(**spec)
    return SensorModel(s, failure_units([], LANES, s.unit))


def read(m: SensorModel, channel: str = "queue", lanes=None) -> dict[str, float]:
    return m.read(CLEAN, lanes or LANES, channel=channel)


# --------------------------------------------------------------------- spec


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "Gauss"}, "kind="),  # wielkość liter ma znaczenie
        ({"unit": "detector"}, "unit="),
        ({"substitution": "nan"}, "substitution="),
        ({"kind": "gauss", "level": -1}, "level="),
        ({"kind": "mask", "level": 1.5}, "UŁAMEK"),
        ({"kind": "gauss", "level": 1, "unit": "road"}, "per ODCZYT"),
        ({"onset": -5}, "onset="),
    ],
)
def test_zly_spec_wywala_sie_przy_budowie(kwargs, match):
    """Literówka w YAML-u ma się wywalić od razu, nie po 200 epizodach dziwnych
    wyników. `unit` przy szumie pomiarowym jest ignorowalny — więc jest błędem,
    nie cichym no-opem."""
    with pytest.raises(ValueError, match=match):
        FailureSpec(**kwargs)


def test_suffix_rozroznia_scenariusze():
    """Bez tego dwa scenariusze awarii nadpisałyby się w `runs/`: `run_id` sklada
    się z agenta, sieci, seeda i yellow — a te są identyczne."""
    seen = {
        FailureSpec(kind=k, level=v, unit=u, substitution=s, onset=t).suffix()
        for k, v, u, s, t in [
            ("gauss", 3.5, "lane", "zero", 0),
            ("gauss", 4.0, "lane", "zero", 0),
            ("urand", 3.5, "lane", "zero", 0),
            ("mask", 0.25, "lane", "zero", 0),
            ("mask", 0.25, "intersection", "zero", 0),
            ("mask", 0.25, "lane", "hold", 120),
            ("drift", 0.2, "road", "zero", 0),
        ]
    }
    assert len(seen) == 7
    assert FailureSpec().suffix() == ""  # czysty bieg zostaje bez sufiksu


# ------------------------------------------------------------ stan TRWAŁY


@pytest.mark.parametrize("level", [0.25, 0.5])
def test_maska_to_ulamek_dokladny(level):
    """Protokół mówi \"maski 25/50%\", więc 25% ma znaczyć 25%, a nie \"średnio
    25%\" — inaczej dotkliwość różniłaby się między seedami i krzywa T2
    mieszałaby efekt metody z rozrzutem losowania."""
    m = model(kind="mask", level=level)
    assert len(m.broken) == round(level * len(LANES))


def test_maska_grupuje_po_jednostce_awarii():
    """`unit='road'` to jedna kamera na wlocie, a nie trzy niezależne pętle:
    albo pada cała droga, albo żaden jej pas. Wariant `intersection` to scenariusz
    kriging z 3.2."""
    m = model(kind="mask", level=0.5, unit="road")
    roads = {lane.rsplit("_", 1)[0] for lane in m.broken}
    assert len(roads) == 2
    for road in roads:
        assert sum(1 for lane in m.broken if lane.startswith(road)) == 3


def test_martwy_detektor_jest_martwy_w_obu_kanalach():
    """Jeden fizyczny czujnik obsługuje kolejkę i pojazdy jadące. Uśmiercenie go
    tylko w jednym kanale to awaria fizycznie niemożliwa, a głowa niepewności
    dostałaby z niej darmową wskazówkę."""
    m = model(kind="mask", level=0.5)
    m.begin_decision()
    q, r = read(m, "queue"), read(m, "running")
    for lane in m.broken:
        assert q[lane] == 0.0 and r[lane] == 0.0
    for lane in set(LANES) - m.broken:
        assert q[lane] == CLEAN[lane] and r[lane] == CLEAN[lane]


def test_zepsuty_detektor_nie_migocze():
    """Awaria trwała trzyma się przez CAŁY epizod. Losowanie maski co decyzję
    symulowałoby uszkodzenie fizycznie nieistniejące i głowa nauczyłaby się
    migotania zamiast awarii."""
    m = model(kind="mask", level=0.25)
    broken = set(m.broken)
    for _ in range(240):  # tyle decyzji ma epizod przy Δt=15
        m.begin_decision()
        out = read(m)
        assert set(m.broken) == broken
        assert all(out[lane] == 0.0 for lane in broken)


def test_reset_przelosowuje_awarie_na_nowy_epizod():
    """Replikacją w Fazie 3 jest N LOSOWAŃ AWARII (ruch jest ustalony w flow.json,
    więc seed polityki nic dla metod bez treningu nie zmienia)."""
    m = model(kind="mask", level=0.25)
    patterns = set()
    for episode in range(10):
        m.reset(episode)
        patterns.add(m.broken)
    assert len(patterns) > 1


def test_wzorzec_awarii_nie_zalezy_od_globalnych_seedow():
    """Gdyby awaria szła z tego samego ziarna co sieć, seedy różniłyby się
    JEDNOCZEŚNIE inicjalizacją polityki i tym, które detektory padły — i w tabeli
    T2 nie dałoby się rozdzielić źródeł rozrzutu."""
    np.random.seed(0)
    a = model(kind="mask", level=0.25).broken
    np.random.seed(12345)
    np.random.random(1000)
    b = model(kind="mask", level=0.25).broken
    assert a == b


def test_drift_jest_multiplikatywny_i_nie_robi_dziur():
    """Scenariusz NIEOBSERWOWALNY: wartości zostają sensowne w całym zakresie
    obciążenia, nie ma ani braków, ani skoków wariancji. Rule-gate jest tu ślepy
    z definicji — i tylko tu UQ zarabia na siebie."""
    m = model(kind="drift", level=0.2)
    m.begin_decision()
    out = read(m)
    assert m.broken == frozenset()
    assert all(m.valid(LANES).values())
    ratios = {lane: out[lane] / CLEAN[lane] for lane in LANES}
    assert len(set(np.round(list(ratios.values()), 6))) > 1  # różne detektory, różny bias
    assert any(abs(r - 1.0) > 0.05 for r in ratios.values())
    m.begin_decision()  # bias jest TRWAŁY, nie losuje się co odczyt
    assert read(m) == out


# ---------------------------------------------------------- stan CHWILOWY


def test_szum_jest_jeden_na_decyzje_i_wspolny_dla_czytelnikow():
    """Pas wyjazdowy skrzyżowania A jest pasem wjazdowym skrzyżowania B, więc ten
    sam detektor jest czytany dwa razy w jednej decyzji. Dwa niezależne losowania
    pozwoliłyby sieci uśrednić szum między sąsiadami i awaria wyszłaby ŁAGODNIEJSZA,
    niż jest naprawdę."""
    m = model(kind="gauss", level=3.5)
    m.begin_decision()
    first = read(m)
    assert read(m) == first, "ten sam detektor, ta sama decyzja, dwie wartości"
    assert read(m, lanes=LANES[:3]) == {k: first[k] for k in LANES[:3]}

    m.begin_decision()
    assert read(m) != first, "nowa decyzja = nowy odczyt"


def test_szum_jest_niezalezny_miedzy_pasami():
    """Jedno losowanie na całą sieć byłoby wspólnym przesunięciem, które EP (różnica
    średnich) w dużej części by skasował."""
    m = model(kind="gauss", level=3.5)
    m.begin_decision()
    deltas = {lane: read(m)[lane] - CLEAN[lane] for lane in LANES}
    assert len(set(np.round(list(deltas.values()), 6))) > len(LANES) // 2


@pytest.mark.parametrize("kind", ["gauss", "urand"])
def test_dotkliwosc_rosnie_z_poziomem(kind):
    """Oś X sweepu z bramki 2.0. Jeśli nie jest monotoniczna, monotoniczność
    niepewności nie ma o czym świadczyć."""
    severity = []
    for level in (0.5, 1.0, 2.0, 4.0):
        m = model(kind=kind, level=level)
        diffs = []
        for _ in range(50):
            m.begin_decision()
            out = read(m)
            diffs += [abs(out[lane] - CLEAN[lane]) for lane in LANES]
        severity.append(float(np.mean(diffs)))
    assert severity == sorted(severity)


def test_zliczenie_nie_bywa_ujemne():
    """Szum o sigma 4 na pasie z kolejką 1 zszedłby poniżej zera — a detektor nie
    umie zaraportować minus dwóch samochodów."""
    m = model(kind="gauss", level=8.0)
    for _ in range(50):
        m.begin_decision()
        out = m.read(dict.fromkeys(LANES, 1.0), LANES, channel="queue")
        assert all(v >= 0.0 for v in out.values())


# --------------------------------------------------------------- podstawienie


def test_read_nie_modyfikuje_wejscia():
    """`values` bywa cachem świata (`lane_waiting_count`), z którego liczy się
    NAGRODA. Modyfikacja w miejscu skaziłaby ją po cichu."""
    m = model(kind="mask", level=0.5)
    m.begin_decision()
    before = dict(CLEAN)
    read(m)
    assert CLEAN == before


def test_hold_trzyma_ostatnia_wartosc_sprzed_awarii():
    """Wariant TRUDNIEJSZY od `zero`: nie zostawia śladu (dokładnego 0 przez wiele
    decyzji), na którym rule-gate z ablacji (d) mógłby wygrać trywialnie."""
    m = model(kind="mask", level=0.5, substitution="hold", onset=3)
    for _ in range(3):  # decyzje 0,1,2 — jeszcze przed onsetem
        m.begin_decision()
        assert read(m) == CLEAN
    m.begin_decision()  # decyzja 3 = awaria
    out = read(m)
    assert m.broken
    for lane in m.broken:
        assert out[lane] == CLEAN[lane], "hold ma trzymać ostatni zdrowy odczyt"


def test_hold_bez_onsetu_degeneruje_sie_do_zera():
    """Detektor martwy od pierwszej decyzji nie ma czego trzymać. Zachowanie
    udokumentowane, żeby nie wyglądało na błąd przy czytaniu wyników."""
    m = model(kind="mask", level=0.5, substitution="hold")
    m.begin_decision()
    out = read(m)
    assert all(out[lane] == 0.0 for lane in m.broken)


def test_mean_podstawia_srednia_zywych():
    m = model(kind="mask", level=0.5, substitution="mean")
    m.begin_decision()
    out = read(m)
    live = [out[lane] for lane in LANES if lane not in m.broken]
    assert out[next(iter(m.broken))] == pytest.approx(float(np.mean(live)))


def test_onset_wlacza_awarie_w_runtime():
    """Narracja deploymentowa z planu Fazy 3: polityka jedzie na czystym, awaria
    zdarza się w trakcie."""
    m = model(kind="mask", level=0.5, onset=2)
    m.begin_decision()
    assert read(m) == CLEAN, "przed onsetem detektory są zdrowe"
    assert not m.live
    m.begin_decision()
    assert not m.live
    m.begin_decision()
    assert m.live
    assert any(read(m)[lane] == 0.0 for lane in m.broken)


# --------------------------------------------------------------- ground truth


def test_report_zapisuje_co_faktycznie_padlo():
    """Sam spec nie wystarcza: ta sama specyfikacja na innej sieci psuje inne
    pasy, a bramka 2.0 mierzy właśnie lokalizację awarii."""
    m = model(kind="mask", level=0.25, unit="road")
    r = m.report()
    assert r["kind"] == "mask" and r["unit"] == "road"
    assert r["n_units"] == 4  # cztery drogi, nie dwanaście pasów
    assert r["n_broken"] == len(m.broken) == 3
    assert sorted(m.broken) == r["broken"]


def test_valid_widzi_dokladnie_zepsute_pasy():
    """To czyta BRAMKA (rule-gate z ablacji (d) i poziom (b) fallbacku) — nigdy
    wektor stanu, żeby `ob_length` zostało takie samo we wszystkich scenariuszach."""
    m = model(kind="mask", level=0.25)
    valid = m.valid(LANES)
    assert {lane for lane, ok in valid.items() if not ok} == set(m.broken)


@pytest.mark.slow
@pytest.mark.parametrize("unit", ["lane", "road", "intersection"])
def test_jednostki_awarii_na_prawdziwym_roadnecie(advanced_colight_world, unit):
    """`intersection` wymaga przypisania pasa do skrzyżowania, DO KTÓREGO dojeżdża
    — na syntetycznych nazwach nie da się tego sprawdzić. Musi być `slow`, bo
    `Intersection.in_roads` zapełnia dopiero `World.__init__` (`sort_roads`),
    a nie samo parsowanie roadnetu."""
    world, _ = advanced_colight_world
    intersections = list(world.intersections)
    lanes = sorted(
        f"{road['id']}_{i}"
        for inter in intersections
        for road in inter.in_roads
        for i in range(len(road["lanes"]))
    )
    groups = failure_units(intersections, lanes, unit)
    assert set(groups) == set(lanes)
    n_units = len(set(groups.values()))
    if unit == "lane":
        assert n_units == len(lanes)
    elif unit == "intersection":
        assert n_units == len(intersections)
    else:
        assert len(intersections) < n_units < len(lanes)


# ------------------------------------------------------- integracja ze stanem


def warm_up(world, steps: int = 300) -> None:
    """CityFlow startuje PUSTĄ siecią, a wtedy wszystkie zliczenia są zerowe
    i każda awaria jest nieodróżnialna od czystego biegu: zamaskowane 0 to nadal
    0, a 0 przemnożone przez rozkalibrowanie to też 0. Bez rozgrzania testy
    integracyjne przechodziłyby na pustym świecie, czyli nie sprawdzałyby nic."""
    zeros = [0] * len(world.intersections)
    for _ in range(steps):
        world.step(zeros)


@pytest.mark.slow
def test_wylaczona_warstwa_nie_zmienia_nic(advanced_colight_world):
    """Regresja o tym samym standardzie, którym zweryfikowano refaktor ATS:
    identyczne BIT W BIT. Warstwa awarii nie ma prawa dotknąć czystego biegu."""
    world, _ = advanced_colight_world
    warm_up(world)
    base = StateBuilder.from_registry(world, world.intersections, failure=None)
    spec = FailureSpec(kind="gauss", level=3.5)
    with_layer = StateBuilder.from_registry(world, world.intersections, failure=spec)
    assert np.array_equal(base.observe(), with_layer.observe(corrupt=False))


@pytest.mark.slow
def test_nagroda_zostaje_czysta_pod_kazda_awaria(advanced_colight_world):
    """INWARIANT CORE Fazy 3: trening i nagroda na czystym, awaria wyłącznie na
    wejściu polityki. Nie jest to kwestia flagi — `rewards()` czyta cache świata,
    nie ścieżkę obserwacji — więc test pilnuje, żeby refaktor tego nie zepsuł."""
    world, _ = advanced_colight_world
    warm_up(world)
    clean = StateBuilder.from_registry(world, world.intersections, failure=None).rewards()
    assert clean.sum() < 0, "pusta sieć uczyniłaby ten test bezprzedmiotowym"
    for spec in (
        FailureSpec(kind="gauss", level=4.0),
        FailureSpec(kind="mask", level=0.5),
        FailureSpec(kind="mask", level=1.0, unit="intersection"),
        FailureSpec(kind="drift", level=0.3, unit="road"),
    ):
        b = StateBuilder.from_registry(world, world.intersections, failure=spec)
        b.observe()  # popsuta obserwacja NIE ma prawa przeciec do nagrody
        assert np.array_equal(b.rewards(), clean), spec


@pytest.mark.slow
@pytest.mark.parametrize(
    "spec",
    [
        FailureSpec(kind="gauss", level=3.5),
        FailureSpec(kind="mask", level=0.5),
        FailureSpec(kind="drift", level=0.3),
    ],
)
def test_awaria_faktycznie_zmienia_obserwacje(advanced_colight_world, spec):
    """Druga strona poprzedniego testu: warstwa ma robić COŚ. Szerokość wiersza
    zostaje ta sama, bo flaga braku nie wchodzi do stanu."""
    world, _ = advanced_colight_world
    warm_up(world)
    b = StateBuilder.from_registry(world, world.intersections, failure=spec)
    dirty, clean = b.observe(), b.observe(corrupt=False)
    assert dirty.shape == clean.shape
    assert np.abs(clean[:, : -b.n_phases]).sum() > 0, "same zera = nie ma czego psuć"
    assert not np.array_equal(dirty, clean), spec
