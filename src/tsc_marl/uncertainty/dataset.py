"""Zbiór (stan → akcja eksperta) dla prototypu 2.0 — behavior cloning, bez PPO.

Po co osobno od treningu: 2.0 ma TANIO potwierdzić premisę („niepewność rośnie
z awarią sensora") i rozstrzygnąć metodę (EDL vs MC-dropout) ZANIM powstanie
cały system. Uczenie nadzorowane na zrzuconych parach izoluje mechanizm od
ryzyka EDL × PPO i chodzi na CPU w minutach zamiast godzin.

EKSPERT = Advanced-MP, i to nie jest wybór z wygody. Jego decyzja jest funkcją
DOKŁADNIE tej informacji, którą niesie nasz stan ATS: p(s) to suma bloku EP po
ruchach fazy (Def. 5), d(s) to suma bloku demand (Def. 10), a Alg. 1 porównuje
`d(a_cur)·W1 > max p(s)`. Zadanie BC jest więc na czystym wejściu REALIZOWALNE —
i dlatego każdy błąd pod awarią można przypisać awarii, a nie pojemności modelu.
Jedyny wyjątek to zatrzask minimalnego zielonego (`current_phase_time < t_min`),
którego w wektorze stanu NIE MA; dlatego zrzucamy `phase_time` obok wejścia
i raportujemy, jak często ta gałąź w ogóle strzela (`hold_min_green`).

EKSPERT DECYDUJE Z CZYSTEGO SILNIKA, A ZAPISUJEMY POPSUTĄ OBSERWACJĘ. To jest
właściwy target („co należało zrobić"), a nie przeoczenie: gdyby ekspert czytał
popsute wejście, głowa kalibrowałaby się do śmiecia. Wychodzi to za darmo,
bo `advanced_mp.py` czyta silnik wprost, a warstwa awarii siedzi wyłącznie
w `StateBuilder` — żaden scenariusz nie może zmienić trajektorii.

WSZYSTKIE SCENARIUSZE W JEDNYM PRZEBIEGU. Jeden świat, jeden ekspert, jeden
`StateBuilder` na scenariusz plus czysty. Trajektoria jest wtedy identyczna dla
wszystkich scenariuszy Z KONSTRUKCJI, a nie przez szczęśliwy zbieg
determinizmów — porównanie scenariuszy jest wolne od szumu trajektorii, i przy
okazji cały zbiór kosztuje tyle, co jeden bieg, a nie tyle, co jeden na
scenariusz.

    tsc dataset -n cityflow_jinan3x4_4ph --episodes 3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.paths import RUNS

EXPERIMENT = "bc"
CLEAN = "clean"

#: Siatka scenariuszy. Poziomy DOBRANE POD BRAMKĘ 2.0(e): głowa uczy się na
#: `gauss` 1,0/2,0, a testowana jest na 3,5/4,0 i na `urand` — czyli na szumie
#: NIEWIDZIANYM w treningu. Bez rozdzielenia poziomów nie da się odróżnić
#: estymatora niepewności od kalibrowanego detektora znanych poziomów.
DEFAULT_SCENARIOS: tuple[FailureSpec, ...] = (
    FailureSpec(),  # czysty — referencja i zbiór treningowy głowy
    FailureSpec(kind="gauss", level=1.0),
    FailureSpec(kind="gauss", level=2.0),
    FailureSpec(kind="gauss", level=3.5),
    FailureSpec(kind="gauss", level=4.0),
    FailureSpec(kind="urand", level=3.5),
    FailureSpec(kind="mask", level=0.25),
    FailureSpec(kind="mask", level=0.5),
    FailureSpec(kind="mask", level=0.25, unit="intersection"),
    FailureSpec(kind="drift", level=0.2),
    FailureSpec(kind="drift", level=0.3),
)


def slug(spec: FailureSpec) -> str:
    return spec.suffix() or CLEAN


@dataclass
class Dataset:
    """Zrzut w kolejności WĘZŁÓW GRAFU (kontrakt z `graph.py`), nie w kolejności
    `world.intersections` — ta sama, w której pracuje rdzeń Fazy 1, więc drugą
    iterację 2.0 (2–3 węzły, uncertainty-weighted attention) da się puścić na
    tym samym pliku bez przenumerowania."""

    x_clean: np.ndarray  # [T, N, F] float32 — obserwacja bez awarii
    x: dict[str, np.ndarray]  # slug -> [T, N, F] float32
    dead: dict[str, np.ndarray]  # slug -> [T, N, n_mov] bool — ruch z martwym detektorem
    y: np.ndarray  # [T, N] int64 — faza wybrana przez eksperta
    phase_time: np.ndarray  # [T, N] int32 — NIE jest wejściem, patrz docstring modułu
    episode: np.ndarray  # [T] int32
    edge_index: np.ndarray  # [2, E] int64 — dla drugiej iteracji 2.0
    #: [N, n_faz, n_mov] bool — które ruchy obsługuje która faza (`ats.phase_incidence`).
    #: Bez tego z wektora stanu nie da się policzyć p(s) ani d(s), czyli nie da się
    #: ani odtworzyć eksperta, ani sprawdzić, czy zadanie BC jest realizowalne.
    incidence: np.ndarray
    #: [T, N] float32 — p(s) najlepszej fazy minus drugiej, liczone z CZYSTEGO stanu.
    #: Miara AMBIWALENCJI: przy zerze dwie fazy są równie dobre i „niepewność
    #: decyzji" nie ma nic wspólnego z awarią sensora. Plan ostrzega przed dokładnie
    #: tym confoundem (2.2: „dwie fazy równie dobre != awaria sensora"), więc trzymamy
    #: go jako ZMIERZONĄ zmienną, a nie jako zastrzeżenie w tekście.
    tie_margin: np.ndarray
    meta: dict[str, Any]

    def save(self, path: Path) -> Path:
        arrays: dict[str, np.ndarray] = {
            "x_clean": self.x_clean,
            "y": self.y,
            "phase_time": self.phase_time,
            "episode": self.episode,
            "edge_index": self.edge_index,
            "incidence": self.incidence,
            "tie_margin": self.tie_margin,
        }
        for name, value in self.x.items():
            arrays[f"x__{name}"] = value
        for name, value in self.dead.items():
            arrays[f"dead__{name}"] = value
        np.savez_compressed(path, **arrays)
        return path


def collect(
    network: str = "cityflow_jinan3x4_4ph",
    scenarios: tuple[FailureSpec, ...] = DEFAULT_SCENARIOS,
    *,
    expert: str = "advancedmp",
    episodes: int = 3,
    decisions: int | None = None,
) -> Dataset:
    """Jeden przebieg, wszystkie scenariusze. -> `Dataset` + statystyki w `meta`."""
    from tsc_marl import libsignal

    libsignal.bootstrap()  # `core.loop` ciągnie `agent.colight` z submodułu

    from tsc_marl.core.features import StateBuilder
    from tsc_marl.core.graph import RoadNetGraph
    from tsc_marl.core.loop import LoopConfig, TrafficEnv, world_order
    from tsc_marl.libsignal import session

    # `graphic: True` jest tu WYMUSZONE: kolejność wierszy w zrzucie ma być tą
    # samą kolejnością węzłów grafu, w której pracuje rdzeń Fazy 1, a bez tego
    # LibSignal w ogóle nie policzy grafu (`Graph_World_Interface` jest warunkowy)
    # — `advancedmp.yml` go nie potrzebuje, bo MP nie jest metodą grafową.
    patch = {"model": {"graphic": True}}
    with session.standalone_world(expert, network, config_patch=patch) as (world, agents, config):
        cfg = LoopConfig.from_registry()
        n_dec = decisions if decisions is not None else cfg.decisions
        t_min = config["model"].get("t_min", 0)

        graph = RoadNetGraph.from_registry()
        order = graph.order_intersections(world)
        to_world = world_order(graph, world)

        clean_builder = StateBuilder.from_registry(world, order, failure=None)
        builders = {slug(s): StateBuilder.from_registry(world, order, failure=s) for s in scenarios}
        env = TrafficEnv(world, graph, clean_builder, cfg.action_interval)

        incidence = _incidence(clean_builder, config["model"].get("drop_always_green", True))
        acc = _Accumulator(builders, incidence)
        att: list[float] = []
        for ep in range(episodes):
            env.reset(ep)
            for agent in agents:
                agent.reset()
            for b in builders.values():
                b.reset_sensors(ep)
            acc.run_episode(agents, env, to_world, n_dec, ep, t_min)
            att.append(float(world.get_average_travel_time()))

        w1 = config["model"].get("w1")
        return acc.finish(graph, network, expert, scenarios, episodes, n_dec, att, t_min, w1)


class _Accumulator:
    """Zbiera wiersze epizod po epizodzie. Osobna klasa, żeby pętla po decyzjach
    była czytelna — to w niej siedzi jedyna kolejnościowa pułapka tego modułu."""

    def __init__(self, builders: dict[str, Any], incidence: np.ndarray) -> None:
        self.builders = builders
        self.incidence = incidence
        self.n_phases = builders[CLEAN].n_phases
        self.x_clean: list[np.ndarray] = []
        self.x: dict[str, list[np.ndarray]] = {name: [] for name in builders}
        self.dead: dict[str, list[np.ndarray]] = {name: [] for name in builders}
        self.y: list[np.ndarray] = []
        self.phase_time: list[np.ndarray] = []
        self.episode: list[np.ndarray] = []
        self.held = 0
        self.total = 0

    def run_episode(self, agents, env, to_world, n_dec, ep, t_min) -> None:
        n = len(to_world)
        for _ in range(n_dec):
            # 1. Obserwacje — WSZYSTKIE z tego samego stanu silnika, przed akcją.
            self.x_clean.append(self.builders[CLEAN].observe(corrupt=False))
            for name, b in self.builders.items():
                self.x[name].append(b.observe())
                self.dead[name].append(_dead_movements(b))

            # 2. Ekspert czyta silnik, nie nasze cechy — awaria go nie dotyczy.
            y_world = np.array([a.get_action(None, None) for a in agents], dtype=np.int64)
            pt = np.array([a.inter_obj.current_phase_time for a in agents], dtype=np.int32)
            self.held += int((pt < t_min).sum())
            self.total += n

            # 3. PRZENUMEROWANIE: agent rank k obsługuje `world.intersections[k]`,
            #    a wiersz tensora idzie po `node_idx`. `to_world[k] = node_idx`,
            #    więc odwrotnością jest przypisanie przez indeksowanie.
            y_node = np.empty(n, dtype=np.int64)
            pt_node = np.empty(n, dtype=np.int32)
            y_node[to_world] = y_world
            pt_node[to_world] = pt
            self.y.append(y_node)
            self.phase_time.append(pt_node)
            self.episode.append(np.full(1, ep, dtype=np.int32))

            env.step(y_node)

    def finish(self, graph, network, expert, scenarios, episodes, n_dec, att, t_min, w1) -> Dataset:
        x = {k: np.stack(v).astype(np.float32) for k, v in self.x.items()}
        x_clean = np.stack(self.x_clean).astype(np.float32)
        body = x_clean.shape[-1] - self.n_phases
        y = np.stack(self.y).astype(np.int64)
        check = expert_check(x_clean, y, self.incidence, w1, self.n_phases)
        meta = {
            "network": network,
            "expert": expert,
            "episodes": episodes,
            "decisions_per_episode": n_dec,
            "n_nodes": int(graph.n_nodes),
            "t_min": t_min,
            "w1": w1,
            # Ile razy zadziałał zatrzask minimalnego zielonego. To jest ta część
            # decyzji eksperta, której NIE DA SIĘ odtworzyć z samego wektora stanu
            # — sufit trafności BC, nie usterka modelu.
            "hold_min_green": self.held / max(self.total, 1),
            # Zadanie BC jest realizowalne z samego stanu — patrz `expert_check`.
            "agreement": check["agreement"],
            "agreement_decidable": check["agreement_decidable"],
            "tie_rate": check["tie_rate"],
            "boundary_rate": check["boundary_rate"],
            "att_per_episode": att,
            "scenarios": {slug(s): s.as_dict() for s in scenarios},
            # Blok fazy odcięty: one-hot nigdy się nie zmienia, więc trzymany
            # w mianowniku rozcieńczałby liczbę i psuł porównanie z `tsc severity`.
            "changed_cells": {
                name: float((np.abs(v[..., :body] - x_clean[..., :body]) > 1e-9).mean())
                for name, v in x.items()
            },
            "dead_movements": {name: float(np.stack(v).mean()) for name, v in self.dead.items()},
        }
        return Dataset(
            x_clean=x_clean,
            x=x,
            dead={k: np.stack(v) for k, v in self.dead.items()},
            y=y,
            tie_margin=check["margin"],
            phase_time=np.stack(self.phase_time).astype(np.int32),
            episode=np.concatenate(self.episode).astype(np.int32),
            edge_index=np.asarray(graph.edge_index, dtype=np.int64),
            incidence=self.incidence,
            meta=meta,
        )


def expert_check(
    x_clean: np.ndarray, y: np.ndarray, incidence: np.ndarray, w1: float, n_phases: int
) -> dict[str, Any]:
    """Odtwórz decyzję Advanced-MP z SAMEGO wektora stanu i porównaj z zapisaną.

    To jest samokontrola każdego zrzutu, a nie ciekawostka: jedna liczba sprawdza
    naraz kolejność cech (blok EP przed demand, one-hot na końcu), incydencję faz,
    przenumerowanie akcji `rank -> node_idx` ORAZ to, że etykieta pochodzi z CZYSTEGO
    stanu. Przesunięcie kolumny nie wywala niczego — psuje tylko tę liczbę.

    Zgodność liczymy na próbkach ROZSTRZYGALNYCH. Reszta to nie błąd stanu, tylko
    remisy, w których Alg. 1 rozstrzyga ostatnim bitem sumowania:
      * `tie`      — dwie fazy o równym p(s); `argmax` wybiera po kolejności,
      * `boundary` — `d(a_cur)·W1` dokładnie równe `max p(s)`, a warunek jest ostry.
    """
    eps = 1e-6
    x = x_clean.astype(np.float64)
    inc = incidence.astype(np.float64)
    n_mov = incidence.shape[-1]
    p_s = np.einsum("npm,tnm->tnp", inc, x[..., :n_mov])
    d_s = np.einsum("npm,tnm->tnp", inc, x[..., n_mov : 2 * n_mov])
    cur = x[..., -n_phases:].argmax(-1)

    top = np.sort(p_s, -1)
    margin = top[..., -1] - top[..., -2]
    d_cur = np.take_along_axis(d_s, cur[..., None], -1)[..., 0] * w1
    pred = np.where(d_cur > p_s.max(-1), cur, p_s.argmax(-1))

    tie = margin <= eps
    boundary = np.abs(d_cur - p_s.max(-1)) <= eps
    decidable = ~tie & ~boundary
    return {
        "agreement": float((pred == y).mean()),
        # NaN, a nie 1.0, gdy wszystko jest remisem: „nie ma czego sprawdzić"
        # to inna informacja niż „sprawdzone i zgadza się".
        "agreement_decidable": (
            float((pred[decidable] == y[decidable]).mean()) if decidable.any() else float("nan")
        ),
        "tie_rate": float(tie.mean()),
        "boundary_rate": float(boundary.mean()),
        "margin": margin.astype(np.float32),
    }


def _incidence(builder, drop_always_green: bool) -> np.ndarray:
    """[N, n_faz, n_mov] bool — patrz `Dataset.incidence`."""
    from tsc_marl.domain import ats

    out = np.zeros((builder.n_nodes, builder.n_phases, builder.n_mov), dtype=bool)
    for i, inter in enumerate(builder.intersections):
        for phase, movements in enumerate(ats.phase_incidence(inter, drop_always_green)):
            out[i, phase, movements] = True
    return out


def _dead_movements(builder) -> np.ndarray:
    """[N, n_mov] — czy RUCH (l,m) jest obsługiwany przez martwy detektor.

    Ground truth lokalizacji dla bramki 2.0, liczony na poziomie ruchu, bo tam
    właśnie siedzi głowa z 2.2. „Martwy" = którykolwiek pas ruchu (wjazdowy albo
    wyjazdowy — e(l,m) patrzy w obie strony) ma zepsuty czujnik. Uwaga: to NIE
    jest to samo, co „cecha się zmieniła" — zamaskowany detektor na pustym pasie
    raportuje 0, czyli wartość prawdziwą, a mimo to jest martwy.
    """
    out = np.zeros((builder.n_nodes, builder.n_mov), dtype=bool)
    sensors = builder.sensors
    if sensors is None or not sensors.broken:
        return out
    broken = sensors.broken
    for i, struct in enumerate(builder.structs):
        for m, (ins, outs) in enumerate(struct.movements):
            out[i, m] = any(lane in broken for lane in (*ins, *outs))
    return out


def build(
    network: str = "cityflow_jinan3x4_4ph",
    *,
    expert: str = "advancedmp",
    episodes: int = 3,
    replace: bool = False,
) -> Path:
    """Zrzut pod zarejestrowanym biegiem `runlog` — zbiór bez proweniencji jest
    bezwartościowy: za pół roku nie da się odtworzyć, z jakiego kodu powstał."""
    from runlog import Experiment
    from tsc_marl.libsignal import experiment as exp_module

    data = collect(network, expert=expert, episodes=episodes)
    params = {k: data.meta[k] for k in ("network", "expert", "episodes", "t_min", "w1")}
    params["scenarios"] = sorted(data.meta["scenarios"])
    run_id = f"{expert}__{network}__ep{episodes}"

    exp = Experiment(EXPERIMENT, root=RUNS)
    with exp.start(run_id, params, repos=exp_module.repos(), replace=replace) as run:
        path = data.save(run.path / "dataset.npz")
        run.result(
            samples=int(data.y.size),
            hold_min_green=data.meta["hold_min_green"],
            agreement=data.meta["agreement"],
            agreement_decidable=data.meta["agreement_decidable"],
            tie_rate=data.meta["tie_rate"],
            boundary_rate=data.meta["boundary_rate"],
            att_per_episode=data.meta["att_per_episode"],
            changed_cells=data.meta["changed_cells"],
            dead_movements=data.meta["dead_movements"],
            dataset=str(path.relative_to(run.path)),
        )
    return run.path
