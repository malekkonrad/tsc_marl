"""Prototyp 2.2 — heteroscedastyczna głowa OBSERWACYJNA: niepewność cech wejściowych.

Po co osobno od 2.0. Bramka 2.0 zmierzyła, że niepewność DECYZJI (Dirichlet nad
fazami) nie estymuje niepewności obserwacji: poza zakresem treningu zawraca,
a dryfu nie widzi. Plan od początku stawiał mechanizm 2.2 gdzie indziej — na
głowie liczonej z CECH WEJŚCIOWYCH, per ruch (l,m), agregowanej na krawędź grafu.
Ten moduł to tani prototyp tego mechanizmu na tym samym zrzucie BC, przed
wpięciem go w attention.

CO GŁOWA DOSTAJE (Załącznik A2 ze spotkania 18.08): pojedynczy odczyt 7 zamiast 4
jest w pełni wiarygodny — skażenie widać wyłącznie przez NIESPÓJNOŚĆ. Wejście to
więc (1) okno `window` ostatnich odczytów węzła (kolejka nie rośnie szybciej niż
saturation flow; na czerwonym może tylko rosnąć — faza jest w one-hocie) i
(2) opcjonalnie średnia odczytów sąsiadów upstream (zachowanie przepływu).

CO GŁOWA ZWRACA: dla każdej cechy ruchu (blok EP i blok demand) średnią μ i skalę σ
rozkładu PRAWDZIWEJ wartości; strata Gaussian NLL względem `x_clean`, który zna
wyłącznie symulator. Człon jest samokalibrujący: zaniżone σ eksploduje kwadratem
błędu, zawyżone kosztuje log σ.

JAK CZYTAĆ WYNIK: u_ruchu = max σ po dwóch cechach ruchu (w jednostkach odchylenia
cechy na czystym), u_krawędzi(j→i) = max albo średnia po ruchach skrzyżowania i,
których droga wjazdowa zaczyna się w j (mini-ablacja z planu). Detekcja = AUROC
u na komórkach POPSUTYCH w scenariuszu vs u na wszystkich komórkach CZYSTYCH —
czyli czy sygnał odróżnia zepsuty odczyt od zdrowego, a nie czy rośnie średnio.

    tsc obshead            # -> results/gate_2_2.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tsc_marl.uncertainty.prototype import TRAIN_SCENARIOS, Loaded, load

#: Błąd odczytu (w pojazdach, na cesze), od którego komórkę uznajemy za popsutą.
#: Martwy detektor jest popsuty ZAWSZE, nawet gdy przypadkiem raportuje prawdę (0 na
#: pustym pasie) — ground truth lokalizacji idzie z warstwy awarii, nie z różnicy cech.
CORRUPT_TAU = 0.5
UNSEEN = ("gauss3p5", "gauss4", "urand3p5", "mask0p5", "mask0p25_intersection")


@dataclass(frozen=True)
class ObsConfig:
    window: int = 4
    neighbors: bool = True
    #: Cechy TRWAŁOŚCI odczytu w oknie: jaki ułamek okna cecha stała w miejscu i jaki była
    #: dokładnie zerem. Martwy detektor z podstawieniem `zero` zostawia ślad właśnie w czasie
    #: (utknięcie), a nie w pojedynczej wartości — zero jest na czystym wartością modalną.
    persistence: bool = False
    hidden: int = 128
    epochs: int = 60
    lr: float = 1e-3
    batch: int = 512
    split: float = 0.67  # jak 2.0 — podział PO CZASIE (patrz `prototype.py`)
    train_scenarios: tuple[str, ...] = TRAIN_SCENARIOS
    seed: int = 0
    min_sigma: float = 0.02  # w jednostkach standaryzowanych — dolna granica σ


# ---------------------------------------------------------------- wejścia


def window_index(episode: np.ndarray, k: int) -> np.ndarray:
    """[T, k]: indeksy t, t−1, …, t−k+1, przycięte do POCZĄTKU EPIZODU.

    Okno nie może sięgać do poprzedniego epizodu: po `world.reset()` ostatni
    odczyt poprzedniego epizodu opisuje zupełnie inny stan ruchu."""
    t = np.arange(len(episode))
    first: dict[int, int] = {}
    for i, e in enumerate(episode.tolist()):
        first.setdefault(e, i)
    start = np.array([first[e] for e in episode.tolist()])
    return np.maximum(t[:, None] - np.arange(k)[None, :], start[:, None])


def neighbor_matrix(edge_index: np.ndarray, n: int) -> np.ndarray:
    """[N, N] — wiersz i uśrednia węzły, z których krawędź WCHODZI do i (upstream).
    Węzeł bez sąsiadów dostaje zera, nie dzielenie przez zero."""
    a = np.zeros((n, n), dtype=np.float32)
    a[edge_index[1], edge_index[0]] = 1.0
    deg = a.sum(1, keepdims=True)
    return np.divide(a, deg, out=np.zeros_like(a), where=deg > 0)


def persistence_features(x: np.ndarray, win: np.ndarray, n_feat: int) -> np.ndarray:
    """[T, N, 2·n_feat]: dla każdej cechy ruchu ułamek kroków okna BEZ ZMIANY wartości
    i ułamek odczytów DOKŁADNIE ZEROWYCH. Na początku epizodu okno jest przycięte
    (powtórzone indeksy), więc „bez zmiany" rośnie tam sztucznie — tak samo na czystym
    i na popsutym wejściu, więc nie zdradza awarii."""
    xs = np.stack([x[win[:, j]][..., :n_feat] for j in range(win.shape[1])])  # [k, T, N, f]
    same = (np.abs(np.diff(xs, axis=0)) < 1e-6).mean(0) if len(xs) > 1 else np.zeros_like(xs[0])
    zero = (np.abs(xs) < 1e-6).mean(0)
    return np.concatenate([same, zero], -1)


def build_inputs(
    x: np.ndarray, win: np.ndarray, adj: np.ndarray | None, n_persist: int = 0
) -> np.ndarray:
    """[T, N, F] -> [T, N, k·F (+F) (+2·n_persist)] — okno własnych odczytów, średnia
    sąsiadów upstream i (opcjonalnie) cechy trwałości pierwszych `n_persist` cech."""
    parts = [x[win[:, j]] for j in range(win.shape[1])]
    if adj is not None:
        parts.append(np.einsum("ij,tjf->tif", adj, x))
    if n_persist:
        parts.append(persistence_features(x, win, n_persist))
    return np.concatenate(parts, -1).astype(np.float32)


# ------------------------------------------------------------- geometria


def _intersections(roadnet: Path):
    """Obiekty `Intersection` LibSignala z samego pliku, bez silnika — ten sam trik
    co `tests/conftest.py`: parsowanie ruchów idzie kodem, który biegnie w symulacji."""
    from tsc_marl import libsignal  # noqa: PLC0415

    libsignal.bootstrap()
    from world.world_cityflow import Intersection  # noqa: PLC0415

    class _Eng:
        def set_tl_phase(self, *a):
            pass

    class _World:
        RIGHT = True
        eng = _Eng()

    raw = json.loads(roadnet.read_text())
    return raw, {d["id"]: Intersection(d, _World()) for d in raw["intersections"]}


def roadnet_of(network: str) -> Path:
    from tsc_marl.paths import CONFIGS, REPO_ROOT  # noqa: PLC0415

    cfg = json.loads((CONFIGS / "sim" / f"{network}.cfg").read_text())
    return REPO_ROOT / "data" / cfg["roadnetFile"]


def movement_sources(network: str, n_mov: int, drop_always_green: bool = True) -> np.ndarray:
    """[N, n_mov] — indeks węzła, z którego wjeżdża ruch (l,m); −1 = brzeg sieci.

    To jest mapowanie ruch → krawędź z planu: droga wjazdowa ruchu zaczyna się
    w sąsiedzie j, więc ruch należy do krawędzi j→i. Kolejność ruchów taka jak
    w `StateBuilder` (`ats.controlled_movements`), kolejność węzłów — grafu."""
    from tsc_marl.core.graph import RoadNetGraph  # noqa: PLC0415
    from tsc_marl.domain import ats  # noqa: PLC0415

    path = roadnet_of(network)
    graph = RoadNetGraph.from_roadnet(path)
    raw, inters = _intersections(path)
    start = {r["id"]: r["startIntersection"] for r in raw["roads"]}
    out = np.full((graph.n_nodes, n_mov), -1, dtype=np.int64)
    for i, nid in enumerate(graph.node_idx2id):
        for m, (ins, _outs) in enumerate(ats.controlled_movements(inters[nid], drop_always_green)):
            out[i, m] = graph.node_id2idx.get(start[ats.road_of(ins[0])], -1)
    return out


# ------------------------------------------------------------------ model


class ObsHead(nn.Module):
    """MLP wspólny dla wszystkich węzłów: [.., D] -> (μ, σ) [.., n_feat]."""

    def __init__(self, d_in: int, n_feat: int, hidden: int, min_sigma: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, n_feat)
        self.sigma = nn.Linear(hidden, n_feat)
        self.min_sigma = min_sigma

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(z)
        return self.mu(h), torch.nn.functional.softplus(self.sigma(h)) + self.min_sigma


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (0.5 * ((y - mu) / sigma) ** 2 + torch.log(sigma)).mean()


@dataclass
class Fitted:
    model: ObsHead
    in_mean: np.ndarray
    in_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    n_feat: int
    adj: np.ndarray | None
    cfg: ObsConfig

    def features(self, data: Loaded, x: np.ndarray) -> np.ndarray:
        win = window_index(data.episode, self.cfg.window)
        adj = neighbor_matrix(data.edge_index, data.n_nodes) if self.cfg.neighbors else None
        n_persist = self.n_feat if self.cfg.persistence else 0
        return (build_inputs(x, win, adj, n_persist) - self.in_mean) / self.in_std

    @torch.no_grad()
    def predict(self, data: Loaded, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (μ, σ) w jednostkach SUROWYCH cech, [T, N, n_feat]."""
        self.model.eval()
        z = torch.from_numpy(self.features(data, x).astype(np.float32))
        mu, sigma = self.model(z)
        return mu.numpy() * self.y_std + self.y_mean, sigma.numpy() * self.y_std

    def save(self, path: str | Path, *, network: str) -> Path:
        """Zapisz KOMPLET potrzebny do inferencji: wagi + standaryzacja + config.

        Same wagi nie wystarczą i to jest cała treść tego formatu. σ wychodzi z
        modelu w jednostkach standaryzowanych, a dopiero `y_std` robi z niego
        „odchylenie cechy na czystym" — czyli jednostkę, w której zmierzone są
        u_ref i τ z Kroku 4. Wczytanie wag z cudzą standaryzacją nie wywala się:
        daje u przeskalowane o stały czynnik, więc próg przestaje znaczyć to,
        co znaczył, a log ρ modulowałoby o złą wielkość.

        `network` w pliku, bo głowa jest ZWIĄZANA Z SIECIĄ: ta z JiNan na
        HangZhou daje AUROC 0,53 (`results/gate_2_2.md`), czyli rzut monetą
        udający detektor. Pomyłka pliku to najcichsza awaria w całym Kroku 5.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": HEAD_FORMAT,
                "network": network,
                "cfg": asdict(self.cfg),
                "n_feat": int(self.n_feat),
                "in_mean": torch.as_tensor(self.in_mean),
                "in_std": torch.as_tensor(self.in_std),
                "y_mean": torch.as_tensor(self.y_mean),
                "y_std": torch.as_tensor(self.y_std),
                "adj": None if self.adj is None else torch.as_tensor(self.adj),
                "state_dict": self.model.state_dict(),
            },
            path,
        )
        return path


def fit(data: Loaded, cfg: ObsConfig | None = None) -> tuple[Fitted, list[dict]]:
    cfg = cfg or ObsConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n_feat = data.ob_length - data.n_phases  # blok EP (+ demand) — bez one-hotu fazy

    win = window_index(data.episode, cfg.window)
    adj = neighbor_matrix(data.edge_index, data.n_nodes) if cfg.neighbors else None
    train = data.mask(cfg.split, train=True)
    n_persist = n_feat if cfg.persistence else 0
    zs = [build_inputs(data.x[s], win, adj, n_persist)[train] for s in cfg.train_scenarios]
    z = np.concatenate(zs).reshape(-1, zs[0].shape[-1])
    y = np.concatenate([data.x_clean[train][..., :n_feat]] * len(zs)).reshape(-1, n_feat)

    # standaryzacja z CZYSTEGO zbioru treningowego — σ wychodzi wtedy w jednostkach
    # „odchylenia cechy na czystym", porównywalnych między cechami
    clean_z = build_inputs(data.x_clean, win, adj, n_persist)[train].reshape(-1, z.shape[-1])
    in_mean, in_std = clean_z.mean(0), np.maximum(clean_z.std(0), 1e-3)
    clean_y = data.x_clean[train][..., :n_feat].reshape(-1, n_feat)
    y_mean, y_std = clean_y.mean(0), np.maximum(clean_y.std(0), 1e-3)

    zt = torch.from_numpy(((z - in_mean) / in_std).astype(np.float32))
    yt = torch.from_numpy(((y - y_mean) / y_std).astype(np.float32))
    model = ObsHead(zt.shape[1], n_feat, cfg.hidden, cfg.min_sigma)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(zt))
        losses = []
        for i in range(0, len(order), cfg.batch):
            idx = torch.from_numpy(order[i : i + cfg.batch])
            mu, sigma = model(zt[idx])
            loss = gaussian_nll(mu, sigma, yt[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        history.append({"epoch": epoch, "nll": float(np.mean(losses))})
    return Fitted(model, in_mean, in_std, y_mean, y_std, n_feat, adj, cfg), history


# ------------------------------------------------- zapis i odczyt wytrenowanej głowy

#: Wersja formatu artefaktu. Podbić, gdy zmienia się ZAWARTOŚĆ pliku (nowe pole albo
#: inne znaczenie istniejącego). Stary plik ma wtedy polec przy wczytaniu, a nie
#: wczytać się w połowie i dać u w innych jednostkach niż to, na czym dobrano τ.
HEAD_FORMAT = 1


def head_path(network: str) -> Path:
    """Domyślne miejsce artefaktu. OSOBNY plik na sieć — patrz `Fitted.save`."""
    from tsc_marl.paths import RUNS  # noqa: PLC0415

    return RUNS / "heads" / f"obs_head__{network}.pt"


def load_head(path: str | Path, *, network: str | None = None) -> Fitted:
    """Odczytaj głowę zapisaną przez `Fitted.save`. -> `Fitted` w trybie eval.

    `network` podany = sprawdzenie zgodności. Nie jest domyślne, bo testy i
    analizy czytają plik bez kontekstu sieci; sterownik z Kroku 5 podaje ZAWSZE.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    if blob.get("format") != HEAD_FORMAT:
        raise ValueError(
            f"format artefaktu {blob.get('format')!r}, oczekiwany {HEAD_FORMAT} — "
            "przeucz głowę (`tsc obshead --save`)"
        )
    if network is not None and blob["network"] != network:
        raise ValueError(
            f"głowa nauczona na {blob['network']!r}, a sieć to {network!r}; "
            "głowa z JiNan na HangZhou ma AUROC 0,53 (patrz results/gate_2_2.md)"
        )

    raw = dict(blob["cfg"])
    raw["train_scenarios"] = tuple(raw["train_scenarios"])
    cfg = ObsConfig(**raw)
    n_feat = int(blob["n_feat"])
    model = ObsHead(int(blob["in_mean"].shape[0]), n_feat, cfg.hidden, cfg.min_sigma)
    model.load_state_dict(blob["state_dict"])  # strict=True — kształty muszą się zgadzać
    model.eval()
    adj = None if blob["adj"] is None else blob["adj"].numpy()
    return Fitted(
        model,
        blob["in_mean"].numpy(),
        blob["in_std"].numpy(),
        blob["y_mean"].numpy(),
        blob["y_std"].numpy(),
        n_feat,
        adj,
        cfg,
    )


# --------------------------------------------- poziom odniesienia u na zdrowej sieci

#: Ile odchyleń ponad średnią z czystego okna to jeszcze „normalne" u tej krawędzi.
#: NIE jest to parametr do strojenia — wybrany raz, na zrzucie BC, przed T2, z tabeli
#: rozrzutu log ρ w grupie softmaxu (patrz `EdgeReference`). Zmiana = wiersz ablacji.
REFERENCE_K = 2.0
REFERENCE_FORMAT = 1


@dataclass(frozen=True)
class EdgeReference:
    """u, które krawędź pokazuje, gdy NIC się nie zepsuło: `mean + k·sd` z czystego okna.

    PO CO TO ISTNIEJE. u nie spada do zera na zdrowych detektorach — kolejka jest
    nieprzewidywalna sama z siebie, więc głowa uczciwie raportuje σ > 0, RÓŻNE dla
    różnych skrzyżowań (σ skaluje się z obciążeniem: `results/gate_2_2.md`, u na
    czystym 0,492 w JiNan vs 0,380 na lżejszym r2000). Bez odjęcia tego poziomu
    `log ρ = −u/τ` karze krawędź za to, że jest ruchliwa, a nie za awarię —
    zmierzone na czystym: rozstęp log ρ w grupie softmaxu **1,13**, czyli najlepszy
    i najgorszy sąsiad różnią się trzykrotnie na ZDROWEJ sieci. To łamie A3a
    („no-op na czystym") i psuje porównywalność T1 z rdzeniem.

    DLACZEGO `mean + 2·sd`, a nie średnia ani kwantyl. Zmierzone na tej samej
    głowie, średni rozstęp log ρ w grupie (τ = 0,38), ogon epizodu:

        odniesienie    clean    mask0p25    mask0p5    gauss2
        średnia        1,132      1,735      1,862     3,307
        q90            0,527      0,901      0,942     2,408
        mean + 2·sd    0,248      0,454      0,446     1,484

    `mean + 2·sd` jest jednocześnie NAJCISZSZE na czystym i ma NAJWIĘKSZY
    kontrast awaria/czyste (mask0p25: 1,8× vs 1,5× dla średniej; gauss2: 6,0×
    vs 2,9×). Kwantyl q90 odpada dodatkowo przez ogon: jedna krawędź miała q90
    = 2,46, czyli stałaby się ślepa na wszystko poniżej katastrofy.

    τ tej tabeli nie zmienia — skaluje wszystkie wiersze liniowo (log ρ ∝ 1/τ),
    więc wybór odniesienia i wybór τ są niezależne.

    KOLEJNOŚĆ: `u_edge` idzie w kolejności `edges`, czyli (CEL i, ŹRÓDŁO j), tak
    jak zwraca `edge_uncertainty` — NIE w kolejności `graph.edge_index`, która
    jest (źródło, cel) i inaczej posortowana. Przepięcie jednej na drugą to
    Krok 4 i jedyne miejsce, gdzie ta permutacja ma prawo się wydarzyć.
    """

    u_edge: np.ndarray  # [E], w kolejności `edges`
    u_self: np.ndarray  # [N], w kolejności węzłów — pętle własne
    edges: tuple[tuple[int, int], ...]
    network: str
    k: float
    scenario: str
    head_sha: str

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            format=np.array(REFERENCE_FORMAT),
            u_edge=self.u_edge,
            u_self=self.u_self,
            edges=np.array(self.edges, dtype=np.int64).reshape(-1, 2),
            network=np.array(self.network),
            k=np.array(self.k),
            scenario=np.array(self.scenario),
            head_sha=np.array(self.head_sha),
        )
        return path


def file_sha(path: str | Path) -> str:
    """Odcisk artefaktu głowy — 16 znaków wystarczy, to nie jest zabezpieczenie
    kryptograficzne, tylko wykrywacz rozjazdu „u_ref z innej głowy niż wagi"."""
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def reference_path(network: str) -> Path:
    from tsc_marl.paths import RUNS  # noqa: PLC0415

    return RUNS / "heads" / f"u_ref__{network}.npz"


def edge_reference(
    fitted: Fitted,
    data: Loaded,
    sources: np.ndarray,
    *,
    network: str,
    head_sha: str,
    k: float = REFERENCE_K,
    scenario: str = "clean",
) -> EdgeReference:
    """Policz odniesienie na TRENINGOWEJ części czystego okna.

    Treningowej, nie całej: gdyby u_ref liczyło się na tych samych krokach, na
    których potem sprawdzamy „no-op na czystym", ta własność byłaby tautologią.
    Ogon epizodu zostaje wolny na sprawdzenie.
    """
    train = data.mask(fitted.cfg.split, train=True)
    _, sigma = fitted.predict(data, data.x[scenario])
    u_mov = movement_uncertainty(sigma / fitted.y_std, sources.shape[1])  # [T, N, n_mov]
    u_edge, edges = edge_uncertainty(u_mov, sources, "mean")

    e, s = u_edge[train], u_mov[train].mean(-1)  # pętla własna: ŚREDNIA PO WSZYSTKICH
    return EdgeReference(  # ruchach węzła, też tych z brzegu sieci (decyzja 2 Kroku 4)
        u_edge=e.mean(0) + k * e.std(0),
        u_self=s.mean(0) + k * s.std(0),
        edges=tuple((int(i), int(j)) for i, j in edges),
        network=network,
        k=float(k),
        scenario=scenario,
        head_sha=head_sha,
    )


def load_reference(
    path: str | Path, *, network: str | None = None, head: str | Path | None = None
) -> EdgeReference:
    """Odczytaj odniesienie. `head` podana = sprawdzenie, że pasuje do TYCH wag."""
    blob = np.load(Path(path), allow_pickle=False)
    if int(blob["format"]) != REFERENCE_FORMAT:
        raise ValueError(f"format u_ref {int(blob['format'])}, oczekiwany {REFERENCE_FORMAT}")
    ref = EdgeReference(
        u_edge=blob["u_edge"],
        u_self=blob["u_self"],
        edges=tuple((int(i), int(j)) for i, j in blob["edges"]),
        network=str(blob["network"]),
        k=float(blob["k"]),
        scenario=str(blob["scenario"]),
        head_sha=str(blob["head_sha"]),
    )
    if network is not None and ref.network != network:
        raise ValueError(f"u_ref policzone na {ref.network!r}, a sieć to {network!r}")
    if head is not None and (sha := file_sha(head)) != ref.head_sha:
        raise ValueError(
            f"u_ref policzone inną głową (odcisk {ref.head_sha}, plik ma {sha}) — "
            "u jest w jednostkach TEJ głowy, więc próg przestaje cokolwiek znaczyć; "
            "przelicz: `tsc obshead --save`"
        )
    return ref


# ------------------------------------------------------- u -> log ρ (wejście warstwy)

#: Skala modulacji: ile jednostek NADMIARU u (ponad `EdgeReference`) to jeden nat
#: w logicie attention. Dobrane na zrzucie BC regułą zapisaną PRZED T2: „scenariusz
#: gauss2 ma dawać średni rozstęp log ρ w grupie softmaxu równy ln 10", czyli
#: najbardziej i najmniej wiarygodny sąsiad różnią się tam dziesięciokrotnie.
#: Wyszło 0,245 → zaokrąglone. Co z tego wychodzi na pozostałych scenariuszach
#: (ogon epizodu, JiNan, głowa okno 12 + trwałość):
#:
#:     scenariusz              średni rozstęp   mediana   q90
#:     clean                   0,38  (×1,5)      0,00     0,00
#:     mask0p25                0,69  (×2,0)      0,00     2,34
#:     mask0p25_intersection   0,48  (×1,6)      0,00     1,36
#:     mask0p5                 0,68  (×2,0)      0,00     2,28
#:     gauss1                  0,89  (×2,4)      0,00     3,38
#:     gauss2                  2,26  (×9,5)      0,00     7,41
#:     drift0p3                0,55  (×1,7)      0,00     0,92
#:
#: MEDIANA ZERO wszędzie, łącznie z awariami: mechanizm jest RZADKI w czasie —
#: milczy, dopóki u nie wyjdzie ponad własną normę krawędzi, i reaguje na wyskoki.
#: Na czystym nawet q90 = 0, czyli w ≥90% kroków modulacji NIE MA wcale; średnia
#: 0,38 bierze się z rzadkich pików. To jest A3a zmierzone, nie założone.
#: `drift0p3` dostaje 0,55 mimo AUROC 0,536 — głowa dryfu nie wykrywa, więc to
#: modulacja w dużej mierze przypadkowa. Do zapisania w rozdziale 4.
DEFAULT_TAU = 0.25


def log_rho(
    u_edge: np.ndarray,
    u_self: np.ndarray,
    ref: EdgeReference,
    edge_index: torch.Tensor,
    *,
    tau: float = DEFAULT_TAU,
) -> torch.Tensor:
    """u -> log ρ [E + N] w kolejności `edge_index` PO `add_self_loops`.

        log ρ = −relu(u − u_ref) / τ

    `relu`, nie samo odejmowanie: krawędź spokojniejsza niż zwykle ma zostać przy
    wadze normalnej, a nie dostać premię. Attention widzi tylko różnice względne,
    więc premia za ciszę byłaby kolejnym kanałem zmieniającym zachowanie na
    czystym — a A3a mówi, że na czystym ma nie być zmiany.

    PUŁAPKA KOLEJNOŚCI, jedyny powód, dla którego ta funkcja w ogóle istnieje.
    `ref.edges` (za `edge_uncertainty`) to pary **(CEL i, ŹRÓDŁO j)**, posortowane
    po celu. `edge_index` to kolumny **(ŹRÓDŁO, CEL)** w kolejności LibSignala.
    Trzy rzeczy naraz: para odwrócona, sortowanie inne, a zbiór krawędzi na
    naszych sieciach jest SYMETRYCZNY — więc pomyłka w kolejności pary NIE rzuci
    `KeyError`. Lookup się uda i przypisze niepewność krawędzi j→i kolumnie i→j:
    pełny, poprawnie długi wektor, w którym każda wartość siedzi na krawędzi
    przeciwnej do swojej. Nic tego nie zgłosi. Stąd jawny słownik i test, który
    ustawia u na JEDNEJ krawędzi i sprawdza, w której kolumnie wylądowało.

    Pętle własne dokleja się na końcu w kolejności węzłów, bo tak robi
    `add_self_loops` — kontrakt `UncertaintyWeightedAttention`.
    """
    if u_edge.shape != ref.u_edge.shape or u_self.shape != ref.u_self.shape:
        raise ValueError(
            f"u ma kształty {u_edge.shape}/{u_self.shape}, a odniesienie "
            f"{ref.u_edge.shape}/{ref.u_self.shape}"
        )
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index ma być [2, E], jest {tuple(edge_index.shape)}")
    if tau <= 0:
        raise ValueError(f"τ musi być dodatnie, jest {tau}")

    kolumny = [(int(src), int(dst)) for src, dst in edge_index.t().tolist()]
    gdzie = {(i, j): k for k, (i, j) in enumerate(ref.edges)}
    try:  # (źródło, cel) -> szukamy u krawędzi (cel, źródło)
        order = [gdzie[(dst, src)] for src, dst in kolumny]
    except KeyError as brak:
        raise ValueError(
            f"krawędź {brak.args[0]!r} z edge_index nie ma swojego u w odniesieniu "
            f"({len(ref.edges)} krawędzi) — u_ref policzone na innej sieci albo na "
            "innym zestawie ruchów (patrz `drop_always_green` w `movement_sources`)"
        ) from brak
    if len(set(order)) != len(ref.edges):
        raise ValueError(
            f"edge_index pokrywa {len(set(order))} z {len(ref.edges)} krawędzi odniesienia"
        )

    nadmiar_e = np.maximum(u_edge - ref.u_edge, 0.0)[order]
    nadmiar_s = np.maximum(u_self - ref.u_self, 0.0)
    out = -np.concatenate([nadmiar_e, nadmiar_s]) / tau
    return torch.from_numpy(out.astype(np.float32))


def group_spread(log_rho: torch.Tensor, edge_index: torch.Tensor, n_nodes: int) -> float:
    """Średni rozstęp (max − min) log ρ w GRUPIE SOFTMAXU węzła: krawędzie wchodzące
    do i plus jego pętla własna.

    To jest jedyna uczciwa miara „ile mechanizm w ogóle robi". Sam poziom log ρ nie
    mówi nic, bo softmax jest niezmienniczy na stałą wspólną dla grupy (test 3 z
    Kroku 2) — wspólne przesunięcie całej grupy warstwa ignoruje. Liczy się WYŁĄCZNIE
    rozpiętość wewnątrz grupy: 0 = attention nie widzi różnicy, ln 10 ≈ 2,3 = najlepszy
    i najgorszy sąsiad różnią się dziesięciokrotnie. Na tej mierze dobrane jest τ.
    """
    lr = log_rho.detach().numpy()
    cel = np.concatenate([edge_index[1].numpy(), np.arange(n_nodes)])
    grupy = [lr[cel == n] for n in range(n_nodes)]
    return float(np.mean([g.max() - g.min() for g in grupy if len(g)]))


class OnlineUncertainty:
    """Głowa w pętli sterowania: bufor okna -> u -> log ρ, jeden krok na raz.

    Offline głowa dostaje cały epizod naraz i indeksuje okno przez `window_index`.
    W pętli sterowania nie ma „całego epizodu" — jest bufor ostatnich odczytów. Ta
    klasa robi z bufora DOKŁADNIE to samo wejście, które zbudowałby tryb offline,
    i o to chodzi w jej testach: `build_inputs` przyjmuje dowolne [T, N, F], więc
    podajemy bufor jako T = `window` z NAJNOWSZYM ODCZYTEM W WIERSZU 0 i indeksem
    okna `[0, 1, …, k−1]` przyciętym do długości bufora.

    ROZGRZEWKA. Na starcie epizodu bufor jest krótszy niż okno. Offline `window_index`
    przycina wtedy do początku epizodu (powtarza najstarszy dostępny indeks) — i my
    musimy powtórzyć to CO DO INDEKSU, nie „jakoś rozsądnie". Inaczej pierwsze kroki
    każdego epizodu mają inny rozkład u niż ten, na którym policzone jest `u_ref`,
    a mechanizm zaczyna epizod od wygaszania losowych sąsiadów.

    CZYSZCZENIE BUFORA to nie higiena, tylko poprawność: odczyt sprzed `world.reset()`
    opisuje zupełnie inny stan ruchu. Sterownik woła `reset()` z `reset_stats()`,
    który `robustness/degrade.py` odpala na starcie każdego epizodu.
    """

    def __init__(
        self,
        head: Fitted,
        ref: EdgeReference,
        sources: np.ndarray,
        edge_index: torch.Tensor,
        *,
        tau: float = DEFAULT_TAU,
    ) -> None:
        if sources.shape[0] != int(edge_index.max()) + 1:
            raise ValueError(
                f"sources ma {sources.shape[0]} węzłów, edge_index widzi "
                f"{int(edge_index.max()) + 1}"
            )
        self.head, self.ref, self.sources, self.edge_index, self.tau = (
            head,
            ref,
            sources,
            edge_index,
            tau,
        )
        self.n_nodes = int(sources.shape[0])
        self.n_mov = int(sources.shape[1])
        self.adj = neighbor_matrix(edge_index.numpy(), self.n_nodes) if head.cfg.neighbors else None
        self.n_persist = head.n_feat if head.cfg.persistence else 0
        self._buf: list[np.ndarray] = []

    def reset(self) -> None:
        self._buf.clear()

    @torch.no_grad()
    def log_rho(self, obs: np.ndarray) -> torch.Tensor:
        """Dołóż odczyt do okna i policz log ρ [E + N] dla TEGO kroku."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[0] != self.n_nodes:
            raise ValueError(f"obserwacja ma {obs.shape[0]} wierszy, graf ma {self.n_nodes}")
        self._buf.insert(0, obs)  # najnowszy z przodu — tak indeksuje `window_index`
        del self._buf[self.head.cfg.window :]

        hist = np.stack(self._buf)  # [m, N, F], m ≤ window, wiersz 0 = najnowszy
        # PEŁNA macierz okien [m, k], nie jeden wiersz: gałąź sąsiadów w `build_inputs`
        # liczy się dla CAŁEGO `x` (`einsum` po wymiarze czasu), więc przy jednym
        # wierszu `win` kawałki wyjścia mają różną wysokość i `concatenate` pada.
        # Liczymy wszystkie m wierszy (MLP na 12 wierszach to nic) i bierzemy wiersz 0.
        m, k = len(self._buf), self.head.cfg.window
        win = np.minimum(np.arange(m)[:, None] + np.arange(k)[None, :], m - 1)
        z = build_inputs(hist, win, self.adj, self.n_persist)[:1]  # [1, N, D]
        if z.shape[-1] != self.head.in_mean.shape[0]:
            raise ValueError(
                f"wejście głowy ma szerokość {z.shape[-1]}, a głowa uczyła się na "
                f"{self.head.in_mean.shape[0]} — inna konfiguracja okna/sąsiadów/trwałości"
            )

        z = (z - self.head.in_mean) / self.head.in_std
        self.head.model.eval()
        _, sigma = self.head.model(torch.from_numpy(z.astype(np.float32)))
        # przez `y_std` tam i z powrotem, DOKŁADNIE jak `Fitted.predict` + `evaluate`:
        # u ma być w tych samych jednostkach co `u_ref`, co do ostatniego bitu
        s_std = (sigma.numpy() * self.head.y_std) / self.head.y_std
        u_mov = movement_uncertainty(s_std, self.n_mov)  # [1, N, n_mov]
        u_edge, _ = edge_uncertainty(u_mov, self.sources, "mean")  # [1, E]
        return log_rho(u_edge[0], u_mov[0].mean(-1), self.ref, self.edge_index, tau=self.tau)


def train_head(
    network: str = "cityflow_jinan3x4_4ph",
    cfg: ObsConfig | None = None,
    out: str | Path | None = None,
) -> tuple[Path, Path]:
    """Głowa + jej poziom odniesienia na zrzucie BC -> (ścieżka głowy, ścieżka u_ref).

    Oba naraz i zawsze, bo rozjazd między nimi jest cichy: u_ref z innej głowy to
    liczby w innych jednostkach, a nie błąd, który coś zgłosi. Osobno od `run()`,
    bo `run()` liczy całą bramkę razem z deep ensemble z 2.0.
    """
    from tsc_marl.paths import RUNS  # noqa: PLC0415
    from tsc_marl.uncertainty import prototype as P  # noqa: PLC0415

    P.cap_threads()
    base = load(RUNS / "bc" / f"advancedmp__{network}__ep3" / "dataset.npz")
    fitted, _ = fit(base, cfg or ObsConfig())
    head = fitted.save(out or head_path(network), network=network)

    sources = movement_sources(network, base.dead["clean"].shape[-1])
    ref = edge_reference(fitted, base, sources, network=network, head_sha=file_sha(head))
    return head, ref.save(reference_path(network))


# ---------------------------------------------------------------- metryki


def auroc(pos: np.ndarray, neg: np.ndarray, cap: int = 200_000, seed: int = 0) -> float:
    """AUROC przez rangi (Mann–Whitney), remisy uśredniane."""
    pos, neg = np.asarray(pos, float).ravel(), np.asarray(neg, float).ravel()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(pos) > cap:
        pos = rng.choice(pos, cap, replace=False)
    if len(neg) > cap:
        neg = rng.choice(neg, cap, replace=False)
    x = np.concatenate([pos, neg])
    ranks = np.empty(len(x))
    ranks[x.argsort(kind="mergesort")] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def movement_uncertainty(sigma_std: np.ndarray, n_mov: int) -> np.ndarray:
    """[.., n_feat] -> [.., n_mov]: max σ po cechach ruchu (EP i, jeśli jest, demand)."""
    blocks = [
        sigma_std[..., b * n_mov : (b + 1) * n_mov] for b in range(sigma_std.shape[-1] // n_mov)
    ]
    return np.max(np.stack(blocks), axis=0)


def edge_uncertainty(u_mov: np.ndarray, sources: np.ndarray, how: str = "max"):
    """u ruchów -> u krawędzi (j→i). -> ([T, E], lista (i, j)). Ruchy z brzegu
    sieci (−1) nie należą do żadnej krawędzi grafu."""
    edges = sorted({(i, int(j)) for i in range(sources.shape[0]) for j in sources[i] if j >= 0})
    agg = np.max if how == "max" else np.mean
    cols = [agg(u_mov[:, i, sources[i] == j], axis=-1) for i, j in edges]
    return np.stack(cols, axis=-1), edges


def corrupted_movements(x, x_clean, dead, n_mov: int) -> np.ndarray:
    """[T, N, n_mov] — ruch z martwym detektorem ALBO z błędem cechy > CORRUPT_TAU.
    `x`, `x_clean` to już same bloki cech ruchów (bez one-hotu fazy)."""
    err = np.abs(x - x_clean)
    big = np.zeros(x.shape[:-1] + (n_mov,), dtype=bool)
    for b in range(err.shape[-1] // n_mov):
        big |= err[..., b * n_mov : (b + 1) * n_mov] > CORRUPT_TAU
    return big | dead.astype(bool)


def evaluate(fitted: Fitted, data: Loaded, sources: np.ndarray, scenarios=None) -> dict[str, Any]:
    """Metryki na OGONIE epizodu dla każdego scenariusza zbioru."""
    n_mov = data.dead["clean"].shape[-1]
    tail = data.mask(fitted.cfg.split, train=False)
    nf = fitted.n_feat
    res: dict[str, Any] = {}
    clean_u = clean_edge = None
    for name in ["clean", *sorted(s for s in data.x if s != "clean")]:
        if scenarios and name not in scenarios and name != "clean":
            continue
        x = data.x[name]
        mu, sigma = fitted.predict(data, x)
        mu, sigma = mu[tail], sigma[tail]
        xc, xt = data.x_clean[tail][..., :nf], x[tail][..., :nf]
        s_std = sigma / fitted.y_std
        u_mov = movement_uncertainty(s_std, n_mov)
        bad = corrupted_movements(xt, xc, data.dead[name][tail], n_mov)
        # BŁĘDNY odczyt = cecha faktycznie odjechała od prawdy. Martwy detektor na pustym
        # pasie raportuje prawdziwe 0 — jest „popsuty", ale żadna głowa nie ma go czym
        # wykryć (zero to wartość modalna). Detekcję liczymy więc też na samych błędnych.
        wrong = corrupted_movements(xt, xc, np.zeros_like(data.dead[name][tail]), n_mov)
        u_max, _ = edge_uncertainty(u_mov, sources, "max")
        u_mean, edges = edge_uncertainty(u_mov, sources, "mean")
        bad_edge = np.stack([bad[:, i, sources[i] == j].any(-1) for i, j in edges], -1)
        if name == "clean":
            clean_u, clean_edge = u_mov, (u_max, u_mean)
        err_mu = mu - xc
        row = {
            "u_mean": float(u_mov.mean()),
            "coverage90": float((np.abs(err_mu) <= 1.645 * sigma).mean()),
            "calib": float(np.sqrt((err_mu**2).mean()) / np.sqrt((sigma**2).mean())),
            "rmse_input": float(np.sqrt(((xt - xc) ** 2).mean())),
            "rmse_mu": float(np.sqrt((err_mu**2).mean())),
            "corrupt_share": float(bad.mean()),
            "wrong_share": float(wrong.mean()),
            "auroc_wrong": auroc(u_mov[wrong], clean_u) if name != "clean" else float("nan"),
            "auroc_mov": auroc(u_mov[bad], clean_u) if name != "clean" else float("nan"),
            "auroc_edge_max": (
                auroc(u_max[bad_edge], clean_edge[0]) if name != "clean" else float("nan")
            ),
            "auroc_edge_mean": (
                auroc(u_mean[bad_edge], clean_edge[1]) if name != "clean" else float("nan")
            ),
        }
        res[name] = row
    return res


# ------------------------------------------------------------------ bramka


def _gauss_series(res: dict[str, Any]) -> list[tuple[float, float]]:
    pts = [(0.0, res["clean"]["u_mean"])]
    for name, r in res.items():
        if name.startswith("gauss"):
            pts.append((float(name[5:].replace("p", ".")), r["u_mean"]))
    return sorted(pts)


def gate(res: dict[str, Any], sep: dict[str, Any]) -> dict[str, Any]:
    gs = _gauss_series(res)
    # kryteria na odczytach BŁĘDNYCH (patrz `evaluate`) — pytanie brzmi „czy głowa widzi
    # zły odczyt", a nie „czy wie, że detektor jest martwy, gdy raportuje prawdę"
    unseen = [res[n]["auroc_wrong"] for n in UNSEEN if n in res]
    seen = [res[n]["auroc_wrong"] for n in ("gauss1", "gauss2", "mask0p25") if n in res]
    drift = [res[n]["auroc_wrong"] for n in res if n.startswith("drift")]
    out = {
        "a_series": gs,
        "a_monotonic": [v for _, v in gs] == sorted(v for _, v in gs),
        "c_coverage_clean": res["clean"]["coverage90"],
        "c_coverage_seen": float(np.mean([res[n]["coverage90"] for n in ("gauss1", "gauss2")])),
        "d_auroc_seen_min": min(seen) if seen else float("nan"),
        "e_auroc_unseen_min": min(unseen) if unseen else float("nan"),
        "e_auroc_drift_max": max(drift) if drift else float("nan"),
        "f_edge_max_vs_mean": float(
            np.nanmean(
                [res[n]["auroc_edge_max"] - res[n]["auroc_edge_mean"] for n in res if n != "clean"]
            )
        ),  # fmt: skip
        "sep": sep,
    }
    out["passed"] = bool(
        out["a_monotonic"] and out["d_auroc_seen_min"] >= 0.8 and out["e_auroc_unseen_min"] >= 0.7
    )
    return out


def separation(fitted: Fitted, base: Loaded, ood: dict[str, Loaded], ens_fn) -> dict[str, Any]:
    """Tabela 2×2 z Załącznika A7: {czyste, popsute} × {ruch treningowy, OOD}.

    `u_obs` ma reagować na WIERSZ (awaria), `u_ep` na KOLUMNĘ (nieznany ruch).
    Jeśli oba reagują na wszystko — nie ma dekompozycji, tylko jeden sygnał
    w dwóch przebraniach, a dwupoziomowy fallback traci uzasadnienie."""
    out: dict[str, Any] = {}
    sets = {"trening": base, **ood}
    for label, data in sets.items():
        tail = data.mask(fitted.cfg.split, train=False)
        n_mov = data.dead["clean"].shape[-1]
        for scen in ("clean", "gauss2", "mask0p5"):
            if scen not in data.x:
                continue
            _, sigma = fitted.predict(data, data.x[scen])
            u_obs = movement_uncertainty(sigma[tail] / fitted.y_std, n_mov).mean()
            u_ep = ens_fn(data, data.x[scen][tail]) if ens_fn else float("nan")
            out[f"{label}|{scen}"] = {"u_obs": float(u_obs), "u_ep": float(u_ep)}
    return out


def _fmt(v: float, f: str = ".3f") -> str:
    return "—" if v is None or not np.isfinite(v) else format(v, f)


def to_markdown(res: dict[str, Any], g: dict[str, Any], cfg: ObsConfig, ood_res: dict) -> str:
    ok = {True: "TAK", False: "NIE"}
    lines = [
        "# Bramka prototypu 2.2 — głowa OBSERWACYJNA (niepewność cech wejściowych)",
        "",
        f"Okno {cfg.window} odczytów{' + średnia sąsiadów upstream' if cfg.neighbors else ''}, "
        f"{cfg.epochs} epok, Gaussian NLL względem `x_clean`, trening na: "
        f"{', '.join(cfg.train_scenarios)}. Ewaluacja na ogonie epizodu. u = σ w jednostkach "
        "odchylenia cechy na czystym; u ruchu = max po jego cechach (EP, demand).",
        "",
        "| scenariusz | tren. | popsute ruchy | błędne ruchy | u średnie | AUROC ruch (popsute) "
        "| AUROC ruch (błędne) | AUROC krawędź (max) "
        "| AUROC krawędź (mean) | pokrycie 90% | RMSE wejścia | RMSE μ |",
        "|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in res.items():
        lines.append(
            f"| {name} | {'x' if name in cfg.train_scenarios else ''} | {r['corrupt_share']:.1%} "
            f"| {r['wrong_share']:.1%} | {r['u_mean']:.3f} | {_fmt(r['auroc_mov'])} "
            f"| {_fmt(r['auroc_wrong'])} | {_fmt(r['auroc_edge_max'])} "
            f"| {_fmt(r['auroc_edge_mean'])} | {r['coverage90']:.1%} | {r['rmse_input']:.3f} "
            f"| {r['rmse_mu']:.3f} |"
        )
    if ood_res:
        lines += ["", "### Te same metryki na zbiorach OOD (ruch spoza treningu)", "",
                  "| zbiór | scenariusz | u średnie | AUROC ruch | pokrycie 90% |",
                  "|---|---|---:|---:|---:|"]  # fmt: skip
        for label, rr in ood_res.items():
            for name in ("clean", "gauss2", "mask0p5", "drift0p3"):
                if name in rr:
                    r = rr[name]
                    lines.append(
                        f"| {label} | {name} | {r['u_mean']:.3f} | {_fmt(r['auroc_mov'])} "
                        f"| {r['coverage90']:.1%} |"
                    )
    sep = g["sep"]
    labels = sorted({k.split("|")[0] for k in sep}, key=lambda s: (s != "trening", s))
    lines += [
        "",
        "## Separacja 2×2 — czy dwa kanały mierzą dwie różne rzeczy",
        "",
        "`u_obs` = średnie u głowy obserwacyjnej, `u_ep` = epistemic deep ensemble "
        "(MI między członkami, prototyp 2.0). Wiersz = stan odczytów, kolumna = ruch.",
        "",
        "| odczyty \\ ruch | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for scen in ("clean", "gauss2", "mask0p5"):
        cells = []
        for lab in labels:
            v = sep.get(f"{lab}|{scen}")
            cells.append(
                "—" if v is None else f"u_obs {v['u_obs']:.3f} · u_ep {_fmt(v['u_ep'], '.4f')}"
            )
        lines.append(f"| {scen} | " + " | ".join(cells) + " |")
    lines += ["", *_separation_findings(sep, labels)]
    gs = g["a_series"]
    lines += [
        "",
        "## Kryteria wyjścia",
        "",
        f"* **(a) monotoniczność** — u po σ szumu ({', '.join(f'{s:g}' for s, _ in gs)}): "
        f"{', '.join(f'{v:.3f}' for _, v in gs)} → **{ok[g['a_monotonic']]}** "
        "(3,5 i 4,0 NIEWIDZIANE w treningu).",
        f"* **(c) kalibracja** — pokrycie przedziału 90% na czystym "
        f"{g['c_coverage_clean']:.1%}, na szumie widzianym {g['c_coverage_seen']:.1%}.",
        f"* **(d) detekcja na widzianym** — min AUROC ruchu (gauss1, gauss2, mask0p25): "
        f"**{_fmt(g['d_auroc_seen_min'])}** (próg 0,8).",
        f"* **(e) detekcja na NIEWIDZIANYM** — min AUROC po {', '.join(UNSEEN)}: "
        f"**{_fmt(g['e_auroc_unseen_min'])}** (próg 0,7); `drift` (inna rodzina): max AUROC "
        f"**{_fmt(g['e_auroc_drift_max'])}**.",
        f"* **(f) agregacja ruch→krawędź** — AUROC(max) − AUROC(mean), średnio: "
        f"{g['f_edge_max_vs_mean']:+.3f} → w attention "
        f"**{'max' if g['f_edge_max_vs_mean'] >= 0 else 'mean'}**.",
        "",
        f"**Bramka 2.2 (głowa obserwacyjna): {'ZALICZONA' if g['passed'] else 'NIEZALICZONA'}.**",
        "",
        "## Czego ta tabela NIE mówi",
        "",
        "* Prototyp jest NADZOROWANY (symulator zna `x_clean`). W pętli sterowania głowa działa "
        "tak samo — trening offline na trajektoriach zamrożonej polityki, inferencja bez prawdy.",
        "* AUROC porównuje komórki popsute z CZYSTEGO scenariusza — nie mierzy lokalizacji "
        "w obrębie jednego epizodu (tam, gdzie wszystkie detektory szumią, nie ma czego "
        "lokalizować).",
        f"* „Popsuty ruch” = martwy detektor albo błąd cechy > {CORRUPT_TAU} pojazdu. Dryf małych "
        "zliczeń rzadko przekracza próg, więc jego AUROC liczy się na nielicznych komórkach.",
    ]
    return "\n".join(lines) + "\n"


def _separation_findings(sep: dict[str, Any], labels: list[str]) -> list[str]:
    def val(lab, scen, key):
        v = sep.get(f"{lab}|{scen}")
        return float("nan") if v is None else v[key]

    ood = [lab for lab in labels if lab != "trening"]
    if not ood:
        return []
    row_obs = np.nanmean([val(lab, "gauss2", "u_obs") - val(lab, "clean", "u_obs")
                          for lab in ["trening", *ood]])  # fmt: skip
    col_obs = np.nanmean(
        [val(lab, "clean", "u_obs") - val("trening", "clean", "u_obs") for lab in ood]
    )
    row_ep = np.nanmean([val(lab, "gauss2", "u_ep") - val(lab, "clean", "u_ep")
                         for lab in ["trening", *ood]])  # fmt: skip
    col_ep = np.nanmean(
        [val(lab, "clean", "u_ep") - val("trening", "clean", "u_ep") for lab in ood]
    )
    out = [
        f"* `u_obs`: przyrost od awarii (wiersz) **{row_obs:+.3f}**, od OOD (kolumna) "
        f"**{col_obs:+.3f}**.",
        f"* `u_ep`: przyrost od awarii **{row_ep:+.4f}**, od OOD **{col_ep:+.4f}**.",
    ]
    if np.isfinite(col_obs) and col_obs < 0:
        out.append(
            "* `u_obs` SPADA na ruchu spoza treningu: σ głowy skaluje się z obciążeniem (mniej "
            "pojazdów = mniejszy możliwy błąd odczytu), a zbiory OOD są lżejsze. Kierunek jest "
            "nieszkodliwy dla fallbacku (brak fałszywych alarmów), ale próg na `u_obs` trzeba "
            "liczyć względem σ przewidywanego na czystym, nie jako stałą bezwzględną."
        )
    obs_ok = np.isfinite(row_obs) and row_obs > 3 * max(abs(col_obs), 1e-9)
    ep_ok = np.isfinite(col_ep) and col_ep > max(row_ep, 0)
    out.append(
        "* Werdykt separacji: "
        + ("`u_obs` reaguje na awarię, nie na OOD — " if obs_ok else "`u_obs` NIE jest czysty — ")
        + (
            "`u_ep` reaguje mocniej na OOD niż na awarię."
            if ep_ok
            else "`u_ep` NIE reaguje na OOD mocniej niż na awarię — dwupoziomowy fallback nie ma "
            "tu empirycznej podstawy na poziomie (a)."
        )  # fmt: skip
    )
    return out


# ------------------------------------------------------- poprawka na maskach

#: Warianty porównywane w próbie poprawy bramki 2.2 na martwych detektorach. Lista
#: i kryterium zapisane PRZED pierwszym uruchomieniem (2026-09-13) — wynik nie może
#: przesunąć progu. Wariant 0 = głowa z bramki 2.2, punkt odniesienia.
MASK_VARIANTS = (
    ("okno 4 (bramka 2.2)", ObsConfig()),
    ("okno 12", ObsConfig(window=12)),
    ("okno 12 + trwałość", ObsConfig(window=12, persistence=True)),
)
MASK_UNSEEN = ("mask0p5", "mask0p25_intersection")
NOISE_UNSEEN = ("gauss3p5", "gauss4", "urand3p5")
MASK_THRESHOLD = 0.70  # AUROC na błędnych odczytach, OBIE niewidziane maski
NOISE_FLOOR = 0.80  # szum niewidziany nie może na tym stracić


def mask_variants(network: str = "cityflow_jinan3x4_4ph") -> Path:
    """Trzy warianty głowy na tym samym zrzucie -> `results/gate_2_2_masks.md`."""
    from tsc_marl.paths import RESULTS, RUNS  # noqa: PLC0415
    from tsc_marl.uncertainty import prototype as P  # noqa: PLC0415

    P.cap_threads()
    base = load(RUNS / "bc" / f"advancedmp__{network}__ep3" / "dataset.npz")
    sources = movement_sources(network, base.dead["clean"].shape[-1])
    results = []
    for label, cfg in MASK_VARIANTS:
        fitted, _ = fit(base, cfg)
        results.append((label, cfg, evaluate(fitted, base, sources)))

    cols = ("gauss1", "gauss2", "gauss4", "urand3p5", "mask0p25", "mask0p5",
            "mask0p25_intersection", "drift0p3")  # fmt: skip
    out = [
        "# Bramka 2.2 — próba poprawy na martwych detektorach",
        "",
        f"Kryterium ustalone PRZED uruchomieniem: wariant zalicza, jeśli AUROC na błędnych "
        f"odczytach ≥ {MASK_THRESHOLD} na OBU niewidzianych maskach ({', '.join(MASK_UNSEEN)}) "
        f"i ≥ {NOISE_FLOOR} na każdym niewidzianym szumie ({', '.join(NOISE_UNSEEN)}). "
        "Trening na clean/gauss1/gauss2/mask0p25, ewaluacja na ogonie epizodu.",
        "",
        "| wariant | " + " | ".join(cols) + " | pokrycie 90% (czyste) | zalicza |",
        "|---|" + "---:|" * len(cols) + "---:|:-:|",
    ]
    passing = []
    for label, _cfg, res in results:
        ok = (
            min(res[n]["auroc_wrong"] for n in MASK_UNSEEN) >= MASK_THRESHOLD
            and min(res[n]["auroc_wrong"] for n in NOISE_UNSEEN) >= NOISE_FLOOR
        )
        if ok:
            passing.append(label)
        cells = " | ".join(_fmt(res[c]["auroc_wrong"]) for c in cols)
        out.append(
            f"| {label} | {cells} | {res['clean']['coverage90']:.1%} | {'TAK' if ok else 'NIE'} |"
        )
    best = max(results, key=lambda r: min(r[2][n]["auroc_wrong"] for n in MASK_UNSEEN))
    out += [
        "",
        "## Werdykt",
        "",
        (
            f"**Bramka 2.2 ZALICZONA** wariantem: {', '.join(passing)}. Ten wariant zastępuje "
            "głowę z bramki 2.2 w mechanizmie Fazy 3."
            if passing
            else "**Bramka 2.2 ZAMKNIĘTA JAKO ZALICZONA CZĘŚCIOWO.** Żaden wariant nie osiąga "
            f"{MASK_THRESHOLD} na obu niewidzianych maskach. Najlepszy na maskach: "
            f"„{best[0]}” (min AUROC "
            f"{min(best[2][n]['auroc_wrong'] for n in MASK_UNSEEN):.3f}). Głowa jest estymatorem "
            "niepewności dla awarii SZUMOWYCH, a na awariach CICHYCH (martwe detektory) działa "
            "słabo — do opisania wprost; ablacja rule-gate vs UQ na maskach zyskuje na wadze."
        ),
    ]
    path = RESULTS / "gate_2_2_masks.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def run(cfg: ObsConfig | None = None, *, network: str = "cityflow_jinan3x4_4ph") -> Path:
    from tsc_marl.paths import RESULTS, RUNS  # noqa: PLC0415
    from tsc_marl.uncertainty import prototype as P  # noqa: PLC0415

    P.cap_threads()
    cfg = cfg or ObsConfig()
    base = load(RUNS / "bc" / f"advancedmp__{network}__ep3" / "dataset.npz")
    n_mov = base.dead["clean"].shape[-1]
    fitted, history = fit(base, cfg)
    res = evaluate(fitted, base, movement_sources(network, n_mov))

    ood: dict[str, Loaded] = {}
    ood_res: dict[str, Any] = {}
    for label in ("r2000", "hz", "d0p25"):
        net = P.OOD_NETWORKS[label]
        path = RUNS / "bc" / f"advancedmp__{net}__ep3" / "dataset.npz"
        if path.is_file():
            ood[label] = load(path)
            ood_res[label] = evaluate(
                fitted, ood[label], movement_sources(net, n_mov),
                scenarios=("gauss2", "mask0p5", "drift0p3"),
            )  # fmt: skip

    pcfg = P.TrainConfig()
    ensemble = P.fit_ensemble(base, pcfg)

    def ens_fn(data, x):
        return float(P.ensemble_uncertainty(ensemble, x, data, pcfg)["ens_epistemic"].mean())

    g = gate(res, separation(fitted, base, ood, ens_fn))
    out = RESULTS / "gate_2_2.md"
    out.write_text(to_markdown(res, g, cfg, ood_res), encoding="utf-8")
    (RESULTS / "gate_2_2.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "gate": g,
                "rows": res,
                "ood": ood_res,
                "history": history[-5:],
            },
            indent=1,
            default=str,
        ),  # fmt: skip
        encoding="utf-8",
    )
    return out
