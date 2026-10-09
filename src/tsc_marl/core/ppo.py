"""MAPPO — pętla uczenia rdzenia Fazy 1 (PPO + CTDE nad grafem).

Ten moduł nie wie nic o CityFlow. Dostaje tensory, oddaje gradienty. Dzięki temu
cały algorytm testuje się w milisekundach na syntetycznym zadaniu, a błąd we
wzorze na GAE nie ukrywa się za trzygodzinnym treningiem.

CO ROBI MAPPO Z PPO. Nic w samych wzorach — to jest zwykłe PPO, w którym
KAŻDY AGENT JEST OSOBNĄ PRÓBKĄ w batchu. Przy współdzielonych wagach (patrz
`nets.py`) 12 skrzyżowań × T decyzji to po prostu 12·T przykładów. Cała różnica
siedzi w tym, co widzi krytyk (2-hop + kontekst globalny) i czego nie widzi
actor (tylko o_i i sąsiedzi) — czyli w `nets.py`, nie tutaj.

PIĘĆ PUŁAPEK, KTÓRE TEN PLIK OBSŁUGUJE JAWNIE:

1. TRUNCATION ≠ TERMINATION. Epizod kończy się po 3600 s, a nie dlatego, że stan
   jest końcowy — ruch płynie dalej. Ustawienie `done=True` na ostatnim kroku
   wstawiłoby V(s_T)=0 dla stanu, który jest całkowicie zwyczajny, i systematycznie
   zaniżało wartości pod koniec epizodu. Dlatego bootstrapujemy `last_value`,
   a `done` zostaje zarezerwowane na prawdziwą terminację (u nas: nigdy).

2. MINIBATCH IDZIE PO CZASIE, NIE PO AGENTACH. Nie wolno wylosować połowy
   skrzyżowań z kroku t — GAT potrzebuje całego grafu, żeby policzyć attention.
   Losujemy zbiór KROKÓW i sklejamy je w jeden blokowo-diagonalny graf.

3. SKALA ZWROTÓW. ATS jest nieznormalizowany (pressure rzędu dziesiątek), więc
   surowa strata wartości potrafi być o rzędy większa od policy loss i zdominować
   wspólny gradient. `ValueNorm` trzyma bieżącą średnią/wariancję zwrotów: krytyk
   uczy się w przestrzeni znormalizowanej, a do GAE wartości wracają na surową
   skalę. To jest to, co w `nets.py::Critic` świadomie NIE zostało wbudowane.

4. GRADIENT W ROLLOUTCIE. Zbieranie danych musi lecieć w `torch.no_grad()` —
   inaczej graf obliczeniowy z całego epizodu (240 decyzji) zostaje w pamięci.
   To odpowiedzialność wołającego, tak samo jak przy `ActorCritic.act`.

5. SKLEJANIE log ρ W MINIBATCHU (Faza 3). Bufor trzyma [T, E+N] — każdy krok
   w jednym kawałku — a warstwa attention oczekuje układu sklejonego grafu:
   najpierw WSZYSTKIE krawędzie k grafów, dopiero potem k·N pętli własnych.
   Naiwne `reshape(-1)` ma poprawną długość i mylny układ, i nie zgłasza tego
   nic. Robi to `tiled_log_rho`, jedyną drogą, i tylko ona.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import Any

import torch
from torch import Tensor, nn

from tsc_marl.core.nets import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    """Hiperparametry startowe. Każdy z uzasadnieniem, bo w 1.6 będziemy je stroić
    i trzeba wiedzieć, co było wyborem, a co wartością domyślną z papieru.

    `gamma` = 0.95 świadomie ZA baseline'em (`configs/tsc/base.yml`), mimo że MAPPO
    zwykle bierze 0.99: dyskonto liczy się w DECYZJACH, a przy Δt=15 s epizod ma ich
    240, więc 0.95 to horyzont ~20 decyzji = 5 minut ruchu. Ta sama wartość co
    w CoLighcie/Advanced-CoLighcie trzyma porównanie na jednej osi.

    `max_grad_norm` = 0.5 to konwencja PPO, nie `grad_clip: 5.0` z `base.yml` —
    tamto jest ustawieniem DQN-a i przy PPO praktycznie nie tnie.
    """

    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    epochs: int = 10
    n_minibatches: int = 4
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    normalize_values: bool = True
    # Awaryjny hamulec: jeśli polityka odjedzie o więcej niż tyle nats na próbkę,
    # przerywamy epoki tego update'u. None = wyłączone (domyślnie — clip wystarcza).
    target_kl: float | None = None

    @classmethod
    def from_registry(cls, **overrides: Any) -> PPOConfig:
        """Hiperparametry z sekcji `model` YAML-a (`configs/tsc/gatppo.yml`).

        Muszą iść przez config, a nie przez argumenty w kodzie: `experiment.build_params`
        liczy odcisk sekcji `model`+`trainer`, więc zmiana `clip_range` ma unieważnić
        skip-if-done. Hiperparametr żyjący tylko w Pythonie po cichu przepuściłby
        bieg z innymi ustawieniami jako „już policzony"."""
        from common.registry import Registry  # noqa: PLC0415 - dopiero po bootstrapie

        param = Registry.mapping["model_mapping"]["setting"].param
        known = {f.name: param[f.name] for f in fields(cls) if f.name in param}
        return cls(**{**known, **overrides})

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma poza (0, 1]: {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda poza [0, 1]: {self.gae_lambda}")
        if self.clip_range <= 0 or self.epochs < 1 or self.n_minibatches < 1:
            raise ValueError("clip_range > 0 oraz epochs, n_minibatches >= 1")


class ValueNorm(nn.Module):
    """Bieżąca normalizacja zwrotów (MAPPO `use_valuenorm`).

    Krytyk przewiduje wartość ZNORMALIZOWANĄ, więc strata wartości ma skalę ~1
    niezależnie od tego, czy nagroda to kolejka (jednostki) czy pressure
    (dziesiątki). Do GAE wartości muszą wrócić na skalę surową — inaczej
    przewagi byłyby liczone w innych jednostkach niż nagrody.

    Statystyki żyją jako bufory, nie parametry: mają się zapisywać w checkpoincie
    (bez nich wznowiony trening liczyłby wartości w innej skali), ale nie mają
    dostawać gradientu.
    """

    def __init__(self, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Algorytm Welforda w wersji wsadowej (Chan) — stabilniejszy numerycznie
        niż trzymanie sumy kwadratów, która przy 200 epizodach po 240 decyzjach
        potrafi stracić precyzję."""
        batch_count = x.numel()
        if batch_count == 0:
            return
        batch_mean, batch_var = x.mean(), x.var(unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)

    def denormalize(self, x: Tensor) -> Tensor:
        return x * torch.sqrt(self.var + self.epsilon) + self.mean


def tiled_edge_index(edge_index: Tensor, n_nodes: int, k: int) -> Tensor:
    """`k` kopii tego samego grafu jako jeden blokowo-diagonalny [2, k·E].

    Robi to samo co `torch_geometric.Batch.from_data_list` (sprawdzone testem),
    ale bez budowania `k` obiektów `Data` na każdy minibatch. Wolno tak skracać
    WYŁĄCZNIE dlatego, że graf jest statyczny — gdyby topologia zmieniała się
    w czasie (Faza 5, sieć z wyłączanymi drogami), trzeba wrócić do `Batch`.
    """
    offsets = (torch.arange(k, device=edge_index.device) * n_nodes).repeat_interleave(
        edge_index.shape[1]
    )
    return edge_index.repeat(1, k) + offsets


def batch_vector(n_nodes: int, k: int, device: torch.device | None = None) -> Tensor:
    """Przypisanie węzeł→graf dla `k` sklejonych grafów; to, czego oczekuje
    `global_mean_pool` w krytyku."""
    return torch.arange(k, device=device).repeat_interleave(n_nodes)


def tiled_log_rho(log_rho: Tensor, n_edges: int, n_nodes: int) -> Tensor:
    """[k, E+N] (log ρ per KROK CZASU) -> [k·(E+N)] w kolejności sklejonego grafu.

    JEDYNE MIEJSCE NA CICHY BŁĄD W CAŁEJ FAZIE 3 — pułapka nr 3 z docstringu
    `uw_attention.py`, tu uderzająca realnie. Kontrakt warstwy brzmi „ta sama
    kolejność co `edge_index` PRZEKAZANY DO `forward`", a ten dla minibatcha
    powstaje z `tiled_edge_index`, czyli:

        wszystkie E krawędzi grafu 0, potem grafu 1, …, potem grafu k−1,
        a NA SAMYM KOŃCU k·N pętli własnych, w kolejności węzłów 0…k·N−1
        (bo `add_self_loops` dokłada je na końcu, już po sklejeniu).

    Czyli bloki krawędzi i bloki pętli są PRZEPLATANE inaczej niż w buforze,
    gdzie każdy krok trzyma swoje [E+N] w jednym kawałku. Naiwne spłaszczenie

        log_rho.reshape(-1)                          # ŹLE
        torch.cat([log_rho[0]] * k)                  # ŹLE

    ma POPRAWNĄ DŁUGOŚĆ k·(E+N), przechodzi asercję w `_forward` i liczy bzdury:
    pierwsze k·E pozycji dostaje pomieszane krawędzie z pętlami. Nic tego nie
    zgłasza — PyG przepuszcza argumenty `propagate` do `message` nietknięte.
    Dlatego rozdzielamy jawnie i testujemy porównaniem z forwardami per krok.
    """
    if log_rho.dim() != 2 or log_rho.shape[1] != n_edges + n_nodes:
        raise ValueError(
            f"log_rho ma być [k, E+N] = [k, {n_edges + n_nodes}], jest {tuple(log_rho.shape)}"
        )
    k = log_rho.shape[0]
    krawedzie = log_rho[:, :n_edges].reshape(k * n_edges)  # graf 0, graf 1, …
    petle = log_rho[:, n_edges:].reshape(k * n_nodes)  # węzeł 0…N−1 grafu 0, potem grafu 1, …
    return torch.cat([krawedzie, petle])


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """GAE(λ) liczone OSOBNO DLA KAŻDEGO AGENTA. -> (przewagi [T,N], zwroty [T,N]).

    To jest miejsce, w którym opłaca się faktoryzowany krytyk: `rewards` i `values`
    mają wymiar agenta, więc przewaga skrzyżowania i zależy od jego własnej nagrody,
    a nie od sumy po całej sieci. Przy krytyku globalnym te tensory byłyby [T, 1]
    i wszyscy dostawaliby tę samą przewagę.

    `dones[t]` = czy s_{t+1} jest stanem TERMINALNYM (nie: czy epizod się urwał).
    Patrz pułapka 1 w docstringu modułu.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError(
            f"rewards {tuple(rewards.shape)}, values {tuple(values.shape)} i dones "
            f"{tuple(dones.shape)} muszą mieć ten sam kształt [T, N]"
        )
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(rewards.shape[0])):
        next_value = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nonterminal * next_value - values[t]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[t] = running
    return advantages, advantages + values


@dataclass
class Minibatch:
    """Jeden minibatch: `k` KROKÓW CZASU sklejonych w jeden graf o `k·N` węzłach."""

    x: Tensor  # [k·N, ob_length]
    edge_index: Tensor  # [2, k·E]
    batch: Tensor  # [k·N]
    actions: Tensor  # [k·N]
    old_log_probs: Tensor  # [k·N]
    old_values: Tensor  # [k·N], skala SUROWA
    advantages: Tensor  # [k·N]
    returns: Tensor  # [k·N], skala SUROWA
    #: [k·(E+N)] w kolejności sklejonego grafu (patrz `tiled_log_rho`) albo None,
    #: gdy bieg idzie bez mechanizmu. None znaczy „warstwa ma liczyć jak oryginał",
    #: a nie „wektor zer" — to są dwie różne ścieżki w `uw_attention` i tylko
    #: pierwsza jest bit w bit zgodna z rdzeniem z Fazy 1.
    log_rho: Tensor | None = None


class RolloutBuffer:
    """Bufor jednego segmentu rolloutu: [T kroków decyzyjnych, N agentów].

    Graf trzymamy RAZ, nie na krok — topologia jest statyczna (`graph.py`), więc
    kopiowanie `edge_index` 240 razy na epizod byłoby czystym marnotrawstwem.

    Kolejność spłaszczania to `[t, i] -> t·N + i` i jest kontraktem: tak samo
    układa węzły `tiled_edge_index`, więc wiersz `t·N + i` tensora `x` opisuje
    agenta `i` w kroku `t`. Pomyłka tutaj przypisałaby przewagę jednego
    skrzyżowania obserwacji drugiego — bez żadnego błędu.
    """

    def __init__(
        self,
        capacity: int,
        n_nodes: int,
        ob_length: int,
        edge_index: Tensor,
        device: torch.device | str = "cpu",
        *,
        with_log_rho: bool = False,
    ) -> None:
        self.capacity = capacity
        self.n_nodes = n_nodes
        self.ob_length = ob_length
        self.device = torch.device(device)
        self.edge_index = edge_index.to(self.device)
        self.n_edges = int(self.edge_index.shape[1])
        self.pos = 0

        shape = (capacity, n_nodes)
        self.obs = torch.zeros((*shape, ob_length), device=self.device)
        # log ρ ZAPISUJEMY, a nie przeliczamy w update. Głowa obserwacyjna widzi
        # OKNO ostatnich odczytów (`OnlineUncertainty`), więc z samego `obs[t]`
        # nie da się jej wyniku odtworzyć — a `evaluate` musi dostać dokładnie ten
        # wektor, który widział `act`, inaczej ratio w pierwszej epoce ≠ 1.
        self.log_rho = (
            torch.zeros((capacity, self.n_edges + n_nodes), device=self.device)
            if with_log_rho
            else None
        )
        self.actions = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros(shape, device=self.device)
        self.values = torch.zeros(shape, device=self.device)
        self.rewards = torch.zeros(shape, device=self.device)
        self.dones = torch.zeros(shape, device=self.device)
        self.advantages = torch.zeros(shape, device=self.device)
        self.returns = torch.zeros(shape, device=self.device)
        self._ready = False

    def __len__(self) -> int:
        return self.pos

    @property
    def full(self) -> bool:
        return self.pos >= self.capacity

    def reset(self) -> None:
        self.pos = 0
        self._ready = False

    def add(
        self,
        obs: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        values: Tensor,
        rewards: Tensor,
        dones: Tensor | bool = False,
        log_rho: Tensor | None = None,
    ) -> None:
        """Jedna DECYZJA (Δt = 15 s symulacji), nie jeden krok symulatora.

        `values` na skali SUROWEJ — denormalizację robi wołający, bo tylko on wie,
        czy krytyk liczył w przestrzeni znormalizowanej.

        `log_rho` [E+N] dla TEGO kroku, w kolejności kontraktu `uw_attention`
        (krawędzie `edge_index`, potem pętle własne w kolejności węzłów).
        Obecność musi się zgadzać z `with_log_rho` — pomyłka w tę stronę jest
        cicha (bieg bez mechanizmu, który miał go mieć), więc jest wyjątkiem.
        """
        if self.full:
            raise IndexError(f"bufor pełny ({self.capacity}) — zawołaj reset() albo update()")
        if obs.shape != (self.n_nodes, self.ob_length):
            raise ValueError(
                f"obs ma być [{self.n_nodes}, {self.ob_length}], jest {tuple(obs.shape)}"
            )
        if (log_rho is None) != (self.log_rho is None):
            raise ValueError(
                "bufor zbudowany z with_log_rho="
                f"{self.log_rho is not None}, a add() dostał log_rho={log_rho is not None}"
            )
        if log_rho is not None and log_rho.shape != (self.n_edges + self.n_nodes,):
            raise ValueError(
                f"log_rho ma być [{self.n_edges + self.n_nodes}] = E + N, "
                f"jest {tuple(log_rho.shape)}"
            )
        t = self.pos
        self.obs[t] = obs
        if self.log_rho is not None:
            self.log_rho[t] = log_rho
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        # `dones` wolno podać skalarnie (epizod kończy się dla wszystkich naraz)
        # albo per agent — drugie przyda się w Fazie 3, gdy część skrzyżowań
        # będzie wypadać ze sterowania.
        self.dones[t] = torch.as_tensor(dones, dtype=torch.float32, device=self.device).expand(
            self.n_nodes
        )
        self.pos += 1
        self._ready = False

    def compute_returns(self, last_value: Tensor, config: PPOConfig) -> None:
        """GAE po zebraniu segmentu. `last_value` = V(s_T) stanu ZA końcem
        segmentu, na skali surowej — bez niego ostatnie kroki epizodu miałyby
        wartość uciętą do zera (pułapka 1)."""
        if self.pos == 0:
            raise RuntimeError("pusty bufor — nie ma z czego liczyć zwrotów")
        adv, ret = compute_gae(
            self.rewards[: self.pos],
            self.values[: self.pos],
            self.dones[: self.pos],
            last_value,
            config.gamma,
            config.gae_lambda,
        )
        self.advantages[: self.pos] = adv
        self.returns[: self.pos] = ret
        self._ready = True

    def minibatches(
        self, n_minibatches: int, generator: torch.Generator | None = None
    ) -> Iterator[Minibatch]:
        """Losowy podział KROKÓW CZASU na minibatche (pułapka 2).

        Permutacja powstaje na CPU i dopiero potem jedzie na urządzenie bufora —
        celowo, z dwóch powodów. Po pierwsze `generator` bywa CPU-owy (tak robią
        testy), a `randperm(generator=cpu_gen, device='cuda')` to twardy błąd.
        Po drugie: dzięki temu ten sam seed daje TĘ SAMĄ kolejność minibatchy na
        CPU i na GPU, więc różnice między urządzeniami zostają w arytmetyce
        zmiennoprzecinkowej, a nie w tym, które próbki wylądowały razem.
        """
        if not self._ready:
            raise RuntimeError("najpierw compute_returns() — inaczej przewagi to zera")
        order = torch.randperm(self.pos, generator=generator).to(self.device)
        size = max(1, math.ceil(self.pos / n_minibatches))
        for start in range(0, self.pos, size):
            idx = order[start : start + size]
            k = idx.numel()
            yield Minibatch(
                # [k, N, ...] -> [k·N, ...]: kontrakt kolejności `[t, i] -> t·N + i`,
                # ten sam, którego trzyma się `tiled_edge_index`.
                x=self.obs[idx].reshape(k * self.n_nodes, self.ob_length),
                edge_index=tiled_edge_index(self.edge_index, self.n_nodes, k),
                batch=batch_vector(self.n_nodes, k, self.device),
                actions=self.actions[idx].reshape(-1),
                old_log_probs=self.log_probs[idx].reshape(-1),
                old_values=self.values[idx].reshape(-1),
                advantages=self.advantages[idx].reshape(-1),
                returns=self.returns[idx].reshape(-1),
                # NIE `self.log_rho[idx].reshape(-1)` — patrz `tiled_log_rho`:
                # to spłaszczenie ma poprawną długość i mylny układ.
                log_rho=(
                    tiled_log_rho(self.log_rho[idx], self.n_edges, self.n_nodes)
                    if self.log_rho is not None
                    else None
                ),
            )


@dataclass
class PPO:
    """Update MAPPO: clipped surrogate + strata wartości + bonus entropijny.

    Dwa optymalizatory, bo actor i krytyk to rozłączne sieci (`nets.py`) i chcemy
    móc dać im różne `lr` — krytyk zwykle znosi wyższe, bo regresja jest łatwiejsza
    od policy gradientu.
    """

    ac: ActorCritic
    config: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        self.actor_opt = torch.optim.Adam(self.ac.actor.parameters(), lr=self.config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.ac.critic.parameters(), lr=self.config.critic_lr)
        # `ValueNorm` trzyma statystyki w buforach, więc musi stać tam, gdzie sieć —
        # inaczej `denormalize` dostaje tensory z dwóch urządzeń. Urządzenie
        # CZYTAMY z `ac`, nie przyjmujemy osobnym polem: dwa źródła prawdy o tym,
        # gdzie liczy PPO, to gwarantowany rozjazd. Optymalizatory powstają PO
        # przeniesieniu sieci (robi to `CoreTrainer.from_world`), więc ich stan
        # rodzi się od razu na właściwym urządzeniu.
        self.device = next(self.ac.parameters()).device
        self.value_norm = ValueNorm().to(self.device) if self.config.normalize_values else None

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def value_of(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        """V(s) na skali SUROWEJ — tej, na której liczy się GAE i nagrody."""
        v = self.ac.value(x, edge_index, batch)
        return self.value_norm.denormalize(v) if self.value_norm else v

    @torch.no_grad()
    def act(
        self,
        x: Tensor,
        edge_index: Tensor,
        *,
        deterministic: bool = False,
        log_rho: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """(akcje, log_prob, wartości surowe). `no_grad` JEST tutaj, bo to jedyna
        ścieżka zbierania danych — w `evaluate` go być nie może (pułapka 4)."""
        actions, log_probs, values = self.ac.act(
            x, edge_index, deterministic=deterministic, log_rho=log_rho
        )
        if self.value_norm:
            values = self.value_norm.denormalize(values)
        return actions, log_probs, values

    # ------------------------------------------------------------------- update

    def update(
        self, buffer: RolloutBuffer, generator: torch.Generator | None = None
    ) -> dict[str, float]:
        """Jeden pełny update na zebranym segmencie. -> metryki diagnostyczne.

        Zwracane liczby to nie ozdoba: `approx_kl` i `clip_fraction` są jedynym
        wczesnym sygnałem, że lr jest za duży (KL rośnie, clip_fraction > ~0.3),
        a `entropy` — że polityka przedwcześnie się zdeterminowała. W 1.5 idą
        do krzywych uczenia obok ATT.
        """
        cfg = self.config
        returns = buffer.returns[: buffer.pos]
        if self.value_norm:
            self.value_norm.update(returns)

        stats = {k: 0.0 for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac")}
        n_batches, stopped = 0, False

        for _ in range(cfg.epochs):
            for mb in buffer.minibatches(cfg.n_minibatches, generator):
                log_probs, entropy, values = self.ac.evaluate(
                    mb.x, mb.edge_index, mb.actions, mb.batch, log_rho=mb.log_rho
                )
                ratio = torch.exp(log_probs - mb.old_log_probs)

                advantages = mb.advantages
                if cfg.normalize_advantage and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                unclipped = ratio * advantages
                clipped = ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                target = mb.returns
                old_values = mb.old_values
                if self.value_norm:
                    target = self.value_norm.normalize(target)
                    old_values = self.value_norm.normalize(old_values)
                # Clipping wartości z tego samego powodu co clipping polityki:
                # jeden update nie ma prawa przesunąć V dalej niż o value_clip_range,
                # bo cel (`returns`) pochodzi jeszcze ze starej polityki.
                v_clipped = old_values + (values - old_values).clamp(
                    -cfg.value_clip_range, cfg.value_clip_range
                )
                value_loss = (
                    0.5 * torch.max((values - target) ** 2, (v_clipped - target) ** 2).mean()
                )

                entropy_mean = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean

                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.actor.parameters(), cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.ac.critic.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                with torch.no_grad():
                    # Estymator k3 Schulmana: nieujemny i mniej zaszumiony niż
                    # zwykłe (old − new), które przy małym KL potrafi wyjść ujemne.
                    approx_kl = ((ratio - 1) - (log_probs - mb.old_log_probs)).mean()
                    clip_frac = ((ratio - 1).abs() > cfg.clip_range).float().mean()
                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy_mean)
                stats["approx_kl"] += float(approx_kl)
                stats["clip_frac"] += float(clip_frac)
                n_batches += 1

                if cfg.target_kl is not None and float(approx_kl) > cfg.target_kl:
                    stopped = True
                    break
            if stopped:
                break

        out = {k: v / max(n_batches, 1) for k, v in stats.items()}
        out["n_minibatches"] = float(n_batches)
        out["early_stopped"] = float(stopped)
        return out
