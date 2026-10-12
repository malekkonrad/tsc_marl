"""Sensor failure layer: what the detectors report instead of the truth.

Failures are injected at the DETECTOR level, into per-lane counts (`q[lane]`,
`running[lane]`) between `windowed_queue`/`effective_running` and the ATS
formulas (`StateBuilder.observe`). Not per vehicle: a real detector reports a
count, not a list. Not per feature: a movement's pressure already mixes several
lanes, and one broken detector is not one broken movement.

Consequences:

* the reward reads the world's cache, not this path, so it stays clean by
  construction;
* every controller reads the same corrupted observation, so all of them face
  the same failure;
* the validity flag never enters the state vector, so `ob_length` is the same
  in every scenario.

Two classes of damage with different lifetimes: persistent (`mask`, `drift`),
drawn once per episode per failure unit, since a broken sensor does not
flicker; and transient (`gauss`, `urand`), drawn every decision.

Substitution is not neutral: a dead lane reporting 0 says "empty approach",
which max-pressure reads as "no green here". `zero` is the default because it
is the masking protocol of RobustLight (Li et al., 2025); `hold` leaves no
trace a rule could catch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tsc_marl.domain.ats import in_road_lanes, road_of

if TYPE_CHECKING:
    from world.world_cityflow import Intersection

FAILURE_KINDS = ("none", "gauss", "urand", "mask", "drift")
FAILURE_UNITS = ("lane", "road", "intersection")
SUBSTITUTIONS = ("zero", "hold", "mean")

#: Reading channels. One physical detector serves both: a lane with a dead
#: sensor reports neither queue nor moving vehicles.
CHANNELS = ("queue", "running")


@dataclass(frozen=True)
class FailureSpec:
    """Failure scenario. Configured in YAML (`model` section), so it is part of
    the run fingerprint and a different scenario is a different run."""

    kind: str = "none"
    level: float = 0.0
    #: What ONE failure is: a single detector, a whole approach, or a whole
    #: intersection. Applies to persistent damage; noise is always per reading.
    unit: str = "lane"
    substitution: str = "zero"
    #: Decision from which the failure is active; 0 = from the start (as in
    #: RobustLight). A later onset is what makes `hold` meaningful: a detector
    #: dead from the first decision has nothing to hold and degenerates to `zero`.
    onset: int = 0
    #: Random stream INDEPENDENT of `--seed`: otherwise two seeds would differ in
    #: network initialisation AND in which detectors failed, and the sources of
    #: variance could not be separated.
    seed: int = 0

    def __post_init__(self) -> None:
        if self.kind not in FAILURE_KINDS:
            raise ValueError(f"kind={self.kind!r}, allowed: {FAILURE_KINDS}")
        if self.unit not in FAILURE_UNITS:
            raise ValueError(f"unit={self.unit!r}, allowed: {FAILURE_UNITS}")
        if self.substitution not in SUBSTITUTIONS:
            raise ValueError(f"substitution={self.substitution!r}, allowed: {SUBSTITUTIONS}")
        if self.level < 0:
            raise ValueError(f"level={self.level}, must be >= 0")
        if self.onset < 0:
            raise ValueError(f"onset={self.onset}, must be >= 0")
        if self.kind == "mask" and self.level > 1:
            raise ValueError(f"level={self.level} for 'mask' is a FRACTION of units, so <= 1")
        # Fail loudly: `unit` means nothing for measurement noise, and a YAML typo
        # must fail at construction.
        if self.kind in ("gauss", "urand") and self.unit != "lane":
            raise ValueError(f"kind={self.kind!r} is per reading; unit must stay 'lane'")

    @property
    def active(self) -> bool:
        return self.kind != "none" and self.level > 0

    @classmethod
    def from_registry(cls, **overrides: Any) -> FailureSpec:
        """From LibSignal's YAML config (`model.failure`)."""
        from common.registry import Registry

        param = Registry.mapping["model_mapping"]["setting"].param
        raw = dict(param.get("failure") or {})
        raw.update(overrides)
        return cls(**raw)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "level": self.level,
            "unit": self.unit,
            "substitution": self.substitution,
            "onset": self.onset,
            "seed": self.seed,
        }

    def suffix(self) -> str:
        """Part of a `run_id`, so two scenarios never share a run directory;
        empty for no failure."""
        if not self.active:
            return ""
        # no dots: the suffix ends up in run ids and npz keys
        base = f"{self.kind}{self.level:g}".replace(".", "p")
        parts = [base] if self.unit == "lane" else [base, self.unit]
        if self.substitution != "zero":
            parts.append(self.substitution)
        if self.onset:
            parts.append(f"t{self.onset}")
        return "_".join(parts)

    def slug(self) -> str:
        """Scenario name used as a key in results: the suffix, or "clean"."""
        return self.suffix() or "clean"


def failure_units(
    intersections: Sequence[Intersection], lanes: Iterable[str], unit: str
) -> dict[str, str]:
    """lane -> failure unit key.

    For `intersection` the detector belongs to the intersection the lane leads
    INTO (where the camera stands). One intersection's out lane is the next one's
    in lane, so the assignment is globally consistent; lanes leaving the network
    have no owner and form their own single-lane unit.
    """
    if unit not in FAILURE_UNITS:
        raise ValueError(f"unit={unit!r}, allowed: {FAILURE_UNITS}")
    lanes = list(lanes)
    if unit == "lane":
        return {lane: lane for lane in lanes}
    if unit == "road":
        return {lane: road_of(lane) for lane in lanes}

    owner: dict[str, str] = {}
    for inter in intersections:
        for lane in in_road_lanes(inter):
            owner[lane] = inter.id
    return {lane: owner.get(lane, lane) for lane in lanes}


class SensorModel:
    """Detector state of the WHOLE network, keyed by lane id.

    One instance per `StateBuilder`, shared by all intersections. This matters:
    A's out lane is B's in lane, so one physical detector is read twice per
    decision. Drawing state or noise per reader would show two neighbours two
    different values, the network would average the noise out, and the failure
    would look MILDER than it is.
    """

    def __init__(self, spec: FailureSpec, groups: Mapping[str, str]) -> None:
        self.spec = spec
        self.groups = dict(groups)
        self._lanes = sorted(self.groups)
        self._broken: frozenset[str] = frozenset()
        self._gain: dict[str, float] = {}
        self._last: dict[tuple[str, str], float] = {}
        self._cache: dict[tuple[str, str], tuple[float, bool]] = {}
        self._noise = np.random.default_rng([spec.seed, 0, 1])
        self.episode = -1
        self._decision = -1
        self.reset(0)

    # --------------------------------------------------------------- episode

    def reset(self, episode: int = 0) -> None:
        """New episode: redraw persistent damage, clear caches, including the
        `hold` memory (values from before `world.reset()` would leak)."""
        self.episode = episode
        self._decision = -1
        self._cache.clear()
        self._last.clear()
        persistent = np.random.default_rng([self.spec.seed, episode, 0])
        self._noise = np.random.default_rng([self.spec.seed, episode, 1])
        self._broken = frozenset()
        self._gain = {}

        if not self.spec.active:
            return
        keys = sorted(set(self.groups.values()))
        if self.spec.kind == "mask":
            # An exact fraction, not a per-unit probability: 25% means 25%.
            n = int(round(self.spec.level * len(keys)))
            chosen = {keys[i] for i in persistent.permutation(len(keys))[:n]}
            self._broken = frozenset(l for l in self._lanes if self.groups[l] in chosen)
        elif self.spec.kind == "drift":
            # Calibration error is multiplicative (the detector counts x% too few),
            # so values stay plausible at any load and the failure is unobservable.
            gains = 1.0 + persistent.normal(0.0, self.spec.level, size=len(keys))
            per_key = dict(zip(keys, np.clip(gains, 0.0, None), strict=True))
            self._gain = {l: float(per_key[self.groups[l]]) for l in self._lanes}

    def begin_decision(self) -> None:
        """A new reading of all detectors, called once per decision by
        `StateBuilder.observe()`; the cache makes one detector give one value per
        decision. Also counts decisions for `onset` (a `corrupt=False` reading
        does not)."""
        self._decision += 1
        self._cache.clear()

    @property
    def live(self) -> bool:
        """Whether the failure is active; never before the first decision."""
        return self._decision >= self.spec.onset  # onset >= 0, so -1 never passes

    # --------------------------------------------------------------- reading

    def read(
        self, values: Mapping[str, float], lanes: Sequence[str], *, channel: str
    ) -> dict[str, float]:
        """Reading of detectors `lanes` on `channel`. Returns a NEW dict: `values`
        may be the world's cache, which `rewards()` reads too."""
        if channel not in CHANNELS:
            raise ValueError(f"channel={channel!r}, allowed: {CHANNELS}")
        clean = {lane: float(values.get(lane, 0.0)) for lane in lanes}
        if not self.spec.active:
            return clean

        reported: dict[str, float] = {}
        valid: dict[str, bool] = {}
        for lane in lanes:
            hit = self._cache.get((channel, lane))
            if hit is None:
                hit = self._measure(lane, clean[lane], channel)
                self._cache[(channel, lane)] = hit
            reported[lane], valid[lane] = hit

        return self._substitute(reported, valid, channel)

    def _measure(self, lane: str, clean: float, channel: str) -> tuple[float, bool]:
        if not self.live:
            # Healthy before onset: the only time `hold` has something to remember.
            self._last[(channel, lane)] = clean
            return clean, True
        if lane in self._broken:
            return 0.0, False  # `_substitute` fills the value in
        value = clean * self._gain.get(lane, 1.0)
        if self.spec.kind == "gauss":
            value += float(self._noise.normal(0.0, self.spec.level))
        elif self.spec.kind == "urand":
            value += float(self._noise.uniform(-self.spec.level, self.spec.level))
        # A count is never negative. Not rounded: EP is fractional anyway, and
        # quantisation would add a second, unmodelled effect.
        value = max(0.0, value)
        self._last[(channel, lane)] = value
        return value, True

    def _substitute(
        self, reported: dict[str, float], valid: dict[str, bool], channel: str
    ) -> dict[str, float]:
        broken = [lane for lane, ok in valid.items() if not ok]
        if not broken:
            return reported
        if self.spec.substitution == "mean":
            live = [reported[lane] for lane, ok in valid.items() if ok]
            fill = float(np.mean(live)) if live else 0.0
            for lane in broken:
                reported[lane] = fill
        elif self.spec.substitution == "hold":
            # a detector dead since the episode start has nothing to hold -> 0
            for lane in broken:
                reported[lane] = self._last.get((channel, lane), 0.0)
        return reported

    # --------------------------------------------------------- ground truth

    @property
    def broken(self) -> frozenset[str]:
        """Lanes with a dead detector in this episode: ground truth for oracles
        and for evaluating failure detection, never part of the state."""
        return self._broken

    @property
    def gain(self) -> Mapping[str, float]:
        """Calibration gain per lane (1.0 = healthy); ground truth for drift."""
        return dict(self._gain)

    def valid(self, lanes: Iterable[str]) -> dict[str, bool]:
        """Whether a detector reports anything; read by oracles, never by the state."""
        return {lane: lane not in self._broken for lane in lanes}

    def report(self) -> dict[str, Any]:
        """For `runlog`: the scenario and what actually failed (the same spec
        breaks different lanes on another network)."""
        return {
            **self.spec.as_dict(),
            "episode": self.episode,
            "n_units": len(set(self.groups.values())),
            "n_broken": len(self._broken),
            "broken": sorted(self._broken),
        }

    def __repr__(self) -> str:
        s = self.spec
        return (
            f"SensorModel(kind={s.kind!r}, level={s.level:g}, unit={s.unit!r}, "
            f"substitution={s.substitution!r}, onset={s.onset}, "
            f"broken={len(self._broken)}/{len(self._lanes)})"
        )


# ------------------------------------------------------------------ failure mix


@dataclass(frozen=True)
class FailureMix:
    """Distribution of failure scenarios over training episodes (failure
    randomization), configured as `model.failure_mix`.

    `sample(episode)` draws from a stream keyed by (mix seed, episode), not from
    a global generator: the schedule does not depend on `--seed` or on how many
    episodes ran before, so two training arms see the same failures and a
    resumed run gets the same scenarios as a continuous one. Which detectors
    fail is then chosen by `SensorModel` from (`spec.seed`, `episode`).
    """

    specs: tuple[FailureSpec, ...]
    weights: tuple[float, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("empty failure mix; give at least one scenario")
        if len(self.specs) != len(self.weights):
            raise ValueError(f"{len(self.specs)} scenarios, {len(self.weights)} weights")
        if any(w < 0 for w in self.weights):
            raise ValueError(f"weights must be non-negative, got {self.weights}")
        if sum(self.weights) <= 0:
            raise ValueError("the sum of weights must be positive")

    @property
    def probabilities(self) -> np.ndarray:
        w = np.asarray(self.weights, dtype=np.float64)
        return w / w.sum()

    def sample(self, episode: int) -> FailureSpec:
        rng = np.random.default_rng([self.seed, int(episode)])
        return self.specs[int(rng.choice(len(self.specs), p=self.probabilities))]

    def as_dict(self) -> dict[str, Any]:
        """For the run parameters, so a run shows what the policy trained on."""
        return {
            "seed": self.seed,
            "scenarios": [
                {"weight": float(w), **s.as_dict()}
                for s, w in zip(self.specs, self.weights, strict=True)
            ],
        }

    def describe(self) -> str:
        return ", ".join(
            f"{s.slug()}={p:.0%}" for s, p in zip(self.specs, self.probabilities, strict=True)
        )

    @classmethod
    def from_config(cls, raw: Any) -> FailureMix | None:
        """`model.failure_mix` -> mix, or None (a run without failures):

            failure_mix:
              seed: 0
              scenarios:
                - {weight: 3, kind: none}
                - {weight: 1, kind: gauss, level: 2.0}

        Unknown scenario fields fail in `FailureSpec`.
        """
        if not raw:
            return None
        if not isinstance(raw, dict) or "scenarios" not in raw:
            raise ValueError("failure_mix must be a mapping with a `scenarios` key")
        specs, weights = [], []
        for entry in raw["scenarios"]:
            item = dict(entry)
            weights.append(float(item.pop("weight", 1.0)))
            specs.append(FailureSpec(**item))
        return cls(tuple(specs), tuple(weights), seed=int(raw.get("seed", 0)))

    @classmethod
    def from_registry(cls) -> FailureMix | None:
        from common.registry import Registry

        param = Registry.mapping["model_mapping"]["setting"].param
        return cls.from_config(param.get("failure_mix"))
