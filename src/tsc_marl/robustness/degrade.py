"""Evaluation of a FROZEN controller over a list of failure scenarios.

A PLAN (`configs/robustness/<name>.yml`) lists networks, controllers (with seeds
for learned ones), failure scenarios and the number of DRAWS per scenario. The
replication unit under failure is a FAILURE DRAW (`episode` -> `SensorModel.reset`),
not a simulator seed: traffic in flow.json is fixed.

One `runlog` run = one controller x network x seed with ALL scenarios inside
(`metrics.csv`: split = scenario, step = draw). One world per process, a new
`StateBuilder` per scenario, so trajectories differ only by what the controller did.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tsc_marl.domain.sensors import FailureSpec
from tsc_marl.paths import CONFIGS

PLANS = CONFIGS / "robustness"
EXPERIMENT = "t2"
YELLOW_BY_MODE = {"realistic": 5, "benchmark": 0}

#: agent -> controller kind; learned ones have a parent in `runs/t1` (checkpoint).
KINDS = {
    "gatppo": "policy",
    "advancedcolight": "colight",
    "efficientmp": "pressure",
    "advancedmp": "pressure",
    "fixedtime": "fixed",
}
#: `ensemble`: several core actors, the first decides, their disagreement is the
#: policy's epistemic signal (plan t2_ood). `policy_uw`, `policy_gate` and
#: `policy_mask` use THE SAME checkpoint as `policy` and differ only by the overlay
#: (UW-attention, per-node gate, phase starvation mask).
LEARNED = ("policy", "colight", "ensemble", "policy_uw", "policy_gate", "policy_mask")
#: Gate kinds. `oracle` reads the truth from the failure layer and is an UPPER
#: BOUND, not a deployable controller; `rule` and `uq` are deployable.
GATE_KINDS = ("oracle", "rule", "uq")
#: Controller spec keys that change the MECHANISM, not just a column label. They
#: MUST enter the run fingerprint (`unit_params`), otherwise a sweep over tau or a
#: threshold computes the first value and skips the rest as done.
MECHANISM_KEYS = ("uw", "gate", "mask")
#: Traffic variant suffix in a network name (`_d0p25`, `_d1p5`, `_r2000`, `_r5816`);
#: `{base}` in a parent template = the network the policy was trained ON.
VARIANT_SUFFIX = re.compile(r"_(?:d\d+(?:p\d+)?|r\d+)$")


def base_network(network: str) -> str:
    return VARIANT_SUFFIX.sub("", network)


#: Columns written to `metrics.csv` of every run. `queue_max`/`queue_p90`/`wait_max`
#: are the TAIL: a network mean cannot see one intersection held in a dead phase.
METRICS = (
    "att",
    "throughput",
    "queue",
    "delay",
    "reward",
    "queue_max",
    "queue_p90",
    "wait_max",
    "wait_max_demand",
)


@dataclass(frozen=True)
class Unit:
    """One plan point = one subprocess = one `runlog` run."""

    controller: str  # label from the plan (key in `controllers`)
    network: str
    seed: int

    def label(self) -> str:
        return f"{self.network:26s} {self.controller:18s} seed={self.seed}"


@dataclass
class Plan:
    name: str
    path: Path
    networks: list[str]
    controllers: dict[str, dict[str, Any]]
    scenarios: tuple[FailureSpec, ...]
    mode: str = "realistic"
    draws: int = 5
    timeout: int = 7200
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def yellow(self) -> int:
        return YELLOW_BY_MODE[self.mode]

    def agent(self, controller: str) -> str:
        return self.controllers[controller].get("agent", controller)

    def kind(self, controller: str) -> str:
        spec = self.controllers[controller]
        if "ensemble" in spec:
            return "ensemble"
        # gate BEFORE UW: a spec with both is a composition where the gate wraps the
        # modulated policy (`GatedController` around `UWPolicyController`)
        if spec.get("gate"):
            return "policy_gate"
        if spec.get("uw"):
            return "policy_uw"
        if spec.get("mask"):
            return "policy_mask"
        return KINDS[self.agent(controller)]

    def baseline(self, controller: str) -> str | None:
        """The column the report PAIRS this controller with, when the default (the
        first `policy` column of the same agent) would be wrong, e.g. two arms
        without a mechanism trained differently (`t2_train_uw`)."""
        return self.controllers[controller].get("baseline")

    def units(self) -> list[Unit]:
        """Learned controllers x their seeds; rules and fixed time are deterministic,
        so computed once (seed 0)."""
        out = []
        for net in self.networks:
            for label, spec in self.controllers.items():
                kind = self.kind(label)
                if kind == "ensemble":
                    seeds = [spec["ensemble"][0]]  # one point: the deciding seed
                elif kind in LEARNED:
                    seeds = spec.get("seeds", [0])
                else:
                    seeds = [0]
                out += [Unit(label, net, int(s)) for s in seeds]
        return out

    def parent_ids(self, unit: Unit) -> list[str]:
        """Training runs in `runs/t1` that provide checkpoints: one for a policy,
        several for an ensemble (the first one decides)."""
        kind = self.kind(unit.controller)
        if kind not in LEARNED:
            return []
        spec = self.controllers[unit.controller]
        template = spec.get("parent", "{agent}__{network}__s{seed}_y{yellow}")
        seeds = spec["ensemble"] if kind == "ensemble" else [unit.seed]
        return [
            template.format(
                agent=self.agent(unit.controller),
                network=unit.network,
                base=base_network(unit.network),
                seed=s,
                yellow=self.yellow,
            )
            for s in seeds
        ]

    def parent_id(self, unit: Unit) -> str | None:
        ids = self.parent_ids(unit)
        return ids[0] if ids else None

    def run_id(self, unit: Unit) -> str:
        return f"{unit.controller}__{unit.network}__s{unit.seed}_y{self.yellow}__{self.name}"

    def scenarios_for(self, unit: Unit) -> tuple[FailureSpec, ...]:
        """Fixed time reads no sensors, so under failure its result is the clean one."""
        if self.kind(unit.controller) == "fixed":
            return tuple(s for s in self.scenarios if not s.active)
        return self.scenarios


def load_plan(path: str | Path) -> Plan:
    """A name (`t2_nomech`) or a path. The clean scenario is always prepended:
    degradation is measured against it."""
    import yaml

    p = Path(path)
    if not p.suffix:
        p = PLANS / f"{p.name}.yml"
    if not p.is_file():
        available = ", ".join(sorted(x.stem for x in PLANS.glob("*.yml"))) or "none"
        raise SystemExit(f"no plan '{path}' ({p}). Available: {available}")
    raw = yaml.safe_load(p.read_text()) or {}
    known = {"name", "networks", "controllers", "scenarios", "mode", "draws", "timeout"}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"plan '{p.name}': unknown fields {sorted(unknown)}")
    specs = [FailureSpec(**s) for s in raw.get("scenarios", [])]
    if not any(not s.active for s in specs):
        specs.insert(0, FailureSpec())
    plan = Plan(
        name=raw.get("name", p.stem),
        path=p.resolve(),
        networks=list(raw["networks"]),
        controllers=dict(raw["controllers"]),
        scenarios=tuple(specs),
        mode=raw.get("mode", "realistic"),
        draws=int(raw.get("draws", 5)),
        timeout=int(raw.get("timeout", 7200)),
    )
    for label, spec in plan.controllers.items():
        if plan.agent(label) not in KINDS:
            raise SystemExit(f"plan '{p.name}': unknown controller '{plan.agent(label)}'")
        if "ensemble" in spec and KINDS[plan.agent(label)] != "policy":
            raise SystemExit(f"plan '{p.name}': ensemble of core actors only ('{label}')")
        if "ensemble" in spec and len(spec["ensemble"]) < 2:
            raise SystemExit(f"plan '{p.name}': ensemble '{label}' needs >= 2 seeds")
        if spec.get("uw") and KINDS[plan.agent(label)] != "policy":
            raise SystemExit(f"plan '{p.name}': UW-attention with the core actor only ('{label}')")
        # a typo in `side`, `u` or `channels` must not silently give the default;
        # imported here because `obs_head` pulls in torch
        if spec.get("uw"):
            from tsc_marl.robustness.training import check_uw

            error = check_uw(spec["uw"])
            if error:
                raise SystemExit(f"plan '{p.name}': UW '{label}': {error}")
        if spec.get("gate"):
            if KINDS[plan.agent(label)] != "policy":
                raise SystemExit(f"plan '{p.name}': gate with the core actor only ('{label}')")
            # a typo in `kind` must not silently build the oracle, the upper bound
            if spec["gate"].get("kind") not in GATE_KINDS:
                raise SystemExit(
                    f"plan '{p.name}': gate '{label}' has kind="
                    f"{spec['gate'].get('kind')!r}, allowed: {GATE_KINDS}"
                )
        if spec.get("mask"):
            if KINDS[plan.agent(label)] != "policy":
                raise SystemExit(f"plan '{p.name}': mask with the core actor only ('{label}')")
            # `k_max` is the WHOLE mechanism; a misspelt key would silently give a default
            if "k_max" not in spec["mask"]:
                raise SystemExit(f"plan '{p.name}': mask '{label}' must give `k_max` (decisions)")
            if int(spec["mask"]["k_max"]) < 1:
                raise SystemExit(f"plan '{p.name}': mask '{label}' has k_max < 1")
        if spec.get("checkpoint", "last") not in ("last", "best"):
            raise SystemExit(
                f"plan '{p.name}': '{label}' has checkpoint={spec['checkpoint']!r}, "
                "allowed: last, best"
            )
        if "checkpoint" in spec and KINDS[plan.agent(label)] != "policy":
            raise SystemExit(f"plan '{p.name}': checkpoint for the core only ('{label}')")
        base = spec.get("baseline")
        if base is not None and base not in plan.controllers:
            raise SystemExit(
                f"plan '{p.name}': '{label}' points to baseline '{base}', which is not "
                f"in `controllers` ({sorted(plan.controllers)})"
            )
        if base is not None and base == label:
            raise SystemExit(f"plan '{p.name}': '{label}' points to itself as baseline")
    slugs = [s.slug() for s in plan.scenarios]
    if len(set(slugs)) != len(slugs):
        raise SystemExit(f"plan '{p.name}': duplicate scenarios {slugs}")
    if plan.mode not in YELLOW_BY_MODE:
        raise SystemExit(f"plan '{p.name}': mode must be one of {sorted(YELLOW_BY_MODE)}")
    return plan


# ------------------------------------------------------------------- loop


def evaluate(
    env: Any,
    controller: Any,
    specs: tuple[FailureSpec, ...],
    draws: int,
    make_builder: Callable[[FailureSpec], Any],
    decisions: int,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    on_trace: Callable[[str, int, list[Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Every scenario x draw = one full test episode -> metric rows.

    The clean scenario is deterministic, so computed ONCE; a failure scenario
    `draws` times, each draw with another `episode`, which picks the broken
    detectors and the noise.
    """
    rows: list[dict[str, Any]] = []
    for spec in specs:
        env.features = make_builder(spec)
        n = draws if spec.active else 1
        for draw in range(n):
            obs = env.reset(draw)
            if hasattr(controller, "reset_stats"):
                controller.reset_stats()
            for _ in range(decisions):
                obs, _, _ = env.step(controller.act(obs))
            row: dict[str, Any] = {"scenario": spec.slug(), "draw": draw, **env.metrics()}
            if hasattr(controller, "episode_stats"):
                row.update(controller.episode_stats())
            sensors = getattr(env.features, "sensors", None)
            row["n_broken"] = len(sensors.broken) if sensors is not None else 0
            rows.append(row)
            if on_row is not None:
                on_row(row)
            # the optional gate trace leaves AFTER the metrics and before the next
            # `reset_stats`, the only moment it is complete
            if on_trace is not None and hasattr(controller, "episode_trace"):
                trace = controller.episode_trace()
                if trace:
                    on_trace(spec.slug(), draw, trace)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Scalars for `result.json`: means over draws, `att__<scenario>`."""
    out: dict[str, float] = {}
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["scenario"], []).append(r)
    for name, group in by.items():
        for metric in (
            "att", "throughput", "ens_mi", "ens_disagree", "uw_spread", "uw_active",
            "gate_share", "gate_switches", "gate_precision", "gate_recall",
        ):  # fmt: skip
            # mean over draws where the metric IS DEFINED: `gate_precision`/`recall`
            # are missing when a draw had nothing to detect
            values = [r[metric] for r in group if metric in r]
            if values:
                out[f"{metric}__{name}"] = sum(values) / len(values)
    if "att__clean" in out:
        out["att"] = out["att__clean"]
    return out
