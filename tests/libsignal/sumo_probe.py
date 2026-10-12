"""SUMO world probe for `test_sumo_world.py`, a separate process because LibSignal's
Registry is global. The last stdout line is a JSON with the results.

Manual run from the repository root:
    .venv/bin/python tests/libsignal/sumo_probe.py sumo_krakow3w_mid
"""

from __future__ import annotations

import json
import sys

DECISIONS = 20  # 300 s of traffic: mechanics, not convergence


def main(network: str) -> dict:
    from tsc_marl import libsignal

    libsignal.bootstrap()
    import numpy as np
    import torch
    from common.registry import Registry
    from utils.logger import build_config

    from tsc_marl.core.loop import CoreTrainer, LoopConfig
    from tsc_marl.libsignal import session
    from tsc_marl.libsignal.bridge import sim_configs_untouched
    from tsc_marl.libsignal.experiment import parse_args
    from tsc_marl.libsignal.sumo_world import SERVING, serving_lanelinks

    session.register_engine_components()
    args = parse_args(["-a", "gatppo", "-w", "sumo", "-n", network])
    with sim_configs_untouched():
        config, _ = build_config(args)
        session.build_interfaces(config)
        world = Registry.mapping["world_mapping"]["sumo"](
            f"configs/sim/{network}.cfg", 4, interface="libsumo"
        )
        steps = DECISIONS * 15
        core = CoreTrainer.from_world(
            world, loop_config=LoopConfig(steps=steps, test_steps=steps, episodes=1)
        )
        env, features = core.env, core.env.features
        ordered = env.graph.order_intersections(world)

        # permissive green: how many lanelinks `g` adds over upstream's G/s
        upstream = SERVING - {"g"}
        permissive_links = 0
        for inter in world.intersections:
            with_g = sum(len(links) for links in serving_lanelinks(inter)[0])
            without_g = sum(
                1
                for phase in inter.green_phases
                for sig, ctl in zip(phase.state, inter.lanelinks, strict=False)
                if sig in upstream and ctl
            )
            permissive_links += with_g - without_g

        before = [id(inter) for inter in world.intersections]
        obs = env.reset(0)
        env.reset(1)
        identity_kept = [id(inter) for inter in world.intersections] == before
        obs = env.reset(2)

        actions_valid = one_hot_ok = True
        for _ in range(DECISIONS):
            actions, _, _ = core.ppo.act(torch.from_numpy(obs), core.edge_index)
            a = actions.numpy()
            actions_valid &= all(a[i] < len(inter.phases) for i, inter in enumerate(ordered))
            obs, rewards, _ = env.step(a)
            phase_block = obs[:, -features.n_phases :]
            expected = [inter.current_phase for inter in ordered]
            one_hot_ok &= phase_block.argmax(axis=1).tolist() == expected
            one_hot_ok &= all(
                c < len(inter.phases) for c, inter in zip(expected, ordered, strict=True)
            )
            assert np.isfinite(rewards).all()
        metrics = env.metrics()

        # a transition AFTER the episode start, where upstream switched without one; the
        # transition phase is recognised by its program index (greens are 0..len-1)
        inter = next(i for i in world.intersections if i.yellow_dict)
        a, b = (int(p) for p in next(iter(inter.yellow_dict)).split("_"))
        transition_seen = False
        for t in range(40):
            wanted = a if (t // 10) % 2 == 0 else b
            world.step([wanted if i is inter else i.current_phase for i in world.intersections])
            transition_seen |= world.eng.trafficlight.getPhase(inter.id) >= len(inter.phases)

    return {
        "world_class": f"{type(world).__module__}.{type(world).__qualname__}",
        "n_edges": env.graph.n_edges,
        "identity_kept": identity_kept,
        "heterogeneous": features.phase_mask() is not None,
        "permissive_links": permissive_links,
        "lanes_known": all(
            lane in world.lane_length for s in features.structs for lane in s.all_lanes
        ),
        "actions_valid": bool(actions_valid),
        "one_hot_matches_phase": bool(one_hot_ok),
        "transition_after_start": transition_seen,
        "att": metrics["att"],
        "throughput": metrics["throughput"],
    }


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1] if len(sys.argv) > 1 else "sumo_krakow3w_mid")))
