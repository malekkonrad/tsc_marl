"""Wejście CLI rdzenia Fazy 1 (GAT + PPO + CTDE) — odpowiednik `train.py` dla rdzenia.

`train.py` oddaje sterowanie LibSignalowi (`Registry.mapping['task_mapping']`), bo
tam agentem jest klasa z Registry. Rdzeń nie jest agentem LibSignala: pętlę ma
własną (`core/loop.py`), a od symulatora potrzebuje tylko `World`. Dlatego to wejście
istnieje osobno — ale przygotowanie procesu dzieli z `train.py` przez
`session.session()` i różni się DOKŁADNIE jednym krokiem: tam, gdzie tamten woła
`TSCTask(runner).run()`, tutaj powstaje `CoreTrainer`.

Dlaczego nie jeden plik z gałęzią w środku: sekwencja interfejsów Registry jest
wrażliwa na kolejność i już raz kosztowała cichy błąd (`root_dir` ignorowany →
`att_last10` przyjmowało wartość `att`, 27% różnicy). `if` w jej środku byłby tym
samym ryzykiem. W `libsignal/session.py` żadnego `if`-a nie ma — jest jedna kanoniczna
kopia sekwencji i trzy różne rzeczy robione PO niej.

Uruchamiaj przez `tsc core` — CLI ustawia PYTHONHASHSEED przed startem
interpretera (patrz `cli.ensure_hashseed` w tym pakiecie).
"""

from __future__ import annotations

import os
import sys
import time

from tsc_marl.libsignal import session

DEFAULT_AGENT = "gatppo"


def parse_args(argv: list[str] | None = None):
    """Ten sam parser co `train.py` — tylko z innym domyślnym agentem.

    Świadomie nie kopiujemy definicji argumentów: `--yellow`, `--episodes`,
    `--run-id` i `--replace` muszą znaczyć to samo w obu wejściach, bo obie
    ścieżki zapisują biegi do jednego `runlog`.
    """
    from tsc_marl.runner.train import parse_args as base_parse_args

    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a in ("-a", "--agent") for a in argv):
        argv += ["-a", DEFAULT_AGENT]
    return base_parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # require_graph: rdzeń bez grafu nie istnieje. Graf buduje warunkowe
    # `Graph_World_Interface` (`model.graphic`), więc zamiast wywoływać je
    # bezwarunkowo — co rozjeżdżałoby sekwencję z pozostałymi wejściami —
    # żądamy, żeby config to deklarował. `configs/tsc/gatppo.yml` ma `graphic: True`.
    with session.session(args, require_graph=True) as s:
        from common.registry import Registry

        from tsc_marl.core.loop import CoreTrainer
        from tsc_marl.core.ppo import PPOConfig
        from tsc_marl.robustness import training
        from tsc_marl.robustness.degrade import base_network

        cmd = Registry.mapping["command_mapping"]["setting"].param
        started = time.time()

        # Faza 3, trening Z mechanizmem. Oba przełączniki czytają się z sekcji
        # `model` (czyli wchodzą do `config_hash`), a nie z flag CLI — bieg
        # z inną mieszanką awarii MUSI unieważnić skip-if-done, inaczej runner
        # uzna go za już policzony. Ustawiają je `overrides` suite'ów
        # `p1_fail` / `p1_uw`; bez nich to jest dokładnie rdzeń z Fazy 1.
        failures = training.FailureMix.from_registry()
        uw = training.uw_config_from_registry()

        # Świat budujemy przez Registry, a nie `from world.world_cityflow import World`,
        # żeby `-w sumo` (Faza 5) działało tą samą ścieżką. To ten sam sposób,
        # w jaki robi to `BaseTrainer`.
        world = Registry.mapping["world_mapping"][cmd["world"]](
            os.path.join("configs/sim", f"{args.network}.cfg"),
            cmd["thread_num"],
            interface=cmd["interface"],
        )

        core = CoreTrainer.from_world(
            world,
            ppo_config=PPOConfig.from_registry(),
            run=s.run,
            logger=s.logger,
            device=s.device,
            failures=failures,
        )
        if uw is not None:
            # PO zbudowaniu rdzenia, bo głowa potrzebuje `n_mov` ze stanu i
            # `edge_index` z grafu. Artefakty są ZAMROŻONE (`no_grad`, poza
            # `parameters()`), a `u_ref` skalibrowane na zrzucie BC — ograniczenie
            # do wypisania w rozdziale 4, patrz `robustness/training.py`.
            # `uw_signal` — ta sama funkcja co w `tsc degrade`, więc `u`, `side`,
            # `channels` i τ z suite'a dają DOKŁADNIE sygnał, którym bieg będzie
            # potem mierzony (`p1_uw` = głowa po stronie celu, `p1_uw_rule` = reguła)
            core.uncertainty = training.UWStream(
                training.uw_signal(uw, base_network(args.network), core.env),
                core.env.graph.edge_index,
                core.device,
                n_nodes=core.env.graph.n_nodes,
            )
        # `model.save_best` (suite'y `k2_*`): zapis checkpointu z najlepszym epizodem
        # testowym obok końcowego `core.pt`. Klucz siedzi w `model`, więc wchodzi do
        # `config_hash` i suite dostaje sufiks — biegi nie nadpiszą `k1_*`.
        if Registry.mapping["model_mapping"]["setting"].param.get("save_best"):
            core.best_path = s.run.engine_dir / "core_best.pt"
        s.logger.info(
            f"rdzeń: {core.env.graph} | ob_length {core.env.features.ob_length} "
            f"| device {core.device}"
        )
        if failures is not None:
            s.logger.info(f"awarie w treningu: {failures.describe()} (seed {failures.seed})")
        if uw is not None:
            s.logger.info(
                f"UW-attention w treningu: u={uw.get('u', 'head')} "
                f"side={uw.get('side', 'target')} tau={uw.get('tau', 'domyślne')}"
            )
        result = core.train()
        core.save(s.run.engine_dir / "core.pt")
        s.logger.info(f"Total time taken: {time.time() - started:.1f}s | {result}")
    print(f"runlog: {s.run.path}")


if __name__ == "__main__":
    main()
