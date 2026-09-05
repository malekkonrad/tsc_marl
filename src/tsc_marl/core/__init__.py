"""Rdzeń Fazy 1 — GAT + PPO + CTDE, w całości nasza implementacja.

Zależy tylko od `domain/` i `paths.py`. NIE importuje `libsignal/` — silnik
dostaje z zewnątrz, od `runner/train_core.py`, który stawia świat i wstrzykuje go
do `CoreTrainer`. Ta granica jest tym, co pozwala testować rdzeń bez symulatora
(`tests/core/` chodzą w ~2 s, patrz marker `slow`).

Kolejność warstw: `graph` (topologia) → `features` (stan/nagroda) → `nets`
(enkoder GAT + actor-critic) → `ppo` (uczenie) → `loop` (spięcie z symulatorem).
"""
