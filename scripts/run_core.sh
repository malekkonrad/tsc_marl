#!/usr/bin/env bash
# Trening rdzenia Fazy 1 (GAT + PPO + CTDE). Cienka nakładka na `run.sh` — cała
# logika (PYTHONHASHSEED przed startem interpretera, wybór interpretera z .venv)
# zostaje tam, żeby obie ścieżki startowały identycznie.
#
# Użycie:
#   scripts/run_core.sh -n cityflow_jinan3x4_4ph --seed 0
#   scripts/run_core.sh -n cityflow_hangzhou4x4 --seed 1 --yellow 0
#   scripts/run_core.sh -n cityflow_jinan3x4_4ph --episodes 1 --steps 150   # smoke
set -euo pipefail
MODULE=tsc_marl.run_core exec "$(dirname "$0")/run.sh" "$@"
