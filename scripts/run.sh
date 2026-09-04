#!/usr/bin/env bash
# Wejście do treningu/ewaluacji.
#
# Istnieje z jednego powodu: PYTHONHASHSEED musi być ustawiony ZANIM wystartuje
# interpreter — ustawiony z poziomu Pythona nie działa, a bez niego `world_sumo`
# jest niedeterministyczny (te same argumenty, inny wynik). Nie wołaj
# `python -m tsc_marl.run` bezpośrednio, chyba że wiesz, że robisz eval CityFlow.
#
# Użycie:
#   scripts/run.sh -a efficientmp -n cityflow_jinan3x4_4ph --seed 0
#   scripts/run.sh -a colight -n cityflow_jinan3x4_4ph --seed 0 --yellow 0
#   scripts/run_core.sh -n cityflow_jinan3x4_4ph --seed 0        # rdzeń GAT+PPO+CTDE
#
# MODULE wybiera wejście: `tsc_marl.run` (agenty LibSignala) albo `tsc_marl.run_core`
# (rdzeń Fazy 1, własna pętla PPO). Ustawianie PYTHONHASHSEED zostaje w JEDNYM miejscu.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f external/LibSignal/run.py ]]; then
  echo "Brak submodułu LibSignal. Napraw:  git submodule update --init --recursive" >&2
  exit 1
fi

# Interpreter z .venv, a nie gołe `python` — inaczej trafiamy w systemowego/pyenv,
# który nie ma zależności (a przy pyenv bez ustawionego `global` w ogóle nie istnieje).
# Aktywowany venv albo jawny PYTHON=... mają pierwszeństwo.
PYTHON="${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Brak interpretera: $PYTHON. Uruchom najpierw:  make setup" >&2
  exit 1
fi

export PYTHONHASHSEED=0
exec "$PYTHON" -m "${MODULE:-tsc_marl.run}" "$@"
