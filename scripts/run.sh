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
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f external/LibSignal/run.py ]]; then
  echo "Brak submodułu LibSignal. Napraw:  git submodule update --init --recursive" >&2
  exit 1
fi

export PYTHONHASHSEED=0
exec python -m tsc_marl.run "$@"
