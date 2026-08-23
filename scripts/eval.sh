#!/usr/bin/env bash
# Ewaluacja zamrożonej polityki z checkpointu — bez treningu.
#
# Istnieje z tego samego powodu co run.sh: PYTHONHASHSEED musi być ustawiony ZANIM
# wystartuje interpreter.
#
# Użycie:
#   scripts/eval.sh --load-run runs/t1/colight__cityflow_jinan3x4_4ph__s0_y5
#   scripts/eval.sh --load-run runs/t1/... --checkpoint last --tag drop25
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Brak interpretera: $PYTHON. Uruchom najpierw:  make setup" >&2
  exit 1
fi

export PYTHONHASHSEED=0
exec "$PYTHON" -m tsc_marl.evaluate "$@"
