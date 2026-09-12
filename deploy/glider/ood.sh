#!/usr/bin/env bash
# Test gałęzi epistemic (plan t2_ood) — uruchamiać PO zakończeniu noc.sh:
#
#     bash deploy/glider/ood.sh
#
# Potrzebuje 5 seedów rdzenia i Advanced-CoLighta (noc.sh je dogrywa). Startuje w tle
# (odporne na wylogowanie) i od razu wraca. Na końcu pisze results/t2_ood.md
# (werdykt: czy gałąź „epistemic → MaxPressure” ma robotę) i results/t2_ood.md z tabelą.
# Wznowienie: to samo polecenie — policzone punkty są pomijane.
#
# Zmienne: J (ile punktów naraz), DRY=1 (tylko wypisz komendy).
set -u
cd "$(dirname "$0")/../.."
TSC=.venv/bin/tsc
N=$(nproc)
J=${J:-$(( N / 5 > 2 ? N / 5 : 2 ))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

run() {
  local rc=0
  echo "$(date '+%F %T')  tsc $*"
  if [ -z "${DRY:-}" ]; then "$TSC" "$@" || rc=$?; fi
  echo "$(date '+%F %T')  koniec: tsc $* (kod $rc)"
}

if [ "${1:-}" = "_run" ]; then
  run degrade run t2_ood -j "$J"
  run degrade report t2_ood
  run degrade ood t2_ood
  exit
fi

if [ -n "${DRY:-}" ]; then
  bash "$0" _run
  exit
fi
if pgrep -f "deploy/glider/noc.sh" > /dev/null; then
  echo "noc.sh jeszcze liczy — ten test potrzebuje jego wyników (seedy 3–4). Uruchom później."
  exit 1
fi
# Brakujący rodzice = punkty zostaną pominięte; lepiej wiedzieć przed startem.
"$TSC" degrade status t2_ood | awk '$2 == "brak" {n++} END {print n+0 " punktów do policzenia"}'

LOG=runs/_suite_logs/ood_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
setsid nohup bash "$0" _run > "$LOG/ood.log" 2>&1 < /dev/null &
cat <<EOF
Ruszyło: $N rdzeni, J=$J.
Postęp:   tail -f $LOG/ood.log
Stan:     $TSC degrade status t2_ood
Wynik:    results/t2_ood.md (werdykt) + results/t2_ood.csv
EOF
