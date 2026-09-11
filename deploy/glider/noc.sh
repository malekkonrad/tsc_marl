#!/usr/bin/env bash
# Nocna kolejka na maszynie z GPU (glider). Jedno wywołanie:
#
#     bash deploy/glider/noc.sh
#
# startuje DWIE kolejki w tle (odporne na wylogowanie) i od razu wraca:
#   gpu — ablacje rdzenia 1.6 (entry: core, karta 0), siatka po siatce;
#   cpu — T2 bez mechanizmu (potrzebuje wag z runs/t1, które są tylko tutaj),
#         potem uczone baseline'y LibSignala (CoLight/PressLight/Advanced-CoLight).
# Kolejność w kolejce CPU = priorytet: T2 rozstrzyga o fallbacku, T1 może poczekać.
#
# WZNOWIENIE: to samo polecenie. Siatki pomijają punkty z aktualnym wynikiem,
# a bieg przerwany na tej maszynie (martwy proces) liczy się od nowa.
#
# Zmienne: CPU_J / GPU_J (ile punktów naraz), DRY=1 (tylko wypisz komendy).
set -u
cd "$(dirname "$0")/../.."
TSC=.venv/bin/tsc
N=$(nproc)
# CityFlow bierze 4 wątki na symulację, torch dostaje 2 — stąd ~6 wątków na punkt
CPU_J=${CPU_J:-$(( N / 6 > 2 ? N / 6 : 2 ))}
GPU_J=${GPU_J:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

run() {
  local rc=0
  echo "$(date '+%F %T')  tsc $*"
  if [ -z "${DRY:-}" ]; then "$TSC" "$@" || rc=$?; fi
  echo "$(date '+%F %T')  koniec: tsc $* (kod $rc)"
}

gpu_queue() {
  for s in p1_abl_meanpool p1_abl_nognn p1_abl_state_ep p1_abl_state_pressure p1_abl_state_queue; do
    run suite run "$s" -j "$GPU_J"
  done
}

cpu_queue() {
  run degrade run t2_nomech -j "$CPU_J"
  run degrade report t2_nomech
  run suite run t1_learned -j "$CPU_J"
  run suite run t1_learned_paper -j "$CPU_J"
}

case "${1:-}" in
  _gpu) gpu_queue; exit ;;
  _cpu) cpu_queue; exit ;;
esac

if [ -z "${DRY:-}" ]; then
  "$TSC" doctor | grep -E "torch|PYTHONHASHSEED" || { echo "tsc doctor nie działa"; exit 1; }
  .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'brak CUDA'" \
    || { echo "torch nie widzi karty — make setup-gpu"; exit 1; }
fi

LOG=runs/_suite_logs/noc_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
if [ -n "${DRY:-}" ]; then
  bash "$0" _gpu; bash "$0" _cpu
  exit
fi
setsid nohup bash "$0" _gpu > "$LOG/gpu.log" 2>&1 < /dev/null &
sleep 20  # rozsunięcie startów obu kolejek (configs/sim/*.cfg — patrz suite.STAGGER_S)
setsid nohup bash "$0" _cpu > "$LOG/cpu.log" 2>&1 < /dev/null &

cat <<EOF
Ruszyło: $N rdzeni, CPU_J=$CPU_J, GPU_J=$GPU_J, OMP_NUM_THREADS=$OMP_NUM_THREADS.
Postęp:   tail -f $LOG/gpu.log $LOG/cpu.log
Stan:     $TSC degrade status t2_nomech ; $TSC suite status t1_learned
Stop:     pkill -f 'deploy/glider/noc.sh' ; pkill -f 'tsc (suite|degrade) run'
EOF
