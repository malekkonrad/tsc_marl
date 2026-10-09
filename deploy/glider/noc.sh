#!/usr/bin/env bash
# Nocna kolejka na maszynie z GPU (glider). Jedno wywołanie:
#
#     bash deploy/glider/noc.sh
#
# startuje w tle (odporne na wylogowanie) TRZY kolejki i od razu wraca:
#   gpu — treningi rdzenia i ablacji (entry: core, karta 0), siatka po siatce;
#   cpu — treningi LibSignala (Advanced-CoLight, CoLight, PressLight), tylko CPU;
#   t2  — ewaluacje pod awarią sensorów. CZEKA, aż gpu i cpu skończą, bo potrzebuje
#         wag z tych treningów (inaczej nowe seedy byłyby pominięte jako „brak rodzica").
#
# Siatki pomijają punkty z aktualnym wynikiem, więc skrypt liczy tylko to, czego brak
# (np. dograne seedy 3–4). WZNOWIENIE po przerwie: to samo polecenie — bieg przerwany
# na tej maszynie (martwy proces w `running`) runlog liczy od nowa.
#
# Zmienne: CPU_J / GPU_J / T2_J (ile punktów naraz), DRY=1 (tylko wypisz komendy).
# PODZIAŁ WŁASNOŚCI: glider liczy BIEGI (`runs/`), raporty robi laptop (`results/`).
# Raport to sekundy liczenia z `runs/`, a `results/` jest katalogiem WERSJONOWANYM,
# którego właścicielem jest laptop — generowanie go tutaj dwa razy zablokowało
# `git pull` („untracked working tree files would be overwritten by merge”) i za
# każdym razem trzeba było ręcznie odkładać pliki na bok.
set -u
cd "$(dirname "$0")/../.."
TSC=.venv/bin/tsc
# OMP_NUM_THREADS= przed `nproc`: ta zmienna JEST przez nproc honorowana, a kolejki
# startują przez `setsid nohup bash "$0" …`, czyli w potomku, który dziedziczy nasz
# własny eksport. Bez tego `nproc` zwraca 1 i J spada do minimum (2) na 32 rdzeniach.
N=$(OMP_NUM_THREADS= OMP_THREAD_LIMIT= nproc)
# CityFlow bierze 4 wątki na symulację, torch dostaje 2 — stąd ~6 wątków na punkt
CPU_J=${CPU_J:-$(( N / 6 > 2 ? N / 6 : 2 ))}
GPU_J=${GPU_J:-2}
T2_J=${T2_J:-$(( N / 5 > 2 ? N / 5 : 2 ))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

GPU_SUITES="p1_gate p1_abl_meanpool p1_abl_nognn p1_abl_state_ep p1_abl_state_pressure p1_abl_state_queue"
CPU_SUITES="p1_reference t1_learned t1_learned_paper"
T2_PLANS="t2_nomech t2_ablations"

run() {
  local rc=0
  echo "$(date '+%F %T')  tsc $*"
  if [ -z "${DRY:-}" ]; then "$TSC" "$@" || rc=$?; fi
  echo "$(date '+%F %T')  koniec: tsc $* (kod $rc)"
}

gpu_queue() {
  for s in $GPU_SUITES; do run suite run "$s" -j "$GPU_J"; done
  touch "$LOG/gpu.done"
}

cpu_queue() {
  for s in $CPU_SUITES; do run suite run "$s" -j "$CPU_J"; done
  touch "$LOG/cpu.done"
}

t2_queue() {
  if [ -z "${DRY:-}" ]; then
    echo "$(date '+%F %T')  czekam na koniec treningów (gpu.done + cpu.done)"
    until [ -f "$LOG/gpu.done" ] && [ -f "$LOG/cpu.done" ]; do sleep 60; done
  fi
  for p in $T2_PLANS; do
    run degrade run "$p" -j "$T2_J"
  done
}

case "${1:-}" in
  _gpu) LOG=$2; gpu_queue; exit ;;
  _cpu) LOG=$2; cpu_queue; exit ;;
  _t2) LOG=$2; t2_queue; exit ;;
esac

if [ -z "${DRY:-}" ]; then
  "$TSC" doctor | grep -E "torch|PYTHONHASHSEED" || { echo "tsc doctor nie działa"; exit 1; }
  .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'brak CUDA'" \
    || { echo "torch nie widzi karty — make setup-gpu"; exit 1; }
fi

LOG=runs/_suite_logs/noc_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
if [ -n "${DRY:-}" ]; then
  bash "$0" _gpu "$LOG"; bash "$0" _cpu "$LOG"; bash "$0" _t2 "$LOG"
  rm -rf "$LOG"
  exit
fi
setsid nohup bash "$0" _gpu "$LOG" > "$LOG/gpu.log" 2>&1 < /dev/null &
sleep 20  # rozsunięcie startów kolejek (configs/sim/*.cfg — patrz suite.STAGGER_S)
setsid nohup bash "$0" _cpu "$LOG" > "$LOG/cpu.log" 2>&1 < /dev/null &
setsid nohup bash "$0" _t2 "$LOG" > "$LOG/t2.log" 2>&1 < /dev/null &

cat <<EOF
Ruszyło: $N rdzeni, GPU_J=$GPU_J, CPU_J=$CPU_J, T2_J=$T2_J, OMP_NUM_THREADS=$OMP_NUM_THREADS.
Postęp:   tail -f $LOG/gpu.log $LOG/cpu.log $LOG/t2.log
Stan:     $TSC suite status p1_gate ; $TSC degrade status t2_ablations
Stop:     pkill -f 'deploy/glider/noc.sh' ; pkill -f 'tsc (suite|degrade) run'
Raporty:  na LAPTOPIE po synchronizacji — rsync -av glider:tsc_marl/runs/ runs/ ; tsc table
EOF
