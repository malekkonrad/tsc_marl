#!/usr/bin/env bash
# REPLIKACJA ZŁOŻENIA NA DRUGIEJ SIECI. Jedno wywołanie:
#
#     bash deploy/glider/hangzhou_dr.sh
#
# PO CO. `results/t2_gate_dr.md` pokazał, że trening z awariami + bramka schodzi przy
# `mask0p5` na 481,1 s wobec FixedTime 507,0 — JEDYNA konfiguracja w tej pracy, która
# bije sterowanie stałocyklowe przy połowie utraconych detektorów. Twierdzenie stoi
# dziś na JEDNEJ sieci i to jest pierwszy cel ataku recenzenta. Ten skrypt dokłada
# HangZhou: 3 biegi GPU (`p1_fail`) + przeliczenie planu na obu sieciach.
#
# CO LICZY, po kolei:
#   1. `p1_fail` — rdzeń trenowany z awariami, HangZhou × seedy 0–2 (GPU).
#      JiNan jest już policzony i zostanie pominięty jako „aktualny".
#   2. `t2_gate_dr` — kwadrat 2×2 {rdzeń czysty, rdzeń DR} × {bez bramki, z bramką},
#      teraz na obu sieciach (CPU, minuty).
#
# CZEGO NIE LICZY: `p1_uw`. Ramię z UW-attention jest zamknięte jako wynik negatywny
# (§4, §5a notatki Fazy 3) i do twierdzenia o złożeniu nie jest potrzebne.
#
# ⚠ WYMAGA ŚWIEŻEGO KODU. 2026-09-20 doszły metryki ogona (`queue_max`, `queue_p90`,
# `wait_max`, `wait_max_demand`) liczone W EPIZODZIE oraz ślad bramki. Bieg na starym
# kodzie policzy się bez nich i trzeba go będzie powtórzyć — skrypt to sprawdza niżej
# i odmawia startu.
#
# PODZIAŁ WŁASNOŚCI: glider liczy BIEGI (`runs/`), raporty robi laptop (`results/`).
# `results/` jest wersjonowane i generowanie go tutaj blokuje `git pull`.
#
# Startuje w tle (odporne na wylogowanie) i od razu wraca. Wznowienie po przerwie:
# to samo polecenie. Zmienne: GPU_J / T2_J (ile punktów naraz), DRY=1 (tylko wypisz).
set -u
cd "$(dirname "$0")/../.."
TSC=.venv/bin/tsc

NETS="cityflow_hangzhou4x4_4ph"   # sieć DOKŁADANA; JiNan jest już policzony
SUITES="p1_fail"
PLAN=t2_gate_dr

N=$(OMP_NUM_THREADS= OMP_THREAD_LIMIT= nproc)
GPU_J=${GPU_J:-2}
T2_J=${T2_J:-$(( N / 5 > 2 ? N / 5 : 2 ))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

run() {
  local rc=0
  echo "$(date '+%F %T')  tsc $*"
  if [ -z "${DRY:-}" ]; then "$TSC" "$@" || rc=$?; fi
  echo "$(date '+%F %T')  koniec: tsc $* (kod $rc)"
}

if [ "${1:-}" = "_run" ]; then
  for s in $SUITES; do run suite run "$s" -j "$GPU_J"; done
  # T2 dopiero PO treningu: kolumny `gatppo_fail` i `gatppo_fail_gate_rule` biorą wagi
  # z `p1_fail`, a punkt bez wag rodzica jest POMIJANY (plan liczy się dalej, tylko bez
  # kolumny, dla której powstał). Sekwencyjnie w tej samej kolejce — nie równolegle.
  run degrade run "$PLAN" -j "$T2_J"
  exit
fi

if [ -n "${DRY:-}" ]; then
  bash "$0" _run
  exit
fi

"$TSC" doctor | grep -E "torch|PYTHONHASHSEED" || { echo "tsc doctor nie działa"; exit 1; }
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'brak CUDA'" \
  || { echo "torch nie widzi karty — make setup-gpu"; exit 1; }
# Strażnik równoległych kolejek — wyklucza po NAZWIE SKRYPTU, nie po PID.
#
# Dlaczego nie po PID: linia poleceń tego skryptu sama zawiera "deploy/glider/", więc
# `pgrep -f` trafia w siebie. Wykluczenie `$$` NIE WYSTARCZA — `$(...)` forkuje
# podpowłokę o INNYM PID, ale tej samej linii poleceń, i to ona zostaje w wyniku.
# Pierwsza wersja tego strażnika blokowała start zawsze, nie podając powodu.
#
# `pgrep -af` wypisuje "PID linia poleceń", więc `grep -v "$self"` zdejmuje i proces
# główny, i fork. Wypisanie trafionych procesów jest celowe: strażnik blokujący bez
# podania, CO blokuje, jest gorszy od braku strażnika.
self=$(basename "$0")
inne=$(pgrep -af "deploy/glider/.*\.sh" | grep -v "$self" || true)
if [ -n "$inne" ]; then
  echo "inna kolejka na gliderze jeszcze liczy:"
  echo "$inne"
  echo "uruchom później albo ubij tamtą kolejkę."
  exit 1
fi

# STRAŻNIK ŚWIEŻOŚCI KODU. Bez tego bieg policzy się bez metryk ogona i będzie
# do powtórzenia — dokładnie ten błąd, który zdarzył się przy `t2_gate_thr`
# (22 z 32 biegów bez kolumn, patrz §8.6 notatki Fazy 3).
.venv/bin/python - <<'PY' || { echo "→ git pull na gliderze i uruchom ponownie"; exit 1; }
import sys
from tsc_marl.robustness.degrade import METRICS
brak = {"queue_max", "queue_p90", "wait_max", "wait_max_demand"} - set(METRICS)
if brak:
    print(f"STARY KOD: brakuje metryk {sorted(brak)} w METRICS")
    sys.exit(1)
print("kod świeży: metryki ogona obecne")
PY

# Próg bramki bierze się z artefaktu skalibrowanego na czystym oknie zrzutu BC.
# Bez niego kolumny bramkowane padają SystemExit dopiero w podprocesie.
brak=0
for net in $NETS; do
  [ -f "runs/heads/gate_ref__$net.npz" ] || { echo "BRAK: runs/heads/gate_ref__$net.npz"; brak=1; }
done
if [ "$brak" = 1 ]; then
  echo "Skopiuj z laptopa:  rsync -av runs/heads/ glider:tsc_marl/runs/heads/"
  exit 1
fi

for s in $SUITES; do "$TSC" suite status "$s" | tail -1; done

LOG=runs/_suite_logs/hangzhou_dr_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
setsid nohup bash "$0" _run > "$LOG/hangzhou_dr.log" 2>&1 < /dev/null &

cat <<EOF

Kolejka wystartowała w tle. Log: $LOG/hangzhou_dr.log
  podgląd:   tail -f $LOG/hangzhou_dr.log
  stan:      $TSC suite status p1_fail ; $TSC degrade status $PLAN

Po zakończeniu przenieś biegi na laptop i TAM zrób raport:
  rsync -av glider:tsc_marl/runs/ runs/
  $TSC degrade report $PLAN
EOF
