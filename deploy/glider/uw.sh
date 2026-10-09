#!/usr/bin/env bash
# T2 dla UW-attention (plan t2_uw) — uruchamiać PO zakończeniu noc.sh:
#
#     bash deploy/glider/uw.sh
#
# Potrzebuje 5 seedów rdzenia (noc.sh je dogrywa) ORAZ artefaktów głowy obserwacyjnej
# dla każdej sieci planu: runs/heads/obs_head__<sieć>.pt + u_ref__<sieć>.npz.
# Głowa jest tania (CPU, ~10 s) — licz ją na laptopie i skopiuj RAZEM z u_ref, żeby
# pomiar szedł na dokładnie tym artefakcie, dla którego zapisane są τ i tabela A3a:
#
#     rsync -av runs/heads/ glider:tsc_marl/runs/heads/
#
# Startuje w tle (odporne na wylogowanie) i od razu wraca. Na końcu pisze
# results/t2_uw.md + results/t2_uw.csv. Wznowienie: to samo polecenie.
#
# Zmienne: J (ile punktów naraz), DRY=1 (tylko wypisz komendy).
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
J=${J:-$(( N / 5 > 2 ? N / 5 : 2 ))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

run() {
  local rc=0
  echo "$(date '+%F %T')  tsc $*"
  if [ -z "${DRY:-}" ]; then "$TSC" "$@" || rc=$?; fi
  echo "$(date '+%F %T')  koniec: tsc $* (kod $rc)"
}

if [ "${1:-}" = "_run" ]; then
  run degrade run t2_uw -j "$J"
  exit
fi

if [ -n "${DRY:-}" ]; then
  bash "$0" _run
  exit
fi
if pgrep -f "deploy/glider/noc.sh" > /dev/null; then
  echo "noc.sh jeszcze liczy — ten plan potrzebuje jego wag (seedy 0–4). Uruchom później."
  exit 1
fi
# Brak głowy = SystemExit dopiero w podprocesie, czyli po cichu pominięte punkty.
# Lepiej sprawdzić tu, zanim ruszy kolejka.
brak=0
for net in cityflow_jinan3x4_4ph cityflow_hangzhou4x4_4ph; do
  for f in "runs/heads/obs_head__$net.pt" "runs/heads/u_ref__$net.npz"; do
    [ -f "$f" ] || { echo "BRAK: $f"; brak=1; }
  done
done
if [ "$brak" = 1 ]; then
  echo "Skopiuj artefakty głowy z laptopa:  rsync -av runs/heads/ glider:tsc_marl/runs/heads/"
  echo "albo policz tutaj:  $TSC obshead -n <sieć> --save --window 12 --persistence"
  exit 1
fi
"$TSC" degrade status t2_uw | awk '$2 == "brak" {n++} END {print n+0 " punktów do policzenia"}'

LOG=runs/_suite_logs/uw_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
setsid nohup bash "$0" _run > "$LOG/uw.log" 2>&1 < /dev/null &
cat <<EOF2
Ruszyło: $N rdzeni, J=$J.
Postęp:   tail -f $LOG/uw.log
Stan:     $TSC degrade status t2_uw
Wynik:    runs/t2/*__t2_uw/ — raport zrób NA LAPTOPIE, po synchronizacji:
          rsync -av glider:tsc_marl/runs/t2/ runs/t2/ && tsc degrade report t2_uw
EOF2
