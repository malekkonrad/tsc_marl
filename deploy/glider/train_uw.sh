#!/usr/bin/env bash
# Faza 3, §3 — TRENING POLITYKI Z MECHANIZMEM. Jedno wywołanie, od zera do raportu:
#
#     bash deploy/glider/train_uw.sh
#
# Odpowiada na pytanie, którego T2 nie rozstrzygnęło: modulacji NIE DA SIĘ dokleić do
# gotowej polityki (`results/t2_uw.md`) — ale czy polityka TRENOWANA z log ρ nauczyłaby
# się na nim polegać? Trzy ramiona, bo inaczej odpowiedź jest nierozstrzygalna:
#
#   1. rdzeń na czystym            `p1_gate`   (już policzony — noc.sh)
#   2. rdzeń z awariami, BEZ UW    `p1_fail`   ← liczy ten skrypt
#   3. rdzeń z awariami, Z UW      `p1_uw`     ← liczy ten skrypt
#   potem T2 na wszystkich trzech: `t2_train_uw` -> results/t2_train_uw.md
#
# Ramię 2 jest KONTROLĄ, nie ozdobą: trening z awariami sam w sobie poprawia odporność,
# więc bez niego całą poprawę dałoby się przypisać domain randomization zamiast
# mechanizmowi. Raport czyta się dwoma odczytami: (2 vs 1) = ile daje trening
# z awariami, (3 vs 2) = ile daje MECHANIZM ponad to.
#
# CO TO KOSZTUJE. Wersja minimalna z notatki: JiNan × 3 seedy × 2 ramiona = 6 biegów.
# Poszerzenie (obie sieci × 5 seedów = 20) to dopisanie pozycji do `grid` w obu
# suite'ach — `run_id` się nie zmienia, więc policzone punkty zostają pominięte.
# Narzut mechanizmu na epizod zmierzony na CPU: ~10% (21,5 s -> 23,6 s), głowa jest
# tania. Na GPU narzut jest względnie WIĘKSZY, bo głowa i tak liczy na CPU.
#
# WYMAGA (skrypt sprawdza i odmawia startu, zamiast po cichu pomijać punkty):
#   * artefaktów zamrożonej głowy dla każdej sieci: runs/heads/obs_head__<sieć>.pt
#     + u_ref__<sieć>.npz. Liczy się je na laptopie (~10 s) i kopiuje RAZEM:
#         rsync -av runs/heads/ glider:tsc_marl/runs/heads/
#   * policzonego `p1_gate` (ramię 1) — inaczej T2 policzy się bez kolumny odniesienia.
#
# Startuje w tle (odporne na wylogowanie) i od razu wraca. Wznowienie po przerwie:
# to samo polecenie. Zmienne: GPU_J / T2_J (ile punktów naraz), DRY=1 (tylko wypisz).
set -u
cd "$(dirname "$0")/../.."
TSC=.venv/bin/tsc

NETS="cityflow_jinan3x4_4ph"     # musi zgadzać się z `grid.network` w obu suite'ach
SUITES="p1_fail p1_uw"
PLAN=t2_train_uw

# OMP_NUM_THREADS= przed `nproc`: ta zmienna JEST przez nproc honorowana, a kolejka
# startuje przez `setsid nohup bash "$0" …`, czyli w potomku, który dziedziczy nasz
# własny eksport. Bez tego `nproc` zwraca 1 i J spada do minimum na 32 rdzeniach.
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
  # T2 dopiero po OBU ramionach: plan ma kolumny z rodzicami w `p1_fail` i `p1_uw`,
  # a punkt bez wag rodzica jest POMIJANY (plan liczy się dalej, tylko bez kolumny,
  # dla której powstał). Sekwencyjnie w tej samej kolejce — nie równolegle.
  run degrade run "$PLAN" -j "$T2_J"
  # Raportu tu NIE MA: glider liczy biegi (`runs/`), raporty robi laptop (`results/`).
  # `results/` jest wersjonowane i generowanie go tutaj dwa razy zablokowało `git pull`.
  exit
fi

if [ -n "${DRY:-}" ]; then
  bash "$0" _run
  exit
fi

"$TSC" doctor | grep -E "torch|PYTHONHASHSEED" || { echo "tsc doctor nie działa"; exit 1; }
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'brak CUDA'" \
  || { echo "torch nie widzi karty — make setup-gpu"; exit 1; }
if pgrep -f "deploy/glider/noc.sh" > /dev/null; then
  echo "noc.sh jeszcze liczy — ramię 1 (p1_gate) stamtąd pochodzi. Uruchom później."
  exit 1
fi

# Brak głowy = SystemExit dopiero w podprocesie, czyli trening ramienia 3 pada po
# starcie, a suite liczy dalej. Lepiej sprawdzić tu, zanim ruszy kolejka.
brak=0
for net in $NETS; do
  for f in "runs/heads/obs_head__$net.pt" "runs/heads/u_ref__$net.npz"; do
    [ -f "$f" ] || { echo "BRAK: $f"; brak=1; }
  done
done
if [ "$brak" = 1 ]; then
  echo "Skopiuj artefakty głowy z laptopa:  rsync -av runs/heads/ glider:tsc_marl/runs/heads/"
  echo "albo policz tutaj:  $TSC obshead -n <sieć> --save --window 12 --persistence"
  exit 1
fi

# Ramię 1 nie jest liczone przez ten skrypt — jest kolumną odniesienia w T2.
if "$TSC" suite status p1_gate | grep -q " brak "; then
  echo "UWAGA: p1_gate ma niepoliczone punkty — kolumna ramienia 1 w $PLAN będzie niepełna."
  echo "       Policz najpierw: $TSC suite run p1_gate -j $GPU_J"
fi

for s in $SUITES; do
  "$TSC" suite status "$s" | tail -1
done

LOG=runs/_suite_logs/train_uw_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
setsid nohup bash "$0" _run > "$LOG/train_uw.log" 2>&1 < /dev/null &

cat <<EOF
Ruszyło: $N rdzeni, GPU_J=$GPU_J, T2_J=$T2_J, OMP_NUM_THREADS=$OMP_NUM_THREADS.
Postęp:   tail -f $LOG/train_uw.log
Stan:     $TSC suite status p1_fail ; $TSC suite status p1_uw
          $TSC degrade status $PLAN
Wynik:    runs/t2/*__$PLAN/ — raport NA LAPTOPIE, po synchronizacji:
          rsync -av glider:tsc_marl/runs/ runs/ && tsc degrade report $PLAN
Stop:     pkill -f 'deploy/glider/train_uw.sh' ; pkill -f 'tsc (suite|degrade) run'

Czy mechanizm w ogóle żyje w treningu — kolumny \`uw_spread\` / \`uw_active\`
w runs/t1/gatppo__*__p1_uw/metrics.csv (split=train). Na starcie będą WYSOKIE
(polityka jest zła, kolejki rosną, u wychodzi ponad u_ref wszędzie) i powinny
opadać wraz z ATT — to jest ta granica u_ref opisana w robustness/training.py.
EOF
