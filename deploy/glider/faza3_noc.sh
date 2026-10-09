#!/usr/bin/env bash
# FAZA 3 — CAŁA NOC JEDNYM POLECENIEM:
#
#     bash deploy/glider/faza3_noc.sh
#
# Spina dwie pozostałe pozycje z `tsc_marl_research/faza3-po-t2.md`:
#
#   1. SWEEP τ (`t2_uw_tau`) — tania asekuracja przy wyniku negatywnym UW-attention.
#      Jedzie z GOTOWYCH wag z `runs/t1`, więc liczy się szybko i NIE ZALEŻY od
#      niczego, co robi punkt 2. Dlatego jest pierwszy: gdyby trening się wywrócił,
#      ta odpowiedź i tak jest w kieszeni.
#
#   2. TRENING Z MECHANIZMEM (`train_uw.sh`) — trzy ramiona: rdzeń na czystym
#      (`p1_gate`, policzony wcześniej), rdzeń z awariami BEZ UW (`p1_fail`),
#      rdzeń z awariami Z UW (`p1_uw`), a na końcu T2 na wszystkich trzech
#      (`t2_train_uw`). Ramię 2 jest KONTROLĄ: bez niego poprawę dałoby się
#      przypisać domain randomization zamiast mechanizmowi.
#
# Kolejność jest istotna także dlatego, że punkt 1 zajmuje CPU, a punkt 2 startuje
# od treningów na GPU — ale sekwencyjnie, nie równolegle, żeby nie bić się o rdzenie
# z symulatorem (CityFlow bierze ~4 wątki na punkt).
#
# GLIDER LICZY BIEGI, RAPORTY ROBI LAPTOP. Ten skrypt celowo nie woła
# `tsc degrade report` — `results/` jest katalogiem wersjonowanym, którego
# właścicielem jest laptop, a generowanie go tutaj dwa razy zablokowało `git pull`.
# Po nocy: rsync runs/ na laptop i tam `tsc degrade report t2_uw_tau` oraz
# `tsc degrade report t2_train_uw`.
#
# Wznowienie po przerwie: to samo polecenie (policzone punkty są pomijane).
# Zmienne: GPU_J / T2_J (ile punktów naraz), DRY=1 (tylko wypisz komendy).
set -u
cd "$(dirname "$0")/../.."
TSC=.venv/bin/tsc

NETS="cityflow_jinan3x4_4ph"   # sieci, dla których musi istnieć zamrożona głowa
TAU_PLAN=t2_uw_tau

# OMP_NUM_THREADS= przed `nproc`: ta zmienna JEST przez nproc honorowana, a kolejka
# startuje w potomku (`setsid nohup bash "$0" …`), który dziedziczy nasz eksport.
# Bez tego `nproc` zwraca 1 i J spada do minimum (2) na 32 rdzeniach.
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
  run degrade run "$TAU_PLAN" -j "$T2_J"
  # Kolejka treningowa w jej własnym skrypcie — `_run` omija jego backgrounding,
  # więc wszystko zostaje w JEDNYM logu i jednej sekwencji. Bez duplikowania jego
  # logiki: gdy zmieni się tam lista suite'ów, tu nie trzeba nic ruszać.
  GPU_J="$GPU_J" T2_J="$T2_J" bash deploy/glider/train_uw.sh _run
  exit
fi

if [ -n "${DRY:-}" ]; then
  bash "$0" _run
  exit
fi

"$TSC" doctor | grep -E "torch|PYTHONHASHSEED" || { echo "tsc doctor nie działa"; exit 1; }
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'brak CUDA'" \
  || { echo "torch nie widzi karty — make setup-gpu"; exit 1; }
if pgrep -f "deploy/glider/(noc|uw|ood)\.sh" > /dev/null; then
  echo "inna kolejka jeszcze liczy — uruchom później (GPU i rdzenie są wspólne)."
  exit 1
fi

# Brak artefaktów głowy = SystemExit dopiero w podprocesie, czyli punkty PO CICHU
# przepadają: sweep τ zostałby bez kolumn UW, a ramię 3 treningu padłoby po starcie.
brak=0
for net in $NETS; do
  for f in "runs/heads/obs_head__$net.pt" "runs/heads/u_ref__$net.npz"; do
    [ -f "$f" ] || { echo "BRAK: $f"; brak=1; }
  done
done
if [ "$brak" = 1 ]; then
  echo "Skopiuj z laptopa:  rsync -av runs/heads/ glider:tsc_marl/runs/heads/"
  exit 1
fi

# Ramię 1 nie jest liczone tutaj — jest kolumną odniesienia w `t2_train_uw`.
if "$TSC" suite status p1_gate | grep -q " brak "; then
  echo "UWAGA: p1_gate ma niepoliczone punkty — kolumna ramienia 1 będzie niepełna."
fi

"$TSC" degrade status "$TAU_PLAN" | awk '$2 == "brak" {n++} END {print n+0 " punktów sweepu τ"}'
for s in p1_fail p1_uw; do "$TSC" suite status "$s" | tail -1; done

LOG=runs/_suite_logs/faza3_$(date +%Y%m%d_%H%M)
mkdir -p "$LOG"
setsid nohup bash "$0" _run > "$LOG/faza3.log" 2>&1 < /dev/null &

cat <<EOF2
Ruszyło: $N rdzeni, GPU_J=$GPU_J, T2_J=$T2_J, OMP_NUM_THREADS=$OMP_NUM_THREADS.
Kolejność: sweep τ ($TAU_PLAN) -> treningi (p1_fail, p1_uw) -> T2 (t2_train_uw).

Postęp:   tail -f $LOG/faza3.log
Stan:     $TSC degrade status $TAU_PLAN ; $TSC suite status p1_uw
          $TSC degrade status t2_train_uw
Stop:     pkill -f 'deploy/glider/(faza3_noc|train_uw)\.sh' ; pkill -f 'tsc (suite|degrade) run'

Po nocy, NA LAPTOPIE:
  rsync -av glider:tsc_marl/runs/ runs/
  uv run tsc degrade report $TAU_PLAN
  uv run tsc degrade report t2_train_uw
EOF2
