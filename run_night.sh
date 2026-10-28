#!/usr/bin/env bash
# Nachtlauf: Experiment 6 (CPU, 3 Worker) und Experiment 7a (M1-GPU) gleichzeitig.
# Erst beide Schnelltests, dann beide Hauptläufe parallel, am Ende Auswertung + Git-Commit.
# Abbruch mit Ctrl-C jederzeit ok, Neustart macht weiter, wo es aufgehört hat.
# Fortschritt: results/NIGHT.txt, results/neurodyn-w*.log, results/critical.log
cd "$(dirname "$0")"

if [ -z "${NCM_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export NCM_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
NW="${NW:-3}"
mkdir -p results/neurodyn results/critical
S=results/NIGHT.txt
echo "Start: $(date '+%F %T')" > "$S"

echo "== Schnelltest Experiment 6 =="
python experiments/neurodyn.py --smoke 2>&1 | tee results/neurodyn-smoke.log
if ! grep -q "SCHNELLTEST OK" results/neurodyn-smoke.log; then echo "Exp. 6 SCHNELLTEST FEHLGESCHLAGEN"; exit 1; fi
echo "== Schnelltest Experiment 7a (lädt beim ersten Mal CIFAR-10, ~170 MB) =="
python experiments/critical_periods.py --smoke 2>&1 | tee results/critical-smoke.log
if ! grep -q "SCHNELLTEST OK" results/critical-smoke.log; then echo "Exp. 7a SCHNELLTEST FEHLGESCHLAGEN"; exit 1; fi
echo "$(date '+%T')  beide Schnelltests ok" >> "$S"

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null; exit 130' INT TERM
for w in $(seq 0 $((NW - 1))); do
  python experiments/neurodyn.py --worker "$w" --nworkers "$NW" > "results/neurodyn-w$w.log" 2>&1 &
  PIDS+=($!)
done
python experiments/critical_periods.py > results/critical.log 2>&1 &
PIDS+=($!)
alive() { for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }

echo "== Hauptläufe gestartet $(date '+%T'). Du kannst das Terminal jetzt in Ruhe lassen. =="
while alive; do
  sleep 120
  n6=$(ls results/neurodyn/*.json 2>/dev/null | wc -l | tr -d ' ')
  n7=$(ls results/critical/*.json 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date '+%T')  Exp. 6: $n6/135 Läufe | Exp. 7a: $n7/26 Läufe | $(tail -1 results/critical.log 2>/dev/null | cut -c1-70)"
done
wait

echo "== Auswertung $(date '+%T') =="
python experiments/neurodyn_report.py > results/neurodyn-report.log 2>&1
python experiments/critical_report.py > results/critical-report.log 2>&1
cat results/neurodyn-summary.md results/critical-summary.md 2>/dev/null
grep -l "Traceback" results/neurodyn-w*.log results/critical.log 2>/dev/null && echo "ACHTUNG: Fehler in den Logs oben"
echo "$(date '+%T')  Auswertung fertig" >> "$S"

git add -A results experiments *.sh *.md .gitignore >/dev/null 2>&1
git commit -q -m "Nachtlauf: Experiment 6 (Neuronendynamik) und 7a (kritische Phasen)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KQ66rEr7ms41phsikjohHs" && echo "Git-Commit erstellt (git push nicht vergessen)"
echo "Ende: $(date '+%F %T')" | tee -a "$S"
