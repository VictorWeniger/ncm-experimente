#!/usr/bin/env bash
# Experiment 6 über Nacht: erst Schnelltest, dann 3 Worker parallel auf der CPU, am Ende Auswertung + Git-Commit.
# Abbrechen mit Ctrl-C ist jederzeit ok: Fertige Läufe bleiben, ein Neustart macht dort weiter.
# Fortschritt: tail -f results/neurodyn-w*.log
cd "$(dirname "$0")"

if [ -z "${NCM_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export NCM_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
NW="${NW:-3}"
mkdir -p results/neurodyn

echo "== Schnelltest $(date '+%T') =="
if ! python experiments/neurodyn.py --smoke 2>&1 | tee results/neurodyn-smoke.log; then
  echo "SCHNELLTEST FEHLGESCHLAGEN, siehe results/neurodyn-smoke.log"; exit 1
fi
grep -q "SCHNELLTEST OK" results/neurodyn-smoke.log || { echo "SCHNELLTEST FEHLGESCHLAGEN"; exit 1; }

echo "== Hauptlauf mit $NW Workern, Start $(date '+%T') =="
trap 'kill "${PIDS[@]}" 2>/dev/null; exit 130' INT TERM
PIDS=()
for w in $(seq 0 $((NW - 1))); do
  python experiments/neurodyn.py --worker "$w" --nworkers "$NW" "$@" > "results/neurodyn-w$w.log" 2>&1 &
  PIDS+=($!)
done
alive() { for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
# Fortschritt im Terminal zeigen, bis alle Worker fertig sind
while alive; do
  sleep 60
  echo "$(date '+%T')  fertig: $(ls results/neurodyn/*.json 2>/dev/null | wc -l | tr -d ' ')/135 Läufe"
done
wait

echo "== Auswertung $(date '+%T') =="
python experiments/neurodyn_report.py | tee results/neurodyn-report.log
grep -h "Traceback\|Error" results/neurodyn-w*.log && echo "ACHTUNG: Fehler in einem Worker-Log"

git add -A results experiments *.sh *.md >/dev/null 2>&1
git commit -q -m "Experiment 6: Neuronen mit Zeitkonstanten und Feuerwahrscheinlichkeit

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KQ66rEr7ms41phsikjohHs" && echo "Git-Commit erstellt (git push nicht vergessen)"
echo "Ende: $(date '+%F %T')"
