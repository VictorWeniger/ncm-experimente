#!/usr/bin/env bash
# Lässt alle offenen Experimente nacheinander laufen (mehrere Stunden).
# Ein Fehler in einem Schritt bricht die anderen nicht ab. Fortschritt: results/STATUS.txt
cd "$(dirname "$0")"
set -o pipefail

# Mac wach halten, solange das Skript läuft
if [ -z "${NCM_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export NCM_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
mkdir -p results
STATUS=results/STATUS.txt
echo "Start: $(date '+%F %T')" > "$STATUS"

step() {  # step <name> <befehl...>
  local name="$1"; shift
  echo "$(date '+%T')  läuft:  $name" >> "$STATUS"
  echo -e "\n==================== $name ($(date '+%T')) ====================\n"
  if "$@" 2>&1 | tee "results/$name.log"; then
    echo "$(date '+%T')  fertig: $name" >> "$STATUS"
  else
    echo "$(date '+%T')  FEHLER: $name (siehe results/$name.log)" >> "$STATUS"
  fi
}

# 0) Downloads (übersprungen, wenn schon da). GPT-2 large wird erst im Text-Schritt geladen.
step download bash -c '
  set -e
  for p in "sudoku/easy/*" "sudoku/medium/*" "tinystories/*"; do
    hf download hazyresearch/ncm-checkpoints --include "$p" --local-dir artifacts
  done
'

# 1) Sudoku zuerst (klein, braucht wenig Arbeitsspeicher): Ein-Pass-Bewertung + Zufallskontrolle, 1000/1000 Bretter
step sudoku-hard-self3 python experiments/selfsignal_sudoku.py --task hard --n 1000 --n_cal 1000 --tag self3
step sudoku-medium-self3 python experiments/selfsignal_sudoku.py --task medium --n 1000 --n_cal 1000 --tag self3
step sudoku-easy-self3 python experiments/selfsignal_sudoku.py --task easy --n 1000 --n_cal 1000 --tag self3

# 2) Text zuletzt (GPT-2 large als Qualitätsmaß braucht viel Arbeitsspeicher): erst Schnelltest, dann voller Lauf
step text-smoke python experiments/textsignal_tinystories.py --seeds 42 --n 32 --cal_seed 1000 --tag text-smoke
if grep -q "fertig: text-smoke" "$STATUS"; then
  step tinystories-text python experiments/textsignal_tinystories.py --tag text
else
  echo "$(date '+%T')  übersprungen: tinystories-text (Schnelltest fehlgeschlagen)" >> "$STATUS"
fi

# Ergebnisse sichern
git add -A results experiments *.sh *.md >/dev/null 2>&1
git commit -q -m "Tageslauf: Ein-Pass-Bewertung, Text-Experiment, easy/medium

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KQ66rEr7ms41phsikjohHs" >/dev/null 2>&1 && echo "$(date '+%T')  Git-Commit erstellt" >> "$STATUS"
echo "Ende: $(date '+%F %T')" >> "$STATUS"
cat "$STATUS"
