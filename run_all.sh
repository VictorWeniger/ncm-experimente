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

# 0) Downloads: Sudoku easy/medium, TinyStories, GPT-2 large (Referenzmodell für die Textqualität, ~3 GB)
step download bash -c '
  set -e
  for p in "sudoku/easy/*" "sudoku/medium/*" "tinystories/*"; do
    hf download hazyresearch/ncm-checkpoints --include "$p" --local-dir artifacts
  done
  python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained(\"gpt2-large\"); AutoModelForCausalLM.from_pretrained(\"gpt2-large\"); print(\"gpt2-large ok\")"
'

# 1) Schneller Test des Text-Skripts (wenige Minuten), damit Fehler früh auffallen
step text-smoke python experiments/textsignal_tinystories.py --seeds 42 --n 32 --cal_seed 1000 --tag text-smoke

# 2) Sudoku hard: Ein-Pass-Bewertung (ehrliche Kosten) + Zufallskontrolle, 1000 Test- / 1000 Kalibrierungsbretter
step sudoku-hard-self3 python experiments/selfsignal_sudoku.py --task hard --n 1000 --n_cal 1000 --tag self3

# 3) Text: TinyStories mit allen 5 Paper-Seeds
step tinystories-text python experiments/textsignal_tinystories.py --tag text

# 4) Gegenprobe auf medium und easy
step sudoku-medium-self3 python experiments/selfsignal_sudoku.py --task medium --n 1000 --n_cal 1000 --tag self3
step sudoku-easy-self3 python experiments/selfsignal_sudoku.py --task easy --n 1000 --n_cal 1000 --tag self3

# Ergebnisse sichern
git add -A results experiments *.sh *.md >/dev/null 2>&1
git commit -q -m "Tageslauf: Ein-Pass-Bewertung, Text-Experiment, easy/medium

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KQ66rEr7ms41phsikjohHs" >/dev/null 2>&1 && echo "$(date '+%T')  Git-Commit erstellt" >> "$STATUS"
echo "Ende: $(date '+%F %T')" >> "$STATUS"
cat "$STATUS"
