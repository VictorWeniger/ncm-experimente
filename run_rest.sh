#!/usr/bin/env bash
# Nachholen nach dem Tageslauf: Sudoku hard und Text, speicherschonend für das M1-MacBook.
# Vorher den Mac neu starten und nur Terminal offen lassen.
cd "$(dirname "$0")"
set -o pipefail
if [ -z "${NCM_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export NCM_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
mkdir -p results
STATUS=results/STATUS-rest.txt
echo "Start: $(date '+%F %T')" > "$STATUS"
step() {
  local name="$1"; shift
  echo "$(date '+%T')  läuft:  $name" >> "$STATUS"
  echo -e "\n==================== $name ($(date '+%T')) ====================\n"
  if "$@" 2>&1 | tee "results/$name.log"; then
    echo "$(date '+%T')  fertig: $name" >> "$STATUS"
  else
    echo "$(date '+%T')  FEHLER: $name (siehe results/$name.log)" >> "$STATUS"
  fi
}
step sudoku-hard-self3 python experiments/selfsignal_sudoku.py --task hard --n 1000 --n_cal 1000 --bs 32 --tag self3
step tinystories-text python experiments/textsignal_tinystories.py --bs 16 --tag text
git add -A results experiments *.sh *.md >/dev/null 2>&1
git commit -q -m "Nachlauf: Sudoku hard und TinyStories-Text

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KQ66rEr7ms41phsikjohHs" >/dev/null 2>&1 && echo "$(date '+%T')  Git-Commit erstellt" >> "$STATUS"
echo "Ende: $(date '+%F %T')" >> "$STATUS"
cat "$STATUS"
