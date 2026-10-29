#!/usr/bin/env bash
# Experiment 8 auf dem Mac: Suche auf den groben Ebenen. Erst Schnelltest, dann Sudoku hard.
# Beispiel: ./run_search.sh            (hard, 500 Rätsel, N = 8)
#           ./run_search.sh medium     (anderer Schwierigkeitsgrad)
cd "$(dirname "$0")"
if [ -z "${NCM_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export NCM_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi
source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
TASK="${1:-hard}"
mkdir -p results

echo "== Schnelltest $(date '+%T') =="
python experiments/coarse_search.py --smoke --task "$TASK" 2>&1 | tee results/search-smoke.log
grep -q "SCHNELLTEST OK" results/search-smoke.log || { echo "SCHNELLTEST FEHLGESCHLAGEN"; exit 1; }
python experiments/train_value.py "results/sudoku-$TASK-search-smoke-dump.pt" > results/search-smoke-value.log 2>&1 || { echo "SCHNELLTEST Prüfer FEHLGESCHLAGEN, siehe results/search-smoke-value.log"; exit 1; }

echo "== Hauptlauf $(date '+%T') =="
# Lauf 1: T = 0,7 wie in Exp. 1/2, alle Abzweigungsorte
python experiments/coarse_search.py --task "$TASK" --n 500 --n_cal 500 --N 8 --temp 0.7 --dump --tag search-t07 2>&1 | tee "results/sudoku-$TASK-search-t07.log"
python experiments/train_value.py "results/sudoku-$TASK-search-t07-dump.pt" 2>&1 | tee "results/sudoku-$TASK-value-t07.log"
# Lauf 2: T = 1,5, damit oben wirklich verschiedene Grundideen entstehen
python experiments/coarse_search.py --task "$TASK" --n 500 --n_cal 500 --N 8 --temp 1.5 --where top1,top3,top6,all,low6 --dump --tag search-t15 2>&1 | tee "results/sudoku-$TASK-search-t15.log"
python experiments/train_value.py "results/sudoku-$TASK-search-t15-dump.pt" 2>&1 | tee "results/sudoku-$TASK-value-t15.log"
cp results/*search* results/*value* ~/Documents/ncm-nachtlauf/ 2>/dev/null && echo "Ergebnisse nach Documents/ncm-nachtlauf kopiert"

git add results/*search* results/*value* experiments/coarse_search.py experiments/train_value.py run_search.sh >/dev/null 2>&1
git commit -q -m "Experiment 8: Suche auf den groben Ebenen ($TASK)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KQ66rEr7ms41phsikjohHs" && echo "Git-Commit erstellt (git push nicht vergessen)"
echo "Ende $(date '+%T')"
