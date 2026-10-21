#!/usr/bin/env bash
# Startet die Experimente. Beispiel: ./run.sh hard
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
TASK="${1:-hard}"
shift || true
mkdir -p results
python experiments/feedback_sudoku.py --task "$TASK" "$@" 2>&1 | tee "results/sudoku-$TASK.log"
