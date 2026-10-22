#!/usr/bin/env bash
# Startet die Experimente. Beispiel: ./run.sh hard
set -euo pipefail
prev=""
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
TASK="${1:-hard}"
shift || true
TAG=""
for arg in "$@"; do [ "$prev" = "--tag" ] && TAG="-$arg"; prev="$arg"; done
mkdir -p results
python experiments/feedback_sudoku.py --task "$TASK" "$@" 2>&1 | tee "results/sudoku-$TASK$TAG.log"
