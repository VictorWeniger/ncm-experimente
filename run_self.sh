#!/usr/bin/env bash
# Experiment: internes Fehlersignal statt Regel-Checker. Beispiel: ./run_self.sh hard
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
TASK="${1:-hard}"
shift || true
mkdir -p results
python experiments/selfsignal_sudoku.py --task "$TASK" "$@" 2>&1 | tee "results/sudoku-$TASK-self2.log"
