#!/usr/bin/env bash
# Mini-Test des Trainingscodes auf dem Mac (wenige Minuten). Trainiert NICHT wirklich,
# sondern prüft nur, dass Daten, Training, Reparatur-Ziele, Speichern und Auswertung laufen.
cd "$(dirname "$0")"
set -eo pipefail
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
mkdir -p results
{
python experiments/sudoku_data.py --n 512 --out data/sudoku/train-smoke --workers 4
python experiments/train_selfcond.py --task hard --train_dir data/sudoku/train-smoke --out runs/smoke \
  --p_self 0.5 --steps 20 --bs 8 --log_every 5 --eval_every 20 --n_val 64
python experiments/feedback_sudoku.py --task hard --ckpt_dir runs/smoke --n 64 --bs 32 --no_levels \
  --variants baseline,loop_k3_t0.7 --tag smoke-train
echo "MINITEST OK"
} 2>&1 | tee results/train-smoke.log
