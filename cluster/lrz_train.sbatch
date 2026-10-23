#!/bin/bash
# Experiment 4 auf den LRZ AI Systems: Daten erzeugen, Kontrolle + Selbst-Konditionierung trainieren, auswerten.
# Abschicken (auf login.ai.lrz.de, im Repo-Ordner):   sbatch cluster/lrz_train.sbatch
#SBATCH -J ncm-selfcond
#SBATCH -p lrz-hgx-a100-80x4           # A100; Alternativen: lrz-dgx-a100-80x8, lrz-hgx-h100-94x4 (sinfo zeigt, was frei ist)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -o logs/ncm-%j.out
#SBATCH -e logs/ncm-%j.err

set -eo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
IMAGE="${IMAGE:-nvcr.io#nvidia/pytorch:24.10-py3}"   # oder Pfad zu einem lokal gespeicherten .sqsh-Image

srun --container-image="$IMAGE" --container-mounts="$REPO:/workspace" --container-workdir=/workspace \
  bash -lc '
set -eo pipefail
export PYTHONUSERBASE=/workspace/.pylibs PATH=/workspace/.pylibs/bin:$PATH PYTHONUNBUFFERED=1
python -m pip install --user -q omegaconf hydra-core regex
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p results runs

# 1) Trainingsdaten (einmalig, ~1-2 min mit 16 Kernen)
[ -f data/sudoku/train/puzzles.pickle ] || \
  python experiments/sudoku_data.py --n 50000 --clues 30,35,40 --out data/sudoku/train --workers ${SLURM_CPUS_PER_TASK:-16}

# 2) Kontrolle (gleiches Training ohne Selbst-Konditionierung) und Selbst-Konditionierung
for RUN in ctrl:0.0 self:0.5; do
  NAME=${RUN%%:*}; PS=${RUN##*:}
  [ -f runs/$NAME/generator.pt ] || \
    python experiments/train_selfcond.py --task hard --out runs/$NAME --p_self $PS \
      --steps 5000 --bs 256 --bf16 2>&1 | tee results/train-$NAME.log
done

# 3) Auswertung: Original, Kontrolle, Selbst - je mit Schleifen, die grobe Ebenen festhalten (k>0) oder nicht (k0)
VARS=baseline,loop_k3,loop_k6,loop_k0_t0.7,loop_k1_t0.7,loop_k3_t0.7,loop_k6_t0.7,sample2
python experiments/feedback_sudoku.py --task hard --n 1000 --no_levels --variants $VARS --tag exp4-base
for NAME in ctrl self; do
  python experiments/feedback_sudoku.py --task hard --n 1000 --no_levels --variants $VARS \
    --ckpt_dir runs/$NAME --tag exp4-$NAME
done
echo "Experiment 4 fertig"
'
