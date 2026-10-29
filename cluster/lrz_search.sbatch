#!/bin/bash
# Experiment 8 auf den LRZ AI Systems: Suche auf den groben Ebenen, groß (N = 32, 1000 Rätsel, hard + medium).
# Start im Repo-Ordner:  mkdir -p logs && sbatch cluster/lrz_search.sbatch
#SBATCH -J ncm-search
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 05:00:00
#SBATCH -o logs/ncm-search-%j.out

srun --container-image=nvcr.io#nvidia/pytorch:24.10-py3 \
     --container-mounts="$PWD":/workspace \
     --container-workdir=/workspace \
     bash -c '
set -e
export PYTHONUSERBASE=/workspace/.pylibs PATH=/workspace/.pylibs/bin:$PATH PYTHONUNBUFFERED=1
pip install --user -q omegaconf hydra-core "transformers>=4.51" "huggingface_hub>=0.34" regex
nvidia-smi --query-gpu=name,memory.total --format=csv
python experiments/coarse_search.py --smoke --task hard
for T in hard medium; do
  python experiments/coarse_search.py --task $T --n 1000 --n_cal 1000 --N 32 --B 1,2,4,8,16 \
         --where top1,top2,top3,top4,top6,top9,all,low3,low6,low9 --bs 32 --dump --tag search-n32 \
         2>&1 | tee results/sudoku-$T-search-n32.log
  python experiments/train_value.py results/sudoku-$T-search-n32-dump.pt 2>&1 | tee results/sudoku-$T-value-n32.log
done
'
