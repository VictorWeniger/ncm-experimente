#!/bin/bash
# Architektur-Vergleich NCM+LLM auf einer A100: A (Baseline), B, C, C-par, D und GPT-2 ohne Plan,
# alle mit demselben Trainingsbudget (20.000 Schritte x 64 Geschichten). Partition ggf. anpassen.
#SBATCH -J ncm-merge
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o results/bench/slurm-merge-%j.log
set -e
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
TR="--steps 20000 --bs 64 --micro_bs 32"
[ -f results/bench/tinystories/ncm.json ] || python experiments/bench_ncm.py --task tinystories
for k in 0 3; do python experiments/hybrid_stage4.py train --k $k $TR; python experiments/hybrid_stage4.py gen --k $k; done
python experiments/hybrid_stage4.py gen --k 3 --teacher
python experiments/merge_arch.py train --arch B --k 3 $TR
python experiments/merge_arch.py gen --arch B --k 3
python experiments/merge_arch.py gen --arch D --k 3
python experiments/merge_arch.py train --arch C --k 3 $TR
python experiments/merge_arch.py gen --arch C --k 3
python experiments/merge_arch.py gen --arch C --k 3 --max_blocks 6 --tag long --n_samples 256 --seeds 42
python experiments/merge_arch.py train --arch C --k 12 --steps 20000 --bs 64 --micro_bs 16
python experiments/merge_arch.py gen --arch C --k 12 --par
python experiments/bench_score.py --task tinystories --judges "gpt2-large,HuggingFaceTB/SmolLM2-1.7B" --ref_bs 32
python experiments/bench_report.py
