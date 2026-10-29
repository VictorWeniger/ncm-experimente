#!/bin/bash
# Alle Benchmarks auf einer A100 (LRZ AI Systems). Vorher einmal ./run_bench.sh setup im Projektordner
# und zusätzlich: hf download hazyresearch/ncm-checkpoints --include "owt/*" --local-dir artifacts
#SBATCH -J ncm-bench
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o results/bench/slurm-%j.log
set -e
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
J="gpt2-large,HuggingFaceTB/SmolLM2-1.7B"
for t in lm1b-small owt tinystories; do
  python experiments/bench_ncm.py --task $t
done
python experiments/bench_hybrid.py --task tinystories --rewriter Qwen/Qwen2.5-1.5B-Instruct --bs 64
python experiments/bench_hybrid.py --task lm1b-small --lm gpt2 --rewriter Qwen/Qwen2.5-1.5B-Instruct --bs 64 \
  --variants llm_only,s1_rewrite,s1_rewrite_rand,s1_replace,s3_spec,s3_pick
python experiments/bench_ncm.py --task xsum --N 8
python experiments/bench_ncm.py --task wmt14 --N 8
# Stufe 4: gleiches Training mit und ohne Plan
for k in 0 3; do
  python experiments/hybrid_stage4.py train --k $k --steps 20000 --bs 64 --micro_bs 64
  python experiments/hybrid_stage4.py gen --k $k
done
python experiments/hybrid_stage4.py gen --k 3 --teacher
for t in lm1b-small owt tinystories xsum wmt14; do
  python experiments/bench_score.py --task $t --judges "$J" --ref_bs 32
done
python experiments/bench_report.py
