#!/usr/bin/env bash
# Erzeugt Geschichten für das paarweise LLM-Urteil (ohne GPT-2, ca. 30–45 min).
cd "$(dirname "$0")"
set -o pipefail
if [ -z "${NCM_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export NCM_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
mkdir -p results
python experiments/textsignal_tinystories.py --bs 16 --skip_ppl --variants baseline,self_gen_llR,rand_gen_llR --tag judge 2>&1 | tee results/tinystories-judge.log
