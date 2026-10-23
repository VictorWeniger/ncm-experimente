#!/usr/bin/env bash
# Nur der Text-Lauf (TinyStories), speicherschonend für das M1-MacBook. Vorher am besten neu starten.
cd "$(dirname "$0")"
set -o pipefail
if [ -z "${NCM_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export NCM_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
mkdir -p results
if python experiments/textsignal_tinystories.py --bs 16 --ref_bs 2 --tag text 2>&1 | tee results/tinystories-text.log; then
  git add -A results experiments *.sh *.md >/dev/null 2>&1
  git commit -q -m "TinyStories-Text-Experiment

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KQ66rEr7ms41phsikjohHs" >/dev/null 2>&1 && echo "Git-Commit erstellt"
fi
