#!/usr/bin/env bash
# Richtet alles für die Sudoku-Experimente ein (macOS, Apple Silicon).
set -euo pipefail
cd "$(dirname "$0")"

TASK="${1:-hard}"   # nur die Checkpoints für diese Schwierigkeit laden
NCM_COMMIT=c70274273f489abc73832b2b98584a8483b6b203

# Python >= 3.10 suchen
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Kein Python >= 3.10 gefunden. Installieren mit: brew install python@3.12"; exit 1
fi
echo "Python: $($PY --version)"

# Upstream-Code (fest auf den getesteten Commit)
if [ ! -d ncm ]; then
  git clone https://github.com/HazyResearch/ncm.git ncm
fi
git -C ncm fetch -q origin "$NCM_COMMIT" 2>/dev/null || true
git -C ncm checkout -q "$NCM_COMMIT"

# Umgebung
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
echo "Installiere Pakete (torch ist ~70 MB, kann dauern) ..."
python -m pip install --progress-bar on torch numpy omegaconf hydra-core "transformers>=4.51" "huggingface_hub>=0.34" regex

# Nur die Sudoku-Checkpoints und -Daten (klein)
hf download hazyresearch/ncm-checkpoints --include "sudoku/${TASK}/*" --local-dir artifacts
hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/*" --local-dir data

python -c "import torch; print('torch', torch.__version__, '| MPS verfügbar:', torch.backends.mps.is_available())"

# Eigenes Git-Repo für die Experimente (Upstream-Code, Checkpoints und venv sind ignoriert)
if [ ! -d .git ]; then
  git init -q
  git add .gitignore PLAN.md README.md setup.sh run.sh experiments/
  git commit -q -m "NCM-Feedback-Experimente: Plan, Setup und Sudoku-Skript

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KQ66rEr7ms41phsikjohHs"
  echo "Git-Repo angelegt."
fi
echo "Setup fertig."
