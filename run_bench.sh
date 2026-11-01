#!/usr/bin/env bash
# Benchmarks auf dem Mac, in zwei Nächten:
#   ./run_bench.sh setup   Pakete + Checkpoints/Testdaten herunterladen (einmalig, ca. 2-3 GB)
#   ./run_bench.sh a       Nacht A: LM1B (NCM + Nachbessern), TinyStories (NCM + Mischformen 1-3), Bewertung
#   ./run_bench.sh b       Nacht B: XSum (1000 Artikel) und WMT14 (alle 3003 Sätze), Bewertung
#   ./run_bench.sh m       Architektur-Vergleich A (Baseline), B, C, C-par, D + GPT-2 ohne Plan (TinyStories).
#                          Mac-Version mit verkürztem Training (STEPS, Standard 1000); voll: cluster/lrz_merge.sbatch
# Jede Aufgabe startet mit einem Schnelltest; schlägt der fehl, wird nur diese Aufgabe übersprungen.
cd "$(dirname "$0")"
if [ -z "${NCM_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export NCM_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi
source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
# 0.7 begrenzte den Grafikspeicher auf 3,7 GB -> Speicherfehler beim Bewerten und Trainieren
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8
mkdir -p results/bench
PART="${1:-a}"
LOG="results/bench/run-$PART.log"

smoke() {  # smoke <skript> <args...>  -> Rückgabe 1 bei Fehler (nur diese Aufgabe wird übersprungen)
  echo "-- Schnelltest $1 ${*:2} ($(date '+%T'))"
  python "experiments/$1" --smoke "${@:2}" > results/bench/smoke.log 2>&1
  if ! grep -q "SCHNELLTEST OK" results/bench/smoke.log; then
    echo "SCHNELLTEST FEHLGESCHLAGEN: $1 ${*:2}  -> diese Aufgabe wird übersprungen"; tail -30 results/bench/smoke.log; return 1
  fi
  grep "Hochrechnung" results/bench/smoke.log || true
}
step() { echo "== $* ($(date '+%T'))"; python "$@" || { echo "FEHLER bei $*  -> Rest dieser Aufgabe übersprungen"; return 1; }; }
realdir() { python -c "import os,sys; print(os.path.realpath(sys.argv[1]))" "$1"; }

case "$PART" in
setup)
  pip install -q rouge_score sacrebleu datasets "huggingface_hub[cli]"
  mkdir -p artifacts data
  # direkt in die echten Ordner laden (Verknüpfungen auf die externe Platte aufgelöst)
  for spec in "tinystories:tinystories/*" "lm1b:lm1b/tokenizer.pt lm1b/small/*" "xsum:xsum/*" "wmt14:wmt14/*"; do
    d="${spec%%:*}"; pats="${spec#*:}"
    mkdir -p "artifacts/$d"
    dest="$(dirname "$(realdir "artifacts/$d")")"
    # shellcheck disable=SC2086
    inc=(); set -f; for p in $pats; do inc+=(--include "$p"); done; set +f
    hf download hazyresearch/ncm-checkpoints "${inc[@]}" --local-dir "$dest"
  done
  hf download hazyresearch/ncm-datasets --repo-type dataset --include "xsum/*" --include "wmt14/*" --local-dir "$(realdir data)"
  # Modelle für Mischformen und Bewertung vorab laden (sonst lädt die Nacht sie)
  python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
for m in ["gpt2", "gpt2-large", "HuggingFaceTB/SmolLM2-360M", "roneneldan/TinyStories-33M",
          "EleutherAI/gpt-neo-125M", "Qwen/Qwen2.5-0.5B-Instruct", "bert-base-uncased", "Qwen/Qwen3-0.6B"]:
    AutoTokenizer.from_pretrained(m)
    if m not in ("bert-base-uncased", "Qwen/Qwen3-0.6B", "EleutherAI/gpt-neo-125M"):
        AutoModelForCausalLM.from_pretrained(m)
    print("ok", m)
PY
  echo "Prüfe Dateien:"
  ok=1
  for f in artifacts/tinystories/generator.pt artifacts/lm1b/tokenizer.pt artifacts/lm1b/small/generator.pt \
           artifacts/xsum/generator.pt artifacts/wmt14/generator.pt data/xsum/passages.pickle; do
    if [ -s "$f" ]; then echo "  ok     $f"; else echo "  FEHLT  $f"; ok=0; fi
  done
  ls data/wmt14 2>/dev/null | sed 's/^/  wmt14: /'
  [ "$ok" = 1 ] && echo "Setup fertig." || echo "Setup unvollständig (siehe FEHLT)."
  exit 0 ;;
a)
  {
  ( smoke bench_ncm.py --task lm1b-small && smoke bench_score.py --task lm1b-small \
    && step experiments/bench_ncm.py --task lm1b-small && step experiments/bench_score.py --task lm1b-small )
  ( smoke bench_ncm.py --task tinystories && smoke bench_hybrid.py --task tinystories && smoke bench_score.py --task tinystories \
    && step experiments/bench_ncm.py --task tinystories && { step experiments/bench_hybrid.py --task tinystories; \
    step experiments/bench_score.py --task tinystories; } )
  } 2>&1 | tee "$LOG" ;;
b)
  {
  # LM1B nachholen, falls Teil a es übersprungen hat (Speicherfehler in der Bewertung, behoben)
  if [ ! -f results/bench/lm1b-small/scores.json ]; then
  ( smoke bench_score.py --task lm1b-small || true
    [ -f results/bench/lm1b-small/ncm.json ] || step experiments/bench_ncm.py --task lm1b-small
    step experiments/bench_score.py --task lm1b-small )
  fi
  ( smoke bench_ncm.py --task xsum --n_full 1000 && smoke bench_score.py --task xsum \
    && step experiments/bench_ncm.py --task xsum --n 1000 && step experiments/bench_score.py --task xsum )
  ( smoke bench_ncm.py --task wmt14 --n_full 3003 && smoke bench_score.py --task wmt14 \
    && step experiments/bench_ncm.py --task wmt14 && step experiments/bench_score.py --task wmt14 )
  } 2>&1 | tee "$LOG" ;;
m)
  ST="${STEPS:-1000}"
  TR="--steps $ST --bs 16 --micro_bs 2"
  {
  [ -f results/bench/tinystories/ncm.json ] || step experiments/bench_ncm.py --task tinystories
  # A (Baseline) und GPT-2 ohne Plan
  ( smoke hybrid_stage4.py train --k 3 && smoke hybrid_stage4.py gen --k 3 \
    && step experiments/hybrid_stage4.py train --k 0 $TR && step experiments/hybrid_stage4.py gen --k 0 \
    && step experiments/hybrid_stage4.py train --k 3 $TR && step experiments/hybrid_stage4.py gen --k 3 )
  # B und D
  ( smoke merge_arch.py train --arch B --k 3 && smoke merge_arch.py gen --arch B --k 3 && smoke merge_arch.py gen --arch D --k 3 \
    && step experiments/merge_arch.py train --arch B --k 3 $TR && step experiments/merge_arch.py gen --arch B --k 3 \
    && step experiments/merge_arch.py gen --arch D --k 3 )
  # C
  ( smoke merge_arch.py train --arch C --k 3 && smoke merge_arch.py gen --arch C --k 3 \
    && step experiments/merge_arch.py train --arch C --k 3 $TR && step experiments/merge_arch.py gen --arch C --k 3 \
    && step experiments/merge_arch.py gen --arch C --k 3 --max_blocks 6 --tag long --n_samples 64 --seeds 42 )
  # C-par
  ( smoke merge_arch.py train --arch C --k 12 && smoke merge_arch.py gen --arch C --k 12 --par \
    && step experiments/merge_arch.py train --arch C --k 12 --steps $ST --bs 16 --micro_bs 1 && step experiments/merge_arch.py gen --arch C --k 12 --par )
  step experiments/bench_score.py --task tinystories
  } 2>&1 | tee "$LOG" ;;
nach)
  # Nachholen, was in der ersten Nacht fehlte (vorhandene Ergebnisse werden nicht neu gerechnet)
  {
  ( [ -f results/bench/lm1b-small/ncm.json ] || step experiments/bench_ncm.py --task lm1b-small ) \
    && step experiments/bench_score.py --task lm1b-small
  [ -f results/bench/tinystories/hybrid.json ] && grep -q s1_rewrite results/bench/tinystories/hybrid.json \
    || step experiments/bench_hybrid.py --task tinystories
  step experiments/bench_score.py --task tinystories
  ( smoke bench_ncm.py --task wmt14 --n_full 3003 && step experiments/bench_ncm.py --task wmt14 \
    && step experiments/bench_score.py --task wmt14 )
  # XSum neu: die erste Nacht lieferte wegen eines Umwandlungsfehlers auf dem Mac leere Texte
  ( step experiments/bench_ncm.py --task xsum --n 1000 && step experiments/bench_score.py --task xsum )
  } 2>&1 | tee "$LOG" ;;
nach2)
  # LM1B neu bewerten (Textprotokoll war leer), TinyStories bewerten
  {
  step experiments/bench_score.py --task lm1b-small
  step experiments/bench_score.py --task tinystories
  } 2>&1 | tee "$LOG" ;;
m2)
  # Teil m ohne Schnelltests nachholen: eine Geschichte gleichzeitig, C-par ist schon fertig
  ST="${STEPS:-1000}"
  TR="--steps $ST --bs 16 --micro_bs 1"
  {
  ( step experiments/hybrid_stage4.py train --k 0 $TR && step experiments/hybrid_stage4.py gen --k 0 )
  ( step experiments/hybrid_stage4.py train --k 3 $TR && step experiments/hybrid_stage4.py gen --k 3 )
  ( step experiments/merge_arch.py train --arch B --k 3 $TR && step experiments/merge_arch.py gen --arch B --k 3 \
    && step experiments/merge_arch.py gen --arch D --k 3 )
  ( step experiments/merge_arch.py train --arch C --k 3 $TR && step experiments/merge_arch.py gen --arch C --k 3 \
    && step experiments/merge_arch.py gen --arch C --k 3 --max_blocks 6 --tag long --n_samples 64 --seeds 42 )
  step experiments/bench_score.py --task tinystories
  step experiments/bench_score.py --task lm1b-small
  } 2>&1 | tee "$LOG" ;;
dk)
  # Kontrollen für D: zufällig ausgelöste Neuversuche (Auswahl nach Plan) und ganz zufällig
  {
  step experiments/merge_arch.py gen --arch D --k 3 --d_mode randtrig
  step experiments/merge_arch.py gen --arch D --k 3 --d_mode rand
  step experiments/bench_score.py --task tinystories
  } 2>&1 | tee "$LOG" ;;
*) echo "Aufruf: ./run_bench.sh setup|a|b|m|m2|nach|nach2|dk"; exit 1 ;;
esac

python experiments/bench_report.py > /dev/null && echo "Übersicht: results/bench/SUMMARY.md"
mkdir -p ~/Documents/ncm-nachtlauf/bench
rsync -a --exclude "*-smoke" --include "*/" --include "*.md" --include "*.log" --include "scores.json" \
  --include "pairs*.json*" --exclude "*" results/bench/ ~/Documents/ncm-nachtlauf/bench/ && echo "Ergebnisse nach Documents/ncm-nachtlauf/bench kopiert"
git add experiments/bench_*.py experiments/hybrid_stage4.py experiments/merge_arch.py run_bench.sh cluster/*.sbatch BENCH_README.md \
  results/bench/*/scores.json results/bench/*/table.md results/bench/SUMMARY.md >/dev/null 2>&1
git commit -q -m "Benchmarks Teil $PART: NCM, Nachbessern, Mischformen

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KQ66rEr7ms41phsikjohHs" && echo "Git-Commit erstellt (git push nicht vergessen)"
echo "Ende $(date '+%T')"
