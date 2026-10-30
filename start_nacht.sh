#!/usr/bin/env bash
# Startet die Nacht: Exp. 8 -> Setup -> Benchmarks a -> b -> Architekturen m, Sprachmodelle auf der externen Platte.
# Aufruf im Projektordner: ./start_nacht.sh        (alles)
#                           ./start_nacht.sh bench  (ohne Exp. 8)
cd "$(dirname "$0")"

# 1. Externe Platte (fest eingestellt)
DISK="/Volumes/Sonstige Backups"
[ -d "$DISK" ] || { echo "Platte $DISK nicht gefunden. Ist sie angeschlossen? (ls /Volumes)"; exit 1; }
BASE="$DISK/ncm"

# 2. Beschreibbar? Genug Platz?
mkdir -p "$BASE/hf-cache" 2>/dev/null && touch "$BASE/hf-cache/.test" 2>/dev/null \
  || { echo "Kann auf $DISK nicht schreiben (NTFS-formatiert oder schreibgeschützt?)"; exit 1; }
rm -f "$BASE/hf-cache/.test"
FREE_EXT=$(df -g "$DISK" | tail -1 | awk '{print $4}')
FREE_INT=$(df -g ~ | tail -1 | awk '{print $4}')
echo "Externe Platte: $DISK  (frei: ${FREE_EXT} GB)"
echo "Interne Platte frei: ${FREE_INT} GB"
[ "$FREE_EXT" -ge 12 ] || { echo "Zu wenig Platz auf der externen Platte (brauche 12 GB)"; exit 1; }
[ "$FREE_INT" -ge 6 ] || { echo "Intern zu wenig Platz (brauche 6 GB frei)"; exit 1; }

# Neue NCM-Checkpoints und Testdaten ebenfalls auf die Platte (per Verknüpfung, Skripte merken nichts davon)
mkdir -p artifacts "$BASE/data"
for d in lm1b xsum wmt14 owt stage4 merge; do
  if [ ! -e "artifacts/$d" ]; then mkdir -p "$BASE/artifacts/$d" && ln -s "$BASE/artifacts/$d" "artifacts/$d"; fi
done
[ -e data ] || ln -s "$BASE/data" data

# 3. Neueste Skripte aus Documents holen
cp ~/Documents/ncm-bench/experiments/*.py experiments/ 2>/dev/null
cp ~/Documents/ncm-bench/run_bench.sh . 2>/dev/null
[ -f ~/Documents/ncm-exp8/run_search.sh ] && cp ~/Documents/ncm-exp8/run_search.sh .
chmod +x run_search.sh run_bench.sh
mkdir -p results

# 4. Los
export HF_HOME="$BASE/hf-cache"
echo "Start $(date '+%H:%M'). Protokoll: results/alles.log. Fenster offen lassen, Mac ans Netzteil."
if [ "${1:-}" = "bench" ]; then S8=":"; else S8="./run_search.sh"; fi
export S8
caffeinate -dimsu bash -c 'export NCM_CAFFEINATED=1; $S8; ./run_bench.sh setup && { ./run_bench.sh a; ./run_bench.sh b; ./run_bench.sh m; }' 2>&1 | tee results/alles.log
