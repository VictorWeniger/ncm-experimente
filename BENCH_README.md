# Benchmarks: NCM, NCM + Nachbessern und Mischformen

Füllt die offenen Zellen der Benchmark-Tabellen im Labor-Artefakt.

## Installieren (einmalig, im Projektordner `~/Projekte/ncm-experimente`)

```bash
cp ~/Documents/ncm-bench/experiments/*.py experiments/
cp ~/Documents/ncm-bench/run_bench.sh ~/Documents/ncm-bench/BENCH_README.md .
mkdir -p cluster && cp ~/Documents/ncm-bench/cluster/lrz_bench.sbatch cluster/
./run_bench.sh setup     # Pakete, Checkpoints (LM1B small, XSum, WMT14), Testdaten, Bewertungsmodelle
```

Danach `./run_bench.sh a` in der ersten Nacht und `./run_bench.sh b` in der zweiten. Ergebnisse landen in
`results/bench/<aufgabe>/table.md`, die Übersicht in `results/bench/SUMMARY.md`. Beides wird zusätzlich
nach `~/Documents/ncm-nachtlauf/bench` kopiert.

## Was gerechnet wird

| Skript | Aufgaben | Varianten |
|---|---|---|
| `bench_ncm.py` | tinystories, lm1b-small, owt, xsum, wmt14 | NCM allein (Paper-Einstellung), Nachbessern mit `gen_llR`, zufälliges Nachbessern als Kontrolle, Best-of-N |
| `bench_hybrid.py` | tinystories (lm1b-small) | reines kleines LLM, Stufe 1 (LLM schreibt markierte Texte neu, plus 2 Kontrollen), Stufe 1b (ersetzen), Stufe 2 (nur der schwächste Satz), Stufe 3 (LLM liest den Entwurf und übernimmt bis zur ersten unplausiblen Stelle), LLM wählt aus N NCM-Entwürfen |
| `hybrid_stage4.py` | tinystories | Stufe 4: GPT-2 small mit den groben NCM-Ebenen als Plan; `--k 0` = gleich trainiertes GPT-2 ohne Plan (faires reines LLM). Nur Cluster. |
| `bench_score.py` | alle | Paper-Metriken + unabhängiges Bewertungsmodell + gepaarter Bootstrap + blinde Paare |
| `bench_report.py` | alle | Übersicht über alle Aufgaben |

## Bewusste Entscheidungen

- **Baseline exakt wie das Paper.** Gleiche Sampling-Einstellung, gleiche Frequenz- und Nachbarschaftsstrafen beim Dekodieren, gleiche Metrik. Die Baseline-Zeile muss also die Paper-Zahl treffen (innerhalb der Streuung über Seeds). Tut sie das nicht, stimmt etwas am Aufbau, und die anderen Zeilen sind nichts wert.
- **XSum und WMT14 dekodiert das Paper gierig.** Neu sampeln ergäbe dort denselben Text. Nachbessern heißt hier: gierig plus N−1 Samples, Auswahl per `gen_llR`. Dazu kommen zwei Kontrollen: zufällige Auswahl (bringt nur das Sampling etwas?) und ein Orakel mit der Referenz (Obergrenze).
- **Zweites Bewertungsmodell aus einer anderen Familie (SmolLM2).** GPT-2 large bevorzugt Texte von GPT-artigen Modellen. Eine Mischform mit GPT-2 könnte sonst nur deshalb gewinnen, weil sie dem Prüfer ähnlich ist. Gewertet wird ein Effekt nur, wenn er unter beiden Prüfern in dieselbe Richtung geht.
- **Entropie immer mit angeben.** Eine niedrigere Perplexity bei deutlich niedrigerer Entropie heißt nur „repetitiver“, nicht „besser“.
- **Kosten in drei Maßen.** GFLOPs pro Sample (Näherung: 2 × Parameter × verarbeitete Tokens), sequenzielle Schritte (NCM-Ebenen + LLM-Tokens; das ist der Parallelitätsvorteil von NCM) und Sekunden auf deinem Rechner.
- **Blinde Paare** (`pairs.jsonl`, Auflösung in `pairs_key.json`): für ein paarweises Urteil durch dich oder ein großes LLM, ohne zu wissen, welcher Text von wem ist.

## Grenzen

- LM1B mit FLM-Protokoll braucht die Roh-Tokens. Für Mischformen gibt es deshalb nur das Text-Protokoll, das aber für alle Varianten identisch ist.
- Stufe 2 schreibt den schwächsten Satz mit einem normalen LLM neu, das nur den Text davor sieht, nicht den danach. Echtes Einfügen mit Kontext auf beiden Seiten wäre ein Ausbau.
- Stufe 4 erzeugt der Einfachheit halber alle NCM-Ebenen, berechnet werden aber nur die Kosten der k groben.
- OWT (1024 Tokens) und Stufe 4 sind auf einem 8-GB-M1 nicht sinnvoll: `cluster/lrz_bench.sbatch`.
