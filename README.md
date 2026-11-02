# NCM-Feedback-Experimente

Inferenz-Experimente mit [Next-Concept Modeling](https://github.com/HazyResearch/ncm) (Kumbong et al., 2026, [arXiv:2610.14162](https://arxiv.org/abs/2610.14162)). NCM erzeugt Text bzw. Sudoku-Lösungen grob → fein über eine Hierarchie diskreter Codes, eine Ebene pro Durchlauf.

**Leitfrage:** Wird NCM besser, wenn es seine eigene Ausgabe prüft und gezielt nachbessert (wie Predictive Coding im Gehirn)? Und wie schneidet es gegen und zusammen mit normalen Sprachmodellen ab?

Alle Läufe auf einem MacBook Air (M1, 8 GB) mit den veröffentlichten NCM-Checkpoints, ohne NCM nachzutrainieren.

## Ergebnisse in einem Satz pro Thema

1. **NCM erkennt seine eigenen Fehler** (`gen_llR`: Ausgabe neu kodieren, per Teacher Forcing bewerten). Sudoku hard 84,3 → 88,3 % durch gezieltes Nachrechnen, 92,0 % mit Suche über 8 Kandidaten. Text: TinyStories −11 %, LM1B −7 % Perplexity; die Auswahl durch das Signal macht mehr als die Hälfte des Effekts aus.
2. **Das Signal misst „typisch“, nicht „treu“.** Bei XSum/WMT14 hilft es nicht. Die pmi-Korrektur (mit minus ohne Vorlage) hilft bei der Übersetzung (66 statt 57 % richtige Paarentscheidungen, +0,9 BLEU in Kombination), nicht bei XSum.
3. **Grob → fein schlägt Token-für-Token nur bei globalen Bedingungen.** Sudoku 88 % gegen ~1 %, aber bei freiem Text ist ein gleich trainiertes kleines autoregressives Modell klar besser (Perplexity 9,8 gegen 16,2) und braucht ein Sechstel der Rechenarbeit.
4. **Mischformen:** Das Signal ist eine gute Weiche für ein LLM (markierte neu schreiben 13,0 gegen zufällige 14,1). Ein gemeinsames Plan-und-Schreib-Netz bringt nach kurzem Training nichts; ein scheinbarer Gewinn (Variante D) wurde durch eine Kontrolle als Artefakt entlarvt.
5. **Biologie:** Zufällige Spikes verbessern Längen-Generalisierung; der Schaden „kritischer Phasen“ erklärt sich vollständig durch die sinkende Lernrate.

Ausführlich mit allen Zahlen: [docs/FINDINGS.md](docs/FINDINGS.md). Das interaktive Labor (Erklärung des Papers, Mathe, alle Experimente): [docs/labor.html](docs/labor.html) im Browser öffnen.

## Experimente

| Nr. | Frage | Code | Start | Ergebnisse | Stand |
|---|---|---|---|---|---|
| 1 | Feedback-Schleife mit externem Regel-Checker (Sudoku) | `experiments/feedback_sudoku.py` | `./run.sh hard` | `results/sudoku-hard-n500.*`, `-kontrolle.*` | ✓ |
| 2 | Internes Fehlersignal statt Regel-Checker | `experiments/selfsignal_sudoku.py` | `./run_self.sh hard` | `results/sudoku-*-self*.json` | ✓ |
| 3 | Dasselbe für Text (TinyStories), blindes Urteil | `experiments/textsignal_tinystories.py`, `judge_*.py` | `./run_text.sh`, `./run_judge.sh` | `results/tinystories-*` | ✓ |
| 4 | Nachtraining mit eigenen Fehlern | `experiments/train_selfcond.py` | `cluster/` | – | wartet auf Cluster |
| 5 | Mischformen NCM + LLM, Stufen 1–3 | `experiments/bench_hybrid.py` | `./run_bench.sh a` | `results/bench/tinystories/` | ✓ |
| 5 | Stufe 4: NCM-Plan als Präfix für GPT-2 (= Variante A) | `experiments/hybrid_stage4.py` | `./run_bench.sh m2` | `results/bench/tinystories/` | ✓ (kurzes Training) |
| 5b | Planer und Schreiber in einem Netz (B, C, C-par, D + Kontrollen) | `experiments/merge_arch.py` | `./run_bench.sh m2`, `dk` | `results/bench/tinystories/` | ✓ (kurzes Training) |
| 6 | Neuronen mit Zeitkonstanten und Spikes | `experiments/neurodyn.py` | `./run_neurodyn.sh` | `results/neurodyn*` | ✓ |
| 7a | Kritische Phasen (CIFAR-10) | `experiments/critical_periods.py` | `./run_night.sh` | `results/critical*` | ✓ |
| 8 | Suche auf groben Ebenen („erst Grundideen, dann ausformulieren“) | `experiments/coarse_search.py`, `train_value.py` | `./run_search.sh` | `results/sudoku-hard-search-*`, `-value-*` | ✓ |
| 9 | Auswahlsignal für bedingte Aufgaben (pmi, MBR, Rückübersetzung) | `experiments/cond_select.py` | `./run_bench.sh sel` | `results/bench/{xsum,wmt14}/select.*` | ✓ |
| B | Paper-Benchmarks: TinyStories, LM1B, XSum, WMT14 (+ Nachbessern) | `experiments/bench_ncm.py`, `bench_score.py` | `./run_bench.sh a`, `b`, `nach` | `results/bench/*/table.md`, `SUMMARY.md` | ✓ (OWT braucht Cluster) |

Welche Datei in `results/` zu welchem Experiment gehört: [results/README.md](results/README.md). Fragestellungen und vorab notierte Erwartungen: [docs/PLAN.md](docs/PLAN.md). Detailprotokoll Exp. 1–3: [docs/RESULTS.md](docs/RESULTS.md).

## Aufbau

| Pfad | Inhalt |
|---|---|
| `experiments/` | alle Experiment-Skripte (gemeinsame Bausteine: `ncm_scoring.py`, `bench_common.py`) |
| `run_*.sh`, `start_nacht.sh` | Startskripte für den Mac (mit Schnelltest vor jedem langen Lauf) |
| `cluster/` | Slurm-Jobs für die LRZ AI Systems (`lrz_bench.sbatch`, `lrz_merge.sbatch`, `lrz_search.sbatch`) |
| `results/` | Ergebnisse (JSON, Tabellen, Logs); große Dateien sind ausgeschlossen, siehe `.gitignore` |
| `docs/` | Ergebnisbericht, Plan, Labor als HTML |
| `BENCH_README.md` | Anleitung und Designentscheidungen der Benchmarks |

## Reproduzieren

```bash
./setup.sh                 # klont NCM (fester Commit), legt .venv an, lädt Sudoku-Checkpoints
./run_bench.sh setup       # Checkpoints/Testdaten für TinyStories, LM1B, XSum, WMT14 + Bewertungsmodelle
./run_bench.sh a           # dann b, nach, m2, dk, sel – siehe BENCH_README.md
```

Auf einem 8-GB-Mac liegen Sprachmodelle und Checkpoints auf einer externen Platte (`start_nacht.sh` richtet Verknüpfungen ein, `HF_HOME` zeigt dorthin). NCM-Code, Checkpoints und Daten sind nicht in diesem Repo, die Skripte laden sie von GitHub bzw. Hugging Face.

## Hinweis

Privates Forschungsprojekt, nicht von den NCM-Autoren. Kleine Stichproben, ein Modell pro Aufgabe, Trainings in Exp. 5/5b stark verkürzt (1000 statt 20.000 Schritte). Bewertet wird immer mit zwei Prüfmodellen aus verschiedenen Familien (GPT-2 large wie im Paper, SmolLM2-360M unabhängig), weil ein Prüfer verwandte Schreiber bevorzugt.
