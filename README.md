# NCM-Feedback-Experimente

Inferenz-Experimente mit [Next-Concept Modeling](https://github.com/HazyResearch/ncm) (Kumbong et al., 2026, [arXiv:2610.14162](https://arxiv.org/abs/2610.14162)). NCM erzeugt Text bzw. Sudoku-Lösungen grob → fein über eine Hierarchie diskreter Codes, in einem einzigen Durchlauf ohne Rückkopplung.

**Frage:** Wird NCM besser, wenn es seine eigene Ausgabe prüft und gezielt nachbessert, ähnlich wie beim Predictive Coding im Gehirn?

## Ergebnisse auf einen Blick

| | ohne Nachbessern | mit internem Fehlersignal | mit externem Regel-Checker |
|---|---|---|---|
| Sudoku hard (1000 Bretter) | 84,3 % gelöst | **88,3 %** | 89,4 % |
| Sudoku medium | 96,9 % | **97,8 %** | 98,1 % |
| TinyStories (GPT-2-Perplexity ↓) | 16,15 | **14,36** (Vielfalt unverändert) | – |

1. **NCM erkennt seine eigenen Fehler ohne Training:** Die Ausgabe wird neu kodiert und die Codes werden per Teacher Forcing in einem Durchlauf unter dem Generator bewertet (`gen_llR`). Das ergibt AUROC 0,95–0,96 bei Sudoku und Spearman 0,78 mit der GPT-2-Qualität bei Text.
2. **Gezieltes Nachrechnen** an den erkannten Stellen bringt fast so viel wie ein externer Regel-Checker, bei ähnlichem Rechenaufwand.
3. **Die Erkennung ist der Kern:** Gleich viel Rechenaufwand, zufällig verteilt, bringt deutlich weniger. Ein blindes paarweises Urteil bestätigt die Auswahl des Signals in ~75 % der Fälle.
4. **Rückkopplung in die groben Konzept-Ebenen bringt ohne Training nichts.** Experiment 4 (Nachtraining mit eigenen fehlerhaften Skizzen) ist vorbereitet.

Details, Statistik (McNemar, Bootstrap) und alle Einschränkungen stehen in [RESULTS.md](RESULTS.md). Fragestellungen und vorab notierte Erwartungen stehen in [PLAN.md](PLAN.md).

## Aufbau

| Pfad | Inhalt |
|---|---|
| `experiments/feedback_sudoku.py` | Exp. 1: Feedback-Schleifen mit Regel-Checker (wie viele grobe Ebenen festhalten, mit/ohne Rauschen) |
| `experiments/selfsignal_sudoku.py` | Exp. 2: interne Fehlersignale, Erkennung (AUROC) und Schleifen ohne Regel-Checker |
| `experiments/textsignal_tinystories.py` | Exp. 3: dasselbe für Text, Qualität über GPT-2 large |
| `experiments/judge_*.py` | Exp. 3b: blinde Paare und Auswertung für das paarweise Urteil |
| `experiments/ncm_scoring.py` | Bewertung gegebener Codes per Teacher Forcing in einem Forward Pass |
| `experiments/train_selfcond.py`, `sudoku_data.py` | Exp. 4: Nachtraining mit Selbst-Konditionierung, Sudoku-Datengenerator |
| `experiments/neurodyn.py`, `neurodyn_report.py` | Exp. 6: kleine RNNs mit Membran-Zeitkonstanten (gemeinsam, pro Neuron, inputabhängig) und Bernoulli-Spikes |
| `experiments/critical_periods.py`, `critical_report.py` | Exp. 7a: kritische Phasen (unscharfe Bilder am Trainingsanfang, CIFAR-10) |
| `cluster/` | Slurm-Job und Anleitung für die LRZ AI Systems |
| `results/` | Rohdaten aller Läufe (JSON, Logs, Paare und Urteile) |

## Reproduzieren

Getestet auf einem MacBook Air (M1, 8 GB) mit PyTorch/MPS.

```bash
./setup.sh            # klont NCM (fester Commit), legt .venv an, lädt die Sudoku-Checkpoints
./run.sh hard         # Experiment 1
./run_self.sh hard    # Experiment 2
./run_text.sh         # Experiment 3 (lädt GPT-2 large, ~3 GB)
./run_neurodyn.sh     # Experiment 6 (nur CPU, kein Download, mehrere Stunden)
./run_night.sh        # Experiment 6 (CPU) und 7a (GPU, lädt CIFAR-10) gleichzeitig
```

Der NCM-Code, die Checkpoints und die Daten liegen nicht in diesem Repo, die Skripte laden sie von GitHub bzw. Hugging Face.

## Hinweis

Privates Forschungsprojekt, nicht von den NCM-Autoren. Die Ergebnisse beruhen auf kleinen Stichproben und einem einzigen Modell pro Aufgabe.
