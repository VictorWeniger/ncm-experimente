# ncm-experimente

Inferenz-Experimente auf [Next-Concept Modeling](https://github.com/HazyResearch/ncm) (Kumbong et al., 2026): Hilft eine Rückkopplung von feinen zu groben Ebenen, ähnlich wie beim Predictive Coding? Details zu Fragestellung, Varianten und Erwartungen stehen in [PLAN.md](PLAN.md), die Ergebnisse in [RESULTS.md](RESULTS.md).

## Benutzung

```bash
./setup.sh          # klont den NCM-Code, legt .venv an, lädt die Sudoku-Checkpoints
./run.sh hard       # alle Varianten auf 2000 Bretter (hard)
./run.sh hard --n 200 --variants baseline,loop_k3   # schneller Test
```

## Aufbau

- `experiments/feedback_sudoku.py`: alle Varianten (baseline, clamp, loop_k*, sample8) plus die Analyse pro Ebene
- `ncm/`: Upstream-Code, wird von `setup.sh` auf einen festen Commit geklont (nicht im Repo)
- `artifacts/`, `data/`: Checkpoints und Testbretter von Hugging Face (nicht im Repo)
- `results/`: JSON und Logs pro Lauf

Der Upstream-Code bleibt unverändert. Das Skript lädt ihn als Bibliothek und tauscht zur Laufzeit nur das Sampling pro Ebene aus.
