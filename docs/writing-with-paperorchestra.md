# Paper schreiben mit PaperOrchestra

Wie aus diesem Repo mit [PaperOrchestra](https://github.com/google-research/paper-orchestra)
(Google Cloud AI Research) ein Paper-Entwurf wird. PaperOrchestra ist ein Multi-Agent-Framework,
das aus einer Ideen- und einer Experiment-Log-Datei ein LaTeX-Manuskript erzeugt, inklusive
Literatur-Review und, wenn gewünscht, selbst generierten Plots.

Lokaler Klon liegt unter `~/paper-orchestra`.

## Was PaperOrchestra als Eingabe braucht

Einen `raw_materials/`-Ordner mit:

- `idea_sparse.md` — Kernidee, Fragestellung, Hypothese.
- `experimental_log.md` — Setup, Experimente, Zahlen, Beobachtungen.
- `figures/` (optional) — fertige Plots als `.pdf`/`.png`. Ohne figures mit `--use_plotting true`
  laufen, dann baut der Plotting-Agent die Figuren selbst.

Dazu ein LaTeX-Template aus `~/paper-orchestra/templates/<conf>` (z. B. `iclr2025`, `cvpr2025`).

## Mapping: dieses Repo -> raw_materials

- `idea_sparse.md`: aus `README.md` + `docs/PLAN.md` ziehen (NCM, Nachbessern, Mischformen,
  die eigentliche Fragestellung).
- `experimental_log.md`: aus `docs/FINDINGS.md`, `docs/RESULTS.md` und `results/critical-summary.md`
  zusammenstellen, plus die relevanten Zahlen aus `results/` (bench/, critical/, neurodyn/, judge/).
  Pro Experiment kurz Setup + Ergebnis: `bench_ncm`, `bench_hybrid`, `critical_periods`,
  `neurodyn`, `train_selfcond`, `train_value`.
- `figures/`: entweder die vorhandenen Plots aus `results/` kopieren, oder `--use_plotting true`
  setzen und die Report-Skripte (`bench_report.py`, `critical_report.py`, `neurodyn_report.py`)
  als Datenquelle nehmen.

Die raw_materials am besten unter `paper/raw_materials/` in diesem Repo pflegen, dann sind Idee
und Log mitversioniert.

## Setup (einmal)

```bash
cd ~/paper-orchestra
conda create -n paper_orchestra python=3.11 -y
conda activate paper_orchestra
pip install -r requirements.txt
```

`.env` im paper-orchestra-Root anlegen, mit mindestens einem Modell-Key:

```env
GEMINI_API_KEY=...            # alternativ VERTEX_AI_PROJECT + VERTEX_AI_LOCATION, oder OPENAI_API_KEY
SEMANTIC_SCHOLAR_API_KEY=...  # fuer die Literatur-Review
```

## Lauf

```bash
cd ~/paper-orchestra
python paper_writing_cli.py \
  --raw_materials_dir ~/ncm-experimente/paper/raw_materials \
  --latex_template_dir templates/iclr2025 \
  --output_dir ~/ncm-experimente/paper/out \
  --use_plotting true
```

Default-Modell ist `gemini-3.1-pro-preview` (ueber `--writer_model_name` / `--reflection_model_name`
umstellbar). `--research_cutoff 2026-11` begrenzt die Literatur auf einen Stand.

## Zu beachten

- Ohne gueltigen Modell-API-Key laeuft nichts.
- Der Dateiname der Idee/des Logs ist per `--idea_filename` / `--experimental_log_filename`
  aenderbar, falls du nicht die Defaults nutzt.
- PaperOrchestra ist kein offiziell von Google supportetes Produkt (siehe Disclaimer in deren README).
