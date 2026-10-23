# Experiment 4 auf den LRZ AI Systems (TUM)

Stand der Recherche: 23.10.2026, Quellen unten. Details wie Partitionsnamen können sich ändern. Vor dem ersten Job auf dem Login-Knoten `sinfo` ausführen.

## 1. Zugang bekommen

Die LRZ AI Systems sind **nicht automatisch für alle Studierenden** freigeschaltet. Man braucht:

1. **eine LRZ-Kennung** (als TUM-Student hast du eine; sie ist an deine TUM-Kennung gekoppelt) und
2. die Mitgliedschaft in einem **„AI-enabled“ LRZ-Projekt**. Das läuft fast immer über einen **Lehrstuhl**: Der „Master User“ des Lehrstuhl-Projekts fügt deine Kennung zur Gruppe `<Projekt-ID>-ai-c` hinzu. Die Gruppe hat kein Mitgliederlimit.

Hat der Lehrstuhl noch kein freigeschaltetes Projekt, stellt der Master User eine Anfrage beim LRZ-Servicedesk. Verlangt wird ein **klarer Forschungsbezug**, alternativ geht es über die Initiative **BayernKI**.

**Praktisch heißt das:** Du brauchst einen Betreuer an einem Lehrstuhl, z.B. für eine Projekt- oder Masterarbeit oder als HiWi. Die Experimente hier sind dafür ein gutes Argument. Fragen kannst du so: *„Ich habe auf eigene Faust Inferenz-Experimente mit Next-Concept Modeling gemacht (Ergebnisse anbei) und würde gern ein kleines Nachtraining machen, ca. 10–20 GPU-Stunden auf einer A100. Hätte der Lehrstuhl ein AI-enabled LRZ-Projekt, in das ich aufgenommen werden könnte?“*

Das LRZ bietet außerdem eine Einführungsschulung an („AI Training Series – Introduction to the LRZ AI Systems“).

## 2. Einloggen und Code hochladen

```bash
ssh <lrz-kennung>@login.ai.lrz.de
```

Vom Mac aus den Projektordner hochladen (ohne die Mac-Python-Umgebung):

```bash
rsync -av --exclude .venv --exclude .git ~/Documents/ncm-experimente <lrz-kennung>@login.ai.lrz.de:~/
```

Die Checkpoints (`artifacts/`) und Testdaten (`data/`) kommen dabei mit. Den Upstream-Code `ncm/` ebenfalls.

## 3. Wie Jobs dort laufen

- Planung über **Slurm**: `sbatch` für Batch-Jobs, `salloc` und `srun` für interaktive Sitzungen.
- Software läuft in **Containern** (Enroot mit dem Slurm-Plugin Pyxis), typischerweise NVIDIAs PyTorch-Container von `nvcr.io`. Den Ordner bindet man mit `--container-mounts` ein.
- GPU-Partitionen (laut LRZ-Doku): `lrz-hgx-h100-94x4` (H100), `lrz-hgx-a100-80x4` und `lrz-dgx-a100-80x8` (A100 80 GB), `lrz-dgx-a100-40x8-mig`, `lrz-dgx-1-v100x8` und `lrz-v100x2` (V100, Standard), dazu P100-Partitionen. **`--gres=gpu:1` ist Pflicht.**

## 4. Experiment 4 starten

Im Repo-Ordner auf dem Login-Knoten:

```bash
mkdir -p logs
sbatch cluster/lrz_train.sbatch
squeue -u $USER            # Status
tail -f logs/ncm-<jobid>.out
```

Das Skript `cluster/lrz_train.sbatch` macht alles in einem Job:
1. installiert drei kleine Python-Pakete in den Projektordner (PyTorch ist im Container),
2. erzeugt 50.000 Sudoku-Trainingsrätsel (30/35/40 Vorgaben, eindeutig lösbar, ohne Überschneidung mit den Testbrettern),
3. trainiert **zwei** Varianten mit je 5000 Schritten: `ctrl` (gleiches Training ohne Selbst-Konditionierung) und `self` (die Hälfte der Beispiele mit eigenen groben Ebenen und Reparatur-Zielen),
4. wertet Original, `ctrl` und `self` mit denselben Feedback-Schleifen aus wie Experiment 1 (1000 Bretter hard).

Geschätzte Laufzeit auf einer A100: grob 1–2 Stunden. Angefordert sind 6 Stunden als Puffer.

Ergebnisse landen in `results/sudoku-hard-exp4-*.json`, die Trainingsverläufe in `results/train-*.log` und `runs/*/log.json`. Zum Auswerten einfach zurück auf den Mac holen:

```bash
rsync -av <lrz-kennung>@login.ai.lrz.de:~/ncm-experimente/results ~/Documents/ncm-experimente/
```

## 5. Falls etwas hakt

- **Kein Internet auf dem Rechenknoten** (`pip install` scheitert): Pakete einmal in einer interaktiven Sitzung installieren und den Container als eigenes Image speichern (LRZ-Doku „Enroot“, Stichwort `enroot export`). Oder im Container ohne `--user`-Install arbeiten und die drei Pakete vorab in `.pylibs` kopieren.
- **Partition voll oder nicht freigeschaltet:** `sinfo` und `sacctmgr show assoc user=$USER` zeigen, was erlaubt ist. In `lrz_train.sbatch` dann `-p` ändern. Auf V100 (`lrz-v100x2`) das Flag `--bf16` entfernen.
- **Zu langsam:** `--steps` halbieren, das reicht für einen ersten Eindruck.

## Quellen

- LRZ AI Systems, Übersicht: https://doku.lrz.de/ai-systems-11484278.html
- Zugang: https://doku.lrz.de/1-access-10746642.html
- Login: https://doku.lrz.de/1-1-ai-systems-login-2498299908.html
- Rechenressourcen und Partitionen: https://doku.lrz.de/1-general-description-and-resources-10746641.html
- Interaktive Jobs mit Containern: https://doku.lrz.de/6-running-applications-as-interactive-jobs-on-the-lrz-ai-systems-10746640.html
- Batch-Jobs: https://doku.lrz.de/5-2-slurm-batch-jobs-introduction-1898974516.html und https://doku.lrz.de/7-running-applications-as-batch-jobs-on-the-lrz-ai-systems-10746643.html
- Schulung: https://doku.lrz.de/2025-06-02-ai-training-series-introduction-to-the-lrz-ai-systems-hdta3s25-1830808949.html
