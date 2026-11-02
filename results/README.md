# Ergebnisse: welche Datei gehört zu welchem Experiment

Die Skripte schreiben direkt nach `results/`, deshalb liegen die Dateien flach. Diese Übersicht ordnet sie zu.

| Experiment | Dateien |
|---|---|
| 1 · Feedback mit Regel-Checker | `sudoku-hard.json/.log`, `sudoku-hard-n500.*`, `sudoku-hard-kontrolle.*`, `sudoku-hard-n2000-baseline.*` |
| 2 · internes Fehlersignal | `sudoku-hard-self.*`, `sudoku-hard-self2.*`, `sudoku-hard-self3.*`, `sudoku-medium-self3.*`, `sudoku-easy-self3.*` |
| 3 · Text (TinyStories) | `tinystories-text.*`, `tinystories-text-smoke.json`, `text-smoke.log` |
| 3b · blindes Urteil | `tinystories-judge*`, `judge/` |
| 6 · Neuronen mit Zeit und Zufall | `neurodyn*` (Zusammenfassung: `neurodyn-summary.md`) |
| 7a · kritische Phasen | `critical*` (Zusammenfassung: `critical-summary.md`) |
| 8 · Suche auf groben Ebenen | `sudoku-hard-search-t07.*`, `sudoku-hard-search-t15.*`, `*-value*`, `search-smoke*`, `level_demo.json` |
| Benchmarks, Exp. 5, 5b, 9 | `bench/<aufgabe>/` mit `table.md` (Tabelle), `scores.json` (alle Zahlen), `pairs.jsonl` + `pairs_key.json` (blinde Paare), `select.md/.json` (Exp. 9); Übersicht `bench/SUMMARY.md`; Protokolle `bench/run-*.log` |

Nicht im Repo (zu groß, lassen sich neu erzeugen): `*-dump.pt` aus Exp. 8, `bench/*/ncm.json`, `hybrid.json`, `stage4.json`, `merge.json` (alle erzeugten Texte), trainierte Modelle unter `artifacts/`.
