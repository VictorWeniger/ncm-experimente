# Was wir herausgefunden haben (Stand 2. 11. 2026)

Neun Experimente und die Text-Benchmarks des Papers, alle auf einem MacBook Air (M1, 8 GB) mit den veröffentlichten NCM-Modellen, ohne NCM selbst nachzutrainieren. Exp. 4 (Nachtraining) wartet auf den Cluster.

---

## 1. NCM erkennt seine eigenen Fehler, und das lässt sich nutzen

**Signal `gen_llR`:** fertige Ausgabe mit dem NCM-Tokenizer neu kodieren, die Codes per Teacher Forcing in einem Durchlauf unter dem Generator bewerten, mittlere Log-Wahrscheinlichkeit. Kein Training, keine Regeln.

| | ohne Nachbessern | mit `gen_llR` | Kontrolle: gleich viel Aufwand, zufällig | externer Regel-Checker |
|---|---|---|---|---|
| Sudoku hard (1000 Bretter) | 84,3 % | **88,3 %** | – | 89,4 % |
| Sudoku medium | 96,9 % | **97,8 %** | – | 98,1 % |
| TinyStories (Perplexity ↓) | 16,15 | **14,36** | 15,45 | – |
| LM1B (Perplexity ↓, FLM wie Paper) | 92,59 | **86,08** | 89,95 | – |

- Erkennung bei Sudoku: AUROC 0,96. Text: Spearman 0,78 mit der GPT-2-Qualität.
- Best-of-4 nach `gen_llR`: TinyStories 12,58, LM1B 80,07 (Entropie unverändert, also nicht einfach repetitiver).
- Beide Prüfmodelle (GPT-2 large, SmolLM2-360M) zeigen dieselbe Richtung.

**Exp. 8, Suche auf den groben Ebenen** (500 Sudokus hard, 8 Kandidaten, Auswahl nach `gen_llR`; gierig 85,2 %):

| Abzweigen auf … | T = 0,7 | T = 1,5 | verschiedene Bretter von 8 |
|---|---|---|---|
| gröbster Ebene | 86,2 % | 86,6 % | 1,1–1,2 |
| oberen 3 Ebenen | 87,4 % | 89,2 % | 1,2–1,4 |
| oberen 6 Ebenen | 90,4 % | 91,2 % | 1,5–1,8 |
| allen Ebenen | 89,8 % | **92,0 %** | 1,8–2,2 |
| nur feinen Ebenen (unten 6) | 88,0 % | 89,2 % | 1,6–2,0 |

- Alles neu bei T = 1,5: 36 Rätsel zusätzlich gelöst, 2 verloren (McNemar p < 0,0001). Nur bei den verdächtigen 15 %: 91,8 % bei ~2,2-fachem statt 8-fachem Aufwand.
- Nur grob abzweigen ist signifikant schlechter als alles neu (p < 0,01): Die groben Ebenen legen bei Sudoku fast nichts fest.
- Halbfertige Bretter lassen sich schlecht bewerten: Teilprüfer AUROC 0,5–0,88, gelerntes Prüfnetz ≤ 0,72, gegenüber 0,96 am fertigen Brett.

## 2. Das Signal misst „typisch“, nicht „treu“

Benchmarks des Papers, nachgemessen (Paper in Klammern): XSum ROUGE-L 27,78 (28,0), WMT14 BLEU 21,64 (22,6). Aus 4 Kandidaten (gierig + 3 Samples) holt ein Orakel 33,8 ROUGE-L bzw. 25,5 BLEU, `gen_llR` wählt kaum besser als der Zufall.

**Exp. 9:** neue Signale auf denselben Kandidaten, Gewichte auf 30 % der Dokumente gewählt, gemessen auf 70 %.

| Regel | WMT14 BLEU | WMT14 chrF | Paar-Trefferquote WMT14 | XSum ROUGE-L | Paar-Trefferquote XSum |
|---|---|---|---|---|---|
| gierig | 21,53 | 49,37 | – | 28,24 | – |
| `gen_llR` | 20,90 | 48,34 | 56,6 % | 28,13 | 54,1 % |
| **pmi** (mit − ohne Vorlage) | 21,76 | 50,36 | **66,0 %** | 27,36 | 51,2 % |
| MBR (Konsens) | 22,05 | 50,65 | 68,6 % | **28,84** | 57,9 % |
| Rückübersetzung | 21,89 | 50,50 | 67,3 % | – | – |
| pmi + MBR + Rückübersetzung | **22,44** | **51,03** | – | 28,70 (ohne RÜ) | – |
| Kombination nur bei verdächtigen 12 % | 21,73 | 49,75 | – | 28,35 | – |
| Orakel | 25,52 | 53,60 | – | 33,83 | – |

- Kontrolle mit fremdem Artikel: Auch bei XSum passt der Kandidat in 99,9 % besser zum eigenen Artikel. pmi misst also echte Nähe zur Vorlage, die ROUGE-Wertung bei XSum belohnt das aber nicht (XSum-Referenzen enthalten oft Information außerhalb des Artikels).

## 3. Grob → fein schlägt Token-für-Token nur bei globalen Bedingungen

| TinyStories | Perplexity GPT-2 large | SmolLM2 | Entropie | Schritte nacheinander | GFLOPs/Geschichte |
|---|---|---|---|---|---|
| NCM (89 Mio.) | 16,15 | 14,55 | 3,97 | 11 | 113 |
| TinyStories-33M (67 Mio., autoregressiv) | **9,82** | **8,81** | 4,02 | 134 | **18** |

Bei Sudoku umgekehrt: Token-für-Token ~1 % (Paper), NCM 88 %. Das Paper vergleicht bei Text nur mit Diffusionsmodellen; gegen die liegt NCM vorn (LM1B 92,6 gegen DUO 97,3).

## 4. Mischformen: Das Signal ist eine gute Weiche, der Plan noch nicht

**Exp. 5, ohne Training** (TinyStories, 5 × 256):

| Variante | GPT-2 large | SmolLM2 | Schritte | GFLOPs |
|---|---|---|---|---|
| LLM (Qwen 0,5B) schreibt markierte 26 % neu | 13,03 | 11,78 | 54 | 313 |
| … zufällige 23 % | 14,06 | 13,11 | 48 | 302 |
| markierte durch frisches LLM-Sample ersetzt | 12,90 | 11,44 | 47 | 227 |
| nur der schwächste Satz neu | 15,70 | 14,23 | 27 | 227 |
| LLM übernimmt Entwurf bis zur ersten unplausiblen Stelle | **10,14** | **8,77** | 91 | 251 |
| LLM wählt aus 4 NCM-Entwürfen | 12,16 | 10,33 | 49 | 961 |

**Exp. 5b, ein Netz plant und schreibt** (GPT-2 small, je 1000 Schritte × 16 Geschichten):

| Variante | GPT-2 large | SmolLM2 | Entropie | Val.-Verlust Text mit echtem Plan |
|---|---|---|---|---|
| GPT-2 ohne Plan | 14,15 | 14,30 | 4,03 | 1,472 |
| A: NCM-Plan als Präfix | 14,02 | 14,04 | 4,02 | 1,477 |
| B: ein Rumpf plant und schreibt | 14,04 | 14,04 | 4,04 | 1,499 |
| D: B + Satz neu bei großem Plan-Abstand | 11,62 | 11,61 | 3,90 | – |
| Kontrolle D: zufällig ausgelöst, Auswahl nach Plan | 11,62 | 11,60 | 3,89 | – |
| Kontrolle D: zufällig ausgelöst und gewählt | **11,37** | **11,28** | 3,89 | – |
| C: Absatz für Absatz | 16,76 | 17,89 | 4,01 | 1,699 |
| C, 6 Blöcke (bis 297 Wörter) | 16,66 | 17,07 | 4,17 | – |
| C-par (Block parallel aus NCM-Decoder) | 334,8 | 307,6 | 3,77 | – |

- Der Plan wird nach 1000 Schritten nicht genutzt (Verlust mit echtem Plan nicht niedriger).
- D ist ein Artefakt des satzweisen Schreibens (kürzer, weniger Vielfalt), nicht des Plans: Die Kontrolle ohne Plan ist gleich gut.
- C hebt die Längengrenze auf, ohne bei langen Geschichten schlechter zu werden.

## 5. Methodische Lehren

- **Kontrollen fangen falsche Effekte:** zufällig statt nach Signal nachbessern (Exp. 2/3), D ohne Plan (Exp. 5b), fremder Artikel (Exp. 9).
- **Zwei Prüfer aus verschiedenen Familien:** GPT-2 large bewertet feingetunte GPT-2-Modelle um 0,13 log-PPL besser als NCM, SmolLM2 nur um 0,01–0,03.

## 6. Biologie-Experimente

- **Exp. 6 (Neuronen mit Zeitkonstanten und Spikes):** Bernoulli-Spikes verbessern die Generalisierung auf längere Sequenzen; Netze lernen längere Zeitkonstanten; heterogene Zeitkonstanten pro Neuron bringen nichts; inputabhängige Zeitkonstanten ≈ GRU. Details: `results/neurodyn-summary.md`.
- **Exp. 7a (kritische Phasen, CIFAR-10):** Dauerschaden durch anfangs unscharfe Bilder ist real, verschwindet aber bei konstanter Lernrate und wird durch eine Lernraten-Spitze vollständig geheilt. Details: `results/critical-summary.md`.

## Offen

1. Hilft ein Plan bei vollem Training (20.000 Schritte)? → `cluster/lrz_merge.sbatch`
2. pmi auf Zusammenfassungen mit treuen Referenzen (z. B. CNN/DailyMail)
3. Exp. 4, Nachtraining mit eigenen Fehlern → Cluster
4. OpenWebText-Benchmark → `cluster/lrz_bench.sbatch`

**Möglicher Kern für ein Paper:** ein Selbstprüf-Signal aus dem Modell selbst, ohne Zusatztraining, das bei Sudoku und freiem Text funktioniert und bei Aufgaben mit Vorlage durch die pmi-Korrektur erweitert wird.
