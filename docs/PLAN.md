# NCM + Feedback: Plan

**Frage:** Wird Next-Concept Modeling (NCM) besser, wenn man ihm eine Rückkopplung von den feinen zu den groben Ebenen gibt, ähnlich wie beim Predictive Coding im Kortex?

Im Original läuft die Generierung genau einmal von grob nach fein, danach ist alles festgelegt. Im Gehirn gehen Fehlersignale dagegen auch wieder nach oben, und die grobe Hypothese wird nachjustiert.

## Warum Sudoku

- Die Metrik ist eindeutig: Ein Brett ist gelöst oder nicht. Das Paper meldet 86,1 % auf „hard“ (30 vorgegebene Zahlen).
- Es gibt ein natürliches **Fehlersignal**: Verstöße gegen die Regeln für Zeilen, Spalten und Boxen sowie Abweichungen von den vorgegebenen Zahlen. Das ist das Gegenstück zum Vorhersagefehler.
- Das Modell ist klein (~25M Generator, ~15M Tokenizer) und läuft auf einem MacBook.
- Es ist **kein Training nötig**, alle Varianten ändern nur die Inferenz.

## Varianten

| Name | Was passiert | Hirn-Analogie |
|---|---|---|
| `baseline` | Originale Generierung, greedy, ein Durchlauf | reiner Feedforward-Sweep |
| `clamp` | Wie baseline, danach werden die vorgegebenen Zahlen hart eingesetzt (ohne Neugenerierung) | Kontrolle: Wie viel bringt allein das Erzwingen des Inputs? |
| `loop_k{k}` | Brett dekodieren → Vorgaben einsetzen → mit dem VQ-VAE **neu kodieren** → die groben Ebenen 1..k behalten → Ebenen k+1..12 neu generieren. Bis zu 4 Runden, **nur für Bretter mit Regelverstößen** (fehlergetrieben), eine Runde wird nur übernommen, wenn sie nicht mehr Fehler hat | Predictive-Coding-Schleife: Fehler unten → grobe Repräsentation wird neu geschätzt → Details neu ausgefüllt |
| `loop_k{k}_t0.7` | Wie oben, aber die Neugenerierung sampelt mit Temperatur 0,7 | Rauschen als Exploration (Sampling-Hypothese, Buesing et al. 2011) |
| `sample8` | Keine Rückkopplung: 8 unabhängige Samples (T=0,7), das mit den wenigsten Regelverstößen gewinnt | parallele Hypothesen statt sequenzieller Korrektur (Kontrolle bei vergleichbarem Rechenaufwand) |

k wird über {1, 3, 6, 9} variiert: Wie viel „Konzept“ hält man fest, wie viel darf neu gedacht werden?

## Was gemessen wird

- **gelöst %:** exakte Übereinstimmung mit der Referenzlösung (Metrik aus dem Paper)
- **gültig %:** das Brett erfüllt alle Regeln
- **Zell-%:** Anteil korrekter Zellen
- **Ebenen-Evaluationen / Brett:** Rechenaufwand (eine volle Generierung = 12), um fair zu vergleichen
- **Analyse pro Ebene:** wie viele Zellen nach Ebene 1, 2, … 12 schon stimmen (wie viel „Lösung“ steckt in den groben Codes?)

## Erwartungen (vorab festgelegt)

1. `baseline` sollte die 86,1 % aus dem Paper reproduzieren (±1 %). Sonst stimmt am Setup etwas nicht.
2. `clamp` bringt wenig, weil das Modell die Vorgaben meist schon richtig übernimmt.
3. Kleines k (viel neu generieren) hilft mehr als großes k, weil Sudoku-Fehler globale Konsistenzfehler sind.
4. Greedy-Schleifen bleiben oft im selben Fixpunkt hängen. Rauschen (`_t0.7`) sollte nötig sein.
5. Die eigentliche Frage ist, ob `loop` besser ist als `sample8` bei gleichem Rechenaufwand. Erst dann bringt die *Rückkopplung* etwas über bloßes *mehr Rechnen* hinaus.

## Was ein Ergebnis bedeuten würde

- Schlägt `loop` `sample8` deutlich: Fehlergetriebene Re-Kodierung ist mehr als Brute-Force-Sampling. Das wäre ein Argument für Rückkopplung zwischen den Ebenen und eine Motivation, sie ins Training einzubauen.
- Gleichstand: Der Gewinn kommt nur aus mehr Rechenzeit plus Verifizierer.
- Keine Verbesserung: Das Modell wurde nie auf eigene Ausgaben als Input trainiert (Exposure Bias). Dann wäre der nächste Schritt das Training mit fehlerbasierter Korruption (Future Work im Paper).

## Grenzen

- Der Verifizierer für die Regeln ist bei Sudoku umsonst, bei Text gibt es so etwas nicht. Bei Text müsste das Fehlersignal aus dem Modell selbst kommen, z.B. aus dem Rekonstruktionsfehler oder der Unsicherheit.
- Nur Inferenz-Tricks auf einem fest trainierten Modell, keine echte Hirnplausibilität (kein lokales Lernen, keine Spikes).
- Ein Datensatz, ein Modell.

## Nächste Schritte, falls es interessant aussieht

1. Gleiches Experiment auf easy/medium.
2. Fehlersignal ohne Verifizierer: Rekonstruktions-Mismatch zwischen alten und neu kodierten Codes.
3. Training mit Selbst-Konditionierung (das Modell sieht im Training eigene Fehler), braucht eine GPU (Colab/LRZ).

---

# Experiment 2: Fehlersignal ohne Regel-Checker

**Frage:** Experiment 1 hat gezeigt, dass fehlergetriebenes Neu-Sampeln hilft. Das Fehlersignal kam dort aber von einem externen Regel-Checker, den es bei Text nicht gibt. Erkennt das Modell *selbst*, wann es falsch liegt?

**Signale** (Details in `experiments/selfsignal_sudoku.py`): Konfidenz des Decoders (unsicherste Zelle bzw. Mittelwert), Likelihood der gewählten Codes laut Generator, Selbstkonsistenz (neu kodiertes Brett gegen generierte Codes), Abgleich mit den Vorgaben im Rätsel (Input-Check, keine Regeln). Als Referenz läuft der Regel-Checker mit.

**Ablauf:**
- *Teil A:* Wie gut trennt jedes Signal gelöste von ungelösten Brettern (AUROC)? Die Schwelle wird auf 500 separaten Kalibrierungs-Brettern festgelegt, nicht auf den Testbrettern.
- *Teil B:* Die Schleife aus Experiment 1 (k0, T=0,7, 4 Runden), gesteuert durch das interne Signal: verdächtige Bretter werden neu versucht, übernommen wird nur ein Versuch mit besserem Signal. Kontrolle bei gleichem Aufwand: `best2_gen_ll` (2 Versuche, der sicherere gewinnt).

**Erwartungen (vorab):**
1. Die Decoder-Konfidenz ist ein schwaches Signal (AUROC ~0,6–0,7): Fehler bei Sudoku sind globale Widersprüche, die einzelne Zelle kann sich trotzdem „sicher“ anfühlen.
2. Die Generator-Likelihood ist besser (AUROC ~0,75), weil sie die ganze Hierarchie abdeckt.
3. Die Selbstkonsistenz ist nahe am Zufall, weil Experiment 1 gezeigt hat, dass das Modell auch auf falsche Ausgaben stabil reagiert.
4. Die Schleife mit dem besten internen Signal holt ungefähr die Hälfte des Gewinns des Regel-Checkers (also ~88 % statt 91 %).

---

# Tageslauf (`./run_all.sh`)

1. **Ehrliche Kosten:** `gen_llR` wird jetzt in **einem** Forward Pass bewertet (Trainingspfad von NCM) statt Ebene für Ebene. Ein Selbsttest prüft am Anfang, dass beide Wege dieselben Werte liefern. Kosten pro Versuch: 12 + 1 NFE statt 24.
2. **Zufallskontrolle `rand_gen_llR`:** Genauso viele Bretter werden neu versucht wie bei `self_gen_llR`, aber zufällig ausgewählt. Damit lässt sich trennen, was die *Erkennung* bringt und was nur das *Auswählen des besseren Versuchs*.
3. **Mehr Bretter:** 1000 Test- / 1000 Kalibrierungsbretter auf hard, medium und easy.

## Experiment 3: Text (TinyStories)

**Frage:** Ist das interne Signal auch bei Text ein Qualitätsmaß, und hilft gezieltes Neu-Sampeln?

- Qualität wie im Paper: Perplexity unter GPT-2 large (niedriger = flüssiger) über 5 Seeds × 256 Geschichten, dazu Unigramm-Entropie (Vielfalt).
- *Teil A:* Spearman-Korrelation zwischen Signal und GPT-2-Qualität pro Geschichte.
- *Teil B:* untere 20 % nach Signal neu sampeln (bis zu 2 Runden, nur Verbesserungen übernehmen), Kontrollen `rand_` und `best2_`.

**Erwartungen (vorab):**
1. `gen_llR` korreliert positiv, aber deutlich schwächer als bei Sudoku (Spearman ~0,3–0,5): Bei Text gibt es viele gute Antworten, und Likelihood misst eher „typisch“ als „gut“.
2. Gezieltes Neu-Sampeln senkt die GenPPL leicht (~5–10 %).
3. **Risiko:** Auswahl nach Likelihood bevorzugt repetitiven Text. Fällt die Entropie deutlich (> 0,05), ist der Gewinn zum Teil Schein.
4. `self_` ist besser als `rand_` bei ähnlichen Kosten. Sonst bringt die Erkennung nichts.

---

# Agenda: Experiment 4, Nachtraining mit eigenen Fehlern

**Warum:** Experiment 1 hat gezeigt, dass das Festhalten der groben Ebenen nichts bringt. Vermutete Ursache: Das Modell hat im Training nur perfekte grobe Codes gesehen und kann mit eigenen fehlerhaften nichts anfangen.

**Plan:**
1. Trainingscode schreiben (fehlt im Upstream-Repo). Die Verlustfunktion steht im Paper (Gl. 10, inklusive Code-Korruption).
2. Sudoku-Trainingsdaten selbst erzeugen (Generator + Löser, 30–40 Vorgaben).
3. Den bestehenden Generator nachtrainieren: Ein Teil der Batches wird mit den *eigenen* generierten groben Codes konditioniert statt mit zufälliger Korruption (Selbst-Konditionierung).
4. Experiment 1 wiederholen. Die Frage: Schlägt `loop_k3_t0.7` jetzt `loop_k0_t0.7`?

**Rechenbedarf:** Nachtrainieren dauert auf einer A100 ~30–60 min. Ziel ist der TUM/LRZ-Cluster: Trainingsskript plus Slurm-Jobskript (1 GPU, ~2 h, ~16 GB RAM, ~5 GB Speicher). Fallback ist ein Colab-Notebook (T4, einige Stunden).

---

# Experiment 4: Nachtraining, Stand 23.10. (Code fertig, wartet auf Cluster)

**Umsetzung** (`experiments/train_selfcond.py`, `experiments/sudoku_data.py`, `cluster/`):
- Trainingsdaten: 50.000 selbst erzeugte, eindeutig lösbare Sudokus (30/35/40 Vorgaben), ohne Überschneidung mit den Testbrettern.
- **Saubere Beispiele:** Paper-Ziel (Gl. 10), Eingabe-Codes zufällig korrumpiert (40 % auf der gröbsten bis 5 % auf der feinsten Ebene).
- **Selbst-Beispiele (50 %):** Das Modell generiert selbst (T = 0,7). Für ein zufälliges k werden die groben Ebenen 0..k−1 aus der eigenen Generierung übernommen. Ziel für die Ebenen ab k sind die **Reparatur-Codes**: die Restquantisierung der echten Lösung, ausgehend von der eigenen (eventuell falschen) groben Rekonstruktion.
- **Kontrolle `ctrl`:** gleich viel Training, nur saubere Beispiele. Sie trennt den Effekt der Selbst-Konditionierung vom Effekt bloßen Weitertrainierens.
- Je 5000 Schritte, Batch 256, AdamW, Lernrate 1e-4 mit Cosinus-Abfall, bf16.

**Erwartungen (vorab):**
1. Die Baseline (greedy, ein Durchlauf) bleibt bei `self` ungefähr gleich (±1 Prozentpunkt). Sinkt sie deutlich, verdrängt das Reparieren das normale Generieren.
2. Bei `self` schlagen Schleifen, die grobe Ebenen festhalten (`loop_k3_t0.7`, `loop_k6_t0.7`), endlich `loop_k0_t0.7`. Bei Original und `ctrl` nicht.
3. Greedy-Schleifen (`loop_k3`, `loop_k6`, ohne Rauschen) kommen bei `self` aus dem Fixpunkt heraus, weil das Modell gelernt hat, aus einer fehlerhaften Skizze etwas anderes zu machen.
4. Klappt das, ist es die Predictive-Coding-Rückkopplung, die in Experiment 1 gefehlt hat: Die grobe Hypothese bleibt stehen, und die feinen Ebenen korrigieren den Fehler.

---

# Agenda: Experiment 5, Mischform aus NCM und normalen LLMs

**Frage:** Lässt sich das Beste aus beiden verbinden? Von NCM die Planung grob → fein, das parallele Erzeugen und das eingebaute Fehlersignal. Von autoregressiven LLMs die Flexibilität bei beliebig langen Texten und die bessere Qualität.

**Mögliche Bauformen (noch zu prüfen):**
1. **Blockweise:** Ein LLM schreibt Satz für Satz bzw. Block für Block (autoregressiv über die Blöcke), jeder Block wird mit NCM grob → fein in wenigen Schritten erzeugt. Analog zu „Block Diffusion“ (Arriola et al. 2025), nur mit NCM statt Diffusion innerhalb des Blocks. Das ist am nächsten an „das Gehirn plant einen Satz grob voraus“.
2. **NCM als Entwurf, LLM als Prüfer (Speculative Decoding):** NCM erzeugt schnell einen ganzen Absatz, ein normales LLM prüft ihn in einem einzigen Durchlauf und übernimmt die passenden Teile. Ähnliche Ideen gibt es mit Diffusionsmodellen als Entwerfer (z.B. DFlash, im NCM-Paper zitiert).
3. **LLM plant, NCM füllt aus:** Ein LLM sagt nur die groben Konzept-Codes voraus (wenige Tokens), NCM füllt die feinen Ebenen parallel. Verwandt mit „Large Concept Models“ (Meta 2024).
4. **Feedback dazwischen:** Das interne Fehlersignal aus Experiment 2/3 entscheidet, welche Blöcke nachgebessert oder vom LLM neu geschrieben werden.

**Erster machbarer Schritt:** Literaturrecherche (Block Diffusion, Large Concept Models, Speculative Decoding mit parallelen Entwerfern) und eine kleine Machbarkeitsstudie mit Variante 2 auf TinyStories. Dafür sind beide Modelle schon da (NCM-TinyStories und GPT-2), kein Training nötig.


---

# Experiment 6: Neuronen mit Zeitverhalten und Feuerwahrscheinlichkeit (bereit, läuft auf dem M1)

**Frage:** Echte Neuronen reagieren nicht sofort. Die Membranspannung baut sich mit einer Zeitkonstante auf und läuft wieder aus (Ionenkanäle öffnen und schließen verzögert), jedes Neuron ist anders schnell, und ob es feuert, ist zufällig. Werden kleine rekurrente Netze damit besser oder robuster?

**Neuronenmodell** (`experiments/neurodyn.py`), pro Zeitschritt mit Leck-Faktor a = exp(−1/τ):
v_t = a·v_{t−1} + (1−a)·(W x_t + U s_{t−1} + b), Feuerwahrscheinlichkeit p_t = sigmoid(v_t).

| Variante | Zeitverhalten | Ausgabe |
|---|---|---|
| `rnn_*` | keins (a = 0), klassisches RNN | |
| `tau_hom_*` | ein gemeinsames lernbares τ (Start ≈ 7 Schritte) | `*_det`: Feuerrate p_t |
| `tau_het_*` | eigenes lernbares τ pro Neuron, Start log-gleichverteilt 1–50 | `*_stoch`: Spike ~ Bernoulli(p_t), Gradient per Straight-Through |
| `tau_dyn_*` | τ hängt vom aktuellen Input und Zustand ab (ähnlich Liquid Time-Constant Networks) | |
| `gru` | Referenz: Standard-GRU | |

**Aufgaben** (synthetisch, kein Download): `freq` (verrauschtes Signal aus langsamer und schneller Schwingung, 4 Klassen, braucht zwei Zeitskalen), `recall` (ein Symbol von 8 merken über 17–77 Schritte), `adding` (zwei markierte Zahlen aus 100 addieren).
**Gemessen:** Test wie im Training, mit stärkerem Input-Rauschen, mit doppelter Folgenlänge. Bei `stoch` zusätzlich dasselbe Netz ohne Zufall. Gelernte τ-Verteilung. 64 Neuronen, 5 Seeds, 135 Läufe.

**Erwartungen (vorab festgelegt):**
1. Zeitkonstanten helfen bei `freq` und `recall` deutlich gegenüber `rnn` (die Membran ist ein eingebautes Kurzzeitgedächtnis).
2. Heterogen ≥ gemeinsam, vor allem bei `freq` (zwei Zeitskalen) und bei doppelter Länge, wie bei Perez-Nieves et al. 2021. Die gelernten τ bleiben breit verteilt.
3. Bernoulli-Spikes kosten im Training etwas Genauigkeit, sind aber unter Rauschen gleich gut oder robuster. Ein stochastisch trainiertes Netz funktioniert auch ohne Zufall.
4. `tau_dyn` liegt nahe an der GRU (beides sind Netze mit Gates).

**Starten:** `./run_neurodyn.sh` (erst Schnelltest ~1–2 min, dann 3 Worker auf der CPU, Auswertung in `results/neurodyn-summary.md`). Fortsetzbar nach Abbruch.

**Anschluss an NCM:** Wenn Bernoulli-Neuronen ohne Verlust trainierbar sind, ist der nächste Schritt Rauschen in den NCM-Aktivierungen statt nur beim Sampling der Codes. Experiment 1 hat gezeigt, dass die Feedback-Schleife ohne Zufall im Fixpunkt hängt.

**Literatur:** Perez-Nieves et al. 2021, *Neural heterogeneity promotes robust learning* (Nat. Commun.). Hasani et al. 2021, *Liquid Time-Constant Networks* (AAAI). Buesing et al. 2011, *Neural dynamics as sampling* (PLoS Comput. Biol.). Bengio et al. 2013 (Straight-Through-Schätzer).

---

# Agenda: Experiment 7, Lernen wie bei Evolution und Entwicklung

**Frage:** Menschen lernen auf zwei Zeitskalen. Die Evolution liefert Bauregeln und angeborene Strukturen (Genom ≈ wenige hundert MB für ~10^14 Synapsen, also keine fertigen Gewichte). Die Entwicklung läuft in Phasen: kritische Perioden, erst Überproduktion von Synapsen, dann Ausdünnen, Schlaf-Konsolidierung, wenig Daten. Was davon hilft künstlichen Netzen?

**Mögliche Teil-Experimente:**
1. **Kritische Phasen** (Mac-machbar): Achille et al. 2019 nachbauen. Ein kleines CNN bekommt in den ersten N Epochen unscharfe Bilder, dann scharfe. Erholt es sich? Dann prüfen, ob Rauschen oder hohe Plastizität am Anfang (wie bei Kindern) das Fenster verändert.
2. **Überproduktion und Ausdünnen:** Großes Netz mit hoher Lernrate starten, dann schrittweise prunen und „stabilisieren“ (Lernrate pro Gewicht senken), gegen ein gleich großes Netz von Anfang an. Bezug: Lottery Ticket Hypothesis (Frankle & Carbin 2019).
3. **Genomischer Flaschenhals:** Gewichte nicht direkt lernen, sondern aus einer kleinen „Bauregel“ erzeugen (Hypernetz mit wenigen Parametern) und über Generationen optimieren. Danach mit wenig Daten nachlernen. Bezug: Zador 2019 (*A critique of pure learning*), Shuvaev et al. 2024 (genomischer Flaschenhals).
4. **Kindgerechte Datenmenge** (Cluster): kleines Sprachmodell nach BabyLM-Regeln (~10–100 Mio. Wörter) mit Entwicklungs-Lehrplan (einfach → schwer, Phasen hoher und niedriger Plastizität) gegen normales Training.

**Erster Schritt:** Literaturrecherche (kritische Phasen in Netzen, Entwicklungs-Curricula, BabyLM-Ergebnisse 2023–2025, genomischer Flaschenhals), dann Teil 1 auf dem Mac.


## Experiment 7a: Kritische Phasen (gebaut, läuft auf der M1-GPU)

`experiments/critical_periods.py`, Auswertung `experiments/critical_report.py`. Kleines CNN (~0,3 Mio. Parameter) auf CIFAR-10, SGD, Lernrate 0,05·0,94^Epoche, alle Läufe enden mit 3 Epochen bei Lernrate 0,0005. Defizit = Bilder auf 8×8 verkleinert und wieder hochskaliert („grauer Star“). 13 Läufe × 2 Seeds:

| Lauf | Was | prüft |
|---|---|---|
| `def0/5/10/20/30/40` | Defizit in den ersten D Epochen, danach 30 normale | Gibt es einen Dauerschaden, und ab wann? |
| `win10`, `win30` | Defizit 10 Epochen lang, aber später | Ist nur der *Anfang* kritisch? |
| `def0/20/40_const` | konstante Lernrate | Ist es nur die sinkende Lernrate? |
| `def20/40_reset` | Lernrate springt nach dem Defizit zurück auf den Startwert | Lässt sich die Phase „wieder öffnen“? |

Für `def0` und `def40` (Seed 0) zusätzlich die Spur der Fisher-Information pro Epoche.

**Erwartungen (vorab festgelegt):**
1. Der Endwert sinkt mit der Länge des Defizits: `def40` liegt mindestens 1 Punkt unter `def0`. `def5` schadet kaum.
2. Dasselbe Defizit später (`win30`) schadet weniger als am Anfang (`def10`).
3. Auch mit konstanter Lernrate bleibt ein Schaden, er ist aber kleiner (wie bei Achille et al.).
4. Die Lernrate zurückzusetzen holt einen Teil, aber nicht alles zurück.
5. Die Fisher-Spur steigt in den ersten Epochen steil an und fällt danach. Mit Defizit ist der Verlauf anders.

**Grenzen:** viel kleineres Netz und kürzeres Training als bei Achille (ResNet-18, 160 Epochen), nur 2 Seeds. Effekte unter ~0,5 Punkten sind Rauschen.

**Starten:** `./run_night.sh` startet Experiment 6 (CPU) und 7a (GPU) gleichzeitig. Einzeln geht `python experiments/critical_periods.py`.
