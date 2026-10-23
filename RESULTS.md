# Ergebnisse: Sudoku hard

Lauf vom 22.10.2026 auf dem MacBook Air (MPS), Upstream-Commit `c70274273f`.

## Reproduktion

Die Baseline mit **2000 Brettern** kommt auf **86,30 %** gelöst. Das Paper meldet 86,1 %. Das Setup stimmt also.

## Hauptlauf (500 Bretter, `results/sudoku-hard-n500.json`)

| Variante | gelöst % | Ebenen/Brett | gelöst je Runde |
|---|---|---|---|
| baseline | 85,2 | 12,0 | |
| clamp | 85,2 | 12,0 | |
| loop_k1 | 85,4 | 18,4 | 85,2 → 85,4 (danach Fixpunkt) |
| loop_k3 | 86,2 | 17,1 | 85,2 → 86,2 (danach Fixpunkt) |
| loop_k6 | 86,2 | 15,4 | 85,2 → 86,2 (danach Fixpunkt) |
| loop_k9 | 86,2 | 13,7 | 85,2 → 86,0 → 86,2 |
| **loop_k1_t0.7** | **90,6** | **17,3** | 85,2 → 87,6 → 89,2 → 89,8 → 90,6 |
| loop_k3_t0.7 | 89,8 | 16,4 | 85,2 → 88,2 → 88,8 → 89,2 → 89,8 |
| loop_k6_t0.7 | 88,0 | 15,1 | 85,2 → 87,4 → 87,6 → 88,0 → 88,0 |
| sample2 | 87,8 | 24,0 | |
| sample8 | 91,8 | 96,0 | |

Statistische Unsicherheit bei n=500: etwa ±1,5 Prozentpunkte pro Zeile (1 SE), also etwa ±2 Prozentpunkte für die Differenz zweier Zeilen.

Zellen korrekt nach Ebene 1…12 (baseline, %): 14,5 · 14,5 · 17,0 · 20,3 · 20,3 · 27,9 · 33,1 · 45,5 · 62,1 · 75,8 · 81,8 · 98,3

## Abgleich mit den Erwartungen aus PLAN.md

1. Baseline reproduziert das Paper: **ja** (86,3 gegenüber 86,1).
2. `clamp` bringt nichts: **ja**, identisch zur Baseline. Das Modell übernimmt die Vorgaben schon von selbst.
3. Kleines k hilft mehr als großes: **ja**, aber nur mit Rauschen. k1 > k3 > k6 (90,6 / 89,8 / 88,0).
4. Greedy-Schleifen bleiben im Fixpunkt hängen: **ja**. Nach Runde 1 ändert sich praktisch nichts mehr (+1 Prozentpunkt).
5. Schlägt `loop` `sample` bei gleichem Rechenaufwand? **Eher ja, aber nicht belastbar.** `loop_k1_t0.7` liegt mit 90,6 % bei **17,3** Evaluationen über `sample2` mit 87,8 % bei **24**. Die Differenz von 2,8 Prozentpunkten entspricht etwa 1,4 Standardfehlern, ist also nicht signifikant. Und `sample8` erreicht mit 91,8 % kaum mehr, braucht dafür aber 5,5-mal so viel Rechenaufwand.

## Interpretation

- **Rückkopplung allein (deterministisch) bringt fast nichts.** Neu kodieren und dann greedy neu generieren landet beim selben Brett. Das Modell ist auf seine eigenen Ausgaben stabil, auch auf die falschen.
- **Rückkopplung + Rauschen + Fehlersignal hilft deutlich:** +5,4 Prozentpunkte gegenüber der Baseline, bei ~45 % mehr Rechenaufwand.
- **Aber: Je *weniger* von der groben Repräsentation festgehalten wird, desto besser.** Das spricht dagegen, dass die *Re-Kodierung der groben Ebenen* der Grund für den Gewinn ist. Plausibler ist: Der Gewinn kommt daher, dass **nur die fehlerhaften Bretter** gezielt neu gesampelt werden (fehlergetriebene Rechenzeit) und eine Verschlechterung nie übernommen wird. `sample2` verteilt seinen Aufwand dagegen auf alle Bretter, auch auf die schon gelösten.
- Die Analyse pro Ebene bestätigt das: Die groben Ebenen tragen kaum Lösungsinformation (nach Ebene 6 sind erst 28 % der Zellen richtig). Sie festzuhalten heißt eher, die Fehler mit festzuhalten.

**Vorläufiges Fazit (vor dem Kontrolllauf):** Der Effekt ist eher „Predictive Coding als *Fehlersignal, das Rechenzeit steuert*“ als „Predictive Coding als *Korrektur der groben Repräsentation*“. Für die zweite Variante müsste das Modell wohl darauf trainiert werden, auf eigenen fehlerhaften Codes zu konditionieren (Future Work im Paper).

## Kontrolllauf (500 Bretter, `results/sudoku-hard-kontrolle.json`)

| Variante | gelöst % | Ebenen/Brett |
|---|---|---|
| baseline | 85,2 | 12,0 |
| **loop_k0_t0.7** (nichts Grobes festhalten) | **91,4** | 17,6 |
| loop_k1_t0.7 | 90,6 | 17,3 |
| sample2 | 87,8 | 24,0 |

Gepaarter McNemar-Test (exakt, zweiseitig) auf denselben 500 Brettern:

| Vergleich | nur A gelöst | nur B gelöst | p |
|---|---|---|---|
| loop_k0_t0.7 vs. sample2 | 21 | 3 | 0,0003 |
| loop_k1_t0.7 vs. sample2 | 16 | 2 | 0,0013 |
| loop_k0_t0.7 vs. loop_k1_t0.7 | 11 | 7 | 0,48 |
| sample2 vs. baseline | 13 | 0 | 0,0002 |
| loop_k0_t0.7 vs. baseline | 31 | 0 | <0,0001 |

## Fazit

1. **Fehlergetriebenes Neu-Sampeln schlägt blindes Sampeln signifikant**, und das bei *weniger* Rechenaufwand (17,6 gegenüber 24 Ebenen pro Brett, p = 0,0003).
2. **Die Rückkopplung in die groben Konzept-Ebenen trägt nichts dazu bei.** Ganz ohne festgehaltene grobe Ebene (k0) ist die Schleife mindestens so gut wie mit (k1). Der Unterschied ist nicht signifikant (p = 0,48), und tendenziell ist k0 sogar besser.
3. Der eigentliche Mechanismus: Ein **Fehlersignal entscheidet, wo nochmal gerechnet wird**, plus „nie verschlechtern“. Das ist im Kern Rejection Sampling mit Verifizierer, angewandt nur dort, wo es nötig ist.

**Für die Hirn-Frage heißt das:** Die Predictive-Coding-artige Korrektur der groben Repräsentation funktioniert mit einem Modell, das nur feedforward trainiert wurde, *nicht*. Was funktioniert, ist das Fehlersignal als Steuerung der Rechenzeit, etwa analog zu „Aufmerksamkeit dort, wo die Vorhersage nicht passt“. Damit die Korrektur der groben Ebenen selbst etwas bringt, müsste sie vermutlich mittrainiert werden.

## Nächste Schritte

1. **Fehlersignal ohne Verifizierer:** Kann das Modell selbst erkennen, wann es falsch liegt (z.B. an niedriger Konfidenz der Logits oder an Mismatch zwischen alten und neu kodierten Codes)? Das wäre die Voraussetzung, damit das auf Text übertragbar ist.
2. **Training mit Selbst-Konditionierung** (das Modell sieht im Training seine eigenen fehlerhaften Codes), um zu testen, ob die Korrektur der groben Ebenen dann etwas bringt. Braucht eine GPU (Colab/LRZ).
3. Die beiden besten Varianten auf alle 2000 Bretter sowie easy/medium.

---

# Experiment 2: Fehlersignal ohne Regel-Checker (`results/sudoku-hard-self.json`)

500 Testbretter, Schwellen auf 500 separaten Kalibrierungsbrettern festgelegt.

## Teil A: Erkennt das Modell seine Fehler?

| Signal | AUROC | markiert % | Treffer % | Fehlalarm % |
|---|---|---|---|---|
| dec_min (unsicherste Zelle) | **0,898** | 18,8 | 74,3 | 9,2 |
| dec_mean | 0,896 | 18,2 | 74,3 | 8,5 |
| gen_ll | 0,888 | 32,4 | 85,1 | 23,2 |
| cycle | 0,860 | 36,8 | 81,1 | 29,1 |
| gen_min | 0,761 | 29,4 | 62,2 | 23,7 |
| clue | 0,500 | 0 | 0 | 0 |
| rules (Orakel) | 1,000 | 14,8 | 100 | 0 |

**Das Modell weiß ziemlich gut, wann es falsch liegt.** Die Unsicherheit des Decoders bei der unsichersten Zelle erkennt 3 von 4 Fehlern bei nur 9 % Fehlalarmen. Damit lagen die Erwartungen 1 und 3 aus dem Plan **falsch**: Die Decoder-Konfidenz ist das *beste* Signal, nicht das schwächste, und die Selbstkonsistenz ist weit besser als Zufall (0,86). `clue` ist wertlos, weil das Modell den Vorgaben nie widerspricht (passt zu `clamp` aus Experiment 1).

## Teil B: Schleife mit internem Signal

| Variante | gelöst % | Ebenen/Brett | p vs. baseline | p vs. verifier |
|---|---|---|---|---|
| baseline | 85,2 | 12,0 | | |
| verifier (Regel-Checker) | 91,4 | 17,6 | <0,0001 | |
| **self_dec_min** | **88,2** | 17,1 | 0,0015 | 0,0015 |
| self_dec_min+clue | 87,8 | 17,5 | 0,0023 | 0,0005 |
| self_cycle | 86,4 | 28,0 | 0,21 | <0,0001 |
| self_gen_ll | 85,4 | 27,6 | 1,0 | <0,0001 |
| best2_gen_ll | 85,2 | 24,0 | 1,0 | <0,0001 |

- **Ohne Regel-Checker funktioniert es auch:** `self_dec_min` holt **+3,0 Prozentpunkte** (signifikant), also knapp die Hälfte des Gewinns mit Regel-Checker (+6,2). Damit trifft Erwartung 4 zu.
- **`gen_ll` und `cycle` scheitern in der Schleife trotz guter AUROC. Das ist ein Designfehler von mir, kein Befund:** Ein greedy erzeugter Versuch hat per Konstruktion die höchste Likelihood seiner eigenen Codes. Ein gesampelter Versuch kann im Vergleich also praktisch nie gewinnen (`best2_gen_ll` ändert kein einziges Brett, die Schleife bleibt bei 85,2 hängen). Der Vergleich misst den Sampling-Pfad, nicht die Qualität des Bretts.
- **Fix für den nächsten Lauf:** `gen_llR` bewertet das *fertige Brett* neu, indem es kodiert und die Codes per Teacher Forcing durch den Generator geschickt werden. Das ist fair zwischen greedy und Samples, kostet aber einen zweiten Durchlauf pro Versuch.

## Lauf 2 mit fairer Likelihood (`results/sudoku-hard-self2.json`)

**Erkennung:** `gen_llR` (fertiges Brett neu kodiert und per Teacher Forcing bewertet) erreicht **AUROC 0,956**: 86,5 % der Fehler werden erkannt, bei nur **2,8 % Fehlalarmen**. Das ist nah am Regel-Checker. Zum Vergleich: dieselbe Likelihood auf dem Sampling-Pfad (`gen_ll`) lag bei 0,888 mit 23 % Fehlalarmen.

| Variante | gelöst % | Ebenen/Brett* |
|---|---|---|
| baseline | 85,2 | 12,0 |
| verifier (Regel-Checker) | 91,4 | 17,6 |
| **self_gen_llR** | **90,0** | 35,8 |
| self_dec_min | 88,2 | 17,1 |
| best2_gen_llR | 87,8 | 48,0 |
| best2_dec_min | 87,4 | 24,0 |

Gepaarte Tests (nur A gelöst / nur B gelöst, p):
- self_gen_llR vs. verifier: 6 / 13, **p = 0,17** (kein signifikanter Unterschied zum Regel-Checker)
- self_gen_llR vs. self_dec_min: 12 / 3, p = 0,035
- self_gen_llR vs. best2_gen_llR: 12 / 1, p = 0,0034 (gezielte Schleife schlägt blinde Auswahl)
- self_dec_min vs. best2_dec_min: 11 / 7, p = 0,48

\* Die Bewertung per Teacher Forcing läuft hier Ebene für Ebene und ist deshalb wie eine volle Generierung gezählt (+12 pro Versuch). Im Training rechnet NCM alle Ebenen in **einem** Forward Pass. Mit diesem Pfad würde die Bewertung grob einen Durchlauf kosten, und `self_gen_llR` läge eher bei ~19 statt 35,8 Ebenen pro Brett, also auf dem Niveau des Regel-Checkers. Das ist noch nicht implementiert, also eine Schätzung.

## Fazit Experiment 2

1. **NCM erkennt seine eigenen Fehler sehr gut**, wenn man das *fertige Ergebnis* bewertet statt den Weg dorthin (AUROC 0,96).
2. **Mit diesem internen Signal kommt die Schleife ohne Regel-Checker auf 90,0 %**, statistisch nicht vom Regel-Checker (91,4 %) zu unterscheiden. Das ist +4,8 Prozentpunkte gegenüber dem Paper-Setup, ohne jedes Training.
3. **Gezielt nachrechnen schlägt blind auswählen** (p = 0,0034). Der Gewinn kommt aus der Kombination aus Fehlererkennung und Rechenzeit nur dort, wo sie gebraucht wird.
4. Die Bewertung „Kodieren, dann Likelihood unter dem Generator“ ist im Kern eine **Vorhersagefehler-Messung**: Wie gut erklärt das generative Modell seine eigene Ausgabe? Das ist die Predictive-Coding-Idee, nur eben als Steuersignal und nicht als Korrektur der Repräsentation.

## Nächste Schritte

1. Bewertung in einem einzigen Forward Pass (Trainingspfad), um die Kosten korrekt zu messen.
2. Gegenprobe auf easy/medium und mit allen 2000 Brettern.
3. Übertragung auf Text (TinyStories): Funktioniert `gen_llR` dort als Signal für schlechte Generierungen?

---

# Tageslauf 22.10. (`./run_day.sh`)

**Technischer Hinweis:** Das MacBook Air (M1) hat sehr wenig gemeinsamen Grafik-/Arbeitsspeicher. Der Lauf **Sudoku hard (16:40–19:35) ist ungültig**: Wiederholt „Insufficient Memory“ auf der GPU, die Baseline weicht innerhalb desselben Laufs ab (84,3 % in Teil A gegen 82,8 % in Teil B auf denselben Brettern), am Ende Absturz. Auch der große Text-Lauf ist bei Batchgröße 64 abgestürzt. Beide laufen mit kleineren Batches und Speicher-Obergrenze neu (`./run_rest.sh`). Medium, easy und der Text-Schnelltest liefen ohne einen einzigen Speicherfehler.

Der Selbsttest der Ein-Pass-Bewertung war in allen Läufen bestanden (max. Abweichung zur Ebene-für-Ebene-Bewertung im Mittel < 1e-6).

## Sudoku medium (1000 Test- / 1000 Kalibrierungsbretter)

Erkennung: gen_llR **AUROC 0,964** (93,5 % der Fehler erkannt, 6,2 % Fehlalarm), dec_min 0,851, cycle 0,899.

| Variante | gelöst % | NFE/Brett | vs. baseline (nur A / nur B, p) |
|---|---|---|---|
| baseline | 96,9 | 12,0 | |
| verifier | 98,1 | 13,2 | 12 / 0, p = 0,0005 |
| **self_gen_llR** | **97,8** | 17,2 | 13 / 4, p = 0,049 |
| self_dec_min | 97,6 | 14,4 | 9 / 2, p = 0,065 |
| **rand_gen_llR** | **96,9** | 18,5 | **0 / 0**, p = 1 |
| best2_gen_llR | 97,4 | 26,0 | 6 / 1, p = 0,125 |

self_gen_llR vs. verifier: 5 / 8, p = 0,58 (kein Unterschied nachweisbar).

**Wichtigster neuer Befund, die Zufallskontrolle:** Genauso viele Neuversuche, aber zufällig verteilt, ändern **kein einziges** Ergebnis. Der Gewinn kommt also aus der **Erkennung**, nicht daraus, dass die Auswahl nach gen_llR an sich bessere Bretter findet.

## Sudoku easy

Die Baseline löst schon 99,7 %, keine Variante unterscheidet sich (alle p = 1). Schwachpunkt: Bei so wenigen Fehlern ist die auf den Kalibrierungsbrettern gelernte Schwelle für gen_llR zu locker. 19 % der Bretter werden neu versucht (NFE 22,9), obwohl nur 0,3 % falsch sind. Das kostet Rechenzeit, schadet aber nicht.

## Text (Schnelltest, nur 32 Geschichten, nicht aussagekräftig)

Läuft technisch durch (GPT-2 large in float16). Der volle Lauf mit 5 × 256 Geschichten steht noch aus (`./run_rest.sh`).

## Sudoku hard, Nachlauf (22.10., 22:34–23:03, ohne Speicherfehler)

1000 Test- / 1000 Kalibrierungsbretter, Bewertung in einem Durchlauf (Selbsttest bestanden). Ersetzt den ungültigen Lauf vom Nachmittag.

Erkennung: gen_llR **AUROC 0,954** (86,0 % der Fehler erkannt, 4,9 % Fehlalarm). dec_min 0,878, cycle 0,853, gen_ll 0,892.

| Variante | gelöst % | NFE/Brett | vs. baseline (nur A / nur B, p) |
|---|---|---|---|
| baseline | 84,3 | 12,0 | |
| verifier | 89,4 | 18,2 | 51 / 0, p < 0,0001 |
| **self_gen_llR** | **88,3** | **20,6** | 42 / 2, p < 0,0001 |
| self_dec_min | 87,5 | 17,2 | 34 / 2, p < 0,0001 |
| rand_gen_llR | 85,6 | 20,7 | 13 / 0, p = 0,0002 |
| best2_gen_llR | 85,9 | 26,0 | 18 / 2, p = 0,0004 |

Paarweise:
- self_gen_llR vs. **rand_gen_llR: 33 / 6, p < 0,0001** (gleicher Aufwand, gezielt gegen zufällig)
- self_gen_llR vs. best2_gen_llR: 29 / 5, p < 0,0001
- self_gen_llR vs. verifier: 11 / 22, p = 0,08
- self_gen_llR vs. self_dec_min: 23 / 15, p = 0,26

**Ehrliche Kosten:** Mit der Bewertung in einem Durchlauf braucht `self_gen_llR` **20,6 NFE pro Brett** statt 35,8 im alten Lauf. Das ist knapp über dem Regel-Checker (18,2) und ein Drittel billiger als `best2` (26,0), bei deutlich mehr gelösten Brettern.

**Was die Zufallskontrolle zeigt:** Auf hard bringen auch zufällige Neuversuche etwas (+1,3 Prozentpunkte), weil bei 16 % Fehlerquote auch zufällig gezogene Bretter manchmal falsch sind. Gezielt ist aber mit gleichem Aufwand **dreimal so wirksam** (+4,0 Prozentpunkte). Auf medium bringen zufällige Neuversuche gar nichts.

**Fazit über alle Schwierigkeitsgrade:** Das Modell erkennt seine Fehler zuverlässig (AUROC 0,95–0,96 auf hard und medium). Nur die Kombination aus Erkennung und gezieltem Nachrechnen bringt den Gewinn, und das bei einem Aufwand nahe dem Regel-Checker. Auf easy gibt es nichts mehr zu holen (Decken-Effekt).

## Text

Der volle Lauf ist auch im Nachlauf abgestürzt: GPU-Speicher beim Bewerten mit GPT-2 large (max. 3,7 GB erlaubt). Fix: NCM wird während der GPT-2-Bewertung auf die CPU ausgelagert und in Häppchen zu 2 bewertet (`./run_text.sh`).
