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
