# lm1b-small

| Variante | Paper-GenPPL | Entropie | PPL gpt2-large | PPL SmolLM2-360M | Δlog-PPL gpt2-large [95 %-KI] | Δlog-PPL SmolLM2-360M [95 %-KI] | GFLOPs/Sample | seq. Schritte | angefasst % | Sekunden |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 92.59 ± 1.40 | 4.312 | 8488.83 | 9174.80 | – | – | 123.56 | 11 | 0 | 559 |
| self | 86.08 ± 1.52 | 4.311 | 8488.83 | 9174.80 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 294.43 | 15 | 19 | 690 |
| rand | 89.95 ± 1.19 | 4.309 | 8488.83 | 9174.80 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 338.43 | 17 | 20 | 796 |
| best4 | 80.07 ± 0.21 | 4.301 | 8488.83 | 9174.80 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 962.64 | 48 | 100 | 2353 |

Negativ bei Δlog-PPL = flüssiger als die NCM-Baseline. Entropie deutlich unter der Baseline = repetitiver.
