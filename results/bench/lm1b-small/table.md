# lm1b-small

| Variante | Paper-GenPPL | Entropie | PPL gpt2-large | PPL SmolLM2-360M | Δlog-PPL gpt2-large [95 %-KI] | Δlog-PPL SmolLM2-360M [95 %-KI] | GFLOPs/Sample | seq. Schritte | angefasst % | Sekunden |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 92.59 ± 1.40 | 4.312 | 2820.79 | 1746.12 | – | – | 123.56 | 11 | 0 | 559 |
| self | 86.08 ± 1.52 | 4.311 | 2324.79 | 1559.91 | -0.091 [-0.141, -0.041] | -0.094 [-0.148, -0.044] | 294.43 | 15 | 19 | 690 |
| rand | 89.95 ± 1.19 | 4.309 | 2776.11 | 1605.23 | -0.031 [-0.068, +0.007] | -0.033 [-0.076, +0.008] | 338.43 | 17 | 20 | 796 |
| best4 | 80.07 ± 0.21 | 4.301 | 1877.40 | 1431.06 | -0.098 [-0.191, -0.011] | -0.073 [-0.177, +0.021] | 962.64 | 48 | 100 | 2353 |

Negativ bei Δlog-PPL = flüssiger als die NCM-Baseline. Entropie deutlich unter der Baseline = repetitiver.
