# Benchmarks: NCM, NCM + Nachbessern, Mischformen

## lm1b-small

| Variante | GenPPL (Paper-Protokoll) | Entropie / Länge | Δ (unabh. Maß) [95 %-KI] | GFLOPs | seq. Schritte |
|---|---|---|---|---|---|
| baseline | 92.59 | 2.437 | – | 123.56 (1.0×) | 11 |
| self | 86.08 | 2.455 | +nan [+nan, +nan] | 294.43 (2.4×) | 15 |
| rand | 89.95 | 2.446 | +nan [+nan, +nan] | 338.43 (2.7×) | 17 |
| best4 | 80.07 | 2.439 | +nan [+nan, +nan] | 962.64 (7.8×) | 48 |

## tinystories

| Variante | GenPPL (Paper-Protokoll) | Entropie / Länge | Δ (unabh. Maß) [95 %-KI] | GFLOPs | seq. Schritte |
|---|---|---|---|---|---|
| baseline | 16.15 | 3.970 | – | 112.77 (1.0×) | 11 |
| self | 14.36 | 3.963 | -0.121 [-0.139, -0.105] | 294.79 (2.6×) | 16 |
| rand | 15.45 | 3.968 | -0.047 [-0.056, -0.038] | 313.24 (2.8×) | 17 |
| best4 | 12.58 | 3.952 | -0.266 [-0.287, -0.246] | 890.98 (7.9×) | 48 |
| llm_only | 9.82 | 4.015 | -0.506 [-0.534, -0.478] | 18.05 (0.2×) | 134 |
| s1_replace | 12.90 | 3.975 | -0.224 [-0.249, -0.201] | 227.48 (2.0×) | 47 |
| s2_sentence | 15.70 | 3.973 | -0.016 [-0.020, -0.011] | 227.40 (2.0×) | 27 |
| s3_spec | 10.14 | 3.987 | -0.498 [-0.519, -0.477] | 250.58 (2.2×) | 91 |
| s3_pick4 | 12.16 | 3.953 | -0.301 [-0.320, -0.282] | 960.75 (8.5×) | 49 |
| s1_rewrite | 13.03 | 4.037 | -0.192 [-0.213, -0.171] | 313.34 (2.8×) | 54 |
| s1_rewrite_rand | 14.06 | 4.015 | -0.118 [-0.133, -0.102] | 302.47 (2.7×) | 48 |
| s1_rewrite_all | 10.00 | 4.171 | -0.456 [-0.477, -0.436] | 564.52 (5.0×) | 167 |
| merge_Cpar_k12 | 334.81 | 3.767 | +nan [+nan, +nan] | 387.73 (3.4×) | 23 |

## wmt14

| Variante | BLEU | Entropie / Länge | Δ (unabh. Maß) [95 %-KI] | GFLOPs | seq. Schritte |
|---|---|---|---|---|---|
| baseline | 21.64 | 0.936 | – | 258.61 (1.0×) | 11 |
| rerank4 | 19.42 | 1.024 | -1.864 [-2.254, -1.479] | 1334.28 (5.2×) | 48 |
| gated4 | 21.51 | 0.949 | -0.141 [-0.349, +0.074] | 509.89 (2.0×) | 18 |
| rand4 | 19.63 | 0.955 | -2.072 [-2.438, -1.712] | 1034.42 (4.0×) | 44 |
| oracle4 | 25.59 | 0.971 | +4.915 [+4.669, +5.176] | 1034.42 (4.0×) | 44 |

## xsum

| Variante | ROUGE-L | Entropie / Länge | Δ (unabh. Maß) [95 %-KI] | GFLOPs | seq. Schritte |
|---|---|---|---|---|---|
| baseline | 27.78 | 20.888 | – | 825.63 (1.0×) | 9 |
| rerank4 | 27.18 | 21.901 | -0.006 [-0.011, -0.001] | 3790.39 (4.6×) | 40 |
| gated4 | 27.73 | 21.108 | -0.000 [-0.003, +0.002] | 1453.61 (1.8×) | 15 |
| rand4 | 26.77 | 20.386 | -0.010 [-0.015, -0.005] | 3302.53 (4.0×) | 36 |
| oracle4 | 33.64 | 20.213 | +0.059 [+0.055, +0.062] | 3302.53 (4.0×) | 36 |

Δ: ohne Bedingung log-Perplexity unter dem unabhängigen Bewertungsmodell (negativ = besser), mit Bedingung Satz-ROUGE-L bzw. Satz-chrF (positiv = besser). GFLOPs relativ zur NCM-Baseline in Klammern.
