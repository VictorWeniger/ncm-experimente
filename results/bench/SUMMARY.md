# Benchmarks: NCM, NCM + Nachbessern, Mischformen

## lm1b-small

| Variante | GenPPL (Paper-Protokoll) | Entropie / Länge | Δ (unabh. Maß) [95 %-KI] | GFLOPs | seq. Schritte |
|---|---|---|---|---|---|
| baseline | 92.59 | 0.693 | – | 123.56 (1.0×) | 11 |
| self | 86.08 | 0.693 | +0.000 [+0.000, +0.000] | 294.43 (2.4×) | 15 |
| rand | 89.95 | 0.693 | +0.000 [+0.000, +0.000] | 338.43 (2.7×) | 17 |
| best4 | 80.07 | 0.693 | +0.000 [+0.000, +0.000] | 962.64 (7.8×) | 48 |

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
