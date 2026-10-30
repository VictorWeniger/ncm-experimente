# Benchmarks: NCM, NCM + Nachbessern, Mischformen

## xsum

| Variante | ROUGE-L | Entropie / Länge | Δ (unabh. Maß) [95 %-KI] | GFLOPs | seq. Schritte |
|---|---|---|---|---|---|
| baseline | 0.00 | 0.000 | – | 825.63 (1.0×) | 9 |
| rerank4 | 0.00 | 0.000 | +0.000 [+0.000, +0.000] | 3790.39 (4.6×) | 40 |
| gated4 | 0.00 | 0.000 | +0.000 [+0.000, +0.000] | 1592.91 (1.9×) | 17 |
| rand4 | 0.00 | 0.000 | +0.000 [+0.000, +0.000] | 3302.53 (4.0×) | 36 |
| oracle4 | 0.00 | 0.000 | +0.000 [+0.000, +0.000] | 3302.53 (4.0×) | 36 |

Δ: ohne Bedingung log-Perplexity unter dem unabhängigen Bewertungsmodell (negativ = besser), mit Bedingung Satz-ROUGE-L bzw. Satz-chrF (positiv = besser). GFLOPs relativ zur NCM-Baseline in Klammern.
