# xsum: Auswahl aus 4 Kandidaten (17 Mess-Dokumente; lambda und Gewichte auf 7 anderen gewählt)

| Regel | R-1 | R-2 | R-L | Δ Satzmetrik [95 %-KI] | trifft besten | anders als gierig |
|---|---|---|---|---|---|---|
| gierig | 34.38 | 13.23 | 28.01 | – | 24 % | 0 % |
| zufällig | 33.57 | 10.04 | 24.42 | -0.0360 [-0.0687, -0.0025] | 12 % | 88 % |
| orakel | 39.24 | 14.77 | 32.59 | +0.0458 [+0.0272, +0.0664] | 100 % | 76 % |
| gen_llR | 34.93 | 13.59 | 26.99 | -0.0103 [-0.0292, +0.0058] | 24 % | 53 % |
| pmi | 33.32 | 11.04 | 25.92 | -0.0210 [-0.0494, +0.0036] | 29 % | 71 % |
| pmi_fremd | 32.14 | 10.99 | 24.54 | -0.0348 [-0.0650, -0.0092] | 18 % | 71 % |
| mbr | 35.62 | 12.27 | 27.94 | -0.0007 [-0.0268, +0.0228] | 29 % | 41 % |
| gen_llR+0.25*pmi | 35.40 | 13.08 | 27.89 | -0.0012 [-0.0252, +0.0183] | 29 % | 53 % |
| kombi(pmi 0, mbr 0.5) | 35.62 | 12.27 | 27.94 | -0.0007 [-0.0268, +0.0228] | 29 % | 41 % |
| pmi nur verdächtige | 34.79 | 12.72 | 27.98 | -0.0004 [-0.0070, +0.0046] | 35 % | 18 % |
| kombi(pmi 0, mbr 0.5) nur verdächtige | 35.20 | 12.72 | 28.10 | +0.0008 [+0.0000, +0.0025] | 29 % | 6 % |
| mbr nur verdächtige | 35.20 | 12.72 | 28.10 | +0.0008 [+0.0000, +0.0025] | 29 % | 6 % |

Paarweise Trefferquote (welcher von zwei Kandidaten ist besser? 50 % = Zufall):

- gen_llR: 51.5 %
- gen_llR_leer: 51.5 %
- pmi: 47.5 %
- pmi_fremd: 51.5 %
- mbr: 52.5 %
