# xsum: Auswahl aus 4 Kandidaten (700 Mess-Dokumente; lambda und Gewichte auf 300 anderen gewählt)

| Regel | R-1 | R-2 | R-L | Δ Satzmetrik [95 %-KI] | trifft besten | anders als gierig |
|---|---|---|---|---|---|---|
| gierig | 35.05 | 13.13 | 28.24 | – | 28 % | 0 % |
| zufällig | 34.04 | 12.24 | 27.39 | -0.0085 [-0.0142, -0.0030] | 26 % | 76 % |
| orakel | 39.72 | 16.44 | 33.83 | +0.0559 [+0.0517, +0.0605] | 100 % | 72 % |
| gen_llR | 35.07 | 13.29 | 28.13 | -0.0012 [-0.0065, +0.0043] | 30 % | 66 % |
| pmi | 34.37 | 12.19 | 27.36 | -0.0088 [-0.0143, -0.0030] | 27 % | 72 % |
| pmi_fremd | 34.11 | 12.01 | 27.38 | -0.0086 [-0.0144, -0.0027] | 27 % | 74 % |
| mbr | 35.41 | 13.31 | 28.84 | +0.0060 [+0.0018, +0.0101] | 32 % | 55 % |
| gen_llR+2*pmi | 34.78 | 12.99 | 27.91 | -0.0033 [-0.0088, +0.0022] | 29 % | 70 % |
| kombi(pmi 0.5, mbr 1) | 35.34 | 13.16 | 28.70 | +0.0046 [+0.0001, +0.0087] | 32 % | 59 % |
| pmi nur verdächtige | 35.07 | 13.13 | 28.14 | -0.0011 [-0.0039, +0.0019] | 28 % | 12 % |
| kombi(pmi 0.5, mbr 1) nur verdächtige | 35.19 | 13.22 | 28.35 | +0.0011 [-0.0012, +0.0036] | 29 % | 11 % |
| mbr nur verdächtige | 35.09 | 13.19 | 28.34 | +0.0010 [-0.0012, +0.0031] | 29 % | 9 % |

Paarweise Trefferquote (welcher von zwei Kandidaten ist besser? 50 % = Zufall):

- gen_llR: 54.1 %
- gen_llR_leer: 54.1 %
- pmi: 51.2 %
- pmi_fremd: 50.1 %
- mbr: 57.9 %
