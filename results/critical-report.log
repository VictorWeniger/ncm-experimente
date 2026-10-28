# Experiment 7a: Kritische Phasen (CIFAR-10, kleines CNN)

Test = Genauigkeit am Ende (scharfe Bilder). Verlust = Punkte schlechter als def0 (gleicher Seed).

| Lauf | Defizit (Epochen) | Lernrate | Seeds | Test % | Verlust ggü. def0 | Test auf unscharfen Bildern % |
|---|---|---|---|---|---|---|
| def0 | 0–0 | decay | 2 | 87.93 ± 0.08 | – | 19.3 |
| def5 | 0–5 | decay | 2 | 87.34 ± 0.21 | +0.58 | 21.8 |
| def10 | 0–10 | decay | 2 | 87.37 ± 0.25 | +0.56 | 24.3 |
| def20 | 0–20 | decay | 2 | 85.98 ± 0.09 | +1.94 | 26.6 |
| def30 | 0–30 | decay | 2 | 85.10 ± 0.23 | +2.82 | 32.1 |
| def40 | 0–40 | decay | 2 | 83.69 ± 0.29 | +4.23 | 35.5 |
| def20_reset | 0–20 | reset | 2 | 88.35 ± 0.11 | -0.42 | 23.8 |
| def40_reset | 0–40 | reset | 2 | 88.25 ± 0.48 | -0.32 | 22.9 |
| def0_const | 0–0 | const | 2 | 87.37 ± 0.06 | +0.56 | 18.1 |
| def20_const | 0–20 | const | 2 | 87.27 ± 0.12 | +0.66 | 21.1 |
| def40_const | 0–40 | const | 2 | 87.59 ± 0.10 | +0.34 | 20.5 |
| win10 | 10–20 | decay | 2 | 87.81 ± 0.11 | +0.12 | 21.4 |
| win30 | 30–40 | decay | 2 | 88.29 ± 0.21 | -0.37 | 21.2 |

## Fragen

- Dauerschaden durch frühes Defizit: def40 83.69% vs def0 87.93% (Differenz -4.23)
- Defizit später (Ep. 10–20) statt am Anfang (0–10): win10 87.81% vs def10 87.37% (Differenz +0.44)
- Defizit spät (Ep. 30–40) statt am Anfang: win30 88.29% vs def10 87.37% (Differenz +0.93)
- Plastizität wieder öffnen (Lernrate zurücksetzen) nach 40 Ep. Defizit: def40_reset 88.25% vs def40 83.69% (Differenz +4.56)
- dasselbe nach 20 Ep.: def20_reset 88.35% vs def20 85.98% (Differenz +2.36)
- Schaden bei konstanter Lernrate: def40_const − def0_const = +0.22 Punkte
- Fisher-Spur def0 (Seed 0): Maximum in Epoche 27. Verlauf (Epoche:Wert) 1:1.04e+03, 6:779, 11:1.75e+03, 16:1.89e+03, 21:1.68e+03, 26:1.87e+03, 31:1.95e+03
- Fisher-Spur def40 (Seed 0): Maximum in Epoche 67. Verlauf (Epoche:Wert) 1:626, 6:766, 11:1.29e+03, 16:1.68e+03, 21:2.48e+03, 26:2.58e+03, 31:4.14e+03, 36:3.53e+03, 41:1.79e+03, 46:2.59e+03, 51:2.75e+03, 56:3.16e+03, 61:3.8e+03, 66:4.67e+03, 71:3.45e+03
