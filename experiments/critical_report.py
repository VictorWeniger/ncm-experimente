"""Auswertung Experiment 7a: results/critical-summary.md
    python experiments/critical_report.py
"""
import glob
import json
import statistics as st
from collections import defaultdict


def main():
    R = defaultdict(dict)
    for f in glob.glob("results/critical/*.json"):
        r = json.load(open(f))
        R[r["name"]][r["seed"]] = r
    if not R:
        print("Noch keine Ergebnisse.")
        return
    base = R.get("def0", {})

    def row(name):
        rs = R.get(name, {})
        if not rs:
            return None
        acc = [r["test"] for r in rs.values()]
        m = st.mean(acc)
        sd = st.stdev(acc) if len(acc) > 1 else 0.0
        drops = [base[s]["test"] - r["test"] for s, r in rs.items() if s in base]
        drop = f"{100 * st.mean(drops):+.2f}" if drops and name != "def0" else "–"
        blur = st.mean(r["test_blurred_input"] for r in rs.values())
        r0 = next(iter(rs.values()))
        return (f"| {name} | {r0['deficit'][0]}–{r0['deficit'][1]} | {r0['schedule']} | {len(rs)} | "
                f"{100 * m:.2f} ± {100 * sd:.2f} | {drop} | {100 * blur:.1f} |")

    L = ["# Experiment 7a: Kritische Phasen (CIFAR-10, kleines CNN)", "",
         "Test = Genauigkeit am Ende (scharfe Bilder). Verlust = Punkte schlechter als def0 (gleicher Seed).", "",
         "| Lauf | Defizit (Epochen) | Lernrate | Seeds | Test % | Verlust ggü. def0 | Test auf unscharfen Bildern % |",
         "|---|---|---|---|---|---|---|"]
    groups = [[f"def{d}" for d in (0, 5, 10, 20, 30, 40)],
              ["def20_reset", "def40_reset"],
              ["def0_const", "def20_const", "def40_const"],
              ["win10", "win30"]]
    for gr in groups:
        for n in gr:
            x = row(n)
            if x:
                L.append(x)
    L += ["", "## Fragen", ""]

    def mean_test(n):
        rs = R.get(n, {})
        return st.mean(r["test"] for r in rs.values()) if rs else None

    def cmp(a, b, text):
        A, B = mean_test(a), mean_test(b)
        if A is not None and B is not None:
            L.append(f"- {text}: {a} {100 * A:.2f}% vs {b} {100 * B:.2f}% (Differenz {100 * (A - B):+.2f})")

    cmp("def40", "def0", "Dauerschaden durch frühes Defizit")
    cmp("win10", "def10", "Defizit später (Ep. 10–20) statt am Anfang (0–10)")
    cmp("win30", "def10", "Defizit spät (Ep. 30–40) statt am Anfang")
    cmp("def40_reset", "def40", "Plastizität wieder öffnen (Lernrate zurücksetzen) nach 40 Ep. Defizit")
    cmp("def20_reset", "def20", "dasselbe nach 20 Ep.")
    d_const = (mean_test("def0_const"), mean_test("def40_const"))
    if None not in d_const:
        L.append(f"- Schaden bei konstanter Lernrate: def40_const − def0_const = {100 * (d_const[1] - d_const[0]):+.2f} Punkte")
    for n in ("def0", "def40"):
        r = R.get(n, {}).get(0)
        if r and r["fisher"]:
            fi = r["fisher"]
            peak = max(range(len(fi)), key=lambda i: fi[i]) + 1
            pts = ", ".join(f"{i + 1}:{fi[i]:.3g}" for i in range(0, len(fi), 5))
            L.append(f"- Fisher-Spur {n} (Seed 0): Maximum in Epoche {peak}. Verlauf (Epoche:Wert) {pts}")
    open("results/critical-summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
