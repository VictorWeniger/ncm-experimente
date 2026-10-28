"""Fasst Experiment 6 zusammen: Mittelwert ± Standardabweichung über Seeds, plus Vorzeichentest
gegen die Vergleichsvariante pro Seed. Schreibt results/neurodyn-summary.md und .json.

    python experiments/neurodyn_report.py
"""
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

ORDER = ["rnn_det", "rnn_stoch", "tau_hom_det", "tau_hom_stoch", "tau_het_det", "tau_het_stoch",
         "tau_dyn_det", "tau_dyn_stoch", "gru"]
# Vergleiche, die die Vorab-Erwartungen in PLAN.md prüfen
PAIRS = [("tau_hom_det", "rnn_det", "Zeitkonstante überhaupt"),
         ("tau_het_det", "tau_hom_det", "heterogen statt gemeinsam"),
         ("tau_dyn_det", "tau_het_det", "inputabhängig statt fest"),
         ("tau_het_stoch", "tau_het_det", "Zufall (Bernoulli) bei heterogen"),
         ("rnn_stoch", "rnn_det", "Zufall (Bernoulli) ohne Zeitkonstante"),
         ("tau_dyn_det", "gru", "bestes Neuronenmodell gegen GRU")]


def sign_p(wins, n):
    """Zweiseitiger exakter Vorzeichentest (Unentschieden schon entfernt)."""
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    runs = defaultdict(dict)
    for f in glob.glob("results/neurodyn/*.json"):
        r = json.load(open(f))
        runs[(r["task"], r["variant"])][r["seed"]] = r
    tasks = sorted({t for t, _ in runs}, key=lambda t: ["freq", "recall", "adding"].index(t))
    lines, summary = ["# Experiment 6: Ergebnisse", ""], {}
    for t in tasks:
        mse = next(next(iter(runs[(t, v)].values())) for v in ORDER if (t, v) in runs)["metric"] == "mse"
        f = (lambda x: f"{x:.4f}") if mse else (lambda x: f"{100 * x:.1f}")
        unit = "MSE (kleiner = besser)" if mse else "Genauigkeit % (größer = besser)"
        lines += [f"## {t}: {unit}", "",
                  "| Variante | Seeds | Test | Rauschen | doppelte Länge | ohne Zufall | tau Median (q10–q90) |",
                  "|---|---|---|---|---|---|---|"]
        for v in ORDER:
            rs = list(runs.get((t, v), {}).values())
            if not rs:
                continue
            def ms(key):
                xs = [r[key] for r in rs if key in r]
                if not xs:
                    return "–", None
                m = st.mean(xs)
                sd = st.stdev(xs) if len(xs) > 1 else 0.0
                return f"{f(m)} ± {f(sd)}", m
            cells = [ms(k) for k in ("test", "test_noise", "test_long", "test_meanfield")]
            taus = [r["tau_q10_q50_q90"] for r in rs if "tau_q10_q50_q90" in r]
            tau = (f"{st.mean(q[1] for q in taus):.1f} ({st.mean(q[0] for q in taus):.1f}–{st.mean(q[2] for q in taus):.1f})"
                   if taus else "–")
            lines.append(f"| {v} | {len(rs)} | " + " | ".join(c[0] for c in cells) + f" | {tau} |")
            summary[f"{t}/{v}"] = {k: c[1] for k, c in zip(("test", "test_noise", "test_long", "test_meanfield"), cells)}
        lines += ["", "Vergleiche (gleiche Seeds, Vorzeichentest über Seeds × 3 Testbedingungen):", ""]
        for a_, b_, name in PAIRS:
            A, B = runs.get((t, a_), {}), runs.get((t, b_), {})
            seeds = sorted(set(A) & set(B))
            if not seeds:
                continue
            wins = n = 0
            diffs = []
            for s in seeds:
                for k in ("test", "test_noise", "test_long"):
                    d = A[s][k] - B[s][k]
                    if mse:
                        d = -d
                    diffs.append(d)
                    if abs(d) > 1e-9:
                        n += 1
                        wins += d > 0
            md = st.mean(diffs)
            md_s = f"{md:+.4f} MSE besser" if mse else f"{100 * md:+.1f} Punkte"
            lines.append(f"- {name} ({a_} vs {b_}): {wins}/{n} besser, Mittel {md_s}, p = {sign_p(wins, n):.3f}")
        lines.append("")
    os.makedirs("results", exist_ok=True)
    open("results/neurodyn-summary.md", "w").write("\n".join(lines))
    json.dump(summary, open("results/neurodyn-summary.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
