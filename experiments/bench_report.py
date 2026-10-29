"""Fasst alle results/bench/<task>/scores.json in einer Übersicht zusammen (results/bench/SUMMARY.md).

Aufruf: python experiments/bench_report.py
"""
import glob
import json
import os

from bench_common import ROOT

base = os.path.join(ROOT, "results", "bench")
out = ["# Benchmarks: NCM, NCM + Nachbessern, Mischformen", ""]
for f in sorted(glob.glob(os.path.join(base, "*", "scores.json"))):
    if "-smoke" in f:
        continue
    d = json.load(open(f))
    S = d["scores"]
    task = d["task"]
    if "rouge1" in S["baseline"]:
        main, unit = "rougeL", "ROUGE-L"
        delta = lambda s: s.get("d_sentence_metric")
    elif "bleu" in S["baseline"]:
        main, unit = "bleu", "BLEU"
        delta = lambda s: s.get("d_sentence_metric")
    else:
        main, unit = "paper_genppl", "GenPPL (Paper-Protokoll)"
        j2 = d["judges"][-1].split("/")[-1]
        delta = lambda s: s.get(f"dlogppl_{j2}")
    out += [f"## {task}", "", f"| Variante | {unit} | Entropie / Länge | Δ (unabh. Maß) [95 %-KI] | GFLOPs | seq. Schritte |",
            "|---|---|---|---|---|---|"]
    b = S["baseline"]
    for k, s in S.items():
        v = s.get(main)
        vs = f"{v:.2f}" if isinstance(v, (int, float)) else "–"
        ex = s.get("entropy", s.get("len_ratio", s.get("words", 0)))
        dd = delta(s)
        ds = "–" if not dd else f"{dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]"
        out.append(f"| {k} | {vs} | {ex:.3f} | {ds} | {s['gflops']:.2f} ({s['gflops'] / max(b['gflops'], 1e-9):.1f}×) | {s['steps']:.0f} |")
    out.append("")
out.append("Δ: ohne Bedingung log-Perplexity unter dem unabhängigen Bewertungsmodell (negativ = besser), "
           "mit Bedingung Satz-ROUGE-L bzw. Satz-chrF (positiv = besser). GFLOPs relativ zur NCM-Baseline in Klammern.")
txt = "\n".join(out)
open(os.path.join(base, "SUMMARY.md"), "w").write(txt + "\n")
print(txt)
