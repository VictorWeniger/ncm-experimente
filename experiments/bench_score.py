"""Benchmark Teil 3: alle erzeugten Texte einer Aufgabe bewerten und als Tabelle ausgeben.

Liest results/bench/<task>/{ncm,hybrid,stage4,merge}.json (was vorhanden ist).

Ohne Bedingung:
  * Paper-Protokoll: GenPPL unter gpt2-large + Unigramm-Entropie, Mittel über die 5 Seeds
    (LM1B: FLM-Protokoll, nur für reine NCM-Varianten möglich, weil es die Roh-Tokens braucht)
  * Text-Protokoll für alle Varianten gleich: Perplexity pro Text unter jedem Bewertungsmodell
    (--judges). Das zweite Modell stammt aus einer anderen Familie als GPT-2, damit Mischformen mit
    GPT-artigen Modellen nicht von ihrer Verwandtschaft mit dem Prüfer profitieren.
  * gepaarter Bootstrap der log-Perplexity gegen die NCM-Baseline
Mit Bedingung: ROUGE-1/2/L + GenPPL (XSum) bzw. BLEU, chrF, Längenverhältnis (WMT14) wie im Paper,
  gepaarter Bootstrap auf Satz-ROUGE-L bzw. Satz-chrF.
Dazu Kosten: GFLOPs/Sample (Näherung), sequenzielle Schritte (NCM-Ebenen + LLM-Tokens), Sekunden.

Außerdem: pairs.jsonl mit blinden Paaren (Baseline gegen jede Variante, Reihenfolge zufällig) für ein
paarweises Urteil durch einen Menschen oder ein großes LLM; die Auflösung steht in pairs_key.json.

Aufruf: python experiments/bench_score.py --task lm1b-small
"""
import argparse
import json
import math
import os
import random

import numpy as np
import torch

from bench_common import (ROOT, RefLM, Task, bleu_scores, bootstrap_diff, entropy_ids, free, pick_device,
                          rouge_scores)

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--judges", default="gpt2-large,HuggingFaceTB/SmolLM2-360M",
                help="Bewertungsmodelle für das Text-Protokoll (erstes = Paper-Referenz)")
ap.add_argument("--ref_bs", type=int, default=4)
ap.add_argument("--n_pairs", type=int, default=100)
ap.add_argument("--device", default="auto")
ap.add_argument("--in_dir", default=os.path.join(ROOT, "results", "bench"))
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

dev = pick_device(a.device)
task_dir = os.path.join(a.in_dir, a.task + ("-smoke" if a.smoke else ""))
V, meta = {}, {}
for f in ("ncm", "hybrid", "stage4", "merge"):
    p = os.path.join(task_dir, f + ".json")
    if os.path.exists(p):
        d = json.load(open(p))
        meta[f] = {k: v for k, v in d.items() if k not in ("variants", "extra", "refs")}
        for k, v in d["variants"].items():
            V[k if f == "ncm" else k] = dict(v, source=f)
        if f == "ncm":
            ncm = d
assert "baseline" in V, "Erst bench_ncm.py laufen lassen."
uncond = ncm["uncond"]
metric = ncm["metric"]
judges = [j for j in a.judges.split(",") if j]
print(f"{a.task}: {len(V)} Varianten: {', '.join(V)}", flush=True)

T = Task(a.task, "cpu")  # nur für den Tokenizer (Entropie, FLM)
pre = T.tk.pre_tokenizer
del T.net
free("cpu")
S = {k: {"source": v["source"], "gflops": float(np.mean(v["flops"])) / 1e9, "steps": float(np.mean(v["nfe"])),
         "seconds": v.get("seconds", 0.0), "retried": 100 * float(np.mean(v.get("retried", [0])))} for k, v in V.items()}

if uncond:
    seeds = V["baseline"]["seed"]
    useeds = list(dict.fromkeys(seeds))
    # -------- Paper-Protokoll
    if metric == "genppl_flm":
        from transformers import AutoTokenizer
        gen_tok = AutoTokenizer.from_pretrained(str(T.CFG.tokenizer.pre_tokenizer.model_id))
        ref = RefLM(str(T.CFG.eval.ref_model), dev)
        for k, v in V.items():
            if "raw_ids" not in v:
                continue
            per = []
            for s in useeds:
                sel = [i for i, x in enumerate(v["seed"]) if x == s]
                texts = gen_tok.batch_decode([v["raw_ids"][i] for i in sel])
                ppl, _ = ref.corpus_ppl_flm(texts, bs=a.ref_bs)
                ent = float(np.mean([entropy_ids(v["raw_ids"][i]) for i in sel]))
                per.append((ppl, ent))
            S[k]["paper_genppl"] = float(np.mean([p[0] for p in per]))
            S[k]["paper_genppl_sd"] = float(np.std([p[0] for p in per], ddof=1)) if len(per) > 1 else 0.0
            S[k]["paper_entropy"] = float(np.mean([p[1] for p in per]))
            print(f"  FLM {k:<16} {S[k]['paper_genppl']:8.2f} ± {S[k]['paper_genppl_sd']:.2f} | Entropie {S[k]['paper_entropy']:.3f}", flush=True)
        ref.close()
    # -------- Text-Protokoll, jedes Bewertungsmodell
    logppl = {}
    for jn in judges:
        J = RefLM(jn, dev)
        short = jn.split("/")[-1]
        logppl[short] = {}
        for k, v in V.items():
            texts = [t if t.strip() else "." for t in v["texts"]]
            if metric != "genppl_flm" and jn == judges[0]:
                # exakt wie das Paper für TinyStories/OWT: generation_perplexity pro Seed
                lp = torch.zeros(len(texts), dtype=torch.float64)
                per_seed = []
                for s_ in useeds:
                    sel = [i for i, x in enumerate(v["seed"]) if x == s_]
                    ppl_i, mp = J.genppl_paper([texts[i] for i in sel], bs=a.ref_bs)
                    lp[sel] = ppl_i.double().log()
                    per_seed.append(mp)
            else:
                nt = J.nll_per_text(texts, bs=a.ref_bs, max_len=1024)
                lp = torch.tensor([s_ / max(n, 1) for s_, n in nt], dtype=torch.float64)
                per_seed = [lp[[i for i, x in enumerate(v["seed"]) if x == s_]].exp().mean().item() for s_ in useeds]
            logppl[short][k] = lp
            S[k][f"ppl_{short}"] = float(np.mean(per_seed))
            S[k][f"ppl_{short}_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
            if jn == judges[0] and metric != "genppl_flm":
                S[k]["paper_genppl"], S[k]["paper_genppl_sd"] = S[k][f"ppl_{short}"], S[k][f"ppl_{short}_sd"]
            print(f"  {short:<14} {k:<16} {S[k][f'ppl_{short}']:8.2f}", flush=True)
        J.close()
        for k in V:
            if k == "baseline":
                continue
            m, lo, hi = bootstrap_diff(logppl[short][k], logppl[short]["baseline"])
            S[k][f"dlogppl_{short}"] = [m, lo, hi]
    # Entropie (Paper: Tokens des NCM-Tokenizers; für alle Varianten gleich berechenbar)
    for k, v in V.items():
        e = [entropy_ids(pre.encode(t)) for t in v["texts"]]
        S[k]["entropy"] = float(np.mean(e))
        if metric != "genppl_flm":
            S[k]["paper_entropy"] = S[k]["entropy"]
        S[k]["words"] = float(np.mean([len(t.split()) for t in v["texts"]]))
else:
    refs = ncm["refs"]
    per_sent = {}
    if metric == "rouge":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        J = RefLM("gpt2", dev)
        for k, v in V.items():
            R = rouge_scores(v["texts"], refs)
            S[k].update({"rouge1": 100 * np.mean(R["rouge1"]), "rouge2": 100 * np.mean(R["rouge2"]),
                         "rougeL": 100 * np.mean(R["rougeL"]),
                         "words": float(np.mean([len(h.split()) for h in v["texts"]]))})
            _, gp = J.genppl_paper([t if t.strip() else "." for t in v["texts"]], bs=a.ref_bs * 4)
            S[k]["genppl_gpt2"] = gp
            per_sent[k] = R["rougeL"]
            print(f"  {k:<12} R1 {S[k]['rouge1']:.2f} R2 {S[k]['rouge2']:.2f} RL {S[k]['rougeL']:.2f} GenPPL {gp:.1f}", flush=True)
        J.close()
    else:
        for k, v in V.items():
            b, sent = bleu_scores(v["texts"], refs)
            S[k].update(b)
            per_sent[k] = sent
            print(f"  {k:<12} BLEU {b['bleu']:.2f} (lc {b['bleu_lc']:.2f}) chrF {b['chrf']:.2f} Länge {b['len_ratio']:.3f}", flush=True)
    for k in V:
        if k != "baseline":
            S[k]["d_sentence_metric"] = list(bootstrap_diff(per_sent[k], per_sent["baseline"]))

# -------- blinde Paare
rng = random.Random(0)
base_t = V["baseline"]["texts"]
pairs, key = [], []
ids = rng.sample(range(len(base_t)), min(a.n_pairs, len(base_t)))
for k, v in V.items():
    if k == "baseline":
        continue
    for i in ids:
        if i >= len(v["texts"]) or v["texts"][i] == base_t[i]:
            continue
        flip = rng.random() < 0.5
        A, B = (v["texts"][i], base_t[i]) if flip else (base_t[i], v["texts"][i])
        pid = f"{k}-{i}"
        item = {"id": pid, "A": A, "B": B}
        if not uncond:
            item["reference"] = ncm["refs"][i]
        pairs.append(item)
        key.append({"id": pid, "variant": k, "variant_is": "A" if flip else "B"})
rng.shuffle(pairs)
with open(os.path.join(task_dir, "pairs.jsonl"), "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
json.dump(key, open(os.path.join(task_dir, "pairs_key.json"), "w"), indent=1)

# -------- Tabelle
lines = [f"# {a.task}", ""]
if uncond:
    js = [j.split("/")[-1] for j in judges]
    head = ["Variante", "Paper-GenPPL", "Entropie"] + [f"PPL {j}" for j in js] + [f"Δlog-PPL {j} [95 %-KI]" for j in js] + \
           ["GFLOPs/Sample", "seq. Schritte", "angefasst %", "Sekunden"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k, s in S.items():
        pg = f"{s['paper_genppl']:.2f} ± {s['paper_genppl_sd']:.2f}" if "paper_genppl" in s else "–"
        row = [k, pg, f"{s.get('paper_entropy', s['entropy']):.3f}"]
        row += [f"{s[f'ppl_{j}']:.2f}" for j in js]
        row += [("–" if k == "baseline" else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*s[f"dlogppl_{j}"])) for j in js]
        row += [f"{s['gflops']:.2f}", f"{s['steps']:.0f}", f"{s['retried']:.0f}", f"{s['seconds']:.0f}"]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Negativ bei Δlog-PPL = flüssiger als die NCM-Baseline. Entropie deutlich unter der Baseline = repetitiver."]
else:
    if metric == "rouge":
        head = ["Variante", "R-1", "R-2", "R-L", "GenPPL", "Wörter", "Δ Satz-R-L [95 %-KI]", "GFLOPs/Dok", "seq. Schritte", "angefasst %"]
    else:
        head = ["Variante", "BLEU", "BLEU lc", "chrF", "Länge", "Δ Satz-chrF [95 %-KI]", "GFLOPs/Dok", "seq. Schritte", "angefasst %"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k, s in S.items():
        d = "–" if k == "baseline" else "{:+.4f} [{:+.4f}, {:+.4f}]".format(*s["d_sentence_metric"])
        if metric == "rouge":
            row = [k, f"{s['rouge1']:.2f}", f"{s['rouge2']:.2f}", f"{s['rougeL']:.2f}", f"{s['genppl_gpt2']:.1f}",
                   f"{s['words']:.1f}", d]
        else:
            row = [k, f"{s['bleu']:.2f}", f"{s['bleu_lc']:.2f}", f"{s['chrf']:.2f}", f"{s['len_ratio']:.3f}", d]
        row += [f"{s['gflops']:.2f}", f"{s['steps']:.0f}", f"{s['retried']:.0f}"]
        lines.append("| " + " | ".join(row) + " |")
table = "\n".join(lines)
print("\n" + table)
open(os.path.join(task_dir, "table.md"), "w").write(table + "\n")
json.dump({"task": a.task, "judges": judges, "scores": S, "meta": meta}, open(os.path.join(task_dir, "scores.json"), "w"), indent=1)
print(f"\nGespeichert: {task_dir}/scores.json, table.md, pairs.jsonl ({len(pairs)} blinde Paare)")
if a.smoke:
    print("SCHNELLTEST OK")
