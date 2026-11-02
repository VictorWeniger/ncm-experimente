"""Exp. 9: Ein Auswahlsignal für bedingte Aufgaben (XSum, WMT14).

Ausgangspunkt (bench_ncm.py): pro Dokument 4 Kandidaten (gierig + 3 Samples). Ein Orakel holt daraus
viel heraus, gen_llR wählt kaum besser als der Zufall. gen_llR fragt "wie typisch ist der Text für mich?",
nicht "wie gut passt er zu dieser Vorlage?".

Neue Signale (alle ohne Training):
  pmi     gen_llR mit Vorlage minus gen_llR mit leerer Vorlage: wie viel wahrscheinlicher wird der Text
          erst durch die Vorlage? (punktweise wechselseitige Information; allgemeine Sätze bekommen wenig)
  mbr     Konsens: mittlere Ähnlichkeit zu den anderen Kandidaten (ROUGE-L bzw. chrF), wie im Paper
  bt      nur WMT14: Kandidat zurück ins Deutsche übersetzen (opus-mt-en-de), chrF zur deutschen Quelle
Kombinationen: gen_llR + lambda * pmi, und Rang-Kombinationen. lambda und Gewichte werden auf den ersten
30 % der Dokumente gewählt und auf den übrigen 70 % gemessen (dort auch alle anderen Regeln).

Aufruf: python experiments/cond_select.py --task xsum
        python experiments/cond_select.py --task wmt14
Braucht results/bench/<task>/ncm.json aus bench_ncm.py.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from bench_common import ROOT, Task, bleu_scores, bootstrap_diff, free, rouge_scores

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["xsum", "wmt14"])
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--tune_frac", type=float, default=0.3)
ap.add_argument("--no_bt", action="store_true", help="Rückübersetzung auslassen")
ap.add_argument("--device", default="auto")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

task_dir = os.path.join(ROOT, "results", "bench", a.task)
src = json.load(open(os.path.join(task_dir, "ncm.json")))
C = src["extra"]["candidates"]  # (n, k) Texte, Kandidat 0 = gierig
refs = src["refs"]
thr = src["extra"]["threshold"]
n, k = len(C), len(C[0])
if a.smoke:
    n = 24
    C, refs = C[:n], refs[:n]
T = Task(a.task, a.device)
dev = T.dev
if dev != "cuda":
    a.bs = min(a.bs, 8)
pfx_all, _, srcs = T.load_data(len(src["refs"]))
pfx_all, srcs = pfx_all[:n], srcs[:n]
print(f"{a.task}: {n} Dokumente x {k} Kandidaten | Gerät {dev}", flush=True)


def to_raw(text):
    ids = T.tk.pre_tokenizer.encode(text)[:T.L - 1] + [T.EOS]
    return ids + [T.PAD] * (T.L - len(ids))


null = [T.EOS] + [T.net.prefix_pad_id if T.net.prefix_pad_id is not None else T.PAD] * (T.P - 1)


SHIFT = np.roll(np.arange(n), 1)  # Kandidaten von Dokument i mit dem Artikel von Dokument i-1 bewerten


@torch.no_grad()
def score_all(use_null, foreign=False):
    out = np.zeros((n, k))
    flat = [(i, j) for i in range(n) for j in range(k)]
    for s in range(0, len(flat), a.bs):
        ch = flat[s:s + a.bs]
        raw = torch.tensor([to_raw(C[i][j]) for i, j in ch], device=dev)
        p = torch.tensor([null if use_null else pfx_all[SHIFT[i] if foreign else i] for i, _ in ch], device=dev)
        v = T.score(p, raw).cpu().numpy()
        for (i, j), x in zip(ch, v):
            out[i, j] = x
        if dev == "mps" and (s // a.bs) % 50 == 0:
            free(dev)
    return out


t0 = time.time()
llr = score_all(False)
print(f"gen_llR mit Vorlage: {time.time() - t0:.0f}s", flush=True)
llr0 = score_all(True)
print(f"gen_llR ohne Vorlage: {time.time() - t0:.0f}s", flush=True)
llrF = score_all(False, foreign=True)
print(f"gen_llR mit fremdem Artikel: {time.time() - t0:.0f}s", flush=True)
pmi = llr - llr0
pmiF = llr - llrF  # wie viel besser passt der Text zum eigenen als zu einem fremden Artikel?
print(f"Diagnose: mittleres gen_llR mit eigener Vorlage {llr.mean():.4f}, leer {llr0.mean():.4f}, fremd {llrF.mean():.4f}; "
      f"Anteil der Kandidaten, die zur eigenen Vorlage besser passen als zur fremden: {(pmiF > 0).mean():.3f}", flush=True)
T.to("cpu")
del T
free(dev)

# ---------------------------------------------------------------- Satz-Metrik (für Orakel, MBR, Auswertung)
if a.task == "xsum":
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sim = lambda h, r: sc.score(r, h)["rougeL"].fmeasure
else:
    import sacrebleu
    sim = lambda h, r: sacrebleu.sentence_chrf(h, [r]).score / 100
gain = np.array([[sim(C[i][j], refs[i]) for j in range(k)] for i in range(n)])
mbr = np.array([[np.mean([sim(C[i][j], C[i][l]) for l in range(k) if l != j]) for j in range(k)] for i in range(n)])
print(f"Satz-Metrik und MBR: {time.time() - t0:.0f}s", flush=True)

signals = {"gen_llR": llr, "gen_llR_leer": llr0, "pmi": pmi, "pmi_fremd": pmiF, "mbr": mbr}

if a.task == "wmt14" and not a.no_bt:
    import sacrebleu
    from transformers import MarianMTModel, MarianTokenizer
    bt_name = "Helsinki-NLP/opus-mt-en-de"
    btok = MarianTokenizer.from_pretrained(bt_name)
    bmod = MarianMTModel.from_pretrained(bt_name).to(dev).eval()
    flat = [C[i][j] for i in range(n) for j in range(k)]
    back = []
    with torch.no_grad():
        for s in range(0, len(flat), 32):
            enc = btok(flat[s:s + 32], return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
            g = bmod.generate(**enc, num_beams=1, max_new_tokens=128)
            back += btok.batch_decode(g, skip_special_tokens=True)
    bt = np.array([[sacrebleu.sentence_chrf(back[i * k + j], [srcs[i]]).score / 100 for j in range(k)] for i in range(n)])
    signals["bt"] = bt
    del bmod
    free(dev)
    print(f"Rückübersetzung: {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- Regeln
rng = np.random.default_rng(0)
idx = rng.permutation(n)
n_tune = int(a.tune_frac * n)
tune, ev = np.sort(idx[:n_tune]), np.sort(idx[n_tune:])
# verdächtig nach dem gierigen gen_llR aus bench_ncm.py (gleiche Tokens wie bei der Schwelle; die hier neu
# tokenisierten Texte liegen systematisch höher und würden nie markiert)
flag = np.array(src["extra"]["gen_llR"][:n]) < thr
print(f"Abstand mit/ohne Vorlage: mittlerer |gen_llR - gen_llR_leer| = {np.abs(pmi).mean():.4f}", flush=True)


def zs(x):
    m, s = x.mean(1, keepdims=True), x.std(1, keepdims=True) + 1e-9
    return (x - m) / s


def pick(score):
    return score.argmax(1)


def mean_gain(p, rows):
    return gain[rows, p[rows]].mean()


rules = {"gierig": np.zeros(n, dtype=int), "zufällig": rng.integers(k, size=n), "orakel": gain.argmax(1)}
for name, sgl in signals.items():
    if name not in ("gen_llR_leer",):
        rules[name] = pick(sgl)
# lambda für gen_llR + lambda * pmi auf dem Abstimm-Teil wählen
grid = [0, 0.25, 0.5, 1, 2, 4, 8]
best_l = max(grid, key=lambda l: mean_gain(pick(llr + l * pmi), tune))
rules[f"gen_llR+{best_l}*pmi"] = pick(llr + best_l * pmi)
# Kombination mit MBR (z-standardisiert innerhalb jedes Dokuments)
combo_feats = {"pmi": zs(pmi), "mbr": zs(mbr)} | ({"bt": zs(signals["bt"])} if "bt" in signals else {})
wgrid = [0, 0.5, 1, 2]
best_w = None
for wp in wgrid:
    for wm in wgrid:
        for wb in (wgrid if "bt" in combo_feats else [0]):
            if wp == wm == wb == 0:
                continue
            s_ = wp * combo_feats["pmi"] + wm * combo_feats["mbr"] + (wb * combo_feats["bt"] if wb else 0)
            g = mean_gain(pick(s_), tune)
            if best_w is None or g > best_w[0]:
                best_w = (g, wp, wm, wb)
_, wp, wm, wb = best_w
s_ = wp * combo_feats["pmi"] + wm * combo_feats["mbr"] + (wb * combo_feats["bt"] if wb else 0)
cname = f"kombi(pmi {wp}, mbr {wm}" + (f", bt {wb})" if "bt" in combo_feats else ")")
rules[cname] = pick(s_)
# nur bei verdächtigen Dokumenten auswählen, sonst gierig
for base in ["pmi", cname, "mbr"]:
    rules[f"{base} nur verdächtige"] = np.where(flag, rules[base], 0)

# ---------------------------------------------------------------- Auswertung auf dem Mess-Teil
hyp = lambda p, rows: [C[i][p[i]] for i in rows]
ref_ev = [refs[i] for i in ev]
res = {"diagnose": {"llr_eigen": float(llr.mean()), "llr_leer": float(llr0.mean()), "llr_fremd": float(llrF.mean()),
                    "anteil_eigen_besser_als_fremd": float((pmiF > 0).mean()),
                    "mittlerer_abstand_eigen_leer": float(np.abs(pmi).mean())},
       "task": a.task, "n_docs": int(n), "n_eval": int(len(ev)), "k": k, "lambda": best_l,
       "kombi_gewichte": {"pmi": wp, "mbr": wm, "bt": wb}, "rules": {}}
base_sent = gain[ev, 0]
rows_out = []
for name, p in rules.items():
    h = hyp(p, ev)
    if a.task == "xsum":
        R = rouge_scores(h, ref_ev)
        main = {"rouge1": 100 * np.mean(R["rouge1"]), "rouge2": 100 * np.mean(R["rouge2"]),
                "rougeL": 100 * np.mean(R["rougeL"])}
    else:
        b, _ = bleu_scores(h, ref_ev)
        main = {"bleu": b["bleu"], "chrf": b["chrf"], "len_ratio": b["len_ratio"]}
    d = bootstrap_diff(gain[ev, p[ev]], base_sent) if name != "gierig" else (0.0, 0.0, 0.0)
    # wie oft trifft die Regel den besten Kandidaten bzw. einen über dem Median
    best_hit = float((gain[ev, p[ev]] >= gain[ev].max(1) - 1e-12).mean())
    changed = float((p[ev] != 0).mean())
    res["rules"][name] = main | {"d_sent": list(d), "trifft_besten": best_hit, "anders_als_gierig": changed}
    rows_out.append((name, main, d, best_hit, changed))

# Wie gut sagt jedes Signal innerhalb eines Dokuments den besseren Kandidaten voraus? (paarweise Trefferquote)
def pair_acc(sgl):
    hit = tot = 0
    for i in ev:
        for j in range(k):
            for l in range(j + 1, k):
                if abs(gain[i, j] - gain[i, l]) < 1e-9:
                    continue
                tot += 1
                hit += (sgl[i, j] > sgl[i, l]) == (gain[i, j] > gain[i, l])
    return hit / max(tot, 1)


res["paar_trefferquote"] = {name: pair_acc(sgl) for name, sgl in signals.items()}

main_key = "rougeL" if a.task == "xsum" else "bleu"
lines = [f"# {a.task}: Auswahl aus {k} Kandidaten ({len(ev)} Mess-Dokumente; lambda und Gewichte auf {n_tune} anderen gewählt)", "",
         f"| Regel | {'R-1 | R-2 | R-L' if a.task == 'xsum' else 'BLEU | chrF | Länge'} | Δ Satzmetrik [95 %-KI] | trifft besten | anders als gierig |",
         "|---|---|---|---|---|---|---|"]
for name, m, d, bh, ch in rows_out:
    vals = [m["rouge1"], m["rouge2"], m["rougeL"]] if a.task == "xsum" else [m["bleu"], m["chrf"], m["len_ratio"]]
    ds = "–" if name == "gierig" else "{:+.4f} [{:+.4f}, {:+.4f}]".format(*d)
    lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" if j < 2 else f"{v:.3f}" if a.task != 'xsum' else f"{v:.2f}"
                                          for j, v in enumerate(vals)) + f" | {ds} | {100 * bh:.0f} % | {100 * ch:.0f} % |")
lines += ["", "Paarweise Trefferquote (welcher von zwei Kandidaten ist besser? 50 % = Zufall):", ""]
lines += [f"- {k_}: {100 * v:.1f} %" for k_, v in res["paar_trefferquote"].items()]
table = "\n".join(lines)
print("\n" + table)
suffix = "-smoke" if a.smoke else ""
open(os.path.join(task_dir, f"select{suffix}.md"), "w").write(table + "\n")
json.dump(res, open(os.path.join(task_dir, f"select{suffix}.json"), "w"), indent=1, ensure_ascii=False)
print(f"\nGespeichert: {task_dir}/select{suffix}.md ({time.time() - t0:.0f}s)")
if a.smoke:
    print("SCHNELLTEST OK")
