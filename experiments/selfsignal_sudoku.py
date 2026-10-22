"""Merkt NCM selbst, wann es falsch liegt? Fehlersignale ohne externen Regel-Checker.

Teil A (Erkennung): Für die Baseline-Lösungen wird für jedes Brett eine Reihe
modellinterner Signale berechnet. Gemessen wird, wie gut jedes Signal gelöste
von ungelösten Brettern trennt (AUROC).

Teil B (Schleife): Die fehlergetriebene Neu-Sampling-Schleife aus dem letzten
Experiment (k0, T=0,7, 4 Runden), aber gesteuert durch ein internes Signal statt
durch den Regel-Checker:
  - ein Brett gilt als "verdächtig", wenn sein Signal unter einer Schwelle liegt
    (Schwelle auf separaten Kalibrierungs-Brettern festgelegt, nicht auf den Testbrettern)
  - ein neuer Versuch wird nur übernommen, wenn sein Signal besser ist

Signale (höher = Modell ist sich sicherer):
  dec_min   kleinste Decoder-Wahrscheinlichkeit über die 81 Zellen (die unsicherste Zelle)
  dec_mean  mittlere log-Wahrscheinlichkeit des Decoders über die Zellen
  gen_ll    mittlere log-Wahrscheinlichkeit der gewählten Codes laut Generator
  gen_llR   wie gen_ll, aber für das fertige Brett neu bewertet: Brett kodieren, dann die Codes per
            Teacher Forcing durch den Generator schicken. Fair zwischen greedy und gesampelten
            Versuchen (gen_ll benachteiligt Samples systematisch). Kostet einen extra Durchlauf.
  gen_min   kleinste Wahrscheinlichkeit eines gewählten Codes
  cycle     Selbstkonsistenz: Brett neu kodieren, Anteil gleicher Codes wie bei der Generierung
  clue      Widerspricht die Ausgabe den Vorgaben im Rätsel? (Abgleich mit dem Input, keine Regeln)
  rules     Regel-Checker (Orakel, nur als Referenz)

Aufruf:
    python experiments/selfsignal_sudoku.py --task hard --n 500 --tag self
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCM = os.path.join(ROOT, "ncm")
sys.path.insert(0, NCM)
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

SEP, BOS = 10, 11
BOARD_LEN = 89
SIGNALS = ["dec_min", "dec_mean", "gen_ll", "gen_llR", "gen_min", "cycle", "clue", "rules"]
DEFAULT_VARIANTS = ["baseline", "verifier",
                    "self_dec_min", "self_gen_llR",
                    "best2_dec_min", "best2_gen_llR"]

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="hard", choices=["easy", "medium", "hard"])
ap.add_argument("--n", type=int, default=500, help="Testbretter")
ap.add_argument("--n_cal", type=int, default=500, help="Kalibrierungsbretter (direkt nach den Testbrettern)")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
ap.add_argument("--device", default="auto")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(ROOT, "results"))
ap.add_argument("--tag", default="self2")
a = ap.parse_args()

dev = a.device if a.device != "auto" else (
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
variants = a.variants.split(",")

# MPS: interpolate(mode="area") nur bei teilbaren Längen -> auf CPU ausweichen
_F_interp = torch.nn.functional.interpolate


def _interp_mps_safe(input, *args, **kwargs):
    mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
    if input.device.type == "mps" and mode == "area":
        return _F_interp(input.cpu(), *args, **kwargs).to(input.device)
    return _F_interp(input, *args, **kwargs)


torch.nn.functional.interpolate = _interp_mps_safe

# ---------------------------------------------------------------- Setup
task = f"sudoku-{a.task}"
CFG = OmegaConf.load(os.path.join(NCM, "config", "eval", f"{task}.yaml"))
ck = os.path.join(ROOT, "artifacts", "sudoku", a.task)
dd = next(d for d in [os.path.join(ROOT, "data", "sudoku", a.task), os.path.join(ROOT, "data", "sudoku")]
          if os.path.exists(os.path.join(d, CFG.data.test)))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(dev).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model)
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net = net.to(dev).eval()
vq = tk.vqvae.vq
LEVELS = list(vq.levels)
K = len(LEVELS)
P = int(CFG.model.prefix_len)

all_items = pickle.load(open(os.path.join(dd, CFG.data.test), "rb"))
all_refs = pickle.load(open(os.path.join(dd, CFG.data.test_refs), "rb"))
dec = lambda ids: tk.pre_tokenizer.decode(list(ids), skip_special_tokens=True)
all_refs = [r if isinstance(r, str) else dec(r) for r in all_refs]
assert len(all_items) >= a.n + a.n_cal, f"nur {len(all_items)} Bretter vorhanden"
TEST = (all_items[:a.n], all_refs[:a.n])
CAL = (all_items[a.n:a.n + a.n_cal], all_refs[a.n:a.n + a.n_cal])


def prefix(item):
    ids = list(item)
    return (ids + [ids[-1]] * P)[:P]


def clue_board(item):
    ids = list(item)
    i = 0
    while i < len(ids) and ids[i] == BOS:
        i += 1
    return ids[i:i + BOARD_LEN]


CELLS = torch.tensor([j for j in range(BOARD_LEN) if (j + 1) % 10 != 0], device=dev)
_g = [[r * 9 + c for c in range(9)] for r in range(9)] + [[r * 9 + c for r in range(9)] for c in range(9)]
_g += [[(br + i) * 9 + bc + j for i in range(3) for j in range(3)] for br in (0, 3, 6) for bc in (0, 3, 6)]
GROUPS = torch.tensor(_g, device=dev)


def rule_errors(board):
    g = board[:, CELLS]
    bad = ((g < 1) | (g > 9)).sum(1)
    oh = F.one_hot(g[:, GROUPS].clamp(0, 10), 11)[..., 1:10].sum(2)
    return bad + (oh - 1).clamp_min(0).sum((1, 2))


def clue_errors(board, clues):
    return ((clues > 0) & (board != clues)).sum(1)


# ---------------------------------------------------------------- Generierung mit Buchführung
_orig_sample = net._sample_level


def generate(pfx, temp, forced=None):
    """Generiert alle Ebenen und merkt sich die (untemperierte) Wahrscheinlichkeit jedes gewählten Codes.
    Mit forced (Liste aller Ebenen) wird nichts gesampelt, sondern die vorgegebenen Codes werden bewertet (Teacher Forcing)."""
    logps = []

    def sample(logits, temperature, top_k, top_p):
        idx = forced[len(logps)] if forced is not None else _orig_sample(logits, temp, 0, 1.0)
        lp = F.log_softmax(logits.float(), -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        logps.append(lp)
        return idx

    net._sample_level = sample
    try:
        with torch.no_grad():
            g = net.generate(None, pfx, temperature=temp, top_k=0, top_p=1.0, cfg_scale=0.0)
    finally:
        del net._sample_level
    return g, torch.cat(logps, dim=1)


def full_z(idx):
    z = torch.zeros(idx.shape[0], LEVELS[-1], vq.embed_dim, device=idx.device)
    s = 0
    for li, l in enumerate(LEVELS):
        e = vq.embeddings[li](idx[:, s:s + l])
        q = vq._up_to_full(e, li, LEVELS[-1]) if li != K - 1 else e
        z = z + vq._apply_quant_resi(q, li)
        s += l
    return z


def split_levels(idx):
    out, s = [], 0
    for l in LEVELS:
        out.append(idx[:, s:s + l])
        s += l
    return out


def attempt(pfx, clues, temp):
    """Ein kompletter Lösungsversuch: Brett (Vorgaben eingesetzt) + alle Signale."""
    g, code_lp = generate(pfx, temp)
    with torch.no_grad():
        logits = tk.vqvae.decoder(full_z(g))
        raw = logits.argmax(-1)
        cell_lp = F.log_softmax(logits.float(), -1).max(-1).values[:, CELLS]
        re = tk.vqvae.encode(raw)
    board = torch.where(clues > 0, clues, raw)
    sig = {
        "dec_min": cell_lp.min(1).values.exp(),
        "dec_mean": cell_lp.mean(1),
        "gen_ll": code_lp.mean(1),
        "gen_llR": generate(pfx, 0.0, forced=split_levels(re))[1].mean(1),
        "gen_min": code_lp.min(1).values.exp(),
        "cycle": (re == g).float().mean(1),
        "clue": -clue_errors(raw, clues).float(),
        "rules": -rule_errors(board).float(),
    }
    return board, sig


def solved_mask(boards, refs):
    return torch.tensor([dec(b.tolist()) == r for b, r in zip(boards, refs)], dtype=torch.bool)


def auroc(score, pos):
    """P(Score eines gelösten Bretts > Score eines ungelösten), Gleichstände zählen halb."""
    s, p = score.double(), pos.bool()
    sp, sn = s[p], s[~p]
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    gt = (sp[:, None] > sn[None, :]).double().mean()
    eq = (sp[:, None] == sn[None, :]).double().mean()
    return (gt + 0.5 * eq).item()


def batches(items):
    for s in range(0, len(items), a.bs):
        yield s, torch.tensor([prefix(it) for it in items[s:s + a.bs]], device=dev), \
            torch.tensor([clue_board(it) for it in items[s:s + a.bs]], device=dev)


def baseline_pass(items, refs):
    boards, sigs = [], {k: [] for k in SIGNALS}
    for _, pfx, clues in batches(items):
        b, sg = attempt(pfx, clues, 0.0)
        boards.append(b.cpu())
        for k in SIGNALS:
            sigs[k].append(sg[k].cpu())
    boards = torch.cat(boards)
    return boards, {k: torch.cat(v) for k, v in sigs.items()}, solved_mask(boards, refs)


def best_threshold(score, solved):
    """Schwelle, die TPR-FPR (Youden) für 'ungelöst' maximiert. Bretter mit Score < Schwelle gelten als verdächtig."""
    cands = torch.unique(score)
    best, best_t = -1.0, cands[0].item()
    bad = ~solved
    for t in cands.tolist():
        flag = score < t
        tpr = (flag & bad).sum().item() / max(bad.sum().item(), 1)
        fpr = (flag & solved).sum().item() / max(solved.sum().item(), 1)
        if tpr - fpr > best:
            best, best_t = tpr - fpr, t
    return best_t, best


# ---------------------------------------------------------------- Varianten
def better(name, new, old):
    """Wird der neue Versuch übernommen?"""
    if name == "verifier":
        return new["rules"] >= old["rules"]
    sig = name.split("_", 1)[1].replace("+clue", "")
    if "+clue" in name:
        return (new["clue"] > old["clue"]) | ((new["clue"] == old["clue"]) & (new[sig] > old[sig]))
    return new[sig] > old[sig]


def suspicious(name, sg, thr):
    if name == "verifier":
        return sg["rules"] < 0
    sig = name.split("_", 1)[1].replace("+clue", "")
    flag = sg[sig] < thr[sig]
    if "+clue" in name:
        flag = flag | (sg["clue"] < 0)
    return flag


def run_variant(name, pfx, clues, thr):
    B = pfx.shape[0]
    board, sg = attempt(pfx, clues, 0.0)
    step = 2 * K if "gen_llR" in name else K  # Teacher-Forcing-Bewertung kostet einen zweiten Durchlauf
    cost = torch.full((B,), float(step), device=dev)
    if name == "baseline":
        return board, cost, [board.clone()], torch.zeros(B, dtype=torch.bool, device=dev)
    if name.startswith("best2_"):
        sig = name[len("best2_"):]
        b2, s2 = attempt(pfx, clues, a.temp)
        take = s2[sig] > sg[sig]
        return torch.where(take[:, None], b2, board), cost + step, [board.clone()], torch.zeros(B, dtype=torch.bool, device=dev)
    hist = [board.clone()]
    ever_flagged = suspicious(name, sg, thr).clone()
    for _ in range(a.rounds):
        act = suspicious(name, sg, thr)
        ever_flagged |= act
        if act.any():
            ai = act.nonzero().squeeze(1)
            nb, ns = attempt(pfx[ai], clues[ai], a.temp)
            old = {k: v[ai] for k, v in sg.items()}
            acc = better(name, ns, old)
            board[ai] = torch.where(acc[:, None], nb, board[ai])
            for k in sg:
                sg[k][ai] = torch.where(acc, ns[k], sg[k][ai])
            cost[ai] += step
        hist.append(board.clone())
    return board, cost, hist, ever_flagged


def mcnemar(a_mask, b_mask):
    n10 = int((a_mask & ~b_mask).sum())
    n01 = int((~a_mask & b_mask).sum())
    n, k = n10 + n01, min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n10, n01, p


# ---------------------------------------------------------------- Lauf
os.makedirs(a.out, exist_ok=True)
out_path = os.path.join(a.out, f"{task}-{a.tag}.json")
print(f"Gerät: {dev} | {task} | Test: {a.n} | Kalibrierung: {a.n_cal} | T={a.temp} | Runden={a.rounds}")
summary = {"task": task, "n": a.n, "n_cal": a.n_cal, "temp": a.temp, "rounds": a.rounds}

# Teil A: Erkennung auf den Testbrettern + Schwellen auf den Kalibrierungsbrettern
t0 = time.time()
_, sig_test, solved_test = baseline_pass(*TEST)
_, sig_cal, solved_cal = baseline_pass(*CAL)
thr, detection = {}, {}
print(f"\nTeil A: Erkennt das Modell seine Fehler? (Baseline gelöst: Test {solved_test.float().mean() * 100:.1f} %, "
      f"Kalibrierung {solved_cal.float().mean() * 100:.1f} %, {time.time() - t0:.0f}s)")
print(f"{'Signal':<10}{'AUROC':>7}{'Schwelle':>12}{'markiert %':>12}{'Treffer %':>11}{'Fehlalarm %':>13}")
for k in SIGNALS:
    t, _ = best_threshold(sig_cal[k], solved_cal)
    thr[k] = t
    flag = sig_test[k] < t
    bad = ~solved_test
    tpr = (flag & bad).sum().item() / max(bad.sum().item(), 1) * 100
    fpr = (flag & solved_test).sum().item() / max(solved_test.sum().item(), 1) * 100
    detection[k] = {"auroc": auroc(sig_test[k], solved_test), "threshold": t,
                    "flagged": flag.float().mean().item() * 100, "tpr": tpr, "fpr": fpr}
    d = detection[k]
    print(f"{k:<10}{d['auroc']:7.3f}{t:12.4f}{d['flagged']:12.1f}{tpr:11.1f}{fpr:13.1f}")
summary["detection"] = detection
json.dump(summary, open(out_path, "w"), indent=2)

# Teil B: Schleifen
print(f"\nTeil B: Schleife mit internem Fehlersignal")
print(f"{'Variante':<20}{'gelöst %':>9}{'Ebenen/Brett':>14}{'je versucht %':>15}{'Zeit s':>8}   gelöst je Runde")
items, refs = TEST
results, masks = {}, {}
for name in variants:
    torch.manual_seed(a.seed)
    t0 = time.time()
    finals, costs, hists, flagged = [], [], None, []
    for _, pfx, clues in batches(items):
        b, c, h, f = run_variant(name, pfx, clues, thr)
        finals.append(b.cpu())
        costs.append(c.cpu())
        flagged.append(f.cpu())
        hists = [[x.cpu()] for x in h] if hists is None else [hh + [x.cpu()] for hh, x in zip(hists, h)]
    boards = torch.cat(finals)
    m = solved_mask(boards, refs)
    masks[name] = m
    per_round = [solved_mask(torch.cat(hh), refs).float().mean().item() * 100 for hh in hists]
    r = {"solved": m.float().mean().item() * 100, "level_evals": torch.cat(costs).mean().item(),
         "retried": torch.cat(flagged).float().mean().item() * 100,
         "seconds": time.time() - t0, "solved_per_round": per_round, "solved_mask": m.int().tolist()}
    results[name] = r
    print(f"{name:<20}{r['solved']:9.2f}{r['level_evals']:14.1f}{r['retried']:15.1f}{r['seconds']:8.0f}   "
          + " → ".join(f"{x:.1f}" for x in per_round), flush=True)
    summary["results"] = results
    json.dump(summary, open(out_path, "w"), indent=2)

# Gepaarte Tests gegen baseline und verifier
print("\nGepaarter McNemar-Test (nur A gelöst / nur B gelöst / p):")
tests = {}
for ref in ("baseline", "verifier"):
    if ref not in masks:
        continue
    for name in variants:
        if name in ("baseline", ref):
            continue
        n10, n01, p = mcnemar(masks[name], masks[ref])
        tests[f"{name} vs {ref}"] = {"only_a": n10, "only_b": n01, "p": p}
        print(f"  {name:<20} vs {ref:<9} {n10:4d} / {n01:4d}   p={p:.4f}")
summary["mcnemar"] = tests
json.dump(summary, open(out_path, "w"), indent=2)
print(f"\nGespeichert: {out_path}")
