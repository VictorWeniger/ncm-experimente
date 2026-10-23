"""Feedback-Experimente für Next-Concept Modeling auf Sudoku.

Vergleicht die originale Grob-zu-fein-Generierung (ein Durchlauf) mit Varianten,
die eine Rückkopplung von der fertigen Lösung zurück zu den groben Ebenen haben
(siehe PLAN.md). Es wird nichts trainiert, nur die Inferenz geändert.

Aufruf (aus dem Repo-Root):
    python experiments/feedback_sudoku.py --task hard --n 2000 --variants all
"""
import argparse
import json
import os
import pickle
import sys
import time

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCM = os.path.join(ROOT, "ncm")
sys.path.insert(0, NCM)
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

SEP, BOS, BLANK = 10, 11, 0
BOARD_LEN = 89
ALL_VARIANTS = ["baseline", "clamp",
                "loop_k1", "loop_k3", "loop_k6", "loop_k9",
                "loop_k1_t0.7", "loop_k3_t0.7", "loop_k6_t0.7",
                "sample2", "sample8"]

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="hard", choices=["easy", "medium", "hard"])
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--rounds", type=int, default=4, help="max. Feedback-Runden")
ap.add_argument("--samples", type=int, default=8, help="N für sampleN")
ap.add_argument("--variants", default="all", help="kommagetrennt oder 'all'")
ap.add_argument("--device", default="auto")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(ROOT, "results"))
ap.add_argument("--tag", default="", help="Suffix für die Ergebnisdatei, z.B. n500")
ap.add_argument("--no_levels", action="store_true", help="Analyse pro Ebene überspringen")
ap.add_argument("--ckpt_dir", default="", help="anderer Checkpoint-Ordner (generator.pt + tokenizer.pt)")
a = ap.parse_args()

if a.device == "auto":
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
else:
    dev = a.device
variants = ALL_VARIANTS if a.variants == "all" else a.variants.split(",")

# MPS kann adaptive_avg_pool1d (= interpolate mode="area") nur bei teilbaren Längen.
# Die Ebenen 1,2,4,6,9,... sind nicht teilbar -> diese eine Operation auf der CPU rechnen.
_F_interp = torch.nn.functional.interpolate


def _interp_mps_safe(input, *args, **kwargs):
    mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
    if input.device.type == "mps" and mode == "area":
        return _F_interp(input.cpu(), *args, **kwargs).to(input.device)
    return _F_interp(input, *args, **kwargs)


torch.nn.functional.interpolate = _interp_mps_safe

# ---------------------------------------------------------------- Setup
task = f"sudoku-{a.task}"
out_name = task + (f"-{a.tag}" if a.tag else "")
CFG = OmegaConf.load(os.path.join(NCM, "config", "eval", f"{task}.yaml"))
ck = a.ckpt_dir or os.path.join(ROOT, "artifacts", "sudoku", a.task)
dd_candidates = [os.path.join(ROOT, "data", "sudoku", a.task), os.path.join(ROOT, "data", "sudoku")]
dd = next(d for d in dd_candidates if os.path.exists(os.path.join(d, CFG.data.test)))

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

items = pickle.load(open(os.path.join(dd, CFG.data.test), "rb"))[:a.n]
refs = pickle.load(open(os.path.join(dd, CFG.data.test_refs), "rb"))[:a.n]
dec = lambda ids: tk.pre_tokenizer.decode([t for t in ids], skip_special_tokens=True)
refs = [r if isinstance(r, str) else dec(r) for r in refs]


def prefix(item):
    ids = list(item)
    return (ids[:P] + [ids[-1]] * P)[:P] if len(ids) < P else ids[:P]


def clue_board(item):
    """Holt das 89-Token-Rätselbrett (0 = leer) aus dem Prompt."""
    ids = list(item)
    i = 0
    while i < len(ids) and ids[i] == BOS:
        i += 1
    b = ids[i:i + BOARD_LEN]
    ok = len(b) == BOARD_LEN and all((b[j] == SEP) == ((j + 1) % 10 == 0) for j in range(BOARD_LEN))
    return b if ok else None


CLUES = [clue_board(it) for it in items]
CLAMP_OK = all(c is not None for c in CLUES)
if not CLAMP_OK:
    print("WARNUNG: Rätselbrett nicht im Prompt gefunden, Vorgaben werden nicht eingesetzt.")
    print("  Beispiel-Prompt:", list(items[0])[:P])
CLUES_T = torch.tensor([c if c is not None else [0] * BOARD_LEN for c in CLUES], device=dev)

# ---------------------------------------------------------------- Bausteine
_orig_sample = net._sample_level


def generate(pfx, forced=None, temp=0.0):
    """Generiert alle K Ebenen. forced: Liste der ersten k Ebenen (je B x l_i), die übernommen statt gesampelt werden."""
    calls = {"i": 0}

    def sample(logits, temperature, top_k, top_p):
        i = calls["i"]
        calls["i"] += 1
        if forced is not None and i < len(forced):
            return forced[i]
        return _orig_sample(logits, temp, 0, 1.0)

    net._sample_level = sample
    try:
        with torch.no_grad():
            g = net.generate(None, pfx, temperature=temp, top_k=0, top_p=1.0, cfg_scale=0.0)
    finally:
        del net._sample_level
    return g


def split_levels(idx):
    out, s = [], 0
    for l in LEVELS:
        out.append(idx[:, s:s + l])
        s += l
    return out


def partial_zhats(idx):
    """Kumulative Rekonstruktion nach jeder Ebene (geklont, da idx_to_zhats in-place addiert)."""
    z = torch.zeros(idx.shape[0], LEVELS[-1], vq.embed_dim, device=idx.device)
    zs, s = [], 0
    for li, l in enumerate(LEVELS):
        e = vq.embeddings[li](idx[:, s:s + l])
        q = vq._up_to_full(e, li, LEVELS[-1]) if li != K - 1 else e
        z = z + vq._apply_quant_resi(q, li)
        zs.append(z.clone())
        s += l
    return zs


def decode_board(idx):
    with torch.no_grad():
        return tk.vqvae.decoder(partial_zhats(idx)[-1]).argmax(-1)


def reencode(board):
    with torch.no_grad():
        return tk.vqvae.encode(board)


def clamp(board, clues):
    if not CLAMP_OK:
        return board
    return torch.where(clues > 0, clues, board)


_cells = [j for j in range(BOARD_LEN) if (j + 1) % 10 != 0]
CELLS = torch.tensor(_cells, device=dev)
_groups = []
for r in range(9):
    _groups.append([r * 9 + c for c in range(9)])
for c in range(9):
    _groups.append([r * 9 + c for r in range(9)])
for br in (0, 3, 6):
    for bc in (0, 3, 6):
        _groups.append([(br + i) * 9 + bc + j for i in range(3) for j in range(3)])
GROUPS = torch.tensor(_groups, device=dev)  # 27 x 9


def errors(board, clues):
    """Fehlersignal pro Brett: doppelte Ziffern in Gruppen + Nicht-Ziffern + verletzte Vorgaben."""
    g = board[:, CELLS]  # B x 81
    bad_digit = ((g < 1) | (g > 9)).sum(1)
    vals = g[:, GROUPS]  # B x 27 x 9
    onehot = torch.nn.functional.one_hot(vals.clamp(0, 10), 11)[..., 1:10].sum(2)  # B x 27 x 9
    dup = (onehot - 1).clamp_min(0).sum((1, 2))
    clue_viol = ((clues > 0) & (board != clues)).sum(1) if CLAMP_OK else torch.zeros_like(dup)
    return bad_digit + dup + clue_viol


def run_variant(name, pfx, clues):
    """Gibt (finales Brett, Ebenen-Evaluationen pro Brett, Verlauf der gelösten Bretter pro Runde) zurück."""
    B = pfx.shape[0]
    g = generate(pfx)
    board = decode_board(g)
    cost = torch.full((B,), float(K), device=dev)
    if name == "baseline":
        return board, cost, [board]
    if name == "clamp":
        return clamp(board, clues), cost, [clamp(board, clues)]
    if name.startswith("sample"):
        n = int(name[len("sample"):])
        best = clamp(board, clues)
        best_err = errors(best, clues)
        for _ in range(n - 1):
            b = clamp(decode_board(generate(pfx, temp=0.7)), clues)
            e = errors(b, clues)
            better = e < best_err
            best = torch.where(better[:, None], b, best)
            best_err = torch.minimum(e, best_err)
        cost += K * (n - 1)
        return best, cost, [best]
    if name.startswith("loop_k"):
        parts = name[len("loop_k"):].split("_t")
        k = int(parts[0])
        temp = float(parts[1]) if len(parts) > 1 else 0.0
        board = clamp(board, clues)
        err = errors(board, clues)
        hist = [board.clone()]
        for _ in range(a.rounds):
            active = err > 0
            if not active.any():
                hist.append(board.clone())
                continue
            ai = active.nonzero().squeeze(1)
            coarse = split_levels(reencode(board[ai]))[:k]
            nb = clamp(decode_board(generate(pfx[ai], forced=coarse, temp=temp)), clues[ai])
            ne = errors(nb, clues[ai])
            accept = ne <= err[ai]
            board[ai] = torch.where(accept[:, None], nb, board[ai])
            err[ai] = torch.where(accept, ne, err[ai])
            cost[ai] += K - k
            hist.append(board.clone())
        return board, cost, hist
    raise ValueError(name)


def to_str(board_row):
    return dec(board_row.tolist())


def score(boards):
    hyps = [to_str(b) for b in boards]
    solved = sum(h == r for h, r in zip(hyps, refs))
    cells = [sum(x == y for x, y in zip(h.replace("|", ""), r.replace("|", ""))) / 81 for h, r in zip(hyps, refs)]
    return hyps, solved, sum(cells) / len(cells)


# ---------------------------------------------------------------- Analyse pro Ebene (baseline)
def level_analysis():
    acc = torch.zeros(K)
    for s in range(0, len(items), a.bs):
        pfx = torch.tensor([prefix(it) for it in items[s:s + a.bs]], device=dev)
        ref_b = torch.tensor([tk.pre_tokenizer.encode(r) for r in refs[s:s + a.bs]], device=dev)
        with torch.no_grad():
            zs = partial_zhats(generate(pfx))
            for li, z in enumerate(zs):
                b = tk.vqvae.decoder(z).argmax(-1)
                acc[li] += (b[:, CELLS] == ref_b[:, CELLS]).float().mean(1).sum().item()
    return (acc / len(items) * 100).tolist()


# ---------------------------------------------------------------- Lauf
os.makedirs(a.out, exist_ok=True)
print(f"Gerät: {dev} | Aufgabe: {task} | Bretter: {len(items)} | Ebenen: {LEVELS} | Vorgaben einsetzbar: {CLAMP_OK}")
summary = {"task": task, "n": len(items), "device": dev, "levels": LEVELS, "rounds": a.rounds, "results": {}}

t0 = time.time()
summary["cell_acc_after_level"] = level_analysis() if not a.no_levels else []
print("Zellen korrekt nach Ebene (%):", " ".join(f"{x:.1f}" for x in summary["cell_acc_after_level"]),
      f"({time.time() - t0:.0f}s)")

print(f"\n{'Variante':<16}{'gelöst %':>9}{'gültig %':>9}{'Zellen %':>9}{'Ebenen/Brett':>14}{'Zeit s':>8}   gelöst je Runde")
for name in variants:
    torch.manual_seed(a.seed)
    t0 = time.time()
    finals, costs, hists = [], [], None
    for s in range(0, len(items), a.bs):
        pfx = torch.tensor([prefix(it) for it in items[s:s + a.bs]], device=dev)
        clues = CLUES_T[s:s + a.bs]
        b, c, h = run_variant(name, pfx, clues)
        finals.append(b.cpu())
        costs.append(c.cpu())
        hists = [[x.cpu()] for x in h] if hists is None else [hh + [x.cpu()] for hh, x in zip(hists, h)]
    boards = torch.cat(finals)
    hyps, solved, cell = score(boards)
    valid = (errors(boards.to(dev), CLUES_T) == 0).float().mean().item() * 100
    per_round = [score(torch.cat(h))[1] / len(items) * 100 for h in hists]
    cost = torch.cat(costs).mean().item()
    dt = time.time() - t0
    r = {"solved_mask": [int(h == ref) for h, ref in zip(hyps, refs)],
         "solved": solved / len(items) * 100, "valid": valid, "cells": cell * 100,
         "level_evals": cost, "seconds": dt, "solved_per_round": per_round}
    summary["results"][name] = r
    print(f"{name:<16}{r['solved']:9.2f}{r['valid']:9.2f}{r['cells']:9.2f}{cost:14.1f}{dt:8.0f}   "
          + " → ".join(f"{x:.1f}" for x in per_round), flush=True)
    json.dump(summary, open(os.path.join(a.out, f"{out_name}.json"), "w"), indent=2)

print(f"\nGespeichert: {os.path.join(a.out, out_name + '.json')}")
