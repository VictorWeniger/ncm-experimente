"""Experiment 4: NCM-Generator nachtrainieren, damit er fehlerhafte grobe Ebenen reparieren kann.

Hintergrund (Experiment 1): Hält man beim Neu-Generieren die groben Ebenen des eigenen,
falschen Bretts fest, bringt das nichts. Das Modell hat im Training nur perfekte grobe Codes
gesehen. Hier lernt es zusätzlich, mit seinen EIGENEN groben Codes umzugehen.

Zwei Arten von Trainingsbeispielen pro Schritt:

  1. "sauber" (Paper-Ziel, Gl. 10): Codes der Lösung als Ziel, Eingaben aus den Lösungs-Codes,
     ein Teil der Eingabe-Codes zufällig ersetzt (Korruption), Verlust über alle Ebenen.

  2. "selbst" (Selbst-Konditionierung, Anteil --p_self): Das Modell generiert selbst
     (Temperatur 0,7, wie in der Feedback-Schleife). Für eine zufällige Ebene k werden die
     groben Ebenen 0..k-1 aus der eigenen Generierung übernommen. Ziel für die Ebenen k..K-1
     sind die "Reparatur-Codes": Restquantisierung der echten Lösung, ausgehend von der
     (eventuell falschen) groben Rekonstruktion. Sie führen von der eigenen Skizze zurück
     zur richtigen Lösung. Verlust nur auf den Ebenen ab k.

Mit --p_self 0 ist das eine reine Kontrolle (gleich viel Training ohne Selbst-Konditionierung).

Aufruf (Beispiel):
    python experiments/train_selfcond.py --task hard --train_dir data/sudoku/train \\
        --out runs/self --p_self 0.5 --steps 5000 --bs 256
"""
import argparse
import json
import math
import os
import pickle
import random
import shutil
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

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="hard", choices=["easy", "medium", "hard"])
ap.add_argument("--init_dir", default="", help="Start-Checkpoint (Standard: artifacts/sudoku/<task>)")
ap.add_argument("--train_dir", default=os.path.join(ROOT, "data", "sudoku", "train"))
ap.add_argument("--out", required=True)
ap.add_argument("--p_self", type=float, default=0.5, help="Anteil der Selbst-Konditionierungs-Beispiele")
ap.add_argument("--self_temp", type=float, default=0.7)
ap.add_argument("--k_min", type=int, default=1, help="kleinste festgehaltene Ebenenzahl bei Selbst-Beispielen")
ap.add_argument("--corrupt_hi", type=float, default=0.4, help="Korruptionsrate gröbste Ebene")
ap.add_argument("--corrupt_lo", type=float, default=0.05, help="Korruptionsrate feinste Eingabe-Ebene")
ap.add_argument("--alpha", type=float, default=0.0, help="Ebenengewichtung w_k ~ l_k^-alpha (0 = jeder Code gleich)")
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr_min", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--log_every", type=int, default=50)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--n_val", type=int, default=500, help="Validierung auf den ersten Testbrettern (nur greedy gelöst %)")
ap.add_argument("--device", default="auto")
ap.add_argument("--bf16", action="store_true", help="bfloat16-Autocast (nur CUDA)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = a.device if a.device != "auto" else (
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
torch.manual_seed(a.seed)
random.seed(a.seed)

if dev == "mps":  # siehe Sudoku-Skripte: interpolate(area) auf MPS nur bei teilbaren Längen
    _F_interp = torch.nn.functional.interpolate

    def _interp_mps_safe(input, *args, **kwargs):
        mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
        if input.device.type == "mps" and mode == "area":
            return _F_interp(input.cpu(), *args, **kwargs).to(input.device)
        return _F_interp(input, *args, **kwargs)

    torch.nn.functional.interpolate = _interp_mps_safe

# ---------------------------------------------------------------- Modell
task = f"sudoku-{a.task}"
CFG = OmegaConf.load(os.path.join(NCM, "config", "eval", f"{task}.yaml"))
init = a.init_dir or os.path.join(ROOT, "artifacts", "sudoku", a.task)
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(init, CFG.checkpoints.tokenizer)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
vqvae = tk.vqvae.to(dev).eval()
for p in vqvae.parameters():
    p.requires_grad_(False)
net = NextConceptTransformer(vqvae_local=vqvae, config=CFG.model)
sd = torch.load(os.path.join(init, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net = net.to(dev)
net.use_flex_attention = False
vq = vqvae.vq
LEVELS = list(vq.levels)
K, L0, LTOT = len(LEVELS), LEVELS[0], sum(LEVELS)
P = net.prefix_len
assert net.query_level0, "Dieses Skript erwartet das Sudoku-Setup (query_level0)"
LEVEL_OF = torch.cat([torch.full((l,), i, dtype=torch.long) for i, l in enumerate(LEVELS)]).to(dev)
STARTS = [sum(LEVELS[:i]) for i in range(K)]
W_LEVEL = torch.tensor([l ** (-a.alpha) for l in LEVELS], device=dev)
CORRUPT = torch.tensor([a.corrupt_hi + (a.corrupt_lo - a.corrupt_hi) * i / max(K - 2, 1) for i in range(K)], device=dev)
VOCAB = [vq.embeddings[i].weight.shape[0] for i in range(K)]

# ---------------------------------------------------------------- Daten
puzzles = pickle.load(open(os.path.join(a.train_dir, "puzzles.pickle"), "rb"))
solutions = pickle.load(open(os.path.join(a.train_dir, "solutions.pickle"), "rb"))
val_dir = os.path.join(ROOT, "data", "sudoku", a.task)
val_p = pickle.load(open(os.path.join(val_dir, CFG.data.test), "rb"))[:a.n_val]
val_s = pickle.load(open(os.path.join(val_dir, CFG.data.test_refs), "rb"))[:a.n_val]
print(f"Gerät {dev} | Training {len(puzzles)} Rätsel | Validierung {len(val_p)} | Ebenen {LEVELS} | p_self {a.p_self}")


def batch():
    idx = random.sample(range(len(puzzles)), a.bs)
    pfx = torch.tensor([puzzles[i][:P] for i in idx], device=dev)
    sol = torch.tensor([solutions[i] for i in idx], device=dev)
    return pfx, sol


# ---------------------------------------------------------------- Bausteine
def logits_for(pfx, idx_in):
    """Ein Forward Pass über alle Ebenen. idx_in: Codes, aus denen die Eingaben gebaut werden."""
    zp = vqvae.encoder.wte(pfx)[:, :P, :] if getattr(CFG.model, "raw_token_prefix", False) else vqvae.encode_z(pfx)[:, :P, :]
    xt = vq.idx_to_ncm_input(idx_in)
    xt = torch.cat([net.build_l0_query_input(idx_in.shape[0], xt.dtype), xt], dim=1)
    x = torch.cat([zp.to(xt.dtype), xt], dim=1)
    pm = (pfx[:, :P] == net.prefix_pad_id) if net.prefix_pad_id is not None else None
    return net(x, prefix_pad_mask=pm)  # (B, LTOT, V)


def corrupt(idx):
    """Ersetzt Eingabe-Codes zufällig durch andere Codes derselben Ebene (Ziele bleiben sauber)."""
    out = idx.clone()
    for i in range(K):
        sl = slice(STARTS[i], STARTS[i] + LEVELS[i])
        m = torch.rand(idx.shape[0], LEVELS[i], device=dev) < CORRUPT[i]
        out[:, sl] = torch.where(m, torch.randint(0, VOCAB[i], m.shape, device=dev), out[:, sl])
    return out


@torch.no_grad()
def repair_targets(sol_tokens, coarse_idx, k):
    """Restquantisierung der echten Lösung, Ebenen 0..k-1 fest vorgegeben (coarse_idx), ab k neu.
    Spiegelt vq.z_to_idx, nur dass die groben Indizes nicht per argmin gewählt werden."""
    z = vqvae._encode_pre_quant(sol_tokens).float()
    B, L, C = z.shape
    z_rest = z.clone()
    out = []
    for i, level in enumerate(LEVELS):
        emb = vq.embeddings[i]
        if i < k:
            ind = coarse_idx[:, STARTS[i]:STARTS[i] + level]
        else:
            flat = vq._down_for_quant(z_rest, i, level).reshape(-1, C) if i != K - 1 else z_rest.reshape(-1, C)
            ind = torch.argmin(vq._distances(flat, emb.weight), dim=1).view(B, level)
        q = vq._up_to_full(emb(ind), i, L) if i != K - 1 else emb(ind)
        z_rest = z_rest - vq._apply_quant_resi(q, i)
        out.append(ind)
    return torch.cat(out, dim=1)


@torch.no_grad()
def self_generate(pfx, temp):
    was = net.training
    net.eval()
    g = net.generate(None, pfx, temperature=temp, top_k=0, top_p=1.0, cfg_scale=0.0)
    net.train(was)
    return g


@torch.no_grad()
def decode(idx):
    z = torch.zeros(idx.shape[0], LEVELS[-1], vq.embed_dim, device=idx.device)
    for i, l in enumerate(LEVELS):
        e = vq.embeddings[i](idx[:, STARTS[i]:STARTS[i] + l])
        q = vq._up_to_full(e, i, LEVELS[-1]) if i != K - 1 else e
        z = z + vq._apply_quant_resi(q, i)
    return vqvae.decoder(z).argmax(-1)


def loss_on(logits, tgt, mask):
    """Gewichtete Kreuzentropie (Gl. 10), nur auf Positionen mit mask=True."""
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
    w = W_LEVEL[LEVEL_OF][None, :].expand_as(ce) * mask
    return (ce * w).sum() / w.sum().clamp_min(1e-8)


@torch.no_grad()
def validate():
    net.eval()
    solved, n = 0, 0
    for s in range(0, len(val_p), 128):
        pfx = torch.tensor([x[:P] for x in val_p[s:s + 128]], device=dev)
        board = decode(net.generate(None, pfx, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0))
        ref = torch.tensor(val_s[s:s + 128], device=dev)
        solved += (board == ref).all(1).sum().item()
        n += len(pfx)
    net.train()
    return solved / n * 100


def lr_at(step):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr_min + 0.5 * (a.lr - a.lr_min) * (1 + math.cos(math.pi * t))


# ---------------------------------------------------------------- Training
os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=2)
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
use_amp = a.bf16 and dev == "cuda"
log = []

v0 = validate()
print(f"Start: greedy gelöst (Validierung) {v0:.1f} %", flush=True)
log.append({"step": 0, "val_solved": v0})
net.train()
t0 = time.time()
run = {"loss_clean": 0.0, "loss_self": 0.0, "n_clean": 0, "n_self": 0, "repair_ok": 0.0, "k": 0.0}

for step in range(a.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    pfx, sol = batch()
    with torch.no_grad():
        tgt_clean = vqvae.encode(sol)
    n_self = int(round(a.bs * a.p_self))
    loss = torch.zeros((), device=dev)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        # 1) sauber + Korruption (alle Ebenen)
        if n_self < a.bs:
            pc, tc = pfx[n_self:], tgt_clean[n_self:]
            lc = loss_on(logits_for(pc, corrupt(tc)), tc, torch.ones_like(tc, dtype=torch.float))
            loss = loss + lc * (a.bs - n_self) / a.bs
            run["loss_clean"] += lc.item()
            run["n_clean"] += 1
        # 2) Selbst-Konditionierung: eigene grobe Ebenen, Reparatur-Ziele für die feinen
        if n_self > 0:
            ps, ss = pfx[:n_self], sol[:n_self]
            k = random.randint(a.k_min, K - 1)
            gen = self_generate(ps, a.self_temp)
            tgt = repair_targets(ss, gen, k)
            mask = (LEVEL_OF >= k).float()[None, :].expand(n_self, -1)
            ls = loss_on(logits_for(ps, tgt), tgt, mask)
            loss = loss + ls * n_self / a.bs
            run["loss_self"] += ls.item()
            run["n_self"] += 1
            run["k"] += k
            run["repair_ok"] += (decode(tgt) == ss).all(1).float().mean().item()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()

    if (step + 1) % a.log_every == 0:
        nc, ns = max(run["n_clean"], 1), max(run["n_self"], 1)
        rec = {"step": step + 1, "lr": lr_at(step), "loss_clean": run["loss_clean"] / nc,
               "loss_self": run["loss_self"] / ns if run["n_self"] else None,
               "repair_target_ok": run["repair_ok"] / ns * 100 if run["n_self"] else None,
               "mean_k": run["k"] / ns if run["n_self"] else None, "sec": time.time() - t0}
        log.append(rec)
        print(f"Schritt {rec['step']:5d} | lr {rec['lr']:.2e} | Verlust sauber {rec['loss_clean']:.4f}"
              + (f" | selbst {rec['loss_self']:.4f} (k≈{rec['mean_k']:.1f}, Reparatur-Ziel korrekt {rec['repair_target_ok']:.0f} %)"
                 if run["n_self"] else "") + f" | {rec['sec']:.0f}s", flush=True)
        run = {key: 0.0 if isinstance(v, float) else 0 for key, v in run.items()}

    if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
        v = validate()
        log.append({"step": step + 1, "val_solved": v})
        print(f"  Validierung nach {step + 1} Schritten: greedy gelöst {v:.1f} % (Start {v0:.1f} %)", flush=True)
        torch.save({"model": net.state_dict(), "step": step + 1, "args": vars(a)}, os.path.join(a.out, "generator.pt"))
        json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=2)

# Tokenizer und Config daneben legen, damit die Auswerte-Skripte den Ordner direkt nutzen können
for f in (CFG.checkpoints.tokenizer, "config.yaml"):
    src = os.path.join(init, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(a.out, f))
print(f"Fertig. Checkpoint: {os.path.join(a.out, 'generator.pt')}")
