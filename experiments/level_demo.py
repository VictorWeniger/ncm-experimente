"""Erzeugt Daten für das interaktive Sudoku im Labor-Artefakt.

Für einige Test-Rätsel (hard) wird pro Ebene k = 1..12 das Brett ausgegeben, das der Decoder aus der
Summe der ersten k Ebenen liest, und zwar
  - "tok": aus den Codes der WAHREN Lösung (Tokenizer: Lösung -> Codes -> erste k Ebenen -> Decoder)
  - "gen": aus den Codes, die der Generator für das Rätsel erzeugt (gierig)
Ausgabe: results/level_demo.json und eine Kopie nach ~/Documents/ncm-nachtlauf/

    python experiments/level_demo.py
"""
import json
import os
import pickle
import shutil
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ncm"))
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
_F_interp = torch.nn.functional.interpolate


def _safe(input, *args, **kwargs):
    mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
    if input.device.type == "mps" and mode == "area":
        return _F_interp(input.cpu(), *args, **kwargs).to(input.device)
    return _F_interp(input, *args, **kwargs)


torch.nn.functional.interpolate = _safe

NCM = os.path.join(ROOT, "ncm")
CFG = OmegaConf.load(os.path.join(NCM, "config", "eval", "sudoku-hard.yaml"))
ck = os.path.join(ROOT, "artifacts", "sudoku", "hard")
dd = next(d for d in [os.path.join(ROOT, "data", "sudoku", "hard"), os.path.join(ROOT, "data", "sudoku")]
          if os.path.exists(os.path.join(d, CFG.data.test)))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(dev).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model)
net.load_state_dict(torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)["model"])
net = net.to(dev).eval()
vq = tk.vqvae.vq
LEVELS = list(vq.levels)
K = len(LEVELS)
P = int(CFG.model.prefix_len)
CUM = [sum(LEVELS[:i]) for i in range(K + 1)]

items = pickle.load(open(os.path.join(dd, CFG.data.test), "rb"))
refs = pickle.load(open(os.path.join(dd, CFG.data.test_refs), "rb"))


def to_ids(r):
    return list(r) if not isinstance(r, str) else tk.pre_tokenizer.encode(r)


@torch.no_grad()
def z_partial(idx, k):
    z = torch.zeros(idx.shape[0], LEVELS[-1], vq.embed_dim, device=idx.device)
    for li in range(k):
        e = vq.embeddings[li](idx[:, CUM[li]:CUM[li + 1]])
        q = vq._up_to_full(e, li, LEVELS[-1]) if li != K - 1 else e
        z = z + vq._apply_quant_resi(q, li)
    return z


@torch.no_grad()
def boards_per_level(idx):
    out = []
    for k in range(1, K + 1):
        out.append(tk.vqvae.decoder(z_partial(idx, k)).argmax(-1).cpu().tolist())
    return out  # K x B x 89


# Rätsel auswählen: 4 vom Generator gelöste und 2 nicht gelöste (die ersten passenden)
sel, n_ok, n_bad = [], 0, 0
with torch.no_grad():
    for s in range(0, 200, 50):
        it = items[s:s + 50]
        pfx = torch.tensor([(list(x) + [list(x)[-1]] * P)[:P] for x in it], device=dev)
        g = net.generate(None, pfx, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0)
        raw = tk.vqvae.decoder(z_partial(g, K)).argmax(-1)
        for j in range(len(it)):
            sol = to_ids(refs[s + j])
            ok = raw[j].tolist() == sol
            if (ok and n_ok < 4) or (not ok and n_bad < 2):
                sel.append((s + j, g[j]))
                n_ok += ok
                n_bad += (not ok)
        if n_ok >= 4 and n_bad >= 2:
            break

demo = []
with torch.no_grad():
    for i, gcodes in sel:
        sol = torch.tensor([to_ids(refs[i])], device=dev)
        tcodes = tk.vqvae.encode(sol)
        puz = list(items[i])
        k0 = 0
        while k0 < len(puz) and puz[k0] == 11:
            k0 += 1
        demo.append({
            "id": i, "puzzle": puz[k0:k0 + 89], "solution": sol[0].tolist(),
            "tok": [b[0] for b in boards_per_level(tcodes)],
            "gen": [b[0] for b in boards_per_level(gcodes[None])],
            "tok_codes": tcodes[0].tolist(), "gen_codes": gcodes.tolist(),
        })
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
path = os.path.join(ROOT, "results", "level_demo.json")
json.dump({"levels": LEVELS, "puzzles": demo}, open(path, "w"))
dst = os.path.expanduser("~/Documents/ncm-nachtlauf")
os.makedirs(dst, exist_ok=True)
shutil.copy(path, dst)
print(f"Fertig: {len(demo)} Rätsel, gespeichert in {path} und {dst}")
