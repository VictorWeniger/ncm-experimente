"""Experiment 8, Teil 2: ein gelernter Prüfer für halbfertige Skizzen ("Value"-Netz wie bei AlphaGo).

Liest den Dump von coarse_search.py --dump. Pro Abzweigungsort (topK) lernt ein kleines Netz aus der
groben Skizze (gepoolte Rekonstruktion der Ebenen 1..K + prior + dec), ob ihre Vervollständigung
das Rätsel löst. Getrennt nach Rätseln: 60 % Training, 40 % Test.

Gemessen auf den Testrätseln:
  - AUROC des gelernten Prüfers gegen prior und dec
  - gelöst %, wenn nur die besten B Skizzen nach Prüfer vervollständigt und davon die beste nach gen_llR
    genommen wird (gleiche Kosten wie dec_B / prior_B in coarse_search.py)

Aufruf:
    python experiments/train_value.py results/sudoku-hard-search-dump.pt
"""
import json
import math
import sys

import torch
import torch.nn as nn


def auroc(score, pos):
    s, p = score.double(), pos.bool()
    sp, sn = s[p], s[~p]
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    return ((sp[:, None] > sn[None, :]).double().mean() + 0.5 * (sp[:, None] == sn[None, :]).double().mean()).item()


def select(part, llr, sol, B):
    keep = part.topk(B, dim=1).indices
    masked = torch.full_like(llr, -1e9)
    masked.scatter_(1, keep, llr.gather(1, keep))
    return sol.gather(1, masked.argmax(1, keepdim=True)).squeeze(1).float().mean().item() * 100


def main(path):
    D = torch.load(path, map_location="cpu", weights_only=False)
    torch.manual_seed(0)
    out = {}
    for w, d in D.items():
        if d["kind"] != "top":
            continue
        X, Y = d["feats"].float(), d["solved"].float()
        R, N, Fdim = X.shape
        perm = torch.randperm(R)
        tr, te = perm[: int(0.6 * R)], perm[int(0.6 * R):]
        mu, sd = X[tr].reshape(-1, Fdim).mean(0), X[tr].reshape(-1, Fdim).std(0) + 1e-6
        Xn = (X - mu) / sd
        net = nn.Sequential(nn.Linear(Fdim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        xtr, ytr = Xn[tr].reshape(-1, Fdim), Y[tr].reshape(-1)
        pw = ((1 - ytr).sum() / ytr.sum().clamp_min(1)).clamp(max=20)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        for ep in range(300):
            net.train()
            idx = torch.randperm(len(xtr))
            for s in range(0, len(idx), 512):
                b = idx[s:s + 512]
                loss = lossf(net(xtr[b]).squeeze(1), ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            v = net(Xn[te].reshape(-1, Fdim)).view(len(te), N)
        sol, llr = d["solved"][te], d["llr"][te]
        r = {"k": d["k"], "train_puzzles": len(tr), "test_puzzles": len(te),
             "auroc": {"value": auroc(v[:, 1:].reshape(-1), sol[:, 1:].reshape(-1)),
                       "prior": auroc(d["prior"][te][:, 1:].reshape(-1), sol[:, 1:].reshape(-1)),
                       "dec": auroc(d["dec"][te][:, 1:].reshape(-1), sol[:, 1:].reshape(-1))},
             "greedy_solved": sol[:, 0].float().mean().item() * 100,
             "coverage": sol.any(1).float().mean().item() * 100, "select": {}}
        for B in [1, 2, 4, 8]:
            if B >= N:
                break
            r["select"][f"B{B}"] = {sc: select(part, llr, sol, B) for sc, part in
                                    (("value", v), ("dec", d["dec"][te]), ("prior", d["prior"][te]))}
        r["select"][f"B{N}"] = {"fill": select(llr, llr, sol, N)}
        out[w] = r
        print(f"[{w}] AUROC Prüfer {r['auroc']['value']:.3f} | dec {r['auroc']['dec']:.3f} | prior {r['auroc']['prior']:.3f} "
              f"| gierig {r['greedy_solved']:.1f} % | Abdeckung {r['coverage']:.1f} %")
        for B, x in r["select"].items():
            print("   " + B + ": " + " | ".join(f"{k} {v:.1f} %" for k, v in x.items()))
    json.dump(out, open(path.replace("-dump.pt", "-value.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
