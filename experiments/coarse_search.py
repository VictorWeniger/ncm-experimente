"""Experiment 8: Suche auf den groben Ebenen ("erst mehrere Grundideen probieren, dann ausformulieren").

Frage: Wird NCM besser, wenn es auf den groben Ebenen mehrere Alternativen ausprobiert, die beste
auswählt und erst dann die feinen Ebenen erzeugt? Und ist es egal, WO man abzweigt?

Für jedes Rätsel werden N Kandidaten erzeugt (Kandidat 0 ist immer die normale gierige Lösung):
  topK   Ebenen 1..K mit Temperatur T gezogen, Ebenen K+1..12 gierig   (Abzweigung oben = Grundidee)
  lowK   Ebenen 1..K gierig (für alle gleich), Ebenen K+1..12 mit T   (Abzweigung unten = Details)
  all    alle Ebenen mit T                                              (komplett neue Versuche)

Analyse-Modus: Alle N Kandidaten werden zu Ende gerechnet und bewertet. Daraus werden alle
Auswahlregeln nachträglich simuliert, mit ehrlicher Kostenrechnung:
  - Teilbewertung der groben Skizze, BEVOR die feinen Ebenen gerechnet werden:
        prior   mittlere log-Wahrscheinlichkeit der groben Codes (billig, aber gierig-verzerrt)
        dec     Decoder-Sicherheit der halbfertigen Skizze (ein Decoder-Durchlauf)
        rand    zufällige Auswahl (Kontrolle)
        oracle  Regel-Checker auf der Vervollständigung (Obergrenze, nicht realisierbar)
  - nur die besten B Skizzen werden zu Ende gerechnet, daraus der beste nach gen_llR
  - B = N heißt: alle vervollständigen ("fill"), Kosten entsprechend
  - gated: nur Rätsel, die gen_llR auf der gierigen Lösung als verdächtig markiert, bekommen die Suche
Zusätzlich: Abdeckung (ist überhaupt ein Kandidat richtig?), Vielfalt, AUROC der Teilbewertungen.

Kosten: NFE (Generator-Durchläufe) und "Token-Einheiten" = Summe über alle Durchläufe von
(Präfix + Länge der Ebene), weil grobe Ebenen viel billiger sind als feine. Dazu echte Laufzeit.

--dump schreibt pro Kandidat Merkmale der groben Skizze + Label (gelöst nach Vervollständigung) für
experiments/train_value.py (gelernter Prüfer für halbfertige Skizzen, Weg 3).

Aufruf:
    python experiments/coarse_search.py --smoke                         # Schnelltest
    python experiments/coarse_search.py --task hard --n 500 --N 8       # Mac
    python experiments/coarse_search.py --task hard --n 1000 --N 32 --dump   # Cluster
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402
from ncm_scoring import teacher_forced_logp  # noqa: E402

SEP, BOS = 10, 11
BOARD_LEN = 89

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="hard", choices=["easy", "medium", "hard"])
ap.add_argument("--n", type=int, default=500, help="Testbretter")
ap.add_argument("--n_cal", type=int, default=500, help="Kalibrierungsbretter für die gated-Schwelle")
ap.add_argument("--N", type=int, default=8, help="Kandidaten pro Rätsel (inkl. gierigem)")
ap.add_argument("--B", default="1,2,4", help="wie viele Skizzen nach der Teilbewertung vervollständigt werden")
ap.add_argument("--where", default="top1,top3,top6,top9,all,low3,low6,low9")
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--bs", type=int, default=0, help="Rätsel pro Batch (0 = automatisch)")
ap.add_argument("--device", default="auto")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dump", action="store_true", help="Merkmale + Labels für train_value.py speichern")
ap.add_argument("--out", default=os.path.join(ROOT, "results"))
ap.add_argument("--tag", default="search")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke:
    a.n, a.n_cal, a.N, a.where, a.tag, a.dump = 16, 16, 4, "top3,all,low6", "search-smoke", True

dev = a.device if a.device != "auto" else (
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
if a.bs == 0:
    a.bs = max(1, (256 if dev == "cuda" else 64) // a.N)

_F_interp = torch.nn.functional.interpolate


def _interp_mps_safe(input, *args, **kwargs):
    mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
    if input.device.type == "mps" and mode == "area":
        return _F_interp(input.cpu(), *args, **kwargs).to(input.device)
    return _F_interp(input, *args, **kwargs)


torch.nn.functional.interpolate = _interp_mps_safe

# ---------------------------------------------------------------- Setup (wie selfsignal_sudoku.py)
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
CUM = [sum(LEVELS[:i]) for i in range(K + 1)]          # Startindex jeder Ebene in der Code-Folge

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


# ---------------------------------------------------------------- Generierung mit Temperatur pro Ebene
_orig_sample = net._sample_level


@torch.no_grad()
def generate(pfx, temps):
    """Alle K Ebenen; temps[i] = Temperatur für Ebene i (0 = gierig).
    Rückgabe: Codes (B x sum(levels)), untemperierte log p je gewähltem Code (B x sum(levels))."""
    logps = []

    def sample(logits, temperature, top_k, top_p):
        i = len(logps)
        idx = _orig_sample(logits, float(temps[i]), 0, 1.0)
        logps.append(F.log_softmax(logits.float(), -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1))
        return idx

    net._sample_level = sample
    try:
        g = net.generate(None, pfx, temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0)
    finally:
        del net._sample_level
    assert len(logps) == K, f"erwartet {K} Ebenen, bekommen {len(logps)}"
    return g, torch.cat(logps, dim=1)


@torch.no_grad()
def z_partial(idx, k):
    """Summe der Rekonstruktion aus den Ebenen 1..k (volle Länge)."""
    z = torch.zeros(idx.shape[0], LEVELS[-1], vq.embed_dim, device=idx.device)
    for li in range(k):
        e = vq.embeddings[li](idx[:, CUM[li]:CUM[li + 1]])
        q = vq._up_to_full(e, li, LEVELS[-1]) if li != K - 1 else e
        z = z + vq._apply_quant_resi(q, li)
    return z


@torch.no_grad()
def decode_board(idx, clues):
    logits = tk.vqvae.decoder(z_partial(idx, K))
    raw = logits.argmax(-1)
    return raw, torch.where(clues > 0, clues, raw)


@torch.no_grad()
def gen_llR(pfx, raw):
    re = tk.vqvae.encode(raw)
    return teacher_forced_logp(net, tk.vqvae, pfx, re).mean(1)


@torch.no_grad()
def dec_partial(idx, k):
    """Decoder-Sicherheit der halbfertigen Skizze (Ebenen 1..k) + gepoolte Merkmale für den Prüfer."""
    z = z_partial(idx, k)
    lp = F.log_softmax(tk.vqvae.decoder(z).float(), -1).max(-1).values[:, CELLS]
    feats = torch.cat([z.mean(1), z.amax(1), z.std(1)], dim=1)
    return lp.mean(1), feats


def solved_mask(boards, refs):
    return torch.tensor([dec(b.tolist()) == r for b, r in zip(boards, refs)], dtype=torch.bool)


def auroc(score, pos):
    s, p = score.double(), pos.bool()
    sp, sn = s[p], s[~p]
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    return ((sp[:, None] > sn[None, :]).double().mean() + 0.5 * (sp[:, None] == sn[None, :]).double().mean()).item()


def mcnemar(x, y):
    n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
    n, k = n10 + n01, min(n10, n01)
    return n10, n01, (min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0)


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------- Kostenmodell
UNIT = [P + l for l in LEVELS]               # Token-Einheiten eines Generator-Durchlaufs für Ebene i
SCORE_UNIT = P + sum(LEVELS)                 # Ein-Pass-Bewertung (gen_llR)
GREEDY_COST = (K + 1, sum(UNIT) + SCORE_UNIT)  # gierig + gen_llR-Bewertung (NFE, Einheiten)


def parse_where(w):
    if w == "all":
        return "top", K
    return ("top" if w.startswith("top") else "low"), int(w[3:])


def cost(kind, k, N, B):
    """(NFE, Einheiten) für eine Suche mit N Skizzen, von denen B vervollständigt werden.
    Kandidat 0 ist der gierige (schon für die gated-Entscheidung berechnet, also inklusive)."""
    coarse_n, coarse_u = k, sum(UNIT[:k])
    fine_n, fine_u = K - k, sum(UNIT[k:])
    if kind == "top":   # N verschiedene grobe Skizzen, B davon fein ausrechnen, B mal bewerten
        return N * coarse_n + B * fine_n + B, N * coarse_u + B * fine_u + B * SCORE_UNIT
    # low: grobe Ebenen einmal (gemeinsam), Abzweigung erst in den feinen; Teilbewertung sinnlos -> B = N
    return coarse_n + N * fine_n + N, coarse_u + N * fine_u + N * SCORE_UNIT


# ---------------------------------------------------------------- Ein Durchgang für alle Rätsel einer Menge
def run_greedy(items, refs):
    sc, sv, boards = [], [], []
    for s in range(0, len(items), a.bs * a.N):
        it = items[s:s + a.bs * a.N]
        pfx = torch.tensor([prefix(x) for x in it], device=dev)
        clues = torch.tensor([clue_board(x) for x in it], device=dev)
        g, _ = generate(pfx, [0.0] * K)
        raw, board = decode_board(g, clues)
        sc.append(gen_llR(pfx, raw).cpu())
        boards.append(board.cpu())
    boards = torch.cat(boards)
    return torch.cat(sc), solved_mask(boards, refs)


def run_where(w, items, refs, greedy_score, greedy_solved):
    """Erzeugt N Kandidaten pro Rätsel (0 = gierig) und gibt alles Nötige für die Auswertung zurück."""
    kind, k = parse_where(w)
    N = a.N
    temps = ([a.temp] * k + [0.0] * (K - k)) if kind == "top" else ([0.0] * k + [a.temp] * (K - k))
    S = {key: [] for key in ("llr", "solved", "prior", "dec", "distinct")}
    feats_all = []
    t0 = time.time()
    for s in range(0, len(items), a.bs):
        it = items[s:s + a.bs]
        R = len(it)
        pfx1 = torch.tensor([prefix(x) for x in it], device=dev)
        clues1 = torch.tensor([clue_board(x) for x in it], device=dev)
        pfx = pfx1.repeat_interleave(N, 0)
        clues = clues1.repeat_interleave(N, 0)
        g, lp = generate(pfx, temps)
        # Kandidat 0 jedes Rätsels = gierig
        g0, lp0 = generate(pfx1, [0.0] * K)
        g = g.view(R, N, -1)
        g[:, 0] = g0
        lp = lp.view(R, N, -1)
        lp[:, 0] = lp0
        g, lp = g.view(R * N, -1), lp.view(R * N, -1)
        raw, board = decode_board(g, clues)
        llr = gen_llR(pfx, raw)
        kk = max(k, 1)
        dsc, feats = dec_partial(g, kk)
        prior = lp[:, :CUM[kk]].mean(1)
        sol = solved_mask(board.cpu(), [r for r in refs[s:s + R] for _ in range(N)])
        S["llr"].append(llr.view(R, N).cpu())
        S["solved"].append(sol.view(R, N))
        S["prior"].append(prior.view(R, N).cpu())
        S["dec"].append(dsc.view(R, N).cpu())
        bv = board.view(R, N, -1).cpu()
        S["distinct"].append(torch.tensor([len({tuple(x.tolist()) for x in bv[i]}) for i in range(R)]))
        if a.dump:
            feats_all.append(torch.cat([feats, prior[:, None], dsc[:, None]], 1).view(R, N, -1).cpu())
        if dev == "mps":
            torch.mps.empty_cache()
    sync()
    out = {key: torch.cat(v) for key, v in S.items()}
    out["seconds"] = time.time() - t0
    if a.dump:
        out["feats"] = torch.cat(feats_all)
    return kind, k, out


def evaluate(kind, k, D, greedy_score, greedy_solved, thr):
    """Simuliert alle Auswahlregeln aus den N vervollständigten Kandidaten."""
    llr, sol, n = D["llr"], D["solved"], D["llr"].shape[0]
    N = llr.shape[1]
    gen = torch.Generator().manual_seed(a.seed)
    res = {"coverage": sol.any(1).float().mean().item() * 100,
           "distinct_boards": D["distinct"].float().mean().item(),
           "candidate_solved": sol.float().mean().item() * 100,
           "auroc_partial": {}}
    flat_sol = sol[:, 1:].reshape(-1)        # nur gezogene Kandidaten
    for sc in ("prior", "dec"):
        res["auroc_partial"][sc] = auroc(D[sc][:, 1:].reshape(-1), flat_sol)
    res["auroc_llr_candidates"] = auroc(llr[:, 1:].reshape(-1), flat_sol)
    rules = {}
    Bs = [int(b) for b in a.B.split(",") if int(b) < N] + [N]
    scorers = ["prior", "dec", "rand", "oracle"] if kind == "top" else ["fill"]
    flagged = greedy_score < thr
    for sc in scorers:
        for B in (Bs if kind == "top" else [N]):
            if sc != "fill" and B == N and sc != "prior":
                continue                      # B = N ist für alle Scorer gleich -> nur einmal ("prior"-Eintrag = fill)
            if kind == "top" and B < N:
                if sc == "oracle":
                    part = sol.float() + 1e-3 * torch.rand(sol.shape, generator=gen)
                elif sc == "rand":
                    part = torch.rand(sol.shape, generator=gen)
                else:
                    part = D[sc].clone()
                keep = part.topk(B, dim=1).indices
            else:
                keep = torch.arange(N).expand(n, N)
            masked = torch.full_like(llr, -1e9)
            masked.scatter_(1, keep, llr.gather(1, keep))
            pick = masked.argmax(1)
            final = sol.gather(1, pick[:, None]).squeeze(1)
            nfe, units = cost(kind, k, N, B)
            name = f"{'fill' if B == N else sc}_B{B}"
            gated = torch.where(flagged, final, greedy_solved)
            frac = flagged.float().mean().item()       # gierig + gen_llR zahlen alle, die Suche nur markierte
            gnfe = GREEDY_COST[0] + frac * nfe
            gunits = GREEDY_COST[1] + frac * units
            rules[name] = {"solved": final.float().mean().item() * 100, "nfe": nfe, "units": units,
                           "gated_solved": gated.float().mean().item() * 100, "gated_nfe": gnfe, "gated_units": gunits,
                           "mask": final.int().tolist(), "gated_mask": gated.int().tolist()}
    res["rules"] = rules
    return res


# ---------------------------------------------------------------- Lauf
os.makedirs(a.out, exist_ok=True)
out_path = os.path.join(a.out, f"{task}-{a.tag}.json")
torch.manual_seed(a.seed)
print(f"Gerät: {dev} | {task} | Test {a.n} | Kalibrierung {a.n_cal} | N={a.N} | T={a.temp} | Batch {a.bs} Rätsel", flush=True)
print(f"Ebenen: {LEVELS} | Token-Einheiten pro Durchlauf: {UNIT}", flush=True)

t0 = time.time()
cal_score, cal_solved = run_greedy(*CAL)
cands = torch.unique(cal_score)
best, thr = -1, cands[0].item()
for t in cands.tolist():
    f = cal_score < t
    j = (f & ~cal_solved).sum().item() / max((~cal_solved).sum().item(), 1) - (f & cal_solved).sum().item() / max(cal_solved.sum().item(), 1)
    if j > best:
        best, thr = j, t
g_score, g_solved = run_greedy(*TEST)
print(f"Gierig: Test {g_solved.float().mean() * 100:.2f} % gelöst | gen_llR-Schwelle {thr:.4f} "
      f"markiert {(g_score < thr).float().mean() * 100:.1f} % ({time.time() - t0:.0f}s)", flush=True)
summary = {"task": task, "n": a.n, "N": a.N, "temp": a.temp, "levels": LEVELS, "units_per_pass": UNIT,
           "greedy": {"solved": g_solved.float().mean().item() * 100, "nfe": GREEDY_COST[0], "units": GREEDY_COST[1],
                      "mask": g_solved.int().tolist()},
           "threshold_gen_llR": thr, "where": {}}
dump = {}
for w in a.where.split(","):
    kind, k, D = run_where(w, *TEST, g_score, g_solved)
    r = evaluate(kind, k, D, g_score, g_solved, thr)
    r["seconds"] = D["seconds"]
    summary["where"][w] = r
    best_rule = max(r["rules"].items(), key=lambda kv: kv[1]["solved"])
    print(f"\n[{w}] {D['seconds']:.0f}s | Abdeckung (irgendein Kandidat richtig) {r['coverage']:.1f} % | "
          f"verschiedene Bretter {r['distinct_boards']:.1f}/{a.N} | Kandidaten gelöst {r['candidate_solved']:.1f} %", flush=True)
    print(f"  AUROC Teilbewertung: prior {r['auroc_partial']['prior']:.3f} | dec {r['auroc_partial']['dec']:.3f} "
          f"| gen_llR (fertig) {r['auroc_llr_candidates']:.3f}")
    print(f"  {'Regel':<14}{'gelöst %':>9}{'NFE':>7}{'Einheiten':>11}{'gated %':>9}{'g-NFE':>7}{'g-Einh.':>9}")
    for name, x in r["rules"].items():
        print(f"  {name:<14}{x['solved']:9.2f}{x['nfe']:7.0f}{x['units']:11.0f}{x['gated_solved']:9.2f}"
              f"{x['gated_nfe']:7.1f}{x['gated_units']:9.0f}")
    if a.dump:
        dump[w] = {"k": k, "kind": kind, "feats": D["feats"], "llr": D["llr"], "solved": D["solved"],
                   "prior": D["prior"], "dec": D["dec"]}
    json.dump(summary, open(out_path, "w"), indent=1)

# Gepaarte Tests: jede Regel gegen gierig, und "oben abzweigen" gegen "unten abzweigen"/"alles neu" (fill)
print("\nMcNemar gegen gierig (nur Regel gelöst / nur gierig gelöst / p):")
gm = g_solved.bool()
tests = {}
for w, r in summary["where"].items():
    for name, x in r["rules"].items():
        m = torch.tensor(x["mask"]).bool()
        n10, n01, p = mcnemar(m, gm)
        tests[f"{w}/{name} vs greedy"] = {"only_a": n10, "only_b": n01, "p": p}
        if name.startswith("fill") or name.startswith("dec"):
            print(f"  {w + '/' + name:<22}{n10:5d}{n01:5d}   p={p:.4f}")
if "all" in summary["where"]:
    am = torch.tensor(summary["where"]["all"]["rules"][f"fill_B{a.N}"]["mask"]).bool()
    print("\nAbzweigungsort gegen 'alles neu' (alle N vervollständigt, gleiche Auswahl nach gen_llR):")
    for w, r in summary["where"].items():
        if w == "all":
            continue
        m = torch.tensor(r["rules"][f"fill_B{a.N}"]["mask"]).bool()
        n10, n01, p = mcnemar(m, am)
        tests[f"{w}/fill vs all/fill"] = {"only_a": n10, "only_b": n01, "p": p}
        print(f"  {w:<8} {r['rules'][f'fill_B{a.N}']['solved']:6.2f} % vs {summary['where']['all']['rules'][f'fill_B{a.N}']['solved']:6.2f} %"
              f"   {n10:4d}/{n01:4d}  p={p:.4f}")
summary["mcnemar"] = tests
json.dump(summary, open(out_path, "w"), indent=1)
if a.dump:
    torch.save(dump, os.path.join(a.out, f"{task}-{a.tag}-dump.pt"))
print(f"\nGespeichert: {out_path}  ({(time.time() - t0) / 60:.1f} min)")
if a.smoke:
    per = sum(r["seconds"] for r in summary["where"].values()) / (len(summary["where"]) * a.n * a.N)
    print(f"SCHNELLTEST OK. Schätzung Mac-Hauptlauf (500 Rätsel, N = 8, 8 Abzweigungsorte): "
          f"~{per * 500 * 8 * 8 / 3600:.1f} h (eher zu hoch, kleine Batches sind ineffizient)")
