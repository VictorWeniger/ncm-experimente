"""Experiment 3: Funktioniert das interne Fehlersignal auch bei Text (TinyStories)?

Bei Sudoku hat die faire Likelihood (gen_llR: Ergebnis neu kodieren, Codes per
Teacher Forcing bewerten) Fehler mit AUROC 0,96 erkannt. Bei Text gibt es kein
"richtig/falsch", deshalb wird Qualität so gemessen wie im Paper: Perplexity unter
GPT-2 large (niedriger = flüssiger), zusammen mit der Unigramm-Entropie (Vielfalt;
fällt sie stark, wird der Text nur repetitiver).

Teil A: Korreliert das interne Signal pro Geschichte mit der GPT-2-Perplexity?
Teil B: Verdächtige Geschichten (untere 20 % des Signals, Schwelle auf separaten
        Kalibrierungs-Samples) werden neu gesampelt, übernommen wird nur ein Versuch
        mit besserem Signal. Kontrollen: zufällig ausgewählte 20 % (rand_) und
        2 Samples für alle, das sicherere gewinnt (best2_).

Aufruf:
    python experiments/textsignal_tinystories.py --seeds 42,0,1,2,3 --n 256
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCM = os.path.join(ROOT, "ncm")
sys.path.insert(0, NCM)
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402
from evaluation.gen_ppl import generation_perplexity  # noqa: E402
from ncm_scoring import make_generate, teacher_forced_logp, check_fast_scoring  # noqa: E402

SIGNALS = ["gen_llR", "gen_ll", "dec_mean", "cycle"]
DEFAULT_VARIANTS = ["baseline", "self_gen_llR", "rand_gen_llR", "best2_gen_llR", "self_dec_mean"]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="42,0,1,2,3", help="Test-Seeds (wie im Paper)")
ap.add_argument("--cal_seed", type=int, default=1000)
ap.add_argument("--n", type=int, default=256, help="Samples pro Seed")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--flag_q", type=float, default=0.2, help="Anteil, der als verdächtig gilt")
ap.add_argument("--ref", default="gpt2-large")
ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
ap.add_argument("--device", default="auto")
ap.add_argument("--out", default=os.path.join(ROOT, "results"))
ap.add_argument("--tag", default="text")
a = ap.parse_args()

dev = a.device if a.device != "auto" else (
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
seeds = [int(s) for s in a.seeds.split(",")]
variants = a.variants.split(",")

_F_interp = torch.nn.functional.interpolate


def _interp_mps_safe(input, *args, **kwargs):
    mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
    if input.device.type == "mps" and mode == "area":
        return _F_interp(input.cpu(), *args, **kwargs).to(input.device)
    return _F_interp(input, *args, **kwargs)


torch.nn.functional.interpolate = _interp_mps_safe

# ---------------------------------------------------------------- Setup
CFG = OmegaConf.load(os.path.join(NCM, "config", "eval", "tinystories.yaml"))
ck = os.path.join(ROOT, "artifacts", "tinystories")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(ck, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(ck, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(dev).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model)
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net = net.to(dev).eval()
L0_P = sd["inference"]["level0_unigram"].double().float().cpu()
del sd
vq = tk.vqvae.vq
LEVELS = list(vq.levels)
K = len(LEVELS)
L0 = LEVELS[0]
NFE_GEN = K - 1  # Ebene 0 kommt aus der Unigramm-Verteilung, nicht vom Generator
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
generate = make_generate(net)


def dec(ids):
    ids = list(ids)
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


def full_z(idx):
    z = torch.zeros(idx.shape[0], LEVELS[-1], vq.embed_dim, device=idx.device)
    s = 0
    for li, l in enumerate(LEVELS):
        e = vq.embeddings[li](idx[:, s:s + l])
        q = vq._up_to_full(e, li, LEVELS[-1]) if li != K - 1 else e
        z = z + vq._apply_quant_resi(q, li)
        s += l
    return z


FAST = True


def attempt(B):
    """Eine Geschichte pro Zeile + Signale. Temperatur wie im Paper-Setup (1,0)."""
    if dev == "mps":
        torch.mps.empty_cache()
    pfx = torch.full((B, 1), EOS, device=dev)
    l0 = torch.multinomial(L0_P, B * L0, replacement=True).view(B, L0).to(dev)
    g, lp = generate(pfx, float(CFG.generation.temperature), idx0=l0)
    with torch.no_grad():
        logits = tk.vqvae.decoder(full_z(g))
        raw = logits.argmax(-1)
        tok_lp = F.log_softmax(logits.float(), -1).max(-1).values
        eos_seen = (raw == EOS).cumsum(1)
        valid = ((eos_seen == 0) | ((eos_seen == 1) & (raw == EOS))).float()
        re = tk.vqvae.encode(raw)
        if FAST:
            llr = teacher_forced_logp(net, tk.vqvae, pfx, re).mean(1)
        else:
            parts = [re[:, s:s + l] for s, l in zip([sum(LEVELS[:i]) for i in range(K)], LEVELS)]
            llr = generate(pfx, 0.0, idx0=parts[0], forced=parts[1:])[1].mean(1)
    sig = {
        "gen_llR": llr,
        "gen_ll": lp.mean(1),
        "dec_mean": (tok_lp * valid).sum(1) / valid.sum(1).clamp_min(1),
        "cycle": (re == g).float().mean(1),
    }
    return raw.cpu(), {k: v.cpu() for k, v in sig.items()}


# ---------------------------------------------------------------- Qualität
rtok = rlm = None


def load_ref():
    global rtok, rlm
    if rlm is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rtok = AutoTokenizer.from_pretrained(a.ref)
        # auf MPS in halber Genauigkeit laden (halber Speicher, sonst swappt ein MacBook Air),
        # die Logits für die Perplexity aber in float32 zurückgeben
        half = dev in ("mps", "cuda")
        m = AutoModelForCausalLM.from_pretrained(a.ref, dtype=torch.float16 if half else torch.float32).to(dev).eval()

        class _Float32Logits(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, **kw):
                out = self.inner(**kw)
                out.logits = out.logits.float()
                return out

        rlm = _Float32Logits(m).eval()
        print(f"Referenzmodell {a.ref} geladen ({'float16' if half else 'float32'})", flush=True)


def quality(texts):
    load_ref()
    r = generation_perplexity(rlm, rtok, [""] * len(texts), texts, device=dev, batch_size=8)
    return r.ppl, r.mean_ppl


def entropy(text):
    ids = tk.pre_tokenizer.encode(text)
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


def spearman(x, y):
    rx = x.argsort().argsort().double()
    ry = y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return (rx * ry).sum().item() / math.sqrt((rx ** 2).sum().item() * (ry ** 2).sum().item())


def auroc(score, pos):
    s, p = score.double(), pos.bool()
    sp, sn = s[p], s[~p]
    gt = (sp[:, None] > sn[None, :]).double().mean()
    eq = (sp[:, None] == sn[None, :]).double().mean()
    return (gt + 0.5 * eq).item()


def initial_pass(seed):
    torch.manual_seed(seed)
    raws, sigs = [], {k: [] for k in SIGNALS}
    for s in range(0, a.n, a.bs):
        r, sg = attempt(min(a.bs, a.n - s))
        raws.append(r)
        for k in SIGNALS:
            sigs[k].append(sg[k])
    return torch.cat(raws), {k: torch.cat(v) for k, v in sigs.items()}


def run_variant(name, raw0, sig0, thr, seed):
    """Erst alle Startversuche (identisch für alle Varianten), dann gezielte Neuversuche."""
    raw, sig = raw0.clone(), {k: v.clone() for k, v in sig0.items()}
    n = raw.shape[0]
    step = NFE_GEN + (1 if FAST else NFE_GEN)  # Generierung + Bewertung
    cost = torch.full((n,), float(step))
    torch.manual_seed(seed + 7777)
    if name == "baseline":
        return raw, cost - (1 if FAST else NFE_GEN), torch.zeros(n, dtype=torch.bool)
    s_name = name.split("_", 1)[1]
    if name.startswith("best2_"):
        rounds, flag = 1, torch.ones(n, dtype=torch.bool)
    elif name.startswith("rand_"):
        rounds, flag = a.rounds, torch.rand(n) < a.flag_q
    else:
        rounds, flag = a.rounds, sig[s_name] < thr[s_name]
    ever = flag.clone()
    for _ in range(rounds):
        if name.startswith("self_"):
            flag = sig[s_name] < thr[s_name]
            ever |= flag
        idx = flag.nonzero().squeeze(1)
        for s in range(0, len(idx), a.bs):
            ii = idx[s:s + a.bs]
            nr, ns = attempt(len(ii))
            acc = ns[s_name] > sig[s_name][ii]
            raw[ii] = torch.where(acc[:, None], nr, raw[ii])
            for k in SIGNALS:
                sig[k][ii] = torch.where(acc, ns[k], sig[k][ii])
            cost[ii] += step
    return raw, cost, ever


# ---------------------------------------------------------------- Lauf
os.makedirs(a.out, exist_ok=True)
out_path = os.path.join(a.out, f"tinystories-{a.tag}.json")
print(f"Gerät: {dev} | TinyStories | Seeds {seeds} x {a.n} | Kalibrierung Seed {a.cal_seed} | Referenz {a.ref}")

torch.manual_seed(123)
_pfx = torch.full((32, 1), EOS, device=dev)
_l0 = torch.multinomial(L0_P, 32 * L0, replacement=True).view(32, L0).to(dev)
_g, _ = generate(_pfx, 1.0, idx0=_l0)
d_mean, d_max = check_fast_scoring(net, tk.vqvae, generate, _pfx, _g, idx0_given=True)
FAST = d_mean < 1e-3
print(f"Selbsttest Ein-Pass-Bewertung: Abweichung {d_mean:.2e} (Mittelwert), {d_max:.2e} (einzelner Code) -> "
      + ("Ein-Pass" if FAST else "WEICHT AB, Ebene für Ebene"))
summary = {"seeds": seeds, "n": a.n, "ref": a.ref, "rounds": a.rounds, "flag_q": a.flag_q, "fast_scoring": FAST}

# Schwellen aus Kalibrierungs-Samples (ohne Qualitätslabel: einfach das untere flag_q-Quantil)
t0 = time.time()
_, sig_cal = initial_pass(a.cal_seed)
thr = {k: torch.quantile(v.double(), a.flag_q).item() for k, v in sig_cal.items()}

# Startversuche pro Seed (für alle Varianten identisch)
init = {s: initial_pass(s) for s in seeds}
print(f"Startversuche fertig ({time.time() - t0:.0f}s), bewerte mit {a.ref} ...")

# Teil A: Korrelation pro Geschichte
all_sig = {k: torch.cat([init[s][1][k] for s in seeds]) for k in SIGNALS}
all_txt = [dec(r.tolist()) for s in seeds for r in init[s][0]]
ppl_base, _ = quality(all_txt)
logppl = ppl_base.log().double()
bad = logppl > torch.quantile(logppl, 1 - a.flag_q)
print(f"\nTeil A: Hängt das interne Signal mit der Textqualität zusammen? ({len(all_txt)} Geschichten)")
print(f"{'Signal':<10}{'Spearman':>10}{'AUROC schlechteste 20 %':>26}")
detection = {}
for k in SIGNALS:
    detection[k] = {"spearman_vs_neg_logppl": spearman(all_sig[k], -logppl), "auroc_bad": auroc(all_sig[k], ~bad)}
    print(f"{k:<10}{detection[k]['spearman_vs_neg_logppl']:10.3f}{detection[k]['auroc_bad']:26.3f}")
summary["detection"] = detection
json.dump(summary, open(out_path, "w"), indent=2)

# Teil B
print(f"\nTeil B: Gezielt neu sampeln")
print(f"{'Variante':<16}{'GenPPL':>8}{'± Seeds':>9}{'Entropie':>10}{'NFE/Sample':>12}{'versucht %':>12}{'Zeit s':>8}")
results, per_sample = {}, {}
for name in variants:
    t0 = time.time()
    ppls, ents, costs, evers, logs = [], [], [], [], []
    for s in seeds:
        raw, cost, ever = run_variant(name, *init[s], thr, s)
        texts = [dec(r.tolist()) for r in raw]
        ppl_i, mean_ppl = quality(texts)
        ppls.append(mean_ppl)
        ents.append(sum(entropy(t) for t in texts) / len(texts))
        costs.append(cost.mean().item())
        evers.append(ever.float().mean().item() * 100)
        logs.append(ppl_i.log())
    g = torch.tensor(ppls)
    r = {"genppl": g.mean().item(), "genppl_sd": g.std().item() if len(g) > 1 else 0.0,
         "entropy": sum(ents) / len(ents), "nfe": sum(costs) / len(costs), "retried": sum(evers) / len(evers),
         "per_seed_genppl": ppls, "seconds": time.time() - t0}
    results[name] = r
    per_sample[name] = torch.cat(logs)
    print(f"{name:<16}{r['genppl']:8.2f}{r['genppl_sd']:9.2f}{r['entropy']:10.3f}{r['nfe']:12.1f}"
          f"{r['retried']:12.1f}{r['seconds']:8.0f}", flush=True)
    summary["results"] = results
    json.dump(summary, open(out_path, "w"), indent=2)

# Gepaarter Vergleich: Differenz der log-Perplexity pro Geschichte, Bootstrap-Konfidenzintervall
if "baseline" in per_sample:
    print("\nPaarweise gegen baseline (mittlere Änderung log-PPL pro Geschichte, 95 %-Bootstrap-KI; negativ = besser):")
    gen = torch.Generator().manual_seed(0)
    paired = {}
    for name, v in per_sample.items():
        if name == "baseline":
            continue
        d = (v - per_sample["baseline"]).double()
        boots = torch.stack([d[torch.randint(len(d), (len(d),), generator=gen)].mean() for _ in range(2000)])
        lo, hi = torch.quantile(boots, 0.025).item(), torch.quantile(boots, 0.975).item()
        paired[name] = {"mean_dlogppl": d.mean().item(), "ci95": [lo, hi]}
        print(f"  {name:<16} {d.mean().item():+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    summary["paired_vs_baseline"] = paired
json.dump(summary, open(out_path, "w"), indent=2)
print(f"\nGespeichert: {out_path}")
