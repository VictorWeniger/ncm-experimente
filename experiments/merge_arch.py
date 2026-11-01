"""Exp. 5b: NCM und LLM in EINER Architektur (TinyStories). Varianten B, C, C-par, D.
Vergleich gegen A = hybrid_stage4.py (zwei getrennte Modelle) und llm_ft (gleiches GPT-2 ohne Plan).

B  "ein Rumpf, zwei Modi": ein GPT-2-Rumpf sieht nacheinander
       [Q0][Eingabe Ebene 1]...[Eingabe Ebene k-1][Plan-Auslese][<bos> Text ...]
   * Ebene j bekommt als Eingabe die kumulative Rekonstruktion der gröberen Ebenen, heruntergemittelt
     auf l_j Positionen (wie NCM). Die Eingabe enthält keine Codes der eigenen Ebene, deshalb darf die
     normale kausale Maske bleiben: alle Codes einer Ebene entstehen in EINEM Durchlauf.
   * Eigene Köpfe pro Ebene sagen die Codes voraus, der GPT-2-Kopf den Text.
   * Plan-Auslese = Rekonstruktion aus allen k Plan-Ebenen, damit der Text auch die feinste Plan-Ebene sieht.
   * Verlust = CE(Text) + lambda * mittlere CE(Codes). Training mit Code-Korruption wie im Paper.
C  "blockweise": dieselbe Maschine, aber pro Block von 64 NCM-Tokens:
       [bisheriger Text][Plan für Block n][<eob> Text von Block n <eob>]  ...  bis <eot>
   Damit gibt es keine feste Längengrenze mehr (128 Tokens bei NCM).
C-par  wie C mit k = alle 12 Ebenen; der Block-Text wird nicht Token für Token geschrieben, sondern vom
   eingefrorenen NCM-Decoder parallel aus der vollständigen Pyramide gelesen (schnellste Variante).
D  "Rückkopplung" (nur Inferenz, auf einem B-Modell): Text Satz für Satz. Nach jedem Satz wird der Text
   bisher mit dem NCM-Tokenizer neu kodiert und mit dem Plan verglichen (Abstand der Rekonstruktionen
   auf den schon geschriebenen Positionen). Ist der Abstand auffällig groß (oberes flag_q-Quantil auf
   Kalibrierung), wird der Satz bis zu R-mal neu geschrieben und der planähnlichste behalten.

Aufrufe:
  python experiments/merge_arch.py train --arch B --k 3
  python experiments/merge_arch.py train --arch C --k 3
  python experiments/merge_arch.py train --arch C --k 12          # = C-par
  python experiments/merge_arch.py gen --arch B --k 3
  python experiments/merge_arch.py gen --arch C --k 3
  python experiments/merge_arch.py gen --arch C --k 12 --par
  python experiments/merge_arch.py gen --arch D --k 3            # nutzt das B-Modell
  python experiments/merge_arch.py gen --arch C --k 3 --max_blocks 6 --tag long   # lange Geschichten
Ergebnisse: results/bench/tinystories/merge.json -> bench_score.py --task tinystories
"""
import argparse
import json
import math
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_common import ROOT, Task, free, n_params, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "gen"])
ap.add_argument("--arch", choices=["B", "C", "D"], required=True)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--par", action="store_true", help="C: Block-Text per NCM-Decoder (braucht --k 12)")
ap.add_argument("--lm", default="gpt2")
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--micro_bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--lam", type=float, default=1.0, help="Gewicht des Code-Verlusts")
ap.add_argument("--code_noise", type=float, default=0.2, help="max. Anteil korrumpierter Codes (gleichverteilt 0..max)")
ap.add_argument("--block", type=int, default=64, help="C: NCM-Tokens pro Block")
ap.add_argument("--max_blocks", type=int, default=2, help="C: Blöcke pro Geschichte beim Erzeugen (2 = 128 Tokens wie NCM)")
ap.add_argument("--max_ctx", type=int, default=192, help="C: max. GPT-2-Tokens Kontext")
ap.add_argument("--max_text", type=int, default=192)
ap.add_argument("--R", type=int, default=2, help="D: max. Neuversuche pro Satz")
ap.add_argument("--d_mode", default="plan", choices=["plan", "randtrig", "rand"],
                help="D: plan = Neuversuch wenn Plan-Abstand groß, bester nach Plan-Abstand; "
                     "randtrig = zufällig ausgelöster Neuversuch (gleiche Rate), bester nach Plan-Abstand; "
                     "rand = zufällig ausgelöst und zufällig gewählt (Kontrolle ganz ohne Plan)")
ap.add_argument("--flag_q", type=float, default=0.2)
ap.add_argument("--n_samples", type=int, default=256)
ap.add_argument("--seeds", default="42,0,1,2,3")
ap.add_argument("--tag", default="")
ap.add_argument("--device", default="auto")
ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "artifacts", "merge"))
ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench", "tinystories"))
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

dev = pick_device(a.device)
if a.smoke:
    a.steps, a.bs, a.micro_bs, a.n_samples, a.seeds, a.warmup = 3, 2, 2, 4, "42", 1
    a.out = a.out + "-smoke"
os.makedirs(a.ckpt_dir, exist_ok=True)
os.makedirs(a.out, exist_ok=True)
train_arch = "B" if a.arch == "D" else a.arch
ckpt = os.path.join(a.ckpt_dir, f"merge_{train_arch}_k{a.k}{'_smoke' if a.smoke else ''}.pt")
if a.par:
    assert a.arch == "C", "--par gibt es nur für C"

T = Task("tinystories", dev)
vq, pre = T.vq, T.tk.pre_tokenizer
LVall = T.levels
K, L = len(LVall), T.L
assert 1 <= a.k <= K
LV = LVall[:a.k]
if a.par:
    assert a.k == K, "C-par braucht die vollständige Pyramide: --k 12"
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.lm)
tok.add_special_tokens({"additional_special_tokens": ["<|eob|>"]})
tok.pad_token = tok.eos_token
EOT = tok.eos_token_id
EOB = tok.convert_tokens_to_ids("<|eob|>")


# ====================================================================== Pyramide
def zcum(codes, upto):
    """Kumulative Rekonstruktion aus den ersten `upto` Ebenen, (B, L, d)."""
    B = codes.shape[0]
    z = torch.zeros(B, L, vq.embed_dim, device=codes.device)
    s = 0
    for li in range(upto):
        l = LVall[li]
        e = vq.embeddings[li](codes[:, s:s + l]).float()
        q = vq._up_to_full(e, li, L) if li != K - 1 else e
        z = z + vq._apply_quant_resi(q, li).float()
        s += l
    return z


def down(z, l):
    if l == z.shape[1]:
        return z
    return F.interpolate(z.transpose(1, 2), size=l, mode="area").transpose(1, 2)


def level_input(codes, j):
    """Eingabe für Ebene j (j >= 1): Rekonstruktion aus Ebenen < j, gemittelt auf l_j Positionen."""
    return down(zcum(codes, j), LVall[j])


def readout(codes):
    return down(zcum(codes, a.k), LV[-1])


def corrupt(codes, rate):
    out, s = codes.clone(), 0
    for li, l in enumerate(LV):
        m = torch.rand(codes.shape[0], l, device=codes.device) < rate[:, None]
        r = torch.randint(vq.embeddings[li].num_embeddings, (codes.shape[0], l), device=codes.device)
        out[:, s:s + l] = torch.where(m, r, out[:, s:s + l])
        s += l
    return out


def encode_ids(id_lists):
    """NCM-BPE-Listen (<= 128) -> Codes aller Ebenen; aufgefüllt mit PAD wie der Tokenizer."""
    x = torch.tensor([(ids + [T.PAD] * L)[:L] for ids in id_lists], device=dev)
    with torch.no_grad():
        return T.tk.vqvae.encode(x)


# ====================================================================== Modell
class Merged(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(a.lm)
        self.lm.resize_token_embeddings(len(tok))
        d = self.lm.config.n_embd
        self.proj = nn.Linear(vq.embed_dim, d)
        self.q0 = nn.Parameter(torch.randn(LVall[0], d) * 0.02)
        self.lvl = nn.Embedding(a.k + 1, d)  # Ebenen 0..k-1 + Plan-Auslese
        self.heads = nn.ModuleList([nn.Linear(d, vq.embeddings[li].num_embeddings) for li in range(a.k)])
        for m in [self.proj, *self.heads]:
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.lvl.weight, std=0.02)

    def wte(self, ids):
        return self.lm.get_input_embeddings()(ids)

    def plan_blocks(self, codes, upto):
        """Eingabe-Blöcke für Ebenen 0..upto-1 (Codes nur der Ebenen < upto-1 werden gebraucht)."""
        B = codes.shape[0] if codes is not None else 1
        out = [self.q0[None].expand(B, -1, -1) + self.lvl.weight[0]]
        for j in range(1, upto):
            out.append(self.proj(level_input(codes, j)) + self.lvl.weight[j])
        return out

    def readout_block(self, codes):
        return self.proj(readout(codes)) + self.lvl.weight[a.k]

    def hidden(self, emb, att):
        pos = (att.cumsum(1) - 1).clamp_min(0)
        return self.lm.transformer(inputs_embeds=emb, attention_mask=att, position_ids=pos).last_hidden_state


def plan_len():
    return sum(LV) + LV[-1]


# GPT-2 hat 1024 Positionen: Kontext und Blocktext so begrenzen, dass Plan + Kontext + Text passen
TXT_MAX = 96 if train_arch == "C" else a.max_text
CTX_MAX = max(0, min(a.max_ctx, 1022 - plan_len() - TXT_MAX - 2)) if train_arch == "C" else 0
assert plan_len() + TXT_MAX + 2 <= 1024, "Plan zu lang für GPT-2"


# ====================================================================== Trainingsdaten
from bench_common import tinystories_first, tinystories_random  # noqa: E402


def make_example(text, rng):
    """-> (ctx_gpt_ids, ncm_block_ids, target_gpt_ids, is_last). B: ganze Geschichte, C: zufälliger Block."""
    ids = pre.encode(text)
    if train_arch == "B":
        blk = ids[:L - 1] + [T.EOS] if len(ids) < L else ids[:L]
        seg = T.dec(blk)
        return [], blk, [EOT] + tok(seg)["input_ids"][:TXT_MAX] + [EOT]
    nb = max(1, math.ceil(len(ids) / a.block))
    n = rng.randrange(nb)
    s0, s1 = n * a.block, min((n + 1) * a.block, len(ids))
    last = s1 >= len(ids)
    blk = ids[s0:s1] + ([T.EOS] if last else [])
    pre_txt = pre.decode(ids[:s0]) if s0 else ""
    seg = pre.decode(ids[:s1])[len(pre_txt):] if s0 else pre.decode(ids[:s1])
    ctx = tok(pre_txt)["input_ids"][-CTX_MAX:] if pre_txt and CTX_MAX else []
    tgt = [EOB] + tok(seg)["input_ids"][:TXT_MAX] + [EOT if last else EOB]
    return ctx, blk, tgt


def batch_loss(model, exs):
    ctxs, blks, tgts = zip(*exs)
    codes = encode_ids(blks)[:, :sum(LV)]
    rate = torch.rand(len(exs), device=dev) * a.code_noise
    noisy = corrupt(codes, rate)
    pb = model.plan_blocks(noisy, a.k)
    ro = model.readout_block(noisy)
    plan = torch.cat(pb + [ro], 1)  # (B, plan_len, d)
    seqs, labels_txt, starts = [], [], []
    for b in range(len(exs)):
        c = torch.tensor(ctxs[b], dtype=torch.long, device=dev)
        t = torch.tensor(tgts[b], dtype=torch.long, device=dev)
        parts = ([model.wte(c)] if len(c) else []) + [plan[b], model.wte(t)]
        seqs.append(torch.cat(parts, 0))
        starts.append(len(c))
        labels_txt.append(t)
    Tm = max(s.shape[0] for s in seqs)
    d = seqs[0].shape[1]
    emb = torch.zeros(len(seqs), Tm, d, device=dev)
    att = torch.zeros(len(seqs), Tm, dtype=torch.long, device=dev)
    for b, s in enumerate(seqs):
        emb[b, :s.shape[0]] = s
        att[b, :s.shape[0]] = 1
    h = model.hidden(emb, att)
    PL = plan_len()
    # Codes: Position der Ebene j sagt Codes der Ebene j voraus
    lc, off = 0.0, 0
    for j, l in enumerate(LV):
        hs = torch.stack([h[b, starts[b] + off: starts[b] + off + l] for b in range(len(seqs))])
        lg = model.heads[j](hs).float()
        lc = lc + F.cross_entropy(lg.reshape(-1, lg.shape[-1]), codes[:, sum(LV[:j]):sum(LV[:j]) + l].reshape(-1))
        off += l
    lc = lc / len(LV)
    # Text: Position i im Textteil sagt Token i+1 voraus
    lt, nt = 0.0, 0
    for b in range(len(seqs)):
        t = labels_txt[b]
        p0 = starts[b] + PL
        lg = model.lm.lm_head(h[b, p0:p0 + len(t) - 1]).float()
        lt = lt + F.cross_entropy(lg, t[1:], reduction="sum")
        nt += len(t) - 1
    lt = lt / nt
    return lt + a.lam * lc, lt.item(), lc.item()


# ====================================================================== Training
if a.mode == "train":
    import random
    del T.net
    T._gen_rec = T._check = T._tf = None  # hielten den Generator über eine Closure im Speicher
    import gc
    gc.collect()
    free(dev)
    model = Merged().to(dev)
    model.lm.gradient_checkpointing_enable()  # spart Aktivierungsspeicher (8-GB-Mac)
    model.lm.config.use_cache = False
    rng = random.Random(0)
    it = tinystories_random("train", seed=1)
    val_txt = tinystories_first("valid", 128 if not a.smoke else 4)
    val = [make_example(t, random.Random(i)) for i, t in enumerate(val_txt)]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    acc = max(1, a.bs // a.micro_bs)
    print(f"Training {train_arch}, k={a.k} ({sum(LV)} Plan-Codes + {LV[-1]} Auslese) | {n_params(model) / 1e6:.0f} M Parameter "
          f"| {a.steps} Schritte x {a.bs}", flush=True)
    t0 = time.time()
    model.train()
    for step in range(a.steps):
        for _ in range(acc):
            exs = [make_example(next(it), rng) for _ in range(a.micro_bs)]
            loss, lt, lc = batch_loss(model, exs)
            (loss / acc).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        del loss
        if dev == "mps":
            torch.mps.empty_cache()  # wechselnde Sequenzlängen zerstückeln sonst den Grafikspeicher
        if step % 100 == 0 or step == a.steps - 1:
            el = time.time() - t0
            print(f"Schritt {step:5d} | Text {lt:.3f} | Codes {lc:.3f} | {el / (step + 1):.2f}s/Schritt | "
                  f"Rest ca. {el / (step + 1) * (a.steps - step - 1) / 3600:.1f} h", flush=True)
        if (step + 1) % 1000 == 0 or step == a.steps - 1:
            model.eval()
            with torch.no_grad():
                vs = [batch_loss(model, val[i:i + a.micro_bs]) for i in range(0, len(val), a.micro_bs)]
            vt, vc = sum(v[1] for v in vs) / len(vs), sum(v[2] for v in vs) / len(vs)
            print(f"  Validierung: Text {vt:.4f} (PPL {math.exp(vt):.2f}) | Codes {vc:.3f}", flush=True)
            model.train()
            torch.save({"model": model.state_dict(), "step": step + 1, "val_text": vt, "val_codes": vc}, ckpt)
    print(f"Gespeichert: {ckpt}")
    if a.smoke:
        print("SCHNELLTEST OK")
    raise SystemExit(0)

# ====================================================================== Erzeugen
model = Merged().to(dev)
sd = torch.load(ckpt, map_location="cpu")
model.load_state_dict(sd["model"])
model.eval()
N_body = n_params(model.lm.transformer)
N_head_txt = model.lm.lm_head.weight.numel()


def fl(tokens):
    return 2 * (N_body + N_head_txt) * tokens


def sample(logits, temp=1.0):
    return torch.multinomial(F.softmax(logits.float() / temp, -1).reshape(-1, logits.shape[-1]), 1).view(logits.shape[:-1])


@torch.no_grad()
def gen_plan(ctx_emb, ctx_att):
    """Plan-Ebenen 0..k-1 erzeugen, eine Ebene pro Durchlauf. ctx_emb: (B, Lc, d) links aufgefüllt."""
    B = ctx_emb.shape[0]
    codes = torch.zeros(B, 0, dtype=torch.long, device=dev)
    for j in range(a.k):
        blocks = model.plan_blocks(codes if j else None, j + 1) if j else model.plan_blocks(None, 1)
        if j == 0:
            blocks = [blocks[0].expand(B, -1, -1)]
        emb = torch.cat([ctx_emb] + blocks, 1)
        att = torch.cat([ctx_att, torch.ones(B, emb.shape[1] - ctx_emb.shape[1], dtype=torch.long, device=dev)], 1)
        h = model.hidden(emb, att)
        cj = sample(model.heads[j](h[:, -LV[j]:]))
        codes = torch.cat([codes, cj], 1)
    return codes


@torch.no_grad()
def write(prefix_emb, prefix_att, max_new, stops):
    out = model.lm.generate(inputs_embeds=prefix_emb, attention_mask=prefix_att, max_new_tokens=max_new,
                            do_sample=True, temperature=1.0, top_k=0, top_p=1.0, pad_token_id=EOT, eos_token_id=stops)
    return out.tolist()


def left_pad(id_lists):
    Lc = max([len(x) for x in id_lists] + [0])
    ids = torch.full((len(id_lists), Lc), EOT, dtype=torch.long, device=dev)
    att = torch.zeros(len(id_lists), Lc, dtype=torch.long, device=dev)
    for b, x in enumerate(id_lists):
        if x:
            ids[b, Lc - len(x):] = torch.tensor(x, device=dev)
            att[b, Lc - len(x):] = 1
    return model.wte(ids), att


def cut(seq, stops):
    for i, t in enumerate(seq):
        if t in stops:
            return seq[:i], t
    return seq, None


@torch.no_grad()
def gen_B(Bn):
    e0, a0 = left_pad([[]] * Bn)
    codes = gen_plan(e0, a0)
    plan = torch.cat(model.plan_blocks(codes, a.k) + [model.readout_block(codes)], 1)
    emb = torch.cat([plan, model.wte(torch.full((Bn, 1), EOT, device=dev))], 1)
    att = torch.ones(Bn, emb.shape[1], dtype=torch.long, device=dev)
    outs = write(emb, att, TXT_MAX, [EOT])
    texts, flops, steps = [], [], []
    for r in outs:
        r, _ = cut(r, [EOT])
        texts.append(tok.decode(r, skip_special_tokens=True).strip())
        flops.append(fl(plan_len() + 1 + len(r)))
        steps.append(a.k + len(r))
    return texts, flops, steps, codes


@torch.no_grad()
def gen_C(Bn):
    ctx = [[] for _ in range(Bn)]
    text = ["" for _ in range(Bn)]
    done = [False] * Bn
    flops, steps = [0.0] * Bn, [0] * Bn
    for n in range(a.max_blocks):
        act = [b for b in range(Bn) if not done[b]]
        if not act:
            break
        ce, ca = left_pad([ctx[b][-CTX_MAX:] if CTX_MAX else [] for b in act])
        codes = gen_plan(ce, ca)
        if a.par:
            with torch.no_grad():
                lg = T.tk.vqvae.decoder(zcum(codes, K))
            ids = lg.argmax(-1).tolist()
            for i, b in enumerate(act):
                r = ids[i]
                end = T.EOS in r[:a.block + 1]
                r = r[:a.block]
                if T.EOS in r:
                    r = r[:r.index(T.EOS)]
                r = [t for t in r if t != T.PAD]
                seg = pre.decode(r)
                if seg and text[b] and not seg.startswith((" ", ".", ",", "!", "?", "'")):
                    seg = " " + seg
                text[b] += seg
                ctx[b] += tok(seg)["input_ids"]
                flops[b] += fl(ce.shape[1] + plan_len()) + 2 * T.N_dec * L
                steps[b] += K
                if end or not r:
                    done[b] = True
            continue
        plan = torch.cat(model.plan_blocks(codes, a.k) + [model.readout_block(codes)], 1)
        emb = torch.cat([ce, plan, model.wte(torch.full((len(act), 1), EOB, device=dev))], 1)
        att = torch.cat([ca, torch.ones(len(act), plan.shape[1] + 1, dtype=torch.long, device=dev)], 1)
        outs = write(emb, att, TXT_MAX, [EOB, EOT])
        for i, b in enumerate(act):
            r, stop = cut(outs[i], [EOB, EOT])
            seg = tok.decode(r, skip_special_tokens=True)
            text[b] += seg
            ctx[b] += r
            flops[b] += fl(ca[i].sum().item() + plan_len() + 1 + len(r))
            steps[b] += a.k + len(r) + 1
            if stop != EOB:
                done[b] = True
    return [t.strip() for t in text], flops, steps


SENT = re.compile(r"[.!?](\"|')?(\s|$)")


@torch.no_grad()
def plan_distance(codes, texts):
    """Abstand Plan <-> neu kodierter Text auf den schon geschriebenen Positionen."""
    ids = [pre.encode(t)[:L] for t in texts]
    re_codes = encode_ids(ids)[:, :sum(LV)]
    zp, zr = zcum(codes, a.k), zcum(re_codes, a.k)
    d = []
    for b, x in enumerate(ids):
        n = max(1, len(x))
        d.append(((zp[b, :n] - zr[b, :n]) ** 2).mean().item())
    return d


SENT_STATS = [0, 0]  # Sätze gesamt, Sätze mit Neuversuch


@torch.no_grad()
def gen_D(Bn, thr=None, calib=False):
    e0, a0 = left_pad([[]] * Bn)
    codes = gen_plan(e0, a0)
    plan = torch.cat(model.plan_blocks(codes, a.k) + [model.readout_block(codes)], 1)
    written = [[] for _ in range(Bn)]
    done = [False] * Bn
    flops = [fl(plan_len()) for _ in range(Bn)]
    steps = [a.k] * Bn
    dists, retried = [], [0] * Bn
    import random as _r
    rng = _r.Random(int(torch.randint(1 << 30, (1,)).item()))
    trig = {}
    while not all(done):
        act = [b for b in range(Bn) if not done[b] and len(written[b]) < a.max_text]
        if not act:
            break
        cands = {b: [] for b in act}
        trig = {b: rng.random() < a.flag_q for b in act}  # nur für die Zufalls-Kontrollen
        for attempt in range(1 + (0 if calib else a.R)):
            if a.d_mode == "plan" or calib:
                todo = [b for b in act if attempt == 0 or (cands[b] and cands[b][-1][2] > thr and len(cands[b]) == attempt)]
            else:
                todo = [b for b in act if attempt == 0 or (trig[b] and len(cands[b]) == attempt)]
            if not todo:
                break
            te, ta = left_pad([[EOT] + written[b] for b in todo])
            pi = torch.stack([plan[b] for b in todo])
            emb = torch.cat([pi, te], 1)
            att = torch.cat([torch.ones(len(todo), pi.shape[1], dtype=torch.long, device=dev), ta], 1)
            # Polster zwischen Plan und Text sind maskiert, Positionen kommen aus der Maske (wie im Training)
            outs = write(emb, att, 48, [EOT])
            news = []
            for i, b in enumerate(todo):
                r = outs[i]
                r_cut, stop = cut(r, [EOT])
                txt = tok.decode(r_cut, skip_special_tokens=True)
                m = SENT.search(txt)
                if m and m.end() < len(txt.rstrip()):
                    keep = tok(txt[:m.end()])["input_ids"]
                    fin = False
                else:
                    keep, fin = r_cut, (stop == EOT)
                flops[b] += fl(len(written[b]) + 1 + len(r))
                steps[b] += len(r)
                if attempt:
                    retried[b] = 1
                news.append((b, keep, fin))
            ds = plan_distance(codes[[b for b, _, _ in news]],
                               [tok.decode(written[b] + k, skip_special_tokens=True) for b, k, _ in news])
            for (b, k, fin), dv in zip(news, ds):
                cands[b].append((k, fin, dv))
                if attempt == 0:
                    dists.append(dv)
        for b in act:
            SENT_STATS[0] += 1
            SENT_STATS[1] += len(cands[b]) > 1
            if a.d_mode == "rand" and not calib:
                k, fin, _ = rng.choice(cands[b])
            else:
                k, fin, _ = min(cands[b], key=lambda c: c[2])
            written[b] += k
            if fin or not k:
                done[b] = True
    texts = [tok.decode(w, skip_special_tokens=True).strip() for w in written]
    return texts, flops, steps, dists, retried


# ---------------------------------------------------------------- Lauf
name = {"B": f"merge_B_k{a.k}", "C": f"merge_C{'par' if a.par else ''}_k{a.k}", "D": f"merge_D_k{a.k}" + ("" if a.d_mode == "plan" else "_" + a.d_mode)}[a.arch]
if a.tag:
    name += "_" + a.tag
thr = None
if a.arch == "D":
    torch.manual_seed(999)
    _, _, _, dcal, _ = gen_D(32 if not a.smoke else 2, calib=True)
    thr = torch.quantile(torch.tensor(dcal), 1 - a.flag_q).item()
    SENT_STATS[:] = [0, 0]
    print(f"D: Schwelle Plan-Abstand (oberes {a.flag_q:.0%}) = {thr:.4f}", flush=True)
seeds = [int(s) for s in a.seeds.split(",")]
bs = 32 if dev == "cuda" else 8
texts, seed_of, flops, steps, retried = [], [], [], [], []
t0 = time.time()
for s in seeds:
    torch.manual_seed(s)
    for b0 in range(0, a.n_samples, bs):
        Bn = min(bs, a.n_samples - b0)
        if a.arch == "B":
            tx, f_, st, _ = gen_B(Bn)
            rt = [0] * Bn
        elif a.arch == "C":
            tx, f_, st = gen_C(Bn)
            rt = [0] * Bn
        else:
            tx, f_, st, _, rt = gen_D(Bn, thr)
        texts += tx
        flops += f_
        steps += st
        retried += rt
        seed_of += [s] * Bn
        if dev == "mps":
            free(dev)
    print(f"Seed {s}: {a.n_samples} Geschichten ({time.time() - t0:.0f}s)", flush=True)
path = os.path.join(a.out, "merge.json")
res = json.load(open(path)) if os.path.exists(path) else {"task": "tinystories", "uncond": True, "variants": {}}
if a.arch == "D":
    print(f"D ({a.d_mode}): {SENT_STATS[0]} Sätze, davon {100 * SENT_STATS[1] / max(SENT_STATS[0], 1):.1f} % neu versucht", flush=True)
res["variants"][name] = {"texts": texts, "seed": seed_of, "flops": flops, "nfe": steps, "retried": retried, "sent_retry_frac": SENT_STATS[1] / max(SENT_STATS[0], 1),
                         "seconds": time.time() - t0, "train_steps": sd.get("step"),
                         "val_text": sd.get("val_text"), "val_codes": sd.get("val_codes")}
json.dump(res, open(path, "w"), ensure_ascii=False)
w = [len(t.split()) for t in texts]
print(f"Gespeichert: {path} (Variante {name}) | Wörter im Mittel {sum(w) / len(w):.0f}, max {max(w)}")
print("Beispiel:", texts[0][:300])
if a.smoke:
    print("SCHNELLTEST OK")
