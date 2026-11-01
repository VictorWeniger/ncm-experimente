"""Exp. 5 Stufe 4: "NCM denkt, das LLM spricht" (TinyStories).

GPT-2 small bekommt die k gröbsten Code-Ebenen einer Geschichte als weiches Präfix (jeder Code über
das eingefrorene NCM-Codebuch eingebettet, linear auf 768 Dimensionen projiziert, plus Ebenen-Einbettung)
und schreibt daraus den Text Token für Token. Beim Erzeugen kommen die groben Ebenen von NCM.

  k = 0  ist dasselbe GPT-2 mit identischem Training, aber ohne Plan, also das faire reine LLM.
  k = 3  sind die Ebenen 1, 8, 16 Codes = 25 Codes (Standard).

Ablauf:
  python experiments/hybrid_stage4.py train --k 3 --steps 6000      # Cluster empfohlen
  python experiments/hybrid_stage4.py train --k 0 --steps 6000      # gleiches Budget ohne Plan
  python experiments/hybrid_stage4.py gen   --k 3                   # 5 Seeds x 256 wie im Paper
  python experiments/hybrid_stage4.py gen   --k 0
  python experiments/bench_score.py --task tinystories              # nimmt stage4.json mit auf

Optional --teacher bei gen: Pläne aus echten Validierungsgeschichten statt von NCM. Das zeigt, wie viel
der Plan überhaupt transportieren kann (Obergrenze für k).
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_common import ROOT, Task, free, n_params, pick_device

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "gen"])
ap.add_argument("--k", type=int, default=3, help="Anzahl grober Ebenen als Plan (0 = ohne Plan)")
ap.add_argument("--lm", default="gpt2")
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--micro_bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--code_noise", type=float, default=0.1, help="Anteil zufällig ersetzter Plan-Codes im Training")
ap.add_argument("--max_text", type=int, default=192, help="max. GPT-2-Tokens pro Geschichte")
ap.add_argument("--n_samples", type=int, default=256)
ap.add_argument("--seeds", default="42,0,1,2,3")
ap.add_argument("--teacher", action="store_true")
ap.add_argument("--device", default="auto")
ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "artifacts", "stage4"))
ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench", "tinystories"))
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

dev = pick_device(a.device)
if a.smoke:
    a.steps, a.bs, a.micro_bs, a.n_samples, a.seeds = 4, 4, 2, 8, "42"
    a.out = a.out + "-smoke"
os.makedirs(a.ckpt_dir, exist_ok=True)
os.makedirs(a.out, exist_ok=True)
ckpt = os.path.join(a.ckpt_dir, f"stage4_k{a.k}{'_smoke' if a.smoke else ''}.pt")

T = Task("tinystories", dev)
vq, pre = T.vq, T.tk.pre_tokenizer
LV = T.levels[:a.k]
NC = sum(LV)


class PlanLM(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.lm = AutoModelForCausalLM.from_pretrained(a.lm)
        d = self.lm.config.n_embd
        self.proj = nn.Linear(vq.embed_dim, d)
        self.lvl = nn.Embedding(max(a.k, 1), d)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.lvl.weight, std=0.02)

    def plan_embeds(self, codes):
        """codes: (B, NC) Codes der Ebenen 0..k-1 -> (B, NC, d)"""
        if a.k == 0:
            return None
        parts, s = [], 0
        for li, l in enumerate(LV):
            with torch.no_grad():
                e = vq.embeddings[li](codes[:, s:s + l]).float()
            parts.append(self.proj(e) + self.lvl.weight[li])
            s += l
        return torch.cat(parts, 1)

    def forward(self, codes, ids, att):
        wte = self.lm.get_input_embeddings()
        x = wte(ids)
        pe = self.plan_embeds(codes)
        if pe is not None:
            x = torch.cat([pe, x], 1)
            att = torch.cat([torch.ones(att.shape[0], NC, dtype=att.dtype, device=att.device), att], 1)
        return self.lm(inputs_embeds=x, attention_mask=att).logits[:, NC:]


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.lm)
tok.pad_token = tok.eos_token
EOT = tok.eos_token_id


def ncm_ids(text):
    ids = pre.encode(text)[:T.L]
    if len(ids) < T.L:
        ids = ids + [T.EOS] + [T.PAD] * (T.L - len(ids) - 1)
    return ids[:T.L]


@torch.no_grad()
def batch_from_texts(texts, noise=0.0):
    ids = torch.tensor([ncm_ids(t) for t in texts], device=dev)
    codes = T.tk.vqvae.encode(ids)[:, :NC] if a.k else None
    if codes is not None and noise > 0:
        for li, (s, l) in enumerate(zip([sum(LV[:i]) for i in range(len(LV))], LV)):
            m = torch.rand(codes.shape[0], l, device=dev) < noise
            r = torch.randint(vq.embeddings[li].num_embeddings, (codes.shape[0], l), device=dev)
            codes[:, s:s + l] = torch.where(m, r, codes[:, s:s + l])
    # Zieltext = was die Codes beschreiben (auf 128 NCM-Tokens gekürzt)
    tgt = [T.dec(r.tolist()) for r in ids.cpu()]
    enc = [[EOT] + tok(t)["input_ids"][:a.max_text] + [EOT] for t in tgt]
    L = max(len(e) for e in enc)
    x = torch.full((len(enc), L), EOT)
    att = torch.zeros(len(enc), L, dtype=torch.long)
    y = torch.full((len(enc), L), -100)
    for j, e in enumerate(enc):
        x[j, :len(e)] = torch.tensor(e)
        att[j, :len(e)] = 1
        y[j, :len(e) - 1] = torch.tensor(e[1:])
    return codes, x.to(dev), att.to(dev), y.to(dev)


# ====================================================================== Training
if a.mode == "train":
    from bench_common import tinystories_first, tinystories_random
    del T.net
    T._gen_rec = T._check = T._tf = None  # hielten den Generator über eine Closure im Speicher
    import gc
    gc.collect()
    free(dev)
    model = PlanLM().to(dev)
    model.lm.gradient_checkpointing_enable()  # spart Aktivierungsspeicher (8-GB-Mac)
    model.lm.config.use_cache = False
    val = tinystories_first("valid", 256 if not a.smoke else 4)
    it = tinystories_random("train", seed=0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    acc = a.bs // a.micro_bs
    print(f"Training Stufe 4, k={a.k} ({NC} Plan-Codes) | {n_params(model) / 1e6:.0f} M Parameter | {a.steps} Schritte x {a.bs}",
          flush=True)

    @torch.no_grad()
    def val_loss():
        model.eval()
        tot = n = 0.0
        for i in range(0, len(val), a.micro_bs):
            c, x, att, y = batch_from_texts(val[i:i + a.micro_bs])
            lg = model(c, x, att)
            l = F.cross_entropy(lg.float().transpose(1, 2), y, reduction="sum", ignore_index=-100)
            tot += l.item()
            n += (y != -100).sum().item()
        model.train()
        return tot / n

    t0 = time.time()
    model.train()
    for step in range(a.steps):
        for _ in range(acc):
            texts = [next(it) for _ in range(a.micro_bs)]
            c, x, att, y = batch_from_texts(texts, a.code_noise)
            lg = model(c, x, att)
            loss = F.cross_entropy(lg.float().transpose(1, 2), y, ignore_index=-100) / acc
            loss.backward()
            loss_val = loss.item() * acc
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        del loss
        if dev == "mps":
            torch.mps.empty_cache()  # wechselnde Sequenzlängen zerstückeln sonst den Grafikspeicher
        if step % 100 == 0 or step == a.steps - 1:
            el = time.time() - t0
            print(f"Schritt {step:5d} | Loss {loss_val:.3f} | {el / (step + 1):.2f}s/Schritt | "
                  f"Rest ca. {el / (step + 1) * (a.steps - step - 1) / 3600:.1f} h", flush=True)
        if (step + 1) % 1000 == 0 or step == a.steps - 1:
            vl = val_loss()
            print(f"  Validierung: Loss {vl:.4f} (PPL {math.exp(vl):.2f})", flush=True)
            torch.save({"model": model.state_dict(), "k": a.k, "step": step + 1, "val_loss": vl}, ckpt)
    print(f"Gespeichert: {ckpt}")
    if a.smoke:
        print("SCHNELLTEST OK")

# ====================================================================== Erzeugen
else:
    model = PlanLM().to(dev)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"])
    model.eval()
    N_lm = n_params(model.lm)
    seeds = [int(s) for s in a.seeds.split(",")]
    texts, seed_of, flops, steps = [], [], [], []
    # Kosten nur für die k groben Ebenen (Ebene 0 kommt bei TinyStories aus der Unigramm-Verteilung)
    f_plan = 2 * T.N_gen * sum(T.P + l for l in LV[1:]) if a.k > 1 else 0
    nfe_plan = max(a.k - 1, 0)
    val = None
    if a.teacher:
        from bench_common import tinystories_first
        val = tinystories_first("valid", len(seeds) * a.n_samples)
    t0 = time.time()
    bs = 32 if dev == "cuda" else 16
    for si, s in enumerate(seeds):
        torch.manual_seed(s)
        for b0 in range(0, a.n_samples, bs):
            B = min(bs, a.n_samples - b0)
            codes = None
            if a.k:
                with torch.no_grad():
                    if a.teacher:
                        off = si * a.n_samples + b0
                        codes = T.tk.vqvae.encode(torch.tensor([ncm_ids(t) for t in val[off:off + B]], device=dev))[:, :NC]
                    else:
                        pfx = torch.full((B, 1), T.EOS, device=dev)
                        g = T.net.generate(T.l0_sample(B), pfx, temperature=T.temp, top_k=T.top_k, top_p=T.top_p)
                        codes = g[:, :NC]
            with torch.no_grad():
                bos = torch.full((B, 1), EOT, device=dev)
                x = model.lm.get_input_embeddings()(bos)
                pe = model.plan_embeds(codes)
                if pe is not None:
                    x = torch.cat([pe, x], 1)
                att = torch.ones(B, x.shape[1], dtype=torch.long, device=dev)
                out = model.lm.generate(inputs_embeds=x, attention_mask=att, max_new_tokens=a.max_text, do_sample=True,
                                        temperature=1.0, top_k=0, top_p=1.0, pad_token_id=EOT, eos_token_id=EOT)
            for r in out.tolist():
                if EOT in r:
                    r = r[:r.index(EOT)]
                texts.append(tok.decode(r, skip_special_tokens=True).strip())
                seed_of.append(s)
                flops.append(f_plan + 2 * N_lm * (NC + 1 + len(r)))
                steps.append(nfe_plan + len(r))
            if dev == "mps":
                free(dev)
        print(f"Seed {s}: {a.n_samples} Geschichten ({time.time() - t0:.0f}s)", flush=True)
    name = ("llm_ft" if a.k == 0 else f"s4_k{a.k}") + ("_teacher" if a.teacher else "")
    path = os.path.join(a.out, "stage4.json")
    res = json.load(open(path)) if os.path.exists(path) else {"task": "tinystories", "uncond": True, "variants": {}}
    res["variants"][name] = {"texts": texts, "seed": seed_of, "flops": flops, "nfe": steps,
                             "retried": [0] * len(texts), "seconds": time.time() - t0,
                             "train_steps": sd.get("step"), "val_loss": sd.get("val_loss")}
    json.dump(res, open(path, "w"), ensure_ascii=False)
    print(f"Gespeichert: {path} (Variante {name})")
    if a.smoke:
        print("SCHNELLTEST OK")
