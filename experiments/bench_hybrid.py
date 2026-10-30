"""Benchmark Teil 2: Mischformen aus NCM und normalem Sprachmodell (Exp. 5, Stufen 1-3), ohne Training.

Liest die Texte aus bench_ncm.py (results/bench/<task>/ncm.json) und schreibt hybrid.json im
selben Format; bewertet wird danach mit bench_score.py.

Varianten (alle auf denselben 5 x 256 Plätzen wie die NCM-Baseline):
  llm_only        nur das kleine Sprachmodell (--lm), ohne NCM
  s1_rewrite      Stufe 1: NCM schreibt, die von gen_llR markierten 20 % schreibt ein Instruct-Modell neu
  s1_rewrite_rand Kontrolle: zufällige 20 % neu schreiben lassen (bringt die Auswahl etwas?)
  s1_rewrite_all  Kontrolle: alles neu schreiben lassen (was macht das Umschreiben allein mit der Metrik?)
  s1_replace      Stufe 1b: markierte 20 % durch ein frisches Sample von --lm ersetzen
  s2_sentence     Stufe 2: in markierten Geschichten nur den Satz mit der schwächsten Decoder-Konfidenz
                  von --lm neu schreiben lassen (Kontext davor, Rest danach bleibt)
  s3_spec         Stufe 3: --lm liest jeden NCM-Entwurf einmal, übernimmt ihn bis zum ersten Token,
                  das es unplausibel findet (unteres spec_q-Quantil), und schreibt ab dort selbst weiter
  s3_pick{N}      Zusatz: --lm wählt aus den N NCM-Kandidaten (aus best{N}) den plausibelsten
                  (Vergleich: best{N} wählt mit NCMs eigenem Signal)

Kosten: FLOPs ~ 2 x Parameter x verarbeitete Tokens, dazu sequenzielle Schritte (NCM-Ebenen + LLM-Tokens).

Aufruf:
  python experiments/bench_hybrid.py --task tinystories
  python experiments/bench_hybrid.py --task lm1b-small --lm gpt2 --variants llm_only,s1_rewrite,s1_replace,s3_spec,s3_pick4
"""
import argparse
import json
import math
import os
import re
import time

import torch
import torch.nn.functional as F

from bench_common import ROOT, Task, free, pick_device, n_params

DEFAULT_LM = {"tinystories": "roneneldan/TinyStories-33M", "lm1b-small": "gpt2"}
ap = argparse.ArgumentParser()
ap.add_argument("--task", default="tinystories")
ap.add_argument("--lm", default="", help="kleines Sprachmodell für Weiterschreiben/Prüfen (Standard je Aufgabe)")
ap.add_argument("--rewriter", default="Qwen/Qwen2.5-0.5B-Instruct", help="Instruct-Modell für Stufe 1")
ap.add_argument("--variants", default="llm_only,s1_rewrite,s1_rewrite_rand,s1_rewrite_all,s1_replace,s2_sentence,s3_spec,s3_pick")
ap.add_argument("--spec_q", type=float, default=0.02, help="Stufe 3: Anteil der NCM-Tokens, die das LLM als unplausibel gelten lässt")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--device", default="auto")
ap.add_argument("--in_dir", default=os.path.join(ROOT, "results", "bench"))
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()

dev = pick_device(a.device)
task_dir = os.path.join(a.in_dir, a.task + ("-smoke" if a.smoke else ""))
src = json.load(open(os.path.join(task_dir, "ncm.json")))
assert src["uncond"], "Die Mischformen sind für die freien Text-Aufgaben gebaut (tinystories, lm1b-small)."
lm_name = a.lm or DEFAULT_LM.get(a.task, "gpt2")
variants = a.variants.split(",")
base = src["variants"]["baseline"]
texts0, seeds = base["texts"], base["seed"]
M = len(texts0)
ex = src["extra"]
llr0 = torch.tensor(ex["gen_llR"])
thr = ex["threshold"]
flag = (llr0 < thr)
q = flag.float().mean().item()
F_NCM = src["flops_generate"] + src["flops_score"]
NFE_NCM = src["nfe_generate"] + src["nfe_score"]
print(f"Gerät {dev} | {a.task} | {M} NCM-Texte | markiert {100 * q:.1f} % | LM {lm_name} | Umschreiber {a.rewriter}", flush=True)

out_path = os.path.join(task_dir, "hybrid.json")
res = {"task": a.task, "uncond": True, "lm": lm_name, "rewriter": a.rewriter, "variants": {}, "extra": {}}
if os.path.exists(out_path):
    old = json.load(open(out_path))
    res["variants"].update(old.get("variants", {}))


def save():
    json.dump(res, open(out_path, "w"), ensure_ascii=False)


# ====================================================================== Sprachmodell-Hilfen
class LM:
    def __init__(self, name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            self.tok = AutoTokenizer.from_pretrained(name)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # TinyStories-Modelle
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        half = dev == "cuda"
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if half else torch.float32).to(dev).eval()
        self.N = n_params(self.m)
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.name = name

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def generate(self, prompts_ids, max_new, sample=True, stop_sentence=False):
        """prompts_ids: Liste von Token-Listen. Rückgabe: (Text der Fortsetzung, #neue Tokens, #Prompt-Tokens)."""
        outs = []
        for i in range(0, len(prompts_ids), a.bs):
            chunk = prompts_ids[i:i + a.bs]
            T = max(len(c) for c in chunk)
            ids = torch.full((len(chunk), T), self.tok.pad_token_id)
            att = torch.zeros(len(chunk), T, dtype=torch.long)
            for j, c in enumerate(chunk):
                ids[j, T - len(c):] = torch.tensor(c)
                att[j, T - len(c):] = 1
            g = self.m.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new,
                                do_sample=sample, temperature=1.0 if sample else None, top_k=0 if sample else None,
                                top_p=1.0 if sample else None, pad_token_id=self.tok.pad_token_id,
                                eos_token_id=self.tok.eos_token_id)
            for j, c in enumerate(chunk):
                new = g[j, T:].tolist()
                if self.tok.eos_token_id in new:
                    new = new[:new.index(self.tok.eos_token_id)]
                txt = self.tok.decode(new, skip_special_tokens=True)
                if stop_sentence:
                    m = re.search(r"[.!?](\"|')?(\s|$)", txt)
                    if m:
                        txt = txt[:m.end()].rstrip()
                outs.append((txt, len(new), len(c)))
            if dev == "mps":
                free(dev)
        return outs

    @torch.no_grad()
    def token_logps(self, texts):
        """Log-Wahrscheinlichkeit jedes Tokens (BOS vorangestellt) + Zeichen-Offsets."""
        res_ = []
        sb = a.bs if dev == "cuda" else 4  # volle Vokabular-Logits sind groß -> auf dem Mac kleine Häppchen
        for i in range(0, len(texts), sb):
            chunk = texts[i:i + sb]
            enc = [self.tok(t, add_special_tokens=False, return_offsets_mapping=True) for t in chunk]
            seqs = [[self.bos] + e["input_ids"] for e in enc]
            T = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), T), self.tok.pad_token_id)
            att = torch.zeros(len(seqs), T, dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            lg = self.m(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits[:, :-1]
            tgt = ids[:, 1:].to(dev).unsqueeze(-1)
            lp = (lg.gather(-1, tgt).squeeze(-1).float() - torch.logsumexp(lg.float(), -1)).cpu()
            del lg
            for j, e in enumerate(enc):
                n = len(e["input_ids"])
                res_.append((lp[j, :n].tolist(), e["offset_mapping"]))
            if dev == "mps":
                free(dev)
        return res_

    def close(self):
        self.m.to("cpu")
        del self.m
        free(dev)


def entry(texts, flops, steps, retried, secs):
    return {"texts": texts, "seed": seeds, "flops": flops, "nfe": steps, "retried": retried, "seconds": secs}


def max_len_tokens(lm):
    L = sorted(len(lm.ids(t)) for t in texts0)
    return max(8, L[int(0.95 * (len(L) - 1))])


# ====================================================================== Kleines LM: llm_only, s1_replace, s2, s3
need_lm = any(v in variants for v in ("llm_only", "s1_replace", "s2_sentence", "s3_spec", "s3_pick"))
if need_lm:
    lm = LM(lm_name)
    max_new = max_len_tokens(lm)
    print(f"{lm_name}: {lm.N / 1e6:.0f} M Parameter, Länge bis {max_new} Tokens", flush=True)
    res["extra"]["lm_params"] = lm.N

    if "llm_only" in variants:
        t0 = time.time()
        torch.manual_seed(0)
        outs = lm.generate([[lm.bos]] * M, max_new)
        res["variants"]["llm_only"] = entry([o[0].strip() for o in outs], [2 * lm.N * (o[1] + o[2]) for o in outs],
                                            [o[1] for o in outs], [0] * M, time.time() - t0)
        print(f"llm_only fertig ({time.time() - t0:.0f}s)", flush=True)
        save()

    if "s1_replace" in variants:
        t0 = time.time()
        torch.manual_seed(1)
        idx = flag.nonzero().squeeze(1).tolist()
        outs = lm.generate([[lm.bos]] * len(idx), max_new)
        texts, flops, steps = list(texts0), [F_NCM] * M, [NFE_NCM] * M
        for i, o in zip(idx, outs):
            texts[i] = o[0].strip()
            flops[i] += 2 * lm.N * (o[1] + o[2])
            steps[i] += o[1]
        res["variants"]["s1_replace"] = entry(texts, flops, steps, flag.int().tolist(), time.time() - t0)
        print(f"s1_replace fertig ({time.time() - t0:.0f}s)", flush=True)
        save()

    if "s2_sentence" in variants:
        # Decoder-Konfidenz pro NCM-Token -> pro Satz; nur den schwächsten Satz neu schreiben
        t0 = time.time()
        T = Task(a.task, "cpu")
        raw_all, tlp_all = ex["raw_ids"], ex["tok_lp"]
        texts, flops, steps = list(texts0), [F_NCM] * M, [NFE_NCM] * M
        jobs = []
        for i in flag.nonzero().squeeze(1).tolist():
            raw, tlp, full = raw_all[i], tlp_all[i], texts0[i]
            n_tok = raw.index(T.EOS) if T.EOS in raw else len(raw)
            ends = [len(T.dec(raw[:k + 1])) for k in range(n_tok)]
            spans = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]*[\"']?\s*", full)]
            if len(spans) < 2:
                spans = [(0, len(full))]
            best, best_s = None, 1e9
            for (s0, s1) in spans:
                vals = [tlp[k] for k in range(n_tok) if s0 < ends[k] <= s1]
                if vals and sum(vals) / len(vals) < best_s:
                    best, best_s = (s0, s1), sum(vals) / len(vals)
            if best is None:
                continue
            jobs.append((i, full[:best[0]], full[best[1]:]))
        del T
        torch.manual_seed(2)
        outs = lm.generate([[lm.bos] + lm.ids(pre) for _, pre, _ in jobs], 60, stop_sentence=True)
        for (i, pre, post), o in zip(jobs, outs):
            mid = o[0].strip()
            texts[i] = (pre + (" " if pre and not pre.endswith(" ") else "") + mid + (" " + post.lstrip() if post else "")).strip()
            flops[i] += 2 * lm.N * (o[1] + o[2])
            steps[i] += o[1]
        res["variants"]["s2_sentence"] = entry(texts, flops, steps, flag.int().tolist(), time.time() - t0)
        print(f"s2_sentence fertig: {len(jobs)} Sätze ersetzt ({time.time() - t0:.0f}s)", flush=True)
        save()

    if "s3_spec" in variants:
        t0 = time.time()
        lps = lm.token_logps(texts0)
        allv = torch.tensor([v for lp, _ in lps for v in lp])
        cut_thr = torch.quantile(allv, a.spec_q).item()
        jobs, texts, flops, steps, cut = [], list(texts0), [], [], []
        for i, (lp, off) in enumerate(lps):
            bad = [k for k, v in enumerate(lp) if v < cut_thr]
            f = F_NCM + 2 * lm.N * (len(lp) + 1)  # NCM-Entwurf + ein LLM-Lesedurchgang
            flops.append(f)
            steps.append(NFE_NCM + 1)
            if bad:
                k = bad[0]
                pre = texts0[i][:off[k][0]]
                jobs.append((i, pre, max(4, len(lp) - k + 8)))
                cut.append(k / max(len(lp), 1))
            else:
                cut.append(1.0)
        torch.manual_seed(3)
        by_len = {}
        for j in jobs:
            by_len.setdefault(min(max_new, j[2]), []).append(j)
        for mx, js in by_len.items():
            outs = lm.generate([[lm.bos] + lm.ids(pre) for _, pre, _ in js], mx)
            for (i, pre, _), o in zip(js, outs):
                texts[i] = (pre + o[0]).strip()
                flops[i] += 2 * lm.N * o[1]  # Präfix steckt schon im Lesedurchgang (KV-Cache)
                steps[i] += o[1]
        res["variants"]["s3_spec"] = entry(texts, flops, steps, [0] * M, time.time() - t0)
        res["extra"]["s3_spec"] = {"logp_threshold": cut_thr, "frac_rewritten": len(jobs) / M,
                                   "mean_accepted_fraction": sum(cut) / len(cut)}
        print(f"s3_spec fertig: {100 * len(jobs) / M:.0f} % ab einer Stelle neu geschrieben, im Mittel "
              f"{100 * sum(cut) / len(cut):.0f} % des Entwurfs übernommen ({time.time() - t0:.0f}s)", flush=True)
        save()

    if "s3_pick" in variants and "candidates" in ex:
        t0 = time.time()
        C = ex["candidates"]
        k = len(C[0])
        flat = [t for c in C for t in c]
        lps = lm.token_logps(flat)
        score = torch.tensor([sum(lp) / max(len(lp), 1) for lp, _ in lps]).view(M, k)
        pick = score.argmax(1)
        ntok = torch.tensor([len(lp) + 1 for lp, _ in lps], dtype=torch.float64).view(M, k).sum(1)
        texts = [C[i][pick[i]] for i in range(M)]
        flops = [k * F_NCM + 2 * lm.N * ntok[i].item() for i in range(M)]
        res["variants"][f"s3_pick{k}"] = entry(texts, flops, [k * NFE_NCM + 1] * M, [1] * M, time.time() - t0)
        best_internal = torch.tensor(ex["cand_llr"]).argmax(1)
        res["extra"]["s3_pick_agree_with_internal"] = (pick == best_internal).float().mean().item()
        print(f"s3_pick{k} fertig: gleiche Wahl wie NCM-Signal in {100 * res['extra']['s3_pick_agree_with_internal']:.0f} % "
              f"({time.time() - t0:.0f}s)", flush=True)
        save()
    lm.close()

# ====================================================================== Stufe 1: Umschreiben mit Instruct-Modell
s1 = [v for v in variants if v.startswith("s1_rewrite")]
if s1:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rt = AutoTokenizer.from_pretrained(a.rewriter)
    rt.padding_side = "left"
    if rt.pad_token is None:
        rt.pad_token = rt.eos_token
    rm = AutoModelForCausalLM.from_pretrained(a.rewriter, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    N_r = n_params(rm)
    kind = "short story" if a.task == "tinystories" else "sentence"

    def prompt(t):
        msg = (f"Rewrite the following {kind} so that it is fluent, grammatical and coherent. Keep the same "
               f"characters, events and roughly the same length. Output only the rewritten {kind}.\n\n{t}")
        r = rt.apply_chat_template([{"role": "user", "content": msg}], tokenize=True, add_generation_prompt=True)
        if isinstance(r, dict) or hasattr(r, "keys"):
            r = r["input_ids"]
        return list(r)

    cache = {}

    @torch.no_grad()
    def rewrite(idx):
        todo = [i for i in idx if i not in cache]
        for s in range(0, len(todo), a.bs):
            ch = todo[s:s + a.bs]
            ps = [prompt(texts0[i]) for i in ch]
            T = max(len(p) for p in ps)
            ids = torch.full((len(ps), T), rt.pad_token_id)
            att = torch.zeros(len(ps), T, dtype=torch.long)
            for j, p in enumerate(ps):
                ids[j, T - len(p):] = torch.tensor(p)
                att[j, T - len(p):] = 1
            mx = int(1.5 * max(len(rt(texts0[i])["input_ids"]) for i in ch)) + 16
            g = rm.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), max_new_tokens=mx, do_sample=False,
                            pad_token_id=rt.pad_token_id)
            for j, i in enumerate(ch):
                new = g[j, T:].tolist()
                n_new = len([t for t in new if t != rt.pad_token_id])
                cache[i] = (rt.decode(new, skip_special_tokens=True).strip(), n_new, len(ps[j]))
            if dev == "mps":
                free(dev)
            print(f"  umgeschrieben {min(s + a.bs, len(todo))}/{len(todo)}", flush=True)

    gen_r = torch.Generator().manual_seed(4)
    sel = {"s1_rewrite": flag.nonzero().squeeze(1).tolist(),
           "s1_rewrite_rand": (torch.rand(M, generator=gen_r) < q).nonzero().squeeze(1).tolist(),
           "s1_rewrite_all": list(range(M))}
    for v in s1:
        t0 = time.time()
        idx = sel[v]
        rewrite(idx)
        texts, flops, steps, ret = list(texts0), [F_NCM] * M, [NFE_NCM] * M, [0] * M
        for i in idx:
            txt, n_new, n_p = cache[i]
            if txt:
                texts[i] = txt
            flops[i] += 2 * N_r * (n_new + n_p)
            steps[i] += n_new
            ret[i] = 1
        res["variants"][v] = entry(texts, flops, steps, ret, time.time() - t0)
        print(f"{v} fertig ({len(idx)} Texte, {time.time() - t0:.0f}s)", flush=True)
        save()
    res["extra"]["rewriter_params"] = N_r
    rm.to("cpu")
    del rm
    free(dev)

save()
print(f"\nGespeichert: {out_path}\nWeiter mit: python experiments/bench_score.py --task {a.task}" + (" --smoke" if a.smoke else ""))
if a.smoke:
    print("SCHNELLTEST OK")
