"""Gemeinsamer Unterbau für die Benchmark-Skripte (bench_ncm, bench_hybrid, hybrid_stage4, bench_score).

Bildet die Evaluation des Papers (ncm/scripts/eval_shard.py + eval_merge.py) nach, läuft aber
auf MPS, CUDA und CPU und liefert zusätzlich das interne Signal gen_llR pro Sample.

Wichtig für die Vergleichbarkeit mit den Paper-Zahlen:
  * Sampling exakt wie in der jeweiligen Config (Temperatur, top_k, top_p, cfg).
  * Dekodierung mit denselben Frequenz-/Nachbarschaftsstrafen (pick_tokens) wie eval_shard.py.
  * Metriken wie eval_merge.py (genppl, genppl_flm, rouge, bleu).
"""
import math
import os
import pickle
import sys
from collections import Counter

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCM = os.path.join(ROOT, "ncm")
sys.path.insert(0, NCM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Wo liegen Checkpoints und Testdaten (wie im README des Papers: artifacts/<task>, data/<task>)
CKPT_DIR = {
    "tinystories": "artifacts/tinystories",
    "lm1b-small": "artifacts/lm1b/small",
    "lm1b-medium": "artifacts/lm1b/medium",
    "lm1b-large": "artifacts/lm1b/large",
    "owt": "artifacts/owt",
    "xsum": "artifacts/xsum",
    "wmt14": "artifacts/wmt14",
}
DATA_DIR = {"xsum": "data/xsum", "wmt14": "data/wmt14"}


def pick_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def free(dev):
    import gc
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()


# MPS kann "area"-Interpolation nicht -> auf CPU umleiten (wie in den bisherigen Skripten)
_F_interp = torch.nn.functional.interpolate


def _interp_mps_safe(input, *args, **kwargs):
    mode = kwargs.get("mode", args[2] if len(args) > 2 else "nearest")
    if input.device.type == "mps" and mode == "area":
        return _F_interp(input.cpu(), *args, **kwargs).to(input.device)
    return _F_interp(input, *args, **kwargs)


torch.nn.functional.interpolate = _interp_mps_safe


def n_params(module, exclude_embeddings=True):
    tot = 0
    for m in module.modules():
        for p in m.parameters(recurse=False):
            if exclude_embeddings and isinstance(m, torch.nn.Embedding):
                continue
            tot += p.numel()
    return tot


class Task:
    """Lädt Tokenizer + Generator einer Aufgabe und stellt Generierung, Signal und Kosten bereit."""

    def __init__(self, task, dev="auto", ckpt=None, data=None):
        from models.next_concept import NextConceptTransformer
        from tokenization.vqvae_tokenizer import VQVAETokenizer
        from ncm_scoring import teacher_forced_logp, make_generate, check_fast_scoring

        self.task = task
        self.dev = dev = pick_device(dev)
        self.CFG = CFG = OmegaConf.load(os.path.join(NCM, "config", "eval", f"{task}.yaml"))
        self.G = G = CFG.generation
        self.ck = ck = os.path.join(ROOT, ckpt or CKPT_DIR[task])
        self.dd = os.path.join(ROOT, data or DATA_DIR.get(task, f"data/{task}"))
        if not os.path.exists(os.path.join(ck, CFG.checkpoints.generator)):
            raise SystemExit(f"Checkpoint fehlt: {os.path.join(ck, CFG.checkpoints.generator)}\n"
                             f"-> siehe BENCH_README.md, Abschnitt 'Herunterladen'")
        CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
        for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
            if CFG.checkpoints.get(name):
                CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
        self.tk = VQVAETokenizer.from_config(CFG.tokenizer)
        self.tk.vqvae = self.tk.vqvae.to(dev).eval()
        self.net = NextConceptTransformer(vqvae_local=self.tk.vqvae, config=CFG.model)
        sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
        self.net.load_state_dict(sd["model"], strict=True)
        self.net = self.net.to(dev).eval()
        del sd

        self.vq = self.tk.vqvae.vq
        self.levels = list(self.vq.levels)
        self.K = len(self.levels)
        self.L = self.levels[-1]
        self.P = int(CFG.model.prefix_len)
        self.EOS = self.tk.pre_tokenizer.eos_token_id
        self.PAD = self.tk.pre_tokenizer.pad_token_id
        self._tf = teacher_forced_logp
        self._gen_rec = make_generate(self.net)
        self._check = check_fast_scoring

        # Sampling-Parameter genau wie eval_shard.py
        self.temp = float(G.get("temperature", 1.0))
        self.top_k = int(G.get("top_k", 1))
        self.top_p = float(G.get("top_p", 0.0))
        self.cfg = float(G.get("cfg_scale", 0.0))
        self.freq = float(G.get("frequency_penalty", 0.0))
        self.mu = float(G.get("frequency_deficit", 0.0))
        adj = str(G.get("adjacency_penalty", "") or "")
        self.ADJ = [float(v) for v in adj.split(",")] if adj else None
        self.ALLOW = self._load_ref(str(G.frequency_allowance)) if G.get("frequency_allowance") else None
        self.L0_P = self._load_ref(str(G.level0_unigram)).float() if G.get("level0_unigram") else None
        self.cut_at_eos = bool(G.get("cut_at_eos", True))
        self.uncond = not CFG.data or not CFG.data.get(CFG.eval.get("split", "test"))
        self.metric = str(CFG.eval.metric)
        self.fast = True

        # Kosten: FLOPs pro verarbeitetem Token ~ 2 x Parameter (ohne Einbettungstabellen;
        # von den Ausgabeköpfen wird pro Position nur einer benutzt)
        heads = getattr(self.net, "lm_heads", None)
        head_all = sum(p.numel() for p in heads.parameters()) if heads is not None else 0
        head_one = head_all / max(len(heads), 1) if heads is not None else 0
        vq_ids = {id(p) for p in self.tk.vqvae.parameters()}
        vq_inside = any(id(p) in vq_ids for p in self.net.parameters())
        self.N_gen = (n_params(self.net) - (n_params(self.tk.vqvae) if vq_inside else 0)
                      - head_all + head_one)
        self.N_enc = n_params(self.tk.vqvae.encoder)
        self.N_dec = n_params(self.tk.vqvae.decoder)

    # ------------------------------------------------------------ Hilfen
    def _load_ref(self, spec):
        if ":" in spec:
            path, ref = spec.split(":", 1)
            obj = torch.load(os.path.join(self.ck, path), map_location="cpu", weights_only=False)
            for k in ref.split("."):
                obj = obj[k]
            return obj.double().cpu()
        return torch.load(os.path.join(self.ck, spec), map_location="cpu").double()

    def dec(self, ids):
        ids = list(ids)
        while ids and ids[0] == self.EOS:  # LM1B: [CLS] steht als Startmarke vorn, nicht als Ende
            ids = ids[1:]
        if self.cut_at_eos and self.EOS in ids:
            ids = ids[:ids.index(self.EOS)]
        return self.tk.pre_tokenizer.decode([t for t in ids if t not in (self.EOS, self.PAD)],
                                            skip_special_tokens=True)

    def l0_sample(self, B):
        if self.L0_P is None:
            return None
        return torch.multinomial(self.L0_P, B * self.levels[0], replacement=True).view(B, -1).to(self.dev)

    # ------------------------------------------------------------ Testdaten (bedingte Aufgaben)
    def load_data(self, n=0):
        CFG = self.CFG
        split = CFG.eval.get("split", "test")
        f_in = os.path.join(self.dd, CFG.data[split])
        f_ref = os.path.join(self.dd, CFG.data[f"{split}_refs"])
        if not os.path.exists(f_in):
            raise SystemExit(f"Testdaten fehlen: {f_in}\n-> siehe BENCH_README.md, Abschnitt 'Herunterladen'")
        items = pickle.load(open(f_in, "rb"))
        refs = pickle.load(open(f_ref, "rb"))
        n = n or int(CFG.eval.get("n_docs", len(items)))
        items, refs = items[:n], refs[:n]
        src_tok = None
        if isinstance(items[0], str):
            from transformers import AutoTokenizer
            src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))

        def prefix(item):
            ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
            return (ids[:self.P] + [self.PAD] * self.P)[:self.P]

        pfx = [prefix(x) for x in items]
        refs = [r if isinstance(r, str) else self.dec(r) for r in refs]
        srcs = [x if isinstance(x, str) else self.tk.pre_tokenizer.decode(list(x), skip_special_tokens=True)
                for x in items]
        return pfx, refs, srcs

    # ------------------------------------------------------------ Dekodierung wie eval_shard.py
    def pick_tokens(self, logits):
        if self.freq <= 0 and not self.ADJ:
            return logits.argmax(-1)
        # float64 gibt es auf MPS nicht -> auf der CPU rechnen
        # erst float32 auf die CPU, dann float64: die direkte Umwandlung MPS -> float64 liefert auf dem Mac Unsinn
        lg = logits.detach().float().cpu().double()
        B, T, V = lg.shape
        freq, mu, ADJ = self.freq, self.mu, self.ADJ
        allow = (self.ALLOW[:V] * T)[None, :] if self.ALLOW is not None else None
        al = ADJ[0] if ADJ else 0.0
        aw = int(ADJ[1]) if ADJ and len(ADJ) > 1 else 1
        ad = ADJ[2] if ADJ and len(ADJ) > 2 else 1.0
        cnt = torch.zeros(B, V, dtype=torch.float64)
        out = torch.empty(B, T, dtype=torch.long)
        hist = []
        for t in range(T):
            step = lg[:, t, :].clone()
            if freq > 0:
                excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
                step -= freq * excess
                if mu > 0 and allow is not None:
                    step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
            for j, h in enumerate(reversed(hist[-aw:])):
                step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64))
            p = step.argmax(-1)
            out[:, t] = p
            hist.append(p)
            cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64))
        return out.to(logits.device)

    # ------------------------------------------------------------ Generieren + bewerten
    @torch.no_grad()
    def generate(self, pfx, temp=None, top_k=None, top_p=None, l0=None):
        """Ein Sample pro Zeile. pfx: (B, P) Long-Tensor auf self.dev.
        Rückgabe: raw (B,L) Tokens, gen_llR (B,), tok_lp (B,L) Decoder-Konfidenz, codes (B,sum levels)."""
        temp = self.temp if temp is None else temp
        top_k = self.top_k if top_k is None else top_k
        top_p = self.top_p if top_p is None else top_p
        B = pfx.shape[0]
        if self.uncond and l0 is None:
            l0 = self.l0_sample(B)
        g = self.net.generate(l0, pfx, temperature=temp, top_k=top_k, top_p=top_p,
                              cfg_scale=self.cfg, cfg_null_eos=self.EOS)
        _, zl = self.vq.idx_to_zhats(g, return_zlevels=True)
        logits = self.tk.vqvae.decoder(zl[-1])
        raw = self.pick_tokens(logits)
        tok_lp = logits.gather(-1, raw.unsqueeze(-1)).squeeze(-1).float() - torch.logsumexp(logits.float(), -1)
        del logits
        llr = self.score(pfx, raw)
        return raw.cpu(), llr.cpu(), tok_lp.cpu(), g.cpu()

    @torch.no_grad()
    def score(self, pfx, raw):
        """gen_llR: Ergebnis neu kodieren und die Codes per Teacher Forcing bewerten (Mittel log p)."""
        re = self.tk.vqvae.encode(raw.to(self.dev))
        if self.fast:
            return self._tf(self.net, self.tk.vqvae, pfx, re).mean(1)
        parts, s = [], 0
        for l in self.levels:
            parts.append(re[:, s:s + l])
            s += l
        if self.uncond or not (self.net.predict_level0 or self.net.query_level0):
            _, lp = self._gen_rec(pfx, 0.0, idx0=parts[0], forced=parts[1:])
        else:
            _, lp = self._gen_rec(pfx, 0.0, forced=parts)
        return lp.mean(1)

    @torch.no_grad()
    def self_test(self, pfx):
        """Prüft, ob die Ein-Pass-Bewertung exakt der Ebene-für-Ebene-Bewertung entspricht."""
        B = pfx.shape[0]
        l0 = self.l0_sample(B) if self.uncond else None
        g = self.net.generate(l0, pfx, temperature=self.temp, top_k=self.top_k, top_p=self.top_p)
        given = self.uncond or not (self.net.predict_level0 or self.net.query_level0)
        d_mean, d_max = self._check(self.net, self.tk.vqvae, self._gen_rec, pfx, g, idx0_given=given)
        self.fast = d_mean < 1e-3
        return d_mean, d_max

    # ------------------------------------------------------------ Kosten pro Sample
    def flops_generate(self):
        """Generator: jede Ebene verarbeitet Präfix + ihre Positionen; danach einmal Decoder."""
        first = 1 if self.uncond and self.L0_P is not None else 0
        toks = sum(self.P + l for l in self.levels[first:])
        return 2 * self.N_gen * toks + 2 * self.N_dec * self.L

    def flops_score(self):
        """Neu kodieren + ein Teacher-Forcing-Durchlauf über alle Ebenen."""
        return 2 * self.N_enc * self.L + 2 * self.N_gen * (self.P + sum(self.levels))

    def nfe_generate(self):
        first = 1 if self.uncond and self.L0_P is not None else 0
        return self.K - first

    def to(self, dev):
        self.net.to(dev)
        self.tk.vqvae.to(dev)


# ====================================================================== Metriken
def entropy_ids(ids):
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


class RefLM:
    """Ein Bewertungsmodell (z.B. gpt2-large wie im Paper oder ein unabhängiges Modell)."""

    def __init__(self, name, dev):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.dev = name, dev
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        half = dev in ("mps", "cuda")
        self.m = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16 if half else torch.float32).to(dev).eval()
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    @torch.no_grad()
    def nll_per_text(self, texts, bs=4, max_len=None):
        """Summe NLL und Tokenzahl pro Text (BOS vorangestellt, alle Texttokens bewertet)."""
        out = []
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = [[self.bos] + self.tok(t, add_special_tokens=False)["input_ids"] for t in chunk]
            if max_len:
                enc = [e[:max_len + 1] for e in enc]
            T = max(len(e) for e in enc)
            ids = torch.full((len(enc), T), self.tok.pad_token_id)
            att = torch.zeros(len(enc), T, dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, :len(e)] = torch.tensor(e)
                att[j, :len(e)] = 1
            ids, att = ids.to(self.dev), att.to(self.dev)
            lg = self.m(input_ids=ids, attention_mask=att).logits.float()
            nll = F.cross_entropy(lg[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
            msk = att[:, 1:].float()
            for j in range(len(enc)):
                out.append(((nll[j] * msk[j]).sum().item(), int(msk[j].sum().item())))
        return out

    @torch.no_grad()
    def genppl_paper(self, texts, bs=4):
        """Wie generation_perplexity im Paper: Mittel der Perplexity pro Text."""
        from evaluation.gen_ppl import generation_perplexity

        class _F32(torch.nn.Module):
            def __init__(s, inner):
                super().__init__()
                s.inner = inner

            def forward(s, **kw):
                o = s.inner(**kw)
                o.logits = o.logits.float()
                return o

        r = generation_perplexity(_F32(self.m).eval(), self.tok, [""] * len(texts), texts,
                                  device=self.dev, batch_size=bs)
        return r.ppl.double(), float(r.mean_ppl)

    @torch.no_grad()
    def corpus_ppl_flm(self, texts, bs=4):
        """FLM-Protokoll (LM1B): erste 128 GPT-2-Tokens, Korpus-Perplexity (eval_merge.genppl_flm)."""
        enc = self.tok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True,
                       padding=True, max_length=128)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        nll_sum = tok_sum = 0.0
        per = []
        eos = self.tok.eos_token_id
        for i in range(0, ids.shape[0], bs):
            s_, m_ = ids[i:i + bs].to(self.dev), attn[i:i + bs].to(self.dev)
            logits = self.m(s_, attention_mask=m_)[0].float().transpose(-1, -2)
            nll = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
            first_eos = (s_ == eos).cumsum(-1) == 1
            valid = (first_eos[..., 1:] + (s_ != eos)[..., 1:]).to(nll.dtype)
            nll_sum += float((nll * valid).sum())
            tok_sum += float(valid.sum())
            per += ((nll * valid).sum(1) / valid.sum(1).clamp_min(1)).tolist()
            del logits, nll
            if self.dev == "mps":
                torch.mps.empty_cache()
        return float(math.exp(nll_sum / tok_sum)), torch.tensor(per, dtype=torch.float64)

    def close(self):
        self.m.to("cpu")
        del self.m
        free(self.dev)


def rouge_scores(hyps, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for h, r in zip(hyps, refs):
        v = sc.score(r, h)
        for k in R:
            R[k].append(v[k].fmeasure)
    return R


def bleu_scores(hyps, refs):
    import sacrebleu
    b = sacrebleu.corpus_bleu(hyps, [refs])
    bl = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True)
    c = sacrebleu.corpus_chrf(hyps, [refs])
    lr = sum(len(h.split()) for h in hyps) / max(sum(len(r.split()) for r in refs), 1)
    sent = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(hyps, refs)]
    return {"bleu": b.score, "bleu_lc": bl.score, "chrf": c.score, "len_ratio": lr}, sent


def bootstrap_diff(a, b, n=2000, seed=0):
    """Mittlere Differenz a-b mit 95%-Bootstrap-KI. Gleiche Länge = gepaart (dieselben Plätze),
    sonst unabhängig (z.B. reines LLM gegen NCM)."""
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    if len(a) == len(b):
        d = a - b
        boots = torch.stack([d[torch.randint(len(d), (len(d),), generator=g)].mean() for _ in range(n)])
        m = d.mean().item()
    else:
        boots = torch.stack([a[torch.randint(len(a), (len(a),), generator=g)].mean()
                             - b[torch.randint(len(b), (len(b),), generator=g)].mean() for _ in range(n)])
        m = (a.mean() - b.mean()).item()
    return m, torch.quantile(boots, 0.025).item(), torch.quantile(boots, 0.975).item()


# ====================================================================== TinyStories als lokale Datei
# Statt Streaming (hing auf dem Mac stundenlang an Hugging Face) einmal die Textdatei laden (1,9 GB, landet
# im HF_HOME, also auf der externen Platte) und daraus per zufälligem Sprung Geschichten ziehen.
_SEP = b"<|endoftext|>"


def tinystories_path(split):
    from huggingface_hub import hf_hub_download
    name = "TinyStories-train.txt" if split == "train" else "TinyStories-valid.txt"
    return hf_hub_download("roneneldan/TinyStories", name, repo_type="dataset")


def tinystories_first(split, n):
    """Die ersten n Geschichten einer Datei (für Validierung)."""
    out, buf = [], b""
    with open(tinystories_path(split), "rb") as f:
        while len(out) < n:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(_SEP)
            buf = parts.pop()
            out += [p.decode("utf-8", "ignore").strip() for p in parts if p.strip()]
    return out[:n]


def tinystories_random(split="train", seed=0, min_chars=40):
    """Endloser Strom zufälliger Geschichten: an eine zufällige Stelle springen, bis zum nächsten
    Trenner vorlesen, die Geschichte danach nehmen."""
    import random
    rng = random.Random(seed)
    path = tinystories_path(split)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        while True:
            f.seek(rng.randrange(max(1, size - 20000)))
            buf = f.read(24000)
            i = buf.find(_SEP)
            if i < 0:
                continue
            j = buf.find(_SEP, i + len(_SEP))
            if j < 0:
                continue
            s = buf[i + len(_SEP):j].decode("utf-8", "ignore").strip()
            if len(s) >= min_chars:
                yield s
