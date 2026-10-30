"""Benchmark Teil 1: NCM allein und NCM + Nachbessern auf den Text-Benchmarks des Papers.

Erzeugt nur Texte (+ Kosten + internes Signal) und speichert sie. Bewertet wird danach mit
bench_score.py, damit nie NCM und das große Bewertungsmodell gleichzeitig im Speicher liegen.

Ohne Bedingung (tinystories, lm1b-small, owt): 5 Seeds x n Samples wie im Paper.
  baseline     Paper-Einstellung
  self         untere flag_q (Schwelle auf Kalibrier-Seed) von gen_llR neu sampeln, max. `rounds` Runden,
               übernommen wird nur ein Versuch mit besserem Signal
  rand         dasselbe, aber zufällig ausgewählte flag_q (Kontrolle: bringt die Auswahl etwas?)
  best{N}      N Samples pro Platz, das mit höchstem gen_llR gewinnt (Kandidaten werden für Stufe 3 gespeichert)

Mit Bedingung (xsum, wmt14): Das Paper dekodiert hier gierig (top_k=1), neu sampeln mit
derselben Einstellung ergäbe exakt denselben Text. Deshalb:
  baseline     gierig wie im Paper
  rerank{N}    gierig + (N-1) Samples bei --temp, Auswahl per gen_llR
  gated{N}     nur wo gen_llR(gierig) unter der Schwelle liegt, sonst gierig (spart Rechenzeit)
  rand{N}      zufälliger Kandidat (Kontrolle: bringt nur das Sampling etwas?)
  oracle{N}    Kandidat mit bester Satz-Metrik gegen die Referenz (Obergrenze, nicht erreichbar)

Aufruf (Beispiele):
  python experiments/bench_ncm.py --task lm1b-small
  python experiments/bench_ncm.py --task xsum --n 1000 --N 4
  python experiments/bench_ncm.py --task tinystories --smoke
"""
import argparse
import json
import os
import time

import torch

from bench_common import ROOT, Task, free, sync

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True,
                choices=["tinystories", "lm1b-small", "lm1b-medium", "lm1b-large", "owt", "xsum", "wmt14"])
ap.add_argument("--seeds", default="", help="Standard: Seeds aus der Config (42,0,1,2,3)")
ap.add_argument("--cal_seed", type=int, default=1000)
ap.add_argument("--n", type=int, default=0, help="ohne Bedingung: Samples pro Seed (Standard 256); "
                                                 "mit Bedingung: Anzahl Testdokumente (Standard: alle)")
ap.add_argument("--n_cal", type=int, default=256, help="Kalibrier-Samples bzw. -Dokumente (vom Testende genommen)")
ap.add_argument("--bs", type=int, default=0, help="Standard: aus der Config")
ap.add_argument("--N", type=int, default=4, help="Kandidaten für best/rerank/gated/rand/oracle")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--flag_q", type=float, default=0.2)
ap.add_argument("--temp", type=float, default=0.7, help="nur bedingte Aufgaben: Temperatur der Zusatzkandidaten")
ap.add_argument("--top_p", type=float, default=0.95, help="nur bedingte Aufgaben")
ap.add_argument("--variants", default="", help="Standard: alle passenden")
ap.add_argument("--device", default="auto")
ap.add_argument("--ckpt", default=None)
ap.add_argument("--data", default=None)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench"))
ap.add_argument("--smoke", action="store_true", help="Schnelltest: 1 Seed x 16 bzw. 16 Dokumente")
ap.add_argument("--n_full", type=int, default=0, help="nur für die Hochrechnung im Schnelltest: geplante Dokumente")
a = ap.parse_args()

T = Task(a.task, a.device, a.ckpt, a.data)
dev = T.dev
bs = a.bs or int(T.CFG.eval.batch_size)
if dev != "cuda":
    # Decoder-Logits (Batch x Länge x Vokabular) begrenzen: WMT14 hat 151.680 Tokens im Vokabular
    V = int(T.tk.vqvae.decoder(torch.zeros(1, T.L, T.vq.embed_dim, device=dev)).shape[-1])
    bs = max(1, min(bs, 16, int(4e7 // (T.L * V))))
seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [int(s) for s in T.CFG.eval.get("seeds", [42])]
N = a.N
if a.smoke:
    seeds, a.n, a.n_cal = seeds[:1], 16, 16
out_dir = os.path.join(a.out, a.task + ("-smoke" if a.smoke else ""))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "ncm.json")
FLM = T.metric == "genppl_flm"

if T.uncond:
    variants = a.variants.split(",") if a.variants else ["baseline", "self", "rand", f"best{N}"]
else:
    variants = a.variants.split(",") if a.variants else ["baseline", f"rerank{N}", f"gated{N}", f"rand{N}", f"oracle{N}"]

print(f"Gerät {dev} | Aufgabe {a.task} | Metrik {T.metric} | Ebenen {T.levels} | Präfix {T.P} | "
      f"Sampling T={T.temp} top_k={T.top_k} top_p={T.top_p} | Strafen freq={T.freq} adj={T.ADJ}", flush=True)
print(f"Parameter (ohne Einbettungen): Generator {T.N_gen / 1e6:.1f} M, Encoder {T.N_enc / 1e6:.1f} M, "
      f"Decoder {T.N_dec / 1e6:.1f} M", flush=True)
F_GEN, F_SCORE, NFE = T.flops_generate(), T.flops_score(), T.nfe_generate()


def run_batches(pfx_all, fn):
    """fn(pfx_batch) -> (raw, llr, tok_lp, ...); sammelt über Batches."""
    outs = None
    for s in range(0, pfx_all.shape[0], bs):
        r = fn(pfx_all[s:s + bs].to(dev))
        outs = [list(x) for x in r] if outs is None else [o + list(x) for o, x in zip(outs, r)]
        if dev == "mps":
            free(dev)
    return [torch.stack(o) for o in outs]


GEN_COUNT = [0, 0.0]


def gen(pfx_all, **kw):
    sync(dev)
    t = time.time()
    r = run_batches(pfx_all, lambda p: T.generate(p, **kw)[:3])
    sync(dev)
    GEN_COUNT[0] += pfx_all.shape[0]
    GEN_COUNT[1] += time.time() - t
    return r


# ---------------------------------------------------------------- Selbsttest
torch.manual_seed(123)
if T.uncond:
    _p = torch.full((min(bs, 8), 1), T.EOS, device=dev)
else:
    _pf, _, _ = T.load_data(8)
    _p = torch.tensor(_pf, device=dev)
d_mean, d_max = T.self_test(_p)
print(f"Selbsttest Ein-Pass-Bewertung: Abweichung {d_mean:.2e} -> {'Ein-Pass' if T.fast else 'Ebene für Ebene (langsamer)'}",
      flush=True)
if not T.fast:
    NFE_SCORE = NFE
    F_SCORE = 2 * T.N_enc * T.L + F_GEN - 2 * T.N_dec * T.L
else:
    NFE_SCORE = 1

res = {"task": a.task, "uncond": T.uncond, "metric": T.metric, "levels": T.levels, "N": N,
       "rounds": a.rounds, "flag_q": a.flag_q, "fast_scoring": T.fast,
       "params": {"gen": T.N_gen, "enc": T.N_enc, "dec": T.N_dec},
       "flops_generate": F_GEN, "flops_score": F_SCORE, "nfe_generate": NFE, "nfe_score": NFE_SCORE,
       "variants": {}, "extra": {}}
t_all = time.time()


def tolist_texts(raw):
    return [T.dec(r.tolist()) for r in raw]


def save():
    json.dump(res, open(out_path, "w"), ensure_ascii=False)


# ====================================================================== ohne Bedingung
if T.uncond:
    n = a.n or int(T.CFG.eval.n_samples)

    def bos(B):
        return torch.full((B, 1), T.EOS, dtype=torch.long)

    torch.manual_seed(a.cal_seed)
    _, llr_cal, _ = gen(bos(a.n_cal))
    thr = torch.quantile(llr_cal.double(), a.flag_q).item()
    res["extra"]["threshold"] = thr
    print(f"Schwelle gen_llR (unteres {a.flag_q:.0%}-Quantil, Kalibrierung): {thr:.4f}", flush=True)

    t0 = time.time()
    init = {}
    for s in seeds:
        torch.manual_seed(s)
        sync(dev)
        ts = time.time()
        init[s] = gen(bos(n))
        sync(dev)
        init[s].append(time.time() - ts)
    t_base = sum(init[s][3] for s in seeds)
    print(f"Startversuche: {len(seeds)} x {n} in {time.time() - t0:.0f}s", flush=True)

    raw0 = torch.cat([init[s][0] for s in seeds])
    llr0 = torch.cat([init[s][1] for s in seeds])
    tlp0 = torch.cat([init[s][2] for s in seeds])
    seed_of = [s for s in seeds for _ in range(n)]
    res["extra"].update({"gen_llR": llr0.tolist(), "tok_lp": [[round(v, 3) for v in r] for r in tlp0.tolist()],
                         "raw_ids": raw0.tolist(), "seed": seed_of})

    def pack(raw, flops, nfe, retried, secs):
        d = {"texts": tolist_texts(raw), "seed": seed_of, "flops": flops, "nfe": nfe, "retried": retried,
             "seconds": secs}
        if FLM:
            d["raw_ids"] = raw.tolist()
        return d

    for name in variants:
        t0 = time.time()
        if name == "baseline":
            M = len(seed_of)
            res["variants"][name] = pack(raw0, [F_GEN] * M, [NFE] * M, [0] * M, t_base)
        elif name in ("self", "rand"):
            raws, flops, nfes, evers = [], [], [], []
            for s in seeds:
                raw, llr = init[s][0].clone(), init[s][1].clone()
                cost_f = torch.full((n,), float(F_GEN + F_SCORE))
                cost_n = torch.full((n,), float(NFE + NFE_SCORE))
                torch.manual_seed(s + 7777)
                flag = (llr < thr) if name == "self" else (torch.rand(n) < a.flag_q)
                ever = flag.clone()
                for _ in range(a.rounds):
                    if name == "self":
                        flag = llr < thr
                        ever |= flag
                    idx = flag.nonzero().squeeze(1)
                    if len(idx) == 0:
                        break
                    nr, nl, _ = gen(bos(len(idx)))
                    acc = nl > llr[idx]
                    raw[idx] = torch.where(acc[:, None], nr, raw[idx])
                    llr[idx] = torch.where(acc, nl, llr[idx])
                    cost_f[idx] += F_GEN + F_SCORE
                    cost_n[idx] += NFE + NFE_SCORE
                raws.append(raw)
                flops += cost_f.tolist()
                nfes += cost_n.tolist()
                evers += ever.int().tolist()
            res["variants"][name] = pack(torch.cat(raws), flops, nfes, evers, t_base + time.time() - t0)
        elif name.startswith("best"):
            k = int(name[4:])
            cands_raw, cands_llr = [raw0], [llr0]
            for j in range(1, k):
                rr, ll = [], []
                for s in seeds:
                    torch.manual_seed(s * 100 + j)
                    r_, l_, _ = gen(bos(n))
                    rr.append(r_)
                    ll.append(l_)
                cands_raw.append(torch.cat(rr))
                cands_llr.append(torch.cat(ll))
            CR, CL = torch.stack(cands_raw, 1), torch.stack(cands_llr, 1)  # (M, k, L), (M, k)
            best = CL.argmax(1)
            raw = CR[torch.arange(len(best)), best]
            M = len(best)
            res["variants"][name] = pack(raw, [k * (F_GEN + F_SCORE)] * M, [k * (NFE + NFE_SCORE)] * M,
                                         [1] * M, t_base + time.time() - t0)
            res["extra"]["candidates"] = [[T.dec(CR[i, j].tolist()) for j in range(k)] for i in range(M)]
            res["extra"]["cand_llr"] = CL.tolist()
            if FLM:
                res["extra"]["cand_raw_ids"] = CR.tolist()
        else:
            print(f"unbekannte Variante {name}, übersprungen")
            continue
        v = res["variants"][name]
        print(f"{name:<10} fertig: {time.time() - t0:6.0f}s | GFLOPs/Sample {sum(v['flops']) / len(v['flops']) / 1e9:8.2f} | "
              f"NFE/Sample {sum(v['nfe']) / len(v['nfe']):5.1f} | neu versucht {100 * sum(v['retried']) / len(v['retried']):5.1f} %",
              flush=True)
        save()

# ====================================================================== mit Bedingung
else:
    pfx_all, refs_all, _ = T.load_data(0)
    n_tot = len(pfx_all)
    n = min(a.n or n_tot - a.n_cal, n_tot - a.n_cal)
    P_eval = torch.tensor(pfx_all[:n])
    P_cal = torch.tensor(pfx_all[n_tot - a.n_cal:])
    refs = refs_all[:n]
    res["refs"] = refs
    res["n_docs"] = n

    torch.manual_seed(a.cal_seed)
    _, llr_cal, _ = gen(P_cal)
    thr = torch.quantile(llr_cal.double(), a.flag_q).item()
    res["extra"]["threshold"] = thr
    print(f"Schwelle gen_llR(gierig) auf {a.n_cal} Kalibrier-Dokumenten: {thr:.4f}", flush=True)

    sync(dev)
    t0 = time.time()
    raw_g, llr_g, tlp_g = gen(P_eval)
    sync(dev)
    t_greedy = time.time() - t0
    print(f"gierig: {n} Dokumente in {t_greedy:.0f}s", flush=True)
    res["extra"].update({"gen_llR": llr_g.tolist()})

    need_cands = any(v != "baseline" for v in variants)
    cand_raw, cand_llr, t_cand = [raw_g], [llr_g], 0.0
    if need_cands:
        for j in range(1, N):
            torch.manual_seed(1000 + j)
            sync(dev)
            ts = time.time()
            r_, l_, _ = gen(P_eval, temp=a.temp, top_k=0, top_p=a.top_p)
            sync(dev)
            t_cand += time.time() - ts
            cand_raw.append(r_)
            cand_llr.append(l_)
            print(f"  Kandidat {j + 1}/{N} fertig ({time.time() - ts:.0f}s)", flush=True)
    CR, CL = torch.stack(cand_raw, 1), torch.stack(cand_llr, 1)
    k_ = CR.shape[1]
    cand_txt = [[T.dec(CR[i, j].tolist()) for j in range(k_)] for i in range(n)]
    res["extra"]["candidates"] = cand_txt
    res["extra"]["cand_llr"] = CL.tolist()

    # Satz-Metrik für das Orakel
    def sent_metric(h, r):
        if T.metric == "rouge":
            from rouge_score import rouge_scorer
            if not hasattr(sent_metric, "sc"):
                sent_metric.sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            return sent_metric.sc.score(r, h)["rougeL"].fmeasure
        import sacrebleu
        return sacrebleu.sentence_chrf(h, [r]).score

    flagged = (llr_g < thr)
    full_f = F_GEN + F_SCORE + (k_ - 1) * (F_GEN + F_SCORE)
    full_n = NFE + NFE_SCORE + (k_ - 1) * (NFE + NFE_SCORE)
    for name in variants:
        if name == "baseline":
            pick = torch.zeros(n, dtype=torch.long)
            flops, nfe, ret, secs = [F_GEN] * n, [NFE] * n, [0] * n, t_greedy
        elif name.startswith("rerank"):
            pick = CL.argmax(1)
            flops, nfe, ret, secs = [full_f] * n, [full_n] * n, [1] * n, t_greedy + t_cand
        elif name.startswith("gated"):
            pick = torch.where(flagged, CL.argmax(1), torch.zeros(n, dtype=torch.long))
            flops = [full_f if f else F_GEN + F_SCORE for f in flagged.tolist()]
            nfe = [full_n if f else NFE + NFE_SCORE for f in flagged.tolist()]
            ret, secs = flagged.int().tolist(), t_greedy + t_cand * flagged.float().mean().item()
        elif name.startswith("rand"):
            g = torch.Generator().manual_seed(5)
            pick = torch.randint(k_, (n,), generator=g)
            flops, nfe, ret, secs = [k_ * F_GEN] * n, [k_ * NFE] * n, [1] * n, t_greedy + t_cand
        elif name.startswith("oracle"):
            pick = torch.tensor([max(range(k_), key=lambda j: sent_metric(cand_txt[i][j], refs[i])) for i in range(n)])
            flops, nfe, ret, secs = [k_ * F_GEN] * n, [k_ * NFE] * n, [1] * n, t_greedy + t_cand
        else:
            print(f"unbekannte Variante {name}, übersprungen")
            continue
        res["variants"][name] = {"texts": [cand_txt[i][pick[i]] for i in range(n)], "flops": flops, "nfe": nfe,
                                 "retried": ret, "seconds": secs, "pick": pick.tolist()}
        changed = (pick != 0).float().mean().item() * 100
        print(f"{name:<10} | GFLOPs/Dok {sum(flops) / n / 1e9:8.2f} | NFE/Dok {sum(nfe) / n:5.1f} | "
              f"anderer Text als gierig {changed:5.1f} %", flush=True)
    save()

save()
print(f"\nGespeichert: {out_path}  ({time.time() - t_all:.0f}s)\n"
      f"Weiter mit: python experiments/bench_score.py --task {a.task}" + (" --smoke" if a.smoke else ""))
if a.smoke:
    per = GEN_COUNT[1] / max(GEN_COUNT[0], 1)
    if T.uncond:
        n_full = len([int(x) for x in T.CFG.eval.get("seeds", [42])]) * int(T.CFG.eval.n_samples)
        total = 256 + n_full * (1 + a.rounds * a.flag_q * 2 + (N - 1))
    else:
        n_full = (a.n_full or int(T.CFG.eval.get("n_docs", 0)))
        total = 256 + n_full * N
    print(f"Hochrechnung voller Lauf ({total:.0f} Generierungen à {per:.3f}s): ca. {per * total / 3600:.1f} h "
          f"(ohne Bewertung)")
    print("SCHNELLTEST OK")
