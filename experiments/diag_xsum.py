"""Diagnose: Warum sind die XSum-Zusammenfassungen leer? Vergleicht MPS und CPU auf 2 Artikeln.
Aufruf: python experiments/diag_xsum.py
"""
import torch

from bench_common import Task

for dev in ["mps", "cpu"]:
    if dev == "mps" and not torch.backends.mps.is_available():
        continue
    T = Task("xsum", dev)
    pfx, refs, srcs = T.load_data(2)
    p = torch.tensor(pfx, device=dev)
    print(f"\n===== {dev} | EOS {T.EOS} PAD {T.PAD} prefix_pad_id {T.net.prefix_pad_id} | "
          f"Präfix-Länge {len(pfx[0])}, davon PAD {sum(t == T.PAD for t in pfx[0])} | Eingabetyp {type(srcs[0]).__name__}")
    print("Artikelanfang:", srcs[0][:150].replace("\n", " "))
    print("Referenz:", refs[0][:150])
    with torch.no_grad():
        g = T.net.generate(None, p, temperature=T.temp, top_k=T.top_k, top_p=T.top_p, cfg_scale=0.0, cfg_null_eos=T.EOS)
        _, zl = T.vq.idx_to_zhats(g, return_zlevels=True)
        lg = T.tk.vqvae.decoder(zl[-1])
    print("Codes Ebene 0..1:", g[0, :10].tolist(), "| NaN in Decoder-Logits:", bool(torch.isnan(lg).any()))
    raw_arg = lg.argmax(-1)
    raw_pen = T.pick_tokens(lg)
    for name, r in [("argmax", raw_arg), ("mit Strafen", raw_pen)]:
        ids = r[0].tolist()
        print(f"{name:>12}: erste Tokens {ids[:12]} | EOS an Position {ids.index(T.EOS) if T.EOS in ids else '-'}")
        print(f"{'':>12}  Text: {T.dec(ids)[:200]!r}")
        print(f"{'':>12}  ohne Abschneiden: {T.tk.pre_tokenizer.decode(ids, skip_special_tokens=True)[:200]!r}")
    print("gen_llR:", T.score(p, raw_pen).tolist())
    del T
