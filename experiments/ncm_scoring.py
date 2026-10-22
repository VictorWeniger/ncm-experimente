"""Gemeinsame Hilfsfunktionen: Codes per Teacher Forcing in EINEM Forward Pass bewerten.

NCM wird so trainiert, dass alle Ebenen in einem Durchlauf vorhergesagt werden
(block-diagonale Attention, jede Ebene sieht nur ihre eigene Eingabe = die
kumulative Rekonstruktion der gröberen Ebenen). Genau diesen Pfad nutzen wir
hier, um die log-Wahrscheinlichkeit gegebener Codes zu berechnen. Das ist
deutlich billiger als die Ebene-für-Ebene-Generierung mit erzwungenen Codes.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def teacher_forced_logp(net, vqvae, pfx, idx):
    """log p(Code) für alle Codes ab Ebene 1 (Ebene 0 wird ausgelassen, sie ist bei
    Text nicht vom Generator, sondern aus einer Unigramm-Verteilung gezogen).

    pfx: (B, >=P) Prompt-Token, idx: (B, sum(levels)) Codes aller Ebenen.
    Rückgabe: (B, sum(levels) - levels[0])
    """
    levels = list(vqvae.vq.levels)
    l0, P = levels[0], net.prefix_len
    cfg = net.config
    if getattr(cfg, "raw_token_prefix", False):
        zp = vqvae.encoder.wte(pfx)[:, :P, :]
    else:
        zp = vqvae.encode_z(pfx)[:, :P, :]
    xt = vqvae.vq.idx_to_ncm_input(idx)  # Eingaben für die Ebenen 1..K-1
    if net.query_level0:
        xt = torch.cat([net.build_l0_query_input(idx.shape[0], xt.dtype), xt], dim=1)
    x = torch.cat([zp.to(xt.dtype), xt], dim=1)
    pm = (pfx[:, :P] == net.prefix_pad_id) if net.prefix_pad_id is not None else None
    logits = net(x, prefix_pad_mask=pm)
    with_l0 = net.query_level0 or net.predict_level0
    tgt = idx if with_l0 else idx[:, l0:]
    lp = F.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp[:, l0:] if with_l0 else lp


def make_generate(net):
    """generate(pfx, temp, idx0=None, forced=None) -> (codes, logp je gewähltem Code ab Ebene 1)

    Zeichnet für jeden gesampelten (oder per forced erzwungenen) Code die untemperierte
    log-Wahrscheinlichkeit auf. idx0: Codes der Ebene 0 vorgeben (Text: Unigramm-Sample).
    """
    orig = net._sample_level

    def generate(pfx, temp, idx0=None, forced=None):
        logps = []

        def sample(logits, temperature, top_k, top_p):
            i = len(logps)
            idx = forced[i] if forced is not None else orig(logits, temp, 0, 1.0)
            logps.append(F.log_softmax(logits.float(), -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1))
            return idx

        net._sample_level = sample
        try:
            with torch.no_grad():
                g = net.generate(idx0, pfx, temperature=temp, top_k=0, top_p=1.0, cfg_scale=0.0)
        finally:
            del net._sample_level
        lp = torch.cat(logps, dim=1)
        if idx0 is None:  # Ebene 0 wurde selbst gesampelt -> für Vergleichbarkeit weglassen
            lp = lp[:, net.levels[0]:]
        return g, lp

    return generate


def split_levels(idx, levels):
    out, s = [], 0
    for l in levels:
        out.append(idx[:, s:s + l])
        s += l
    return out


@torch.no_grad()
def check_fast_scoring(net, vqvae, generate, pfx, idx, idx0_given):
    """Vergleicht Ein-Pass-Bewertung mit Ebene-für-Ebene Teacher Forcing. Gibt max. Abweichung zurück."""
    levels = list(vqvae.vq.levels)
    parts = split_levels(idx, levels)
    if idx0_given:
        _, slow = generate(pfx, 0.0, idx0=parts[0], forced=parts[1:])
    else:
        _, slow = generate(pfx, 0.0, forced=parts)
    fast = teacher_forced_logp(net, vqvae, pfx, idx)
    return (slow.mean(1) - fast.mean(1)).abs().max().item(), (slow - fast).abs().max().item()
