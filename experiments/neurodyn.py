"""Experiment 6: Neuronen mit Zeitverhalten und Feuerwahrscheinlichkeit.

Frage: Werden kleine rekurrente Netze besser oder robuster, wenn ihre Neuronen
  (a) eine Membran-Zeitkonstante haben (Spannung baut sich langsam auf und läuft aus,
      wie beim Öffnen/Schließen von Ionenkanälen),
  (b) jedes Neuron seine eigene, lernbare Zeitkonstante hat (heterogen wie im Gehirn),
  (c) die Zeitkonstante vom aktuellen Input abhängt (ähnlich Liquid Time-Constant Networks),
  (d) Neuronen nur mit einer Wahrscheinlichkeit feuern (Bernoulli-Spikes statt Rate)?

Neuronenmodell (pro Zeitschritt t, Leck-Faktor a = exp(-1/tau)):
    v_t = a * v_{t-1} + (1 - a) * (W x_t + U s_{t-1} + b)     # Membran
    p_t = sigmoid(v_t)                                       # Feuerwahrscheinlichkeit
    s_t = p_t            (det: Feuerrate)
    s_t ~ Bernoulli(p_t) (stoch: Spike ja/nein, Gradient per Straight-Through)
  rnn:       a = 0 (kein Gedächtnis in der Membran, klassisches RNN)
  tau_hom:   ein gemeinsames lernbares tau für alle Neuronen
  tau_het:   ein lernbares tau pro Neuron, Start log-gleichverteilt in [1, 50]
  tau_dyn:   a_t = sigmoid(A x_t + B s_{t-1} + c), Start wie tau_het (inputabhängig)
  gru:       Referenz (Standard-GRU aus PyTorch)

Aufgaben (synthetisch, kein Download nötig):
  adding: zwei markierte Zahlen in einer Folge von 100 merken und addieren (Regression)
  recall: ein Symbol (1 von 8) irgendwann zeigen, am Ende abfragen (Gedächtnis über Verzögerung)
  freq:   verrauschtes Signal aus langsamer + schneller Schwingung, 4 Klassen (zwei Zeitskalen)

Gemessen: Test im Trainingsbereich, mit stärkerem Input-Rauschen, mit doppelter Länge.
Ergebnis pro Lauf: results/neurodyn/<task>-<variant>-s<seed>.json (fertige Läufe werden übersprungen)

Aufruf:
    python experiments/neurodyn.py --worker 0 --nworkers 3           # Anteil eines Workers
    python experiments/neurodyn.py --smoke                           # Schnelltest (~1-2 min)
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

TASKS = ["freq", "recall", "adding"]
CELLS = ["rnn", "tau_hom", "tau_het", "tau_dyn"]
VARIANTS = [f"{c}_{m}" for c in CELLS for m in ("det", "stoch")] + ["gru"]
STEPS = {"adding": 3000, "recall": 2000, "freq": 1500}


# ----------------------------------------------------------------------------- Aufgaben
class Task:
    def __init__(self, name):
        self.name = name
        self.T = {"adding": 100, "recall": 80, "freq": 150}[name]
        self.n_in = {"adding": 2, "recall": 9, "freq": 1}[name]
        self.n_out = {"adding": 1, "recall": 8, "freq": 4}[name]
        self.regression = name == "adding"
        self.noise = {"adding": 0.0, "recall": 0.0, "freq": 0.3}[name]   # Rauschen im Training
        self.noise_hard = {"adding": 0.2, "recall": 0.5, "freq": 0.8}[name]  # Robustheitstest

    def batch(self, B, g, T=None, noise=None):
        T = T or self.T
        noise = self.noise if noise is None else noise
        if self.name == "adding":
            vals = torch.rand(B, T, generator=g)
            mark = torch.zeros(B, T)
            half = T // 2
            i = (torch.rand(B, generator=g) * half).long()
            j = half + (torch.rand(B, generator=g) * (T - half)).long()
            mark[torch.arange(B), i] = 1
            mark[torch.arange(B), j] = 1
            y = (vals * mark).sum(1, keepdim=True)
            x = torch.stack([vals, mark], -1)
            if noise > 0:
                x[..., 0] += noise * torch.randn(B, T, generator=g)
            return x, y
        if self.name == "recall":
            x = torch.zeros(B, T, 9)
            sym = (torch.rand(B, generator=g) * 8).long()
            start = (torch.rand(B, generator=g) * (T - 20)).long()   # mind. ~17 Schritte Verzögerung
            for k in range(3):                                       # Symbol 3 Schritte lang zeigen
                x[torch.arange(B), start + k, sym] = 1
            x[:, -1, 8] = 1                                          # Abfrage-Signal am Ende
            if noise > 0:
                x[..., :8] += noise * torch.randn(B, T, 8, generator=g)
            return x, sym
        # freq
        t = torch.arange(T).float()
        cls = (torch.rand(B, generator=g) * 4).long()
        Ps = torch.tensor([40.0, 60.0])[cls // 2]
        Pf = torch.tensor([5.0, 7.0])[cls % 2]
        ph1 = torch.rand(B, 1, generator=g) * 2 * math.pi
        ph2 = torch.rand(B, 1, generator=g) * 2 * math.pi
        amp = 0.5 + torch.rand(B, 2, generator=g)
        sig = amp[:, :1] * torch.sin(2 * math.pi * t / Ps[:, None] + ph1) \
            + amp[:, 1:] * torch.sin(2 * math.pi * t / Pf[:, None] + ph2)
        sig = sig + noise * torch.randn(B, T, generator=g)
        return sig[..., None], cls


# ----------------------------------------------------------------------------- Modelle
class NeuroRNN(nn.Module):
    def __init__(self, cell, stoch, n_in, n_hid, n_out, tau_range=(1.0, 50.0)):
        super().__init__()
        self.cell, self.stoch, self.H = cell, stoch, n_hid
        self.W = nn.Linear(n_in, n_hid)
        self.U = nn.Linear(n_hid, n_hid, bias=False)
        nn.init.orthogonal_(self.U.weight, gain=1.0)
        lo, hi = math.log(tau_range[0]), math.log(tau_range[1])
        if cell == "tau_hom":
            self.log_tau = nn.Parameter(torch.tensor([(lo + hi) / 2]))   # geometrisches Mittel ~7
        elif cell == "tau_het":
            self.log_tau = nn.Parameter(torch.empty(n_hid).uniform_(lo, hi))
        elif cell == "tau_dyn":
            tau0 = torch.exp(torch.empty(n_hid).uniform_(lo, hi))
            a0 = torch.exp(-1.0 / tau0).clamp(1e-3, 1 - 1e-3)
            self.A = nn.Linear(n_in, n_hid)
            self.Bm = nn.Linear(n_hid, n_hid, bias=False)
            nn.init.zeros_(self.A.weight)
            nn.init.zeros_(self.Bm.weight)
            with torch.no_grad():
                self.A.bias.copy_(torch.log(a0) - torch.log1p(-a0))   # logit(a0)
        self.out = nn.Linear(n_hid, n_out)

    def tau(self):
        if self.cell in ("tau_hom", "tau_het"):
            return torch.exp(self.log_tau).clamp(min=1.0).expand(self.H)
        return None

    def forward(self, x, sample=True):
        B, T, _ = x.shape
        drive_x = self.W(x)                                    # alle Zeitschritte auf einmal
        ax = self.A(x) if self.cell == "tau_dyn" else None
        v = torch.zeros(B, self.H, device=x.device)
        s = torch.zeros(B, self.H, device=x.device)
        a = None
        if self.cell in ("tau_hom", "tau_het"):
            a = torch.exp(-1.0 / self.tau())
        rate_sum = 0.0
        for t in range(T):
            drive = drive_x[:, t] + self.U(s)
            if self.cell == "rnn":
                v = drive
            elif self.cell == "tau_dyn":
                at = torch.sigmoid(ax[:, t] + self.Bm(s))
                v = at * v + (1 - at) * drive
            else:
                v = a * v + (1 - a) * drive
            p = torch.sigmoid(v)
            if self.stoch and sample:
                spk = torch.bernoulli(p.detach())
                s = p + (spk - p).detach()                     # Straight-Through-Gradient
            else:
                s = p
            rate_sum = rate_sum + p.detach().mean()
        self.last_rate = float(rate_sum / T)
        return self.out(v)


class GRUNet(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super().__init__()
        self.rnn = nn.GRU(n_in, n_hid, batch_first=True)
        self.out = nn.Linear(n_hid, n_out)
        self.last_rate = float("nan")

    def forward(self, x, sample=True):
        h, _ = self.rnn(x)
        return self.out(h[:, -1])


def build(variant, task, n_hid):
    if variant == "gru":
        return GRUNet(task.n_in, n_hid, task.n_out)
    cell, mode = variant.rsplit("_", 1)
    return NeuroRNN(cell, mode == "stoch", task.n_in, n_hid, task.n_out)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- Training und Test
def loss_fn(task, pred, y):
    return F.mse_loss(pred, y) if task.regression else F.cross_entropy(pred, y)


@torch.no_grad()
def evaluate(model, task, n, T=None, noise=None, sample=True, n_samples=1, seed=999):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    scores = []
    for _ in range(max(1, n // 256)):
        x, y = task.batch(256, g, T=T, noise=noise)
        outs = [model(x, sample=sample) for _ in range(n_samples)]
        if task.regression:
            pred = torch.stack(outs).mean(0)
            scores.append(F.mse_loss(pred, y).item())
        else:
            prob = torch.stack([o.softmax(-1) for o in outs]).mean(0)
            scores.append((prob.argmax(-1) == y).float().mean().item())
    model.train()
    return sum(scores) / len(scores)


def run_one(task_name, variant, seed, steps, n_hid, bs, lr, n_test, out_dir):
    torch.manual_seed(seed)
    task = Task(task_name)
    model = build(variant, task, n_hid)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    g = torch.Generator().manual_seed(10_000 + seed)
    curve, t0, run_loss = [], time.time(), 0.0
    for it in range(1, steps + 1):
        x, y = task.batch(bs, g)
        loss = loss_fn(task, model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += loss.item()
        if it % 100 == 0 or it == steps:
            k = 100 if it % 100 == 0 else it % 100
            curve.append([it, run_loss / k])
            run_loss = 0.0
    train_s = time.time() - t0

    stoch = variant.endswith("_stoch")
    ns = 4 if stoch else 1
    res = {
        "task": task_name, "variant": variant, "seed": seed, "steps": steps, "hidden": n_hid,
        "params": n_params(model), "train_s": round(train_s, 1),
        "metric": "mse" if task.regression else "acc",
        "test": evaluate(model, task, n_test, n_samples=ns),
        "test_noise": evaluate(model, task, n_test, noise=task.noise_hard, n_samples=ns),
        "test_long": evaluate(model, task, n_test, T=2 * task.T, n_samples=ns),
        "curve": curve, "rate": getattr(model, "last_rate", None),
    }
    if stoch:  # dasselbe Netz ohne Zufall (nur Feuerwahrscheinlichkeit weitergeben)
        res["test_meanfield"] = evaluate(model, task, n_test, sample=False)
        res["test_1sample"] = evaluate(model, task, n_test, n_samples=1)
    if isinstance(model, NeuroRNN) and model.tau() is not None:
        tau = model.tau().detach().contiguous()
        q = torch.quantile(tau, torch.tensor([0.1, 0.5, 0.9]))
        res["tau_q10_q50_q90"] = [round(v, 2) for v in q.tolist()]
        res["tau_all"] = [round(v, 2) for v in tau.tolist()]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{task_name}-{variant}-s{seed}.json")
    json.dump(res, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return res


def fmt(res):
    m = res["metric"]
    f = (lambda v: f"{v:.4f}") if m == "mse" else (lambda v: f"{100 * v:5.1f}%")
    extra = f" | ohne Zufall {f(res['test_meanfield'])}" if "test_meanfield" in res else ""
    tau = f" | tau q10/50/90 {res['tau_q10_q50_q90']}" if "tau_q10_q50_q90" in res else ""
    return (f"{res['task']:6s} {res['variant']:13s} s{res['seed']}  {m} test {f(res['test'])} | "
            f"Rauschen {f(res['test_noise'])} | doppelte Länge {f(res['test_long'])}{extra}{tau} "
            f"| {res['train_s']:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n_test", type=int, default=2048)
    ap.add_argument("--steps_scale", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/neurodyn")
    ap.add_argument("--smoke", action="store_true", help="Schnelltest: alle Varianten, 20 Schritte")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    if a.smoke:
        t0 = time.time()
        for tn in TASKS:
            for v in VARIANTS:
                r = run_one(tn, v, 0, 20, 16, 16, a.lr, 256, "results/neurodyn-smoke")
                print(fmt(r), flush=True)
        # Zeitschätzung für den vollen Lauf: ein echter Mini-Lauf mit voller Größe
        tt = {}
        for tn in TASKS:
            s = time.time()
            run_one(tn, "tau_dyn_stoch", 0, 30, a.hidden, a.bs, a.lr, 256, "results/neurodyn-smoke")
            tt[tn] = (time.time() - s) / 30
        per_seed = sum(tt[tn] * STEPS[tn] * a.steps_scale * len(VARIANTS) for tn in TASKS)
        print(f"\nSCHNELLTEST OK ({time.time() - t0:.0f}s). Geschätzt pro Seed (alle Aufgaben/Varianten, 1 Worker): "
              f"{per_seed / 3600:.1f} h (eher Obergrenze, GRU ist schneller)")
        return

    jobs = [(s, tn, v) for s in [int(x) for x in a.seeds.split(",")]   # seed-weise: frühe Seeds komplett zuerst
            for tn in a.tasks.split(",") for v in a.variants.split(",")]
    mine = jobs[a.worker::a.nworkers]
    for i, (s, tn, v) in enumerate(mine):
        path = os.path.join(a.out, f"{tn}-{v}-s{s}.json")
        if os.path.exists(path):
            continue
        r = run_one(tn, v, s, int(STEPS[tn] * a.steps_scale), a.hidden, a.bs, a.lr, a.n_test, a.out)
        print(f"[w{a.worker} {i + 1}/{len(mine)} {time.strftime('%H:%M')}] {fmt(r)}", flush=True)


if __name__ == "__main__":
    main()
