"""Experiment 7a: Kritische Phasen in künstlichen Netzen (nach Achille, Rovere & Soatto 2019).

Idee aus der Entwicklungsbiologie: Bekommt ein Kätzchen/Kind in den ersten Wochen kein scharfes Bild
(z.B. bei grauem Star oder Schielen), bleibt das Sehen auf diesem Auge dauerhaft schlechter, auch wenn
das Problem später behoben wird. Achille et al. fanden dasselbe bei Netzen: Unscharfe Bilder am Anfang
des Trainings schaden dauerhaft, später fast gar nicht.

Hier auf CIFAR-10 mit einem kleinen CNN (passt auf die M1-GPU):
  defD          Defizit (unscharf: 8x8 hochskaliert) in den ersten D Epochen, danach 30 normale Epochen
  winS          Defizit nur in Epoche S..S+10 (später im Training), danach 30 normale Epochen
  defD_const    wie defD, aber konstante Lernrate (ist es nur die sinkende Lernrate?)
  defD_reset    wie defD, aber nach dem Defizit springt die Lernrate zurück auf den Startwert
                ("Plastizität wieder öffnen", analog zu Versuchen, kritische Phasen im Gehirn neu zu öffnen)
Alle Läufe enden mit 3 Epochen kleiner Lernrate, damit alle gleich gut "auskonvergiert" sind.
Für def0 und def40 (Seed 0) wird zusätzlich die Spur der Fisher-Information pro Epoche gemessen
(Achille: steigt am Anfang stark an und fällt dann, die kritische Phase liegt im Anstieg).

Ergebnis pro Lauf: results/critical/<name>-s<seed>.json, fertige Läufe werden übersprungen.
    python experiments/critical_periods.py --smoke        # Schnelltest + Zeitschätzung
    python experiments/critical_periods.py                # alles
"""
import argparse
import json
import os
import pickle
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
AFTER = 30       # normale Epochen nach dem Defizit
TAIL = 3         # Abschluss-Epochen mit kleiner Lernrate
FISHER_RUNS = {("def0", 0), ("def40", 0)}


def runs_for_seed(seed):
    """Reihenfolge = Priorität: erst die Kernreihe, dann die Kontrollen."""
    r = [(f"def{d}", 0, d, "decay") for d in (0, 10, 20, 40, 5, 30)]
    r += [(f"def{d}_reset", 0, d, "reset") for d in (40, 20)]
    r += [(f"def{d}_const", 0, d, "const") for d in (0, 40, 20)]
    r += [(f"win{s}", s, s + 10, "decay") for s in (10, 30)]
    return [(n, a, b, sch, seed) for n, a, b, sch in r]


# ----------------------------------------------------------------------------- Daten
def load_cifar(root="data/cifar10"):
    npz = os.path.join(root, "cifar10.npz")
    if not os.path.exists(npz):
        os.makedirs(root, exist_ok=True)
        tgz = os.path.join(root, "cifar-10-python.tar.gz")
        if not os.path.exists(tgz):
            print("Lade CIFAR-10 (~170 MB) ...", flush=True)
            urllib.request.urlretrieve(URL, tgz + ".part")
            os.replace(tgz + ".part", tgz)
        xs, ys = [], []
        with tarfile.open(tgz) as tf:
            def get(name):
                d = pickle.load(tf.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")
                return d[b"data"].reshape(-1, 3, 32, 32), np.array(d[b"labels"])
            for i in range(1, 6):
                x, y = get(f"data_batch_{i}")
                xs.append(x)
                ys.append(y)
            xte, yte = get("test_batch")
        np.savez(npz, xtr=np.concatenate(xs), ytr=np.concatenate(ys), xte=xte, yte=yte)
    d = np.load(npz)
    return (torch.from_numpy(d["xtr"]), torch.from_numpy(d["ytr"]).long(),
            torch.from_numpy(d["xte"]), torch.from_numpy(d["yte"]).long())


def prep(xb, dev, augment, blur):
    x = xb.to(dev).float().div_(255)
    if augment:
        B = x.shape[0]
        xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
        oy = torch.randint(0, 9, (B,), device=dev)
        ox = torch.randint(0, 9, (B,), device=dev)
        ar = torch.arange(32, device=dev)
        rows = (oy[:, None] + ar)[:, :, None]            # B,32,1
        cols = (ox[:, None] + ar)[:, None, :]            # B,1,32
        x = xp.permute(0, 2, 3, 1)[torch.arange(B, device=dev)[:, None, None], rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(B, device=dev) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
    if blur:  # Defizit: auf 8x8 verkleinern und wieder hochskalieren
        x = F.interpolate(F.avg_pool2d(x, 4), size=32, mode="bilinear", align_corners=False)
    return (x - MEAN.to(dev)) / STD.to(dev)


# ----------------------------------------------------------------------------- Modell
def conv_bn(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    def __init__(self, w=32, n_cls=10):
        super().__init__()
        self.f = nn.Sequential(conv_bn(3, w), conv_bn(w, 2 * w), nn.MaxPool2d(2),
                               conv_bn(2 * w, 4 * w), nn.MaxPool2d(2),
                               conv_bn(4 * w, 4 * w), nn.MaxPool2d(2),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(4 * w, n_cls)

    def forward(self, x):
        return self.fc(self.f(x))


# ----------------------------------------------------------------------------- Training
def lr_at(epoch, d_end, schedule, total, lr0, decay):
    if epoch >= total - TAIL:
        return lr0 * 0.01
    if schedule == "const":
        return lr0
    if schedule == "reset" and epoch >= d_end:
        return lr0 * decay ** (epoch - d_end)
    return lr0 * decay ** epoch


@torch.no_grad()
def evaluate(model, x, y, dev, blur=False, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(prep(x[i:i + bs], dev, False, blur)).argmax(1).cpu() == y[i:i + bs]).sum().item()
    model.train()
    return correct / len(x)


def fisher_trace(model, x, y, dev, n=200):
    """Spur der empirischen Fisher-Information: mittlere quadrierte Gradientennorm von log p(ŷ|x),
    ŷ aus der Modellverteilung gezogen, je ein Beispiel pro Backward-Pass (BN im Eval-Modus)."""
    model.eval()
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(x), generator=g)[:n]
    tot = 0.0
    for i in idx:
        xb = prep(x[i:i + 1], dev, False, False)
        logp = model(xb).log_softmax(1)
        yh = torch.multinomial(logp.exp().detach(), 1).item()
        model.zero_grad(set_to_none=True)
        logp[0, yh].backward()
        tot += sum((p.grad ** 2).sum().item() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    model.train()
    return tot / n


def run_one(name, d_start, d_end, schedule, seed, data, dev, a, out_dir, n_train=None, epochs_override=None):
    xtr, ytr, xte, yte = data
    if n_train:
        xtr, ytr = xtr[:n_train], ytr[:n_train]
    torch.manual_seed(seed)
    model = SmallCNN(a.width).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    total = epochs_override or (d_end + AFTER + TAIL)
    xtr_dev, ytr_dev = xtr.to(dev), ytr.to(dev)
    hist, fisher, t0 = [], [], time.time()
    g = torch.Generator().manual_seed(seed)
    for ep in range(total):
        blur = d_start <= ep < d_end
        lr = lr_at(ep, d_end, schedule, total, a.lr, a.decay)
        for pg in opt.param_groups:
            pg["lr"] = lr
        perm = torch.randperm(len(xtr), generator=g).to(dev)
        tl, nb = 0.0, 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            b = perm[i:i + a.bs]
            loss = F.cross_entropy(model(prep(xtr_dev[b], dev, True, blur)), ytr_dev[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item() if nb % 20 == 0 else 0.0
            nb += 1
        acc = evaluate(model, xte, yte, dev)
        hist.append({"epoch": ep + 1, "lr": round(lr, 5), "blur": blur, "test": acc,
                     "train_loss_sampled": round(tl / max(1, (nb + 19) // 20), 4)})
        if (name, seed) in FISHER_RUNS and not epochs_override:
            fisher.append(fisher_trace(model, xtr, ytr, dev))
        print(f"    {name} s{seed} Ep {ep + 1}/{total} lr {lr:.4f} {'UNSCHARF ' if blur else ''}test {100 * acc:.1f}%",
              flush=True)
    res = {"name": name, "seed": seed, "deficit": [d_start, d_end], "schedule": schedule, "epochs": total,
           "test": hist[-1]["test"], "test_blurred_input": evaluate(model, xte, yte, dev, blur=True),
           "hist": hist, "fisher": fisher, "minutes": round((time.time() - t0) / 60, 1),
           "width": a.width, "lr0": a.lr, "decay": a.decay}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}-s{seed}.json")
        json.dump(res, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--only", default="", help="Komma-Liste von Laufnamen, leer = alle")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--decay", type=float, default=0.94)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    ("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--out", default="results/critical")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device)
    data = load_cifar()
    print(f"CIFAR-10: {len(data[0])} Training, {len(data[2])} Test, Gerät {dev}", flush=True)

    if a.smoke:
        t0 = time.time()
        r = run_one("def1", 0, 1, "decay", 0, data, dev, a, None, n_train=2048, epochs_override=2)
        fisher_trace(SmallCNN(a.width).to(dev), data[0], data[1], dev, n=20)
        t1 = time.time()
        run_one("time", 0, 0, "decay", 0, data, dev, a, None, n_train=10240, epochs_override=1)
        per_epoch = (time.time() - t1) * len(data[0]) / 10240
        n_ep = sum(b + AFTER + TAIL for _, _, b, _, _ in runs_for_seed(0))
        print(f"\nSCHNELLTEST OK ({t1 - t0:.0f}s, Test {100 * r['test']:.1f}% nach 2 Mini-Epochen). "
              f"~{per_epoch:.0f}s pro Epoche, {n_ep} Epochen pro Seed -> ~{n_ep * per_epoch / 3600:.1f} h pro Seed")
        return

    only = set(filter(None, a.only.split(",")))
    for seed in [int(s) for s in a.seeds.split(",")]:
        for name, ds, de, sch, s in runs_for_seed(seed):
            if only and name not in only:
                continue
            if os.path.exists(os.path.join(a.out, f"{name}-s{s}.json")):
                continue
            print(f"[{time.strftime('%H:%M')}] Start {name} Seed {s}", flush=True)
            r = run_one(name, ds, de, sch, s, data, dev, a, a.out)
            print(f"[{time.strftime('%H:%M')}] FERTIG {name} s{s}: Test {100 * r['test']:.2f}% "
                  f"({r['minutes']} min)", flush=True)
            if dev.type == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
