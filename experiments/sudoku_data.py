"""Erzeugt Sudoku-Trainingsdaten im selben Format wie die NCM-Testdaten.

Format (wie data/sudoku/<task>/puzzles.pickle bzw. solutions.pickle):
  Rätsel:  [11] + 89 Token + [11]   (11 = BOS, 0 = leer, 1–9 Ziffern, 10 = Zeilentrenner)
  Lösung:  89 Token                 (Ziffern, 10 = Zeilentrenner)

Jedes Rätsel hat eine eindeutige Lösung (prüfbar abschaltbar mit --no_unique).
Rätsel, die zufällig mit Testbrettern übereinstimmen, werden verworfen.

Aufruf:
    python experiments/sudoku_data.py --n 50000 --clues 30,35,40 --out data/sudoku/train --workers 16
"""
import argparse
import os
import pickle
import random
from multiprocessing import Pool

SEP, BOS = 10, 11
ALL = (1 << 9) - 1


def box(r, c):
    return (r // 3) * 3 + c // 3


def solve_count(grid, limit=2):
    """Zählt Lösungen bis `limit` (Bitmasken-Backtracking mit MRV). grid: Liste mit 81 Werten, 0 = leer."""
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    empty = []
    for i, v in enumerate(grid):
        r, c = divmod(i, 9)
        if v:
            b = 1 << (v - 1)
            if rows[r] & b or cols[c] & b or boxes[box(r, c)] & b:
                return 0
            rows[r] |= b
            cols[c] |= b
            boxes[box(r, c)] |= b
        else:
            empty.append(i)
    count = 0

    def rec(empty):
        nonlocal count
        if not empty:
            count += 1
            return count >= limit
        best, best_opts, best_n = None, 0, 10
        for j, i in enumerate(empty):
            r, c = divmod(i, 9)
            opts = ALL & ~(rows[r] | cols[c] | boxes[box(r, c)])
            n = bin(opts).count("1")
            if n < best_n:
                best, best_opts, best_n = j, opts, n
                if n <= 1:
                    break
        if best_n == 0:
            return False
        i = empty[best]
        rest = empty[:best] + empty[best + 1:]
        r, c = divmod(i, 9)
        bx = box(r, c)
        opts = best_opts
        while opts:
            b = opts & -opts
            opts ^= b
            rows[r] |= b
            cols[c] |= b
            boxes[bx] |= b
            stop = rec(rest)
            rows[r] ^= b
            cols[c] ^= b
            boxes[bx] ^= b
            if stop:
                return True
        return False

    rec(empty)
    return count


def random_full_grid(rng):
    """Vollständig gelöstes Sudoku: Basismuster + zufällige gültigkeitserhaltende Permutationen."""
    base = lambda r, c: (3 * (r % 3) + r // 3 + c) % 9
    bands = rng.sample(range(3), 3)
    rows = [b * 3 + r for b in bands for r in rng.sample(range(3), 3)]
    stacks = rng.sample(range(3), 3)
    cols = [s * 3 + c for s in stacks for c in rng.sample(range(3), 3)]
    digits = rng.sample(range(1, 10), 9)
    g = [digits[base(r, c)] for r in rows for c in cols]
    if rng.random() < 0.5:  # transponieren
        g = [g[c * 9 + r] for r in range(9) for c in range(9)]
    return g


def make_puzzle(args):
    seed, clues, unique = args
    rng = random.Random(seed)
    sol = random_full_grid(rng)
    puz = sol[:]
    order = list(range(81))
    rng.shuffle(order)
    filled = 81
    for i in order:
        if filled <= clues:
            break
        v = puz[i]
        puz[i] = 0
        if unique and solve_count(puz, 2) != 1:
            puz[i] = v
        else:
            filled -= 1
    if filled != clues:
        return None
    return puz, sol


def to_tokens(grid81):
    out = []
    for r in range(9):
        out += grid81[r * 9:(r + 1) * 9]
        if r < 8:
            out.append(SEP)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--clues", default="30,35,40", help="Anzahl Vorgaben, zufällig gemischt")
    ap.add_argument("--out", default="data/sudoku/train")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no_unique", action="store_true")
    ap.add_argument("--test_dirs", default="data/sudoku/easy,data/sudoku/medium,data/sudoku/hard",
                    help="Testbretter, die nicht im Training vorkommen dürfen")
    a = ap.parse_args()

    clue_opts = [int(c) for c in a.clues.split(",")]
    test = set()
    for d in a.test_dirs.split(","):
        p = os.path.join(d, "puzzles.pickle")
        if os.path.exists(p):
            test |= {tuple(x) for x in pickle.load(open(p, "rb"))}
    print(f"{len(test)} Testbretter werden ausgeschlossen")

    rng = random.Random(a.seed)
    puzzles, solutions = [], []
    tries = 0
    with Pool(a.workers) as pool:
        while len(puzzles) < a.n:
            need = int((a.n - len(puzzles)) * 1.2) + 10
            jobs = [(a.seed * 1_000_003 + tries + i, rng.choice(clue_opts), not a.no_unique) for i in range(need)]
            tries += need
            for res in pool.imap_unordered(make_puzzle, jobs, chunksize=64):
                if res is None:
                    continue
                puz, sol = res
                item = [BOS] + to_tokens(puz) + [BOS]
                if tuple(item) in test:
                    continue
                puzzles.append(item)
                solutions.append(to_tokens(sol))
                if len(puzzles) >= a.n:
                    break
            print(f"  {len(puzzles)}/{a.n}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    pickle.dump(puzzles, open(os.path.join(a.out, "puzzles.pickle"), "wb"))
    pickle.dump(solutions, open(os.path.join(a.out, "solutions.pickle"), "wb"))
    print(f"Gespeichert: {len(puzzles)} Rätsel in {a.out}")


if __name__ == "__main__":
    main()
