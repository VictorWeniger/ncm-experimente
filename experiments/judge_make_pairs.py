import json, random, sys
d = json.load(open(sys.argv[1]))
V = d["variants"]
rng = random.Random(2026)
pools = {}
for cond in ("self_gen_llR", "rand_gen_llR"):
    pool = []
    for s, b in V["baseline"].items():
        for i, (tb, tc) in enumerate(zip(b["texts"], V[cond][s]["texts"])):
            if tb != tc and tb.strip() and tc.strip():
                pool.append((s, i, tb, tc))
    pools[cond] = pool
    print(cond, "geänderte Geschichten:", len(pool))
n = min(60, *(len(p) for p in pools.values()))
items = []
for cond, pool in pools.items():
    for s, i, tb, tc in rng.sample(pool, n):
        items.append((cond, s, i, tb, tc))
rng.shuffle(items)
pairs, key = [], []
for k, (cond, s, i, tb, tc) in enumerate(items):
    flip = rng.random() < 0.5
    A, B = (tc, tb) if flip else (tb, tc)
    pairs.append({"id": k, "A": A, "B": B})
    key.append({"id": k, "cond": cond, "seed": s, "idx": i, "new_is": "A" if flip else "B"})
json.dump(pairs, open("pairs.json", "w"), ensure_ascii=False, indent=0)
json.dump(key, open("KEY_DO_NOT_READ.json", "w"))
print("Paare:", len(pairs))
