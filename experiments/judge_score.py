import json, math
key = {k["id"]: k for k in json.load(open("KEY_DO_NOT_READ.json"))}
v = dict(l.split() for l in open("verdicts.txt"))
res = {}
for i, choice in v.items():
    k = key[int(i)]
    win = choice == k["new_is"]
    res.setdefault(k["cond"], []).append(win)
def binom_p(w, n):
    k = min(w, n - w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
out = {}
for c, l in res.items():
    w, n = sum(l), len(l)
    out[c] = {"new_preferred": w, "n": n, "rate": w / n, "p_vs_50": binom_p(w, n)}
    print(f"{c}: neue Geschichte bevorzugt {w}/{n} = {w/n:.0%}, p (vs. 50 %) = {binom_p(w, n):.4f}")
a, b = res["self_gen_llR"], res["rand_gen_llR"]
pa, pb = sum(a)/len(a), sum(b)/len(b)
p = (sum(a)+sum(b))/(len(a)+len(b)); se = math.sqrt(p*(1-p)*(1/len(a)+1/len(b)))
z = (pa-pb)/se
pz = math.erfc(abs(z)/math.sqrt(2))
print(f"Unterschied gezielt vs. zufällig: {pa-pb:+.0%}, z = {z:.2f}, p = {pz:.4f}")
out["diff"] = {"delta": pa-pb, "z": z, "p": pz}
json.dump(out, open("judge_result.json", "w"), indent=2)
