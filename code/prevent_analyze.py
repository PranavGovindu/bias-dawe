"""Side result (prevent.py arms): per-arm table, per-item exact McNemar vs (a), failure modes.
Usage: python3 prevent_analyze.py a=../results/stage1/items_sft_v1_s0.jsonl.gz b=../results/prevent/items_prev_b_rows1x.jsonl.gz [...]"""
import gzip
import json
import re
import sys
from collections import Counter
from math import comb


def load(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        return {(r["harness"], r["id"]): r for r in map(json.loads, f)}


def acc(items, h, cat=None):
    xs = [r for (hh, _), r in items.items() if hh == h and (cat is None or r["cat"] == cat)]
    return 100 * sum(r["ok"] for r in xs) / len(xs), len(xs)


def mcnemar(a, x, h):
    keys = [k for k in a if k[0] == h]
    b01 = sum(1 for k in keys if not a[k]["ok"] and x[k]["ok"])  # fixed by x
    b10 = sum(1 for k in keys if a[k]["ok"] and not x[k]["ok"])  # broken by x
    n, k = b01 + b10, min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return b01, b10, p


def mode(r):
    if r["ok"]:
        return "ok"
    t = r["text"].lstrip()
    if r["harness"] == "v11_plan":
        if not t.startswith("<plan>"):
            return "skip_plan_json" if t.startswith("{") else "skip_plan_other"
        rest = re.sub(r"^<plan>.*?</plan>", "", t, flags=re.S).strip()
        if not rest:
            return "plan_then_stop"
    if r["fmt_fail"]:
        return "format"
    if r["n_pred"] is not None and r["n_gold"] is not None:
        if r["n_pred"] < r["n_gold"]:
            return "too_few_calls"
        if r["n_pred"] > r["n_gold"]:
            return "too_many_calls"
    return "wrong_args_or_name"


arms = dict(a.split("=", 1) for a in sys.argv[1:])
data = {k: load(v) for k, v in arms.items()}
ref = data.get("a")
print("| Arm | v1 | v4 | v4 parallel | v11 |")
print("|---|---:|---:|---:|---:|")
for k, it in data.items():
    print(f"| {k} | {acc(it, 'v1')[0]:.1f} | {acc(it, 'v4_format')[0]:.1f} | {acc(it, 'v4_format', 'parallel')[0]:.1f} | "
          f"{acc(it, 'v11_plan')[0]:.1f} |")
if ref:
    print("\nMcNemar vs a (fixed / broken / exact p):")
    for k, it in data.items():
        if k == "a":
            continue
        s = []
        for h in ("v1", "v4_format", "v11_plan"):
            b01, b10, p = mcnemar(ref, it, h)
            s.append(f"{h} +{b01}/-{b10} p={p:.3g}")
        print(f"  {k}: " + "; ".join(s))
print("\nFailure modes:")
for h in ("v4_format", "v11_plan", "v1"):
    for k, it in data.items():
        c = Counter(mode(r) for (hh, _), r in it.items() if hh == h)
        c.pop("ok", None)
        print(f"  {h:9s} {k}: " + ", ".join(f"{m} {n}" for m, n in c.most_common()))
