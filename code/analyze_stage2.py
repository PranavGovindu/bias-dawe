"""Stage-2 analysis over per-item scorer output (scoring/score_items.py). Light; stdlib only.

python analyze_stage2.py   (reads ../results/stage1/items_*.jsonl.gz, ../results/stage2/items_*.jsonl.gz and the
                            finish info in ../results/stage2/finish.jsonl.gz)
Writes ../runs/analysis/stage2_analysis.json and prints markdown tables.
"""
import gzip
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
S1 = R / "stage1"
S2 = R / "stage2"
OUT = REPO / "runs" / "analysis"
H = ("v1", "v4_format", "v11_plan")
BIASES = [-2, -4, -6, -8, -10, -12, -16]


def _open(p):
    p = Path(p)
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def load_items(p):
    with _open(p) as f:
        return {(r["id"], r["harness"]): r for r in map(json.loads, f)}


def load_finish(d, name):
    """Finish info ({finish, n_out}) for one pass, from <d>/finish.jsonl.gz (a compact copy of the generator's preds
    rows). Empty if the pass has none (the stage-1 preds carried no finish info)."""
    out = {}
    p = Path(d) / "finish.jsonl.gz"
    if not p.exists():
        return out
    with _open(p) as f:
        for r in map(json.loads, f):
            if r["name"] == name:
                out[(r["id"], r["harness"])] = r
    return out


def acc(it, h, cat=None):
    rs = [r for (i, hh), r in it.items() if hh == h and (cat is None or r["cat"] == cat)]
    return 100 * sum(r["ok"] for r in rs) / len(rs) if rs else float("nan"), len(rs)


def mcnemar(a, b, h):
    """a, b: item dicts. Returns (n_a_only, n_b_only, exact two-sided p)."""
    keys = [k for k in a if k[1] == h and k in b]
    x = sum(1 for k in keys if a[k]["ok"] and not b[k]["ok"])
    y = sum(1 for k in keys if b[k]["ok"] and not a[k]["ok"])
    n = x + y
    if n == 0:
        return x, y, 1.0
    k = min(x, y)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return x, y, min(1.0, p)


def trailing(text, h):
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if h == "v4_format":
        j = t.rfind("</tool_call>")
        return j >= 0 and t[j + len("</tool_call>"):].strip() != ""
    if h == "v11_plan" or h == "v1":
        j = t.rfind("}")
        return j >= 0 and t[j + 1:].strip().strip("`").strip() != ""
    return False


def overcont(it, fin, h):
    """Over-continuation on single-call items of harness h: too many calls, trailing text, hit max tokens."""
    rs = [(k, r) for k, r in it.items() if k[1] == h and r["n_gold"] == 1]
    many = sum(1 for k, r in rs if r["n_pred"] is not None and r["n_pred"] > 1)
    trail = sum(1 for k, r in rs if trailing(r["text"], h))
    maxed = sum(1 for k, r in rs if fin.get(k, {}).get("finish") == "length")
    return {"n": len(rs), "too_many": many, "trailing": trail, "max_tokens": maxed}


def fewcalls(it, h):
    rs = [r for (i, hh), r in it.items() if hh == h and r["n_gold"] and r["n_gold"] > 1]
    return {"n_multi": len(rs), "too_few": sum(1 for r in rs if r["n_pred"] is not None and r["n_pred"] < r["n_gold"]),
            "too_many": sum(1 for r in rs if r["n_pred"] is not None and r["n_pred"] > r["n_gold"])}


def main():
    arms = {"U (sft, stage 1)": (S1 / "items_sft_v1_s0.jsonl.gz", (S1, "sft_v1_s0")),
            "base (F a=0)": (S1 / "items_base.jsonl.gz", (S1, "base"))}
    for name in ["B0"] + [f"B{b}" for b in BIASES] + ["A", "C", "D", "E", "F25", "F50", "F75"]:
        p = S2 / f"items_{name}.jsonl.gz"
        if p.exists():
            arms[name] = (p, (S2, name))
    it = {n: load_items(p) for n, (p, _) in arms.items()}
    fin = {n: load_finish(*f) for n, (_, f) in arms.items()}
    U = it["U (sft, stage 1)"]
    out = {"table": {}, "mcnemar_vs_U": {}, "overcont": {}, "fewcalls": {}}
    print("| arm | v1 | v4 | v4 parallel | v11 | v4 gain | v1 cost |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    u1, u4 = acc(U, "v1")[0], acc(U, "v4_format")[0]
    for n, items in it.items():
        row = {h: acc(items, h)[0] for h in H}
        row["v4_parallel"] = acc(items, "v4_format", "parallel")[0]
        row["n"] = {h: acc(items, h)[1] for h in H}
        out["table"][n] = row
        f = lambda v: "—" if v != v else f"{v:.1f}"
        print(f"| {n} | {f(row['v1'])} | {f(row['v4_format'])} | {f(row['v4_parallel'])} | {f(row['v11_plan'])} | "
              f"{row['v4_format'] - u4:+.1f} | {u1 - row['v1']:+.1f} |")
    print("\nMcNemar vs U (arm-only wins / U-only wins / exact p):")
    for n, items in it.items():
        if n.startswith("U"):
            continue
        out["mcnemar_vs_U"][n] = {h: mcnemar(items, U, h) for h in H if any(k[1] == h for k in items)}
        print(n, {h: v for h, v in out["mcnemar_vs_U"][n].items()})
    print("\nOver-continuation on single-call items (too many calls / trailing text / hit max tokens):")
    for n in it:
        out["overcont"][n] = {h: overcont(it[n], fin[n], h) for h in ("v1", "v4_format")}
        out["fewcalls"][n] = fewcalls(it[n], "v4_format")
        print(n, out["overcont"][n], "v4 multi:", out["fewcalls"][n])
    # pre-registered best-B rule (STAGE2.md): max v4 among b with v1 >= U v1 - 2; ties -> smaller |b|
    cand = [b for b in BIASES if f"B{b}" in it and out["table"][f"B{b}"]["v1"] >= u1 - 2]
    best = max(cand, key=lambda b: (round(out["table"][f"B{b}"]["v4_format"], 6), -abs(b)))
    out["best_B"] = best
    print(f"\nbest B (rule) = {best}")
    pairs = [("A", f"B{best}"), ("E", f"B{best}"), ("A", "E"), ("D", "A"), ("A", "C"), ("F25", "A"), ("F50", "A"), ("F75", "A"),
             ("B0", "U (sft, stage 1)")] + [("A", f"B{b}") for b in BIASES]
    out["pairs"] = {}
    print("Pairwise McNemar (first-only wins / second-only wins / exact p):")
    for a, b in pairs:
        if a in it and b in it:
            out["pairs"][f"{a} vs {b}"] = {h: mcnemar(it[a], it[b], h) for h in H}
            same = {h: sum(it[a][k]["text"] == it[b][k]["text"] for k in it[a] if k[1] == h and k in it[b]) for h in H}
            out["pairs"][f"{a} vs {b}"]["same_text"] = same
            print(f"{a} vs {b}", {h: out["pairs"][f"{a} vs {b}"][h] for h in H}, "same text", same)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage2_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
