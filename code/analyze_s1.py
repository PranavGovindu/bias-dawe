"""Seed-1 replicate analysis (stdlib only).

python analyze_s1.py select   -> picks b from TRAIN scores only (PREDICTIONS.md rule); writes ../runs/analysis/b_choice.json
                                (the committed choice, made before any seed-1 test score, is ../results/stage2_s1/b_choice.json)
python analyze_s1.py test     -> test table, McNemar, over-continuation (uses the committed b_choice.json)
"""
import json
import sys
from pathlib import Path

from analyze_stage2 import OUT, R, acc, fewcalls, load_finish, load_items, mcnemar, overcont

S = R / "stage2_s1"
U_ITEMS = R / "stage1" / "items_prev_a_s1.jsonl.gz"  # seed-1 unpatched: prevent.py arm a1
TRAIN_B = [-2, -4, -6, -8, -10, -12]
TEST_B = [-2, -4, -6, -8, -10, -12, -16]
H = ("v1", "v4_format", "v11_plan")


def select():
    ref = load_items(S / "items_Btr0.jsonl.gz")
    v1_0, n1 = acc(ref, "v1")
    v4_0, n4 = acc(ref, "v4_format")
    rows = {}
    print(f"train (n v1={n1}, v4={n4}); b=0: v1 {v1_0:.2f}, v4 {v4_0:.2f}")
    for b in TRAIN_B:
        it = load_items(S / f"items_Btr{b}.jsonl.gz")
        rows[b] = {"v1": acc(it, "v1")[0], "v4": acc(it, "v4_format")[0]}
        print(f"b={b:4d}: v1 {rows[b]['v1']:.2f} ({rows[b]['v1'] - v1_0:+.2f})  v4 {rows[b]['v4']:.2f} ({rows[b]['v4'] - v4_0:+.2f})")
    ok = [b for b in TRAIN_B if rows[b]["v1"] >= v1_0 - 1e-9]
    flag = None
    if not ok:
        flag = "no b keeps v1 train accuracy; rule undefined -> taking the b with the smallest v1 loss, then highest v4"
        ok = [min(TRAIN_B, key=lambda b: (v1_0 - rows[b]["v1"], -rows[b]["v4"], abs(b)))]
    best = max(ok, key=lambda b: (round(rows[b]["v4"], 9), -abs(b)))
    out = {"rule": "highest v4 train accuracy among b in {-2..-12} with no v1 train loss; ties -> smaller |b|",
           "b0": {"v1": v1_0, "v4": v4_0}, "by_b": rows, "eligible": ok, "chosen_b": best, "flag": flag}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "b_choice.json").write_text(json.dumps(out, indent=1))
    print("eligible", ok, "-> chosen b =", best, flag or "")


def test():
    choice = json.loads((S / "b_choice.json").read_text())
    bc = choice["chosen_b"]
    arms = {"U (a1, prevent_01)": U_ITEMS, "B0 (regen)": S / "items_B0.jsonl.gz"}
    for b in TEST_B:
        arms[f"B{b}"] = S / f"items_B{b}.jsonl.gz"
    arms["A"] = S / "items_A.jsonl.gz"
    arms["D"] = S / "items_D.jsonl.gz"
    it = {n: load_items(p) for n, p in arms.items()}
    fin = {n: load_finish(S, n.split(' ')[0]) for n in arms}
    U = it["U (a1, prevent_01)"]
    u1, u4 = acc(U, "v1")[0], acc(U, "v4_format")[0]
    out = {"chosen_b": bc, "table": {}, "vsU": {}, "pairs": {}, "overcont": {}, "fewcalls": {}}
    print("| arm | v1 | v4 | v4 parallel | v11 | v4 gain | v1 cost |\n|---|---:|---:|---:|---:|---:|---:|")
    for n, x in it.items():
        r = {h: acc(x, h)[0] for h in H}
        r["par"] = acc(x, "v4_format", "parallel")[0]
        out["table"][n] = r
        f = lambda v: "—" if v != v else f"{v:.1f}"
        tag = " (chosen)" if n == f"B{bc}" else ""
        print(f"| {n}{tag} | {f(r['v1'])} | {f(r['v4_format'])} | {f(r['par'])} | {f(r['v11_plan'])} | "
              f"{r['v4_format'] - u4:+.1f} | {u1 - r['v1']:+.1f} |")
    print("\nvs U (arm-only / U-only / p):")
    for n, x in it.items():
        if n.startswith("U"):
            continue
        out["vsU"][n] = {h: mcnemar(x, U, h) for h in H if any(k[1] == h for k in x)}
        print(n, out["vsU"][n])
    print("\npairs (first-only / second-only / p) + same text:")
    for a, b in [("A", f"B{bc}"), ("D", "A"), ("B0 (regen)", "U (a1, prevent_01)")] + [("A", f"B{x}") for x in TEST_B]:
        res = {h: mcnemar(it[a], it[b], h) for h in H if any(k[1] == h for k in it[b])}
        res["same_text"] = {h: sum(it[a][k]["text"] == it[b][k]["text"] for k in it[a] if k[1] == h and k in it[b]) for h in H}
        out["pairs"][f"{a} vs {b}"] = res
        print(f"{a} vs {b}", res)
    print("\nover-continuation, single-call (too many / trailing / max tokens) and v4 multi-call too few/too many:")
    for n in it:
        out["overcont"][n] = {h: overcont(it[n], fin[n], h) for h in ("v1", "v4_format")}
        out["fewcalls"][n] = fewcalls(it[n], "v4_format")
        print(n, out["overcont"][n], out["fewcalls"][n])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis_s1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
