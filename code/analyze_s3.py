"""Stage-3 analysis (stdlib only).
python analyze_s3.py select <s0|s1>  -> (c, b) for B2 and B2-first from TRAIN only (STAGE3_SPEC rule) -> ../runs/analysis/choice_<s>.json
                                        (the committed choice, recorded before any test item was scored, is ../results/stage3/<s>/choice.json)
python analyze_s3.py test <s0|s1>    -> test tables, McNemar, checks (uses the committed choice.json)"""
import json
import sys
from pathlib import Path

from analyze_stage2 import OUT, R, acc, fewcalls, load_finish, load_items, mcnemar, overcont

H = ("v1", "v4_format", "v11_plan")
CS, BS = (0, 2, 4, 8), (-4, -8, -12)
REF = {"s0": {"U": R / "stage1" / "items_sft_v1_s0.jsonl.gz", "A1": R / "stage2" / "items_A.jsonl.gz"},
       "s1": {"U": R / "stage1" / "items_prev_a_s1.jsonl.gz",  # seed-1 unpatched: prevent.py arm a1
              "A1": R / "stage2_s1" / "items_A.jsonl.gz"}}


def name(tag, c, b):
    return f"{tag}_c{c}_b{b}"


def select(seed):
    S = R / "stage3" / seed
    out = {"rule": "max mean(v1, v4, v11) TRAIN accuracy; ties -> smaller |c|, then smaller |b|"}
    for mode, tag in (("B2", "B2tr"), ("B2first", "B2ftr")):
        rows = {}
        for c in CS:
            for b in BS:
                t = "B2tr" if c == 0 else tag  # B2-first at c = 0 is B2 at c = 0 (no '<' push)
                it = load_items(S / f"items_{name(t, c, b)}.jsonl.gz")
                a = {h: acc(it, h)[0] for h in H}
                a["mean"] = sum(a[h] for h in H) / 3
                rows[f"{c},{b}"] = a
        best = max(rows, key=lambda k: (round(rows[k]["mean"], 9), -abs(int(k.split(",")[0])), -abs(int(k.split(",")[1]))))
        out[mode] = {"train": rows, "chosen": [int(x) for x in best.split(",")]}
        print(f"\n{mode} train (v1 / v4 / v11 / mean):")
        for k, a in rows.items():
            print(f"  c={k.split(',')[0]:>2} b={k.split(',')[1]:>3}: {a['v1']:5.1f} {a['v4_format']:5.1f} {a['v11_plan']:5.1f}  mean {a['mean']:5.2f}{'  <- chosen' if k == best else ''}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"choice_{seed}.json").write_text(json.dumps(out, indent=1))


def v11_classes(it):
    rs = [r for k, r in it.items() if k[1] == "v11_plan" and not r["ok"]]
    t = lambda r: r["text"].lstrip()
    return {"no_plan": sum(1 for r in rs if not t(r).startswith("<plan>")),
            "plan_only": sum(1 for r in rs if t(r).startswith("<plan>") and r["text"].strip().endswith("</plan>")),
            "other": sum(1 for r in rs if t(r).startswith("<plan>") and not r["text"].strip().endswith("</plan>"))}


def v1_lt_start(it):
    return sum(1 for k, r in it.items() if k[1] == "v1" and r["text"].lstrip().startswith("<"))


def test(seed):
    S = R / "stage3" / seed
    ch = json.loads((S / "choice.json").read_text())
    cb2, cbf = ch["B2"]["chosen"], ch["B2first"]["chosen"]
    arms = {"U": REF[seed]["U"], "A1": REF[seed]["A1"], "A1chk": S / "items_A1chk.jsonl.gz", "Uchk": S / "items_Uchk.jsonl.gz",
            "A2": S / "items_A2.jsonl.gz", "D2": S / "items_D2.jsonl.gz", "C2": S / "items_C2.jsonl.gz",
            f"B2 {cb2} (chosen)": S / f"items_{name('B2te', *cb2)}.jsonl.gz",
            f"B2first {cbf} (chosen)": S / f"items_{name('B2te' if cbf[0] == 0 else 'B2fte', *cbf)}.jsonl.gz"}
    it = {n: load_items(p) for n, p in arms.items()}
    grid = {}
    for c in CS:
        for b in BS:
            grid[("B2", c, b)] = load_items(S / f"items_{name('B2te', c, b)}.jsonl.gz")
            if c:
                grid[("B2first", c, b)] = load_items(S / f"items_{name('B2fte', c, b)}.jsonl.gz")
    U = it["U"]
    u = {h: acc(U, h)[0] for h in H}
    out = {"chosen": {"B2": cb2, "B2first": cbf}, "table": {}, "grid": {}, "mcnemar": {}, "checks": {}, "classes": {}}
    print(f"\n[{seed}] | arm | v1 | v4 | v4 parallel | v11 | v1 '<'-start | v11 no-plan / plan-only / other |")
    for n, x in it.items():
        r = {h: acc(x, h)[0] for h in H}
        r["par"] = acc(x, "v4_format", "parallel")[0]
        r["v1_lt"] = v1_lt_start(x)
        r["v11_classes"] = v11_classes(x)
        out["table"][n] = r
        print(f"| {n} | {r['v1']:.1f} | {r['v4_format']:.1f} | {r['par']:.1f} | {r['v11_plan']:.1f} | {r['v1_lt']} | "
              f"{r['v11_classes']['no_plan']} / {r['v11_classes']['plan_only']} / {r['v11_classes']['other']} |")
    print(f"\n[{seed}] test grid (v1 / v4 / v11), U = {u['v1']:.1f} / {u['v4_format']:.1f} / {u['v11_plan']:.1f}:")
    for (m, c, b), x in grid.items():
        r = {h: acc(x, h)[0] for h in H}
        r["v1_lt"] = v1_lt_start(x)
        out["grid"][f"{m} c={c} b={b}"] = r
        print(f"  {m:8s} c={c} b={b:>3}: {r['v1']:5.1f} {r['v4_format']:5.1f} {r['v11_plan']:5.1f}   v1 '<'-start {r['v1_lt']}")
    bn, fn = f"B2 {cb2} (chosen)", f"B2first {cbf} (chosen)"
    pairs = [("A2", fn), ("A2", bn), ("D2", "A2"), ("A2", "A1"), ("D2", "A1"), ("C2", "U"), ("A2", "U"), ("D2", "U"),
             (fn, "U"), (bn, "U"), ("A1chk", "A1"), ("Uchk", "U")]
    print(f"\n[{seed}] McNemar (first-only / second-only / p) and identical texts per harness:")
    for a, b in pairs:
        res = {h: mcnemar(it[a], it[b], h) for h in H}
        res["same_text"] = {h: sum(it[a][k]["text"] == it[b][k]["text"] for k in it[a] if k[1] == h and k in it[b]) for h in H}
        out["mcnemar"][f"{a} vs {b}"] = res
        print(f"  {a} vs {b}: " + "; ".join(f"{h} {res[h][0]}/{res[h][1]} p={res[h][2]:.2g}" for h in H) + f"  same {res['same_text']}")
    chk = load_items(S / "items_B2fchk_c0_b-8.jsonl.gz")
    ref = grid[("B2", 0, -8)]
    out["checks"]["two_pass_c0_vs_one_pass"] = {h: sum(chk[k]["text"] == ref[k]["text"] for k in chk if k[1] == h) for h in H}
    print(f"\n[{seed}] two-pass c=0 vs one-pass (0,-8) identical texts:", out["checks"]["two_pass_c0_vs_one_pass"])
    fins = {n: load_finish(S, p) for n, p in (("A2", "A2"), ("D2", "D2"), (fn, name("B2fte", *cbf) if cbf[0] else name("B2te", *cbf)), (bn, name("B2te", *cb2)))}
    for n, f in fins.items():
        out["classes"][n] = {"overcont": {h: overcont(it[n], f, h) for h in ("v1", "v4_format")}, "v4_multi": fewcalls(it[n], "v4_format")}
        print(f"  {n}: over-continuation {out['classes'][n]['overcont']}  v4 multi {out['classes'][n]['v4_multi']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"analysis_s3_{seed}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]](sys.argv[2])
