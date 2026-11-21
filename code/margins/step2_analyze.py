"""Step 2 analysis: seed-0 rerun (sft_v1_s0r) and seed 2 vs seeds 0/1 (3060 states, same path), per-item test outcomes,
and the seed-1 +12 constant test. usage: python3 step2_analyze.py > analysis.txt
(committed output: ../../results/step2/analysis.txt)"""
import json
from collections import Counter
import numpy as np
from analyze_lib import RES, jopen

LT, GJ = 27, (85806, 7082)
q = lambda xs, p: float(np.quantile(np.array(xs), p))


def zmap(r):
    z = {int(t): v for t, v in r["top"]}
    for t, v in r["z"].items():
        z.setdefault(int(t), v)
    return z


def m_lt(r):  # '<' minus best rival (plan start); its negative = '<' deficit (JSON reply start)
    z = zmap(r)
    return z[LT] - max(v for t, v in z.items() if t != LT)


def m_gt2(r):  # after 'plan': best non-JSON-merged minus best JSON-merged
    z = zmap(r)
    return max(v for t, v in z.items() if t not in GJ) - max(z[t] for t in GJ)


def load_run(run):
    """{(set, pos, id): row} with set in v1/train, v1/test, v11/train, v11/test; pos in start/after_plantok."""
    M_ = RES / "margins"
    files = {"s0": [M_ / "rs_margins_s0.jsonl.gz", M_ / "rs_margins_s0_v11.jsonl.gz"],
             "s1": [M_ / "rs_margins_s1.jsonl.gz", M_ / "rs_margins_s1_v11.jsonl.gz"]}.get(run, [M_ / f"rs_margins_{run}.jsonl.gz"])
    out = {}
    for f in files:
        for l in jopen(f):
            r = json.loads(l)
            out[(f"{r['harness']}/{r['split']}", r["pos"], r["id"])] = r
    return out


RUNS = ["s0", "sft_v1_s0r", "s1", "sft_v1_s2"]
M = {r: load_run(r) for r in RUNS}


def series(run, st, pos, fn):
    return {k[2]: fn(v) for k, v in M[run].items() if k[0] == st and k[1] == pos}


def pr(*a):
    print(*a, flush=True)


MEAS = [("plan start ('<' - best)", "v11", "start", m_lt), ("after 'plan' (non-JSON - JSON)", "v11", "after_plantok", m_gt2),
        ("JSON reply start '<' deficit", "v1", "start", lambda r: -m_lt(r))]
pr("=" * 100, "\nA. Margin distributions per run (3060 states)\n")
for name, h, pos, fn in MEAS:
    for split in ("train", "test"):
        for run in RUNS:
            v = list(series(run, f"{h}/{split}", pos, fn).values())
            a = np.array(v)
            pr(f"  {name:32s} {split:5s} {run:11s} n {len(v)} median {np.median(a):+6.2f} q05 {q(v,.05):+6.2f} q95 {q(v,.95):+6.2f} "
               f"%<0 {100*(a<0).mean():5.1f} %|m|<4 {100*(abs(a)<4).mean():5.1f}")
pr("=" * 100, "\nB. Per-item differences vs seed 0 (same items)\n")
for name, h, pos, fn in MEAS:
    for split in ("train", "test"):
        base = series("s0", f"{h}/{split}", pos, fn)
        for run in ("sft_v1_s0r", "s1", "sft_v1_s2"):
            o = series(run, f"{h}/{split}", pos, fn)
            ids = sorted(base)
            x, y = np.array([base[i] for i in ids]), np.array([o[i] for i in ids])
            d = y - x
            rk = lambda a: np.argsort(np.argsort(a))
            pr(f"  {name:32s} {split:5s} {run:11s} - s0: mean {d.mean():+6.2f} SD {d.std():5.2f} median {np.median(d):+6.2f} "
               f"Pearson {np.corrcoef(x,y)[0,1]:.3f} Spearman {np.corrcoef(rk(x),rk(y))[0,1]:.3f}")
pr("=" * 100, "\nC. Test outcomes (unpatched; their scorer per item) vs plan-start margins\n")
items = {"s0": RES / "stage3/s0/items_Uchk.jsonl.gz", "sft_v1_s0r": RES / "step2/items_sft_v1_s0r.jsonl.gz",
         "s1": RES / "stage3/s1/items_Uchk.jsonl.gz", "sft_v1_s2": RES / "step2/items_sft_v1_s2.jsonl.gz"}
for run in RUNS:
    it = {(r["id"], r["harness"]): r for r in map(json.loads, jopen(items[run]))}
    acc = {h: 100 * np.mean([r["ok"] for k, r in it.items() if k[1] == h]) for h in ("v1", "v4_format", "v11_plan")}
    m = series(run, "v11/test", "start", m_lt)
    opened = {i: it[(i, "v11_plan")]["text"].startswith("<") for i in m}
    res = [(m[i], opened[i]) for i in m if abs(m[i]) >= 0.25]
    agree = sum((x > 0) == o for x, o in res)
    pr(f"  {run:11s} v1 {acc['v1']:5.1f} v4 {acc['v4_format']:5.1f} v11 {acc['v11_plan']:5.1f}; v11 opens {sum(opened.values())}/204, "
       f"predicted (m>0) {sum(x > 0 for x in m.values())}; sign agree {agree}/{len(res)} (validation)")
a = {r["id"]: r["text"].startswith("<") for r in map(json.loads, jopen(items["s0"])) if r["harness"] == "v11_plan"}
b = {r["id"]: r["text"].startswith("<") for r in map(json.loads, jopen(items["sft_v1_s0r"])) if r["harness"] == "v11_plan"}
pr(f"\n  same-item v11 first token, rerun vs seed 0 (U): both open {sum(a[i] and b[i] for i in a)}, s0 only {sum(a[i] and not b[i] for i in a)}, s0r only {sum(b[i] and not a[i] for i in a)}, neither {sum(not a[i] and not b[i] for i in a)}")
ta = {r["id"]: r["text"] for r in map(json.loads, jopen(items["s0"])) if r["harness"] == "v1"}
tb = {r["id"]: r["text"] for r in map(json.loads, jopen(items["sft_v1_s0r"])) if r["harness"] == "v1"}
pr(f"  identical v1 test texts, rerun vs seed 0: {sum(ta[i] == tb[i] for i in ta)}/204")

pr("=" * 100, "\nD. The seed-1 +12 test: first-token (c, b) = (12, -8) vs control (8, -8), test items\n")
m_ps = series("s1", "v11/test", "start", m_lt)
m_ap = series("s1", "v11/test", "after_plantok", m_gt2)
for c in (8, 12):
    it = {(r["id"], r["harness"]): r for r in map(json.loads, jopen(RES / f"step2/items_s1_c{c}.jsonl.gz"))}
    v11 = {k[0]: r for k, r in it.items() if k[1] == "v11_plan"}
    op = [i for i in v11 if v11[i]["text"].startswith("<plan")]
    js = [i for i in op if v11[i]["text"].startswith("<plan>{")]
    fails = [i for i in v11 if not v11[i]["ok"]]
    nop = [i for i in fails if not v11[i]["text"].startswith("<plan")]
    acc = {h: 100 * np.mean([r["ok"] for k, r in it.items() if k[1] == h]) for h in ("v1", "v4_format", "v11_plan")}
    pr(f"  c={c:2d}: v1 {acc['v1']:.1f} v4 {acc['v4_format']:.1f} v11 {acc['v11_plan']:.1f}; opens {len(op)}/204; v1 replies starting '<' "
       f"{sum(r['text'].startswith('<') for k, r in it.items() if k[1] == 'v1')}; '<plan>{{' {len(js)} ({100*len(js)/len(op):.1f}% of openers); "
       f"v11 failures {len(fails)}: no-plan {len(nop)}, '<plan>{{' {sum(i in js for i in fails)}, other {len(fails)-len(nop)-sum(i in js for i in fails)}")
    po = [m_ps[i] + c > 0 for i in v11]
    pr(f"        per-item (validation): opens predicted by m+c>0: {sum(po)}; agree {sum((m_ps[i] + c > 0) == v11[i]['text'].startswith('<') for i in v11)}/204; "
       f"'<plan>{{' among openers predicted by after-'plan' m<0: {sum(m_ap[i] < 0 for i in op)}, agree {sum((m_ap[i] < 0) == (i in js) for i in op)}/{len(op)}")
it8 = {r["id"]: r for r in map(json.loads, jopen(RES / "step2/items_s1_c8.jsonl.gz")) if r["harness"] == "v11_plan"}
s3 = {r["id"]: r for r in map(json.loads, jopen(RES / "stage3/s1/items_B2fte_c8_b-8.jsonl.gz")) if r["harness"] == "v11_plan"}
pr(f"  control vs stage-3 B2-first (8,-8): identical v11 texts {sum(it8[i]['text'] == s3[i]['text'] for i in it8)}/204, outcome flips {sum(it8[i]['ok'] != s3[i]['ok'] for i in it8)}")
