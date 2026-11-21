"""neartie_01 step-1 analysis (stdlib + numpy). Reads ../../results/margins/margins_s{0,1}.jsonl.gz (output of margins.py)
and the per-item files in ../../results/; prints every table of the step-1 analysis.
usage: python3 analyze.py > analysis.txt   (committed output: ../../results/margins/analysis.txt)"""
import json, os, statistics as st
from collections import Counter, defaultdict
import numpy as np
from analyze_lib import *

DW = str(RES)
PV = str(RES / "prevent")
TOKNAME = {27: "'<'", 4754: "'{\"'", 90: "'{'", 248046: "eos", 198: "'\\n'", 271: "'\\n\\n'", 2069: "'},'", 13587: "']}'"}
BINS = [-1e9, -12, -8, -4, -2, 0, 2, 4, 8, 12, 1e9]
BLAB = ["<-12", "-12..-8", "-8..-4", "-4..-2", "-2..0", "0..2", "2..4", "4..8", "8..12", ">12"]
R = {s: load(s) for s in ("s0", "s1")}


def sel(s, harness, kind):
    return [r for r in R[s] if r["harness"] == harness and r["kind"] == kind]


def primary(r, add=None):
    """the pre-registered primary margin for this decision kind (positive = the gold/trained-correct choice)."""
    if r["kind"] == "plan_start":
        return m_gold(r, add)[0]
    if r["kind"] in ("after_plan", "continue") and r["harness"] != "v1":
        return m_eot(r, add)[0]
    if r["kind"] == "stop" and r["harness"] == "v4_format":
        return -m_eot(r, add)[0]
    return m_gold(r, add)[0]  # v1 continue / stop: gold merged token vs best rival


def hist(xs):
    c = [0] * len(BLAB)
    for x in xs:
        for i in range(len(BLAB)):
            if BINS[i] < x <= BINS[i + 1]:
                c[i] += 1
    return c


def summ(xs):
    xs = list(xs)
    a = np.array(xs)
    return {"n": len(xs), "median": float(np.median(a)), "q10": q(xs, .1), "q25": q(xs, .25), "q75": q(xs, .75),
            "q90": q(xs, .9), "frac_neg": float((a < 0).mean()), "frac_abs_lt4": float((abs(a) < 4).mean()),
            "frac_unresolved_lt0.25": float((abs(a) < 0.25).mean())}


def pr(*a):
    print(*a, flush=True)


KINDS = [("v11_plan", "plan_start", "target '<' - best rival"), ("v11_plan", "after_plan", "best non-eos - eos"),
         ("v4_format", "continue", "best non-eos - eos"), ("v4_format", "stop", "eos - best non-eos"),
         ("v1", "continue", "gold - best rival"), ("v1", "stop", "gold - best rival")]

# ---------------------------------------------------------------------------------------------------------------- 1
pr("=" * 100, "\n1. Margin histograms (train items, teacher-forced; primary margin, logits)\n")
pr("kind | seed | " + " | ".join(BLAB) + " | median | q10 / q90 | %neg | %|m|<4")
S = {}
for h, k, lab in KINDS:
    for s in ("s0", "s1"):
        xs = [primary(r) for r in sel(s, h, k)]
        S[(s, h, k)] = summ(xs)
        d = S[(s, h, k)]
        pr(f"{h}/{k} [{lab}] | {s} | " + " | ".join(str(c) for c in hist(xs)) +
           f" | {d['median']:+.2f} | {d['q10']:+.2f} / {d['q90']:+.2f} | {100*d['frac_neg']:.1f} | {100*d['frac_abs_lt4']:.1f}")

pr("\nSecond column: gold-minus-best-rival at after_plan / v4 continue (gold = '\\n'), and m_eot at v1 points")
for h, k in (("v11_plan", "after_plan"), ("v4_format", "continue"), ("v1", "continue"), ("v1", "stop")):
    for s in ("s0", "s1"):
        rs = sel(s, h, k)
        if h == "v1":
            xs = [(-1 if k == "stop" else 1) * m_eot(r)[0] for r in rs]
            lab = "eos margin (continue: non-eos - eos; stop: eos - non-eos)"
        else:
            xs = [m_gold(r)[0] for r in rs]
            lab = "gold '\\n' - best rival"
        d = summ(xs)
        pr(f"  {h}/{k} {s} [{lab}]: median {d['median']:+.2f}, q10 {d['q10']:+.2f}, %neg {100*d['frac_neg']:.1f}")

pr("\nTop rival identity")
for h, k, _ in KINDS:
    for s in ("s0", "s1"):
        rs = sel(s, h, k)
        if k == "plan_start" or h == "v1":
            c = Counter(m_gold(r)[1] for r in rs if m_gold(r)[0] < 0)
            c_all = Counter(m_gold(r)[1] for r in rs)
            pr(f"  {h}/{k} {s}: rival of gold (all) {[(TOKNAME.get(t, t), n) for t, n in c_all.most_common(4)]}; when m<0 {[(TOKNAME.get(t, t), n) for t, n in c.most_common(3)]}")
        else:
            c = Counter(m_eot(r)[1] for r in rs)
            g = Counter(m_gold(r)[1] for r in rs if m_gold(r)[0] < 0)
            pr(f"  {h}/{k} {s}: best non-eos {[(TOKNAME.get(t, t), n) for t, n in c.most_common(4)]}; rival of gold when gold loses {[(TOKNAME.get(t, t), n) for t, n in g.most_common(3)]}")

# ---------------------------------------------------------------------------------------------------------------- 2
pr("=" * 100, "\n2. Pipeline validation: plan-start margin vs vLLM train first token (B2tr / B2ftr, per item)\n")


def items(path):
    return {(r["id"], r["harness"]): r for r in map(json.loads, jopen(path))}


for s in ("s0", "s1"):
    ps = {r["id"]: r for r in sel(s, "v11_plan", "plan_start")}
    for c in (0, 2, 4, 8):
        for fam in ("B2tr", "B2ftr"):
            if c == 0 and fam == "B2ftr":
                continue
            it = items(f"{DW}/stage3/{s}/items_{fam}_c{c}_b-4.jsonl.gz")
            agree = dis = unres = 0
            bad = []
            for i, r in ps.items():
                m = m_gold(r)[0] + c
                opened = it[(i, "v11_plan")]["text"].startswith("<")
                if abs(m) < 0.25:
                    unres += 1
                    continue
                if (m > 0) == opened:
                    agree += 1
                else:
                    dis += 1
                    bad.append(round(m, 2))
            pr(f"  {s} {fam} c={c}: agree {agree}, disagree {dis} (margins {sorted(bad)[:8]}), unresolved |m|<0.25: {unres}; "
               f"predicted open {sum(m_gold(r)[0]+c > 0 for r in ps.values())}/595, observed {sum(it[(i,'v11_plan')]['text'].startswith('<') for i in ps)}/595")

# ---------------------------------------------------------------------------------------------------------------- 3
pr("=" * 100, "\n3. Seed shift (same 595 train items; seed 1 minus seed 0, primary margin)\n")
SH = {}
for h, k, _ in KINDS:
    a = {(r["id"], r["k"]): primary(r) for r in sel("s0", h, k)}
    b = {(r["id"], r["k"]): primary(r) for r in sel("s1", h, k)}
    keys = sorted(set(a) & set(b))
    x, y = np.array([a[i] for i in keys]), np.array([b[i] for i in keys])
    d = y - x
    SH[(h, k)] = (d.mean(), d.std(), np.median(d))
    pr(f"  {h}/{k}: n {len(keys)}, mean diff {d.mean():+.2f}, SD diff {d.std():.2f}, median diff {np.median(d):+.2f}, "
       f"Pearson r {np.corrcoef(x, y)[0,1]:.2f}, Spearman {np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0,1]:.2f}")
ps_mean = abs(SH[("v11_plan", "plan_start")][0])
den = max(max(abs(SH[k][0]), SH[k][1]) for k in (("v1", "continue"), ("v1", "stop")))
pr(f"  R = |mean diff plan_start| / max(|mean|, SD) over v1 continue/stop = {ps_mean:.2f} / {den:.2f} = {ps_mean/den:.2f}")
den4 = max(abs(SH[("v4_format", "stop")][0]), SH[("v4_format", "stop")][1])
pr(f"  (with v4 stop as the yardstick instead: {ps_mean:.2f} / {den4:.2f} = {ps_mean/den4:.2f})")
pr("  plan_start pairwise '<' - '{\"' (not best-rival), seed diff:")
a = {r["id"]: pair(r, LT, LBR) for r in sel("s0", "v11_plan", "plan_start")}
b = {r["id"]: pair(r, LT, LBR) for r in sel("s1", "v11_plan", "plan_start")}
d = np.array([b[i] - a[i] for i in a])
pr(f"    mean {d.mean():+.2f}, SD {d.std():.2f}; per-logit: mean z('<') diff {np.mean([sel('s1','v11_plan','plan_start')[j]['z']['27']-sel('s0','v11_plan','plan_start')[j]['z']['27'] for j in range(595)]):+.2f}, "
   f"mean z('{{\"') diff {np.mean([sel('s1','v11_plan','plan_start')[j]['z']['4754']-sel('s0','v11_plan','plan_start')[j]['z']['4754'] for j in range(595)]):+.2f}")

# ---------------------------------------------------------------------------------------------------------------- 4
pr("=" * 100, "\n4. Constant size implied by the margins vs what worked\n")
for s in ("s0", "s1"):
    ps = [primary(r) for r in sel(s, "v11_plan", "plan_start")]
    ap = [primary(r) for r in sel(s, "v11_plan", "after_plan")]
    vc = [primary(r) for r in sel(s, "v4_format", "continue")]
    vs = [primary(r) for r in sel(s, "v4_format", "stop")]
    pr(f"  {s} plan_start: c to open 50/80/90/95/99%: " + " / ".join(f"{max(0,-q(ps, 1-p)):.2f}" for p in (.5, .8, .9, .95, .99)))
    for c in (2, 4, 6, 8, 10, 12, 16):
        pr(f"     c={c:>2}: open {100*np.mean(np.array(ps)+c>0):5.1f}%")
    pr(f"  {s} after_plan: |b| to continue 50/80/90/95/99%: " + " / ".join(f"{max(0,-q(ap, 1-p)):.2f}" for p in (.5, .8, .9, .95, .99)))
    pr(f"  {s} v4 continue: |b| to continue 50/80/90/95/99%: " + " / ".join(f"{max(0,-q(vc, 1-p)):.2f}" for p in (.5, .8, .9, .95, .99)))
    pr(f"  {s} v4 stop: |b| at which 1/2/5/10/25% of stop points cross: " + " / ".join(f"{q(vs, p):.2f}" for p in (.01, .02, .05, .1, .25)))
    for b in (2, 4, 6, 8, 10, 12, 16):
        pr(f"     b=-{b:>2}: v4 continue cleared {100*np.mean(np.array(vc)+b>0):5.1f}%, after_plan cleared {100*np.mean(np.array(ap)+b>0):5.1f}%, "
           f"v4 stop crossed {100*np.mean(np.array(vs)-b<0):5.1f}%")

# ---------------------------------------------------------------------------------------------------------------- 5
pr("=" * 100, "\n5. Teacher-forced margins vs free-running train outcomes (per item)\n")


def v4_item_preds(s):
    cont = defaultdict(list)
    stop = {}
    for r in sel(s, "v4_format", "continue"):
        cont[r["id"]].append(primary(r))
    for r in sel(s, "v4_format", "stop"):
        stop[r["id"]] = primary(r)
    return cont, stop


def conf(pred, obs):
    tp = sum(p and o for p, o in zip(pred, obs)); fp = sum(p and not o for p, o in zip(pred, obs))
    fn = sum(o and not p for p, o in zip(pred, obs)); tn = sum((not p) and (not o) for p, o in zip(pred, obs))
    return tp, fp, fn, tn


for s in ("s0", "s1"):
    cont, stop = v4_item_preds(s)
    if s == "s0":
        runs = [(b, f"{DW}/stage3/s0/items_B2tr_c0_b-{b}.jsonl.gz") for b in (4, 8, 12)]
    else:
        runs = [(0, f"{DW}/stage2_s1/items_Btr0.jsonl.gz")] + [(b, f"{DW}/stage2_s1/items_Btr-{b}.jsonl.gz") for b in (2, 4, 6, 8, 10, 12)]
    for b, p in runs:
        it = items(p)
        multi = sorted(cont)
        pf = [min(cont[i]) + b < 0 for i in multi]
        of = [it[(i, "v4_format")]["n_pred"] < it[(i, "v4_format")]["n_gold"] for i in multi]
        tp, fp, fn, tn = conf(pf, of)
        allid = sorted(stop)
        po = [stop[i] - b < 0 for i in allid]
        oo = [it[(i, "v4_format")]["n_pred"] > it[(i, "v4_format")]["n_gold"] for i in allid]
        tp2, fp2, fn2, tn2 = conf(po, oo)
        pr(f"  {s} b=-{b:>2}: too few (152 multi): pred/obs {sum(pf)}/{sum(of)}, agree {tp+tn}/152 = {100*(tp+tn)/152:.1f}% "
           f"(tp {tp} fp {fp} fn {fn}); too many (595): pred/obs {sum(po)}/{sum(oo)}, agree {100*(tp2+tn2)/595:.1f}% (tp {tp2} fp {fp2} fn {fn2})")

pr("\n  after_plan, teacher-forced on gold plans vs free-running 'plan then stop' (items that opened '<plan>'):")
for s in ("s0", "s1"):
    ap = {r["id"]: primary(r) for r in sel(s, "v11_plan", "after_plan")}
    for c, b in ((0, 4), (8, 4), (8, 8), (8, 12)):
        it = items(f"{DW}/stage3/{s}/items_B2tr_c{c}_b-{b}.jsonl.gz")
        ids = [i for i in ap if it[(i, "v11_plan")]["text"].lstrip().startswith("<plan>") and "</plan>" in it[(i, "v11_plan")]["text"]]
        pred = [ap[i] + b < 0 for i in ids]
        obs = [it[(i, "v11_plan")]["text"].strip().endswith("</plan>") for i in ids]
        tp, fp, fn, tn = conf(pred, obs)
        pr(f"   {s} c={c} b=-{b}: n {len(ids)} closed plans; pred stop {sum(pred)}, obs stop {sum(obs)}, agree {100*(tp+tn)/max(1,len(ids)):.1f}% (tp {tp} fp {fp} fn {fn})")

# ---------------------------------------------------------------------------------------------------------------- 6
pr("=" * 100, "\n6. Margins under the stage-2/3 edits (output side only; A1/A2 in-sample, D1/D2 = their release, C2 random)\n")
for s in ("s0", "s1"):
    for h, k in (("v11_plan", "plan_start"), ("v11_plan", "after_plan"), ("v4_format", "continue"), ("v4_format", "stop"),
                 ("v1", "continue"), ("v1", "stop")):
        rs = sel(s, h, k)
        line = []
        for name, eds in (("none", []), ("A1", ["A1_eos"]), ("D1", ["D1_eos"]), ("A2", ["A2_27", "A2_eos"]), ("D2", ["D2_27", "D2_eos"]), ("C2", ["C2_27", "C2_eos"])):
            xs = np.array([primary(r, edit_add(r, eds)) for r in rs])
            line.append(f"{name} {100*np.mean(xs > 0):5.1f}%")
        pr(f"  {s} {h}/{k}: % correct side: " + ", ".join(line))
    for nm in ("A2_27", "D2_27", "C2_27"):
        v = [r["shift"][nm] for r in sel(s, "v11_plan", "plan_start")]
        pr(f"  {s} plan_start h.delta {nm}: mean {np.mean(v):+.2f}, SD {np.std(v):.2f}, min {min(v):+.2f}, max {max(v):+.2f}")
    for nm in ("A1_eos", "D1_eos", "A2_eos", "D2_eos"):
        for h, k in (("v4_format", "continue"), ("v4_format", "stop"), ("v11_plan", "after_plan")):
            v = [r["shift"][nm] for r in sel(s, h, k)]
            pr(f"  {s} {h}/{k} h.delta {nm}: mean {np.mean(v):+.2f}, SD {np.std(v):.2f}, min {min(v):+.2f}, max {max(v):+.2f}")
    pr(f"  {s} near-tie test at plan_start under D2: items moved to '<' (m<0 -> m'>0): margins before (quantiles) ")
    ps = sel(s, "v11_plan", "plan_start")
    moved = [m_gold(r)[0] for r in ps if m_gold(r)[0] < 0 and m_gold(r, edit_add(r, ["D2_27", "D2_eos"]))[0] > 0]
    stay = [m_gold(r)[0] for r in ps if m_gold(r)[0] < 0 and m_gold(r, edit_add(r, ["D2_27", "D2_eos"]))[0] <= 0]
    pr(f"     moved n={len(moved)} median m {np.median(moved) if moved else float('nan'):+.2f} min {min(moved) if moved else float('nan'):+.2f}; "
       f"not moved n={len(stay)} median m {np.median(stay) if stay else float('nan'):+.2f} max {max(stay) if stay else float('nan'):+.2f}")

# ---------------------------------------------------------------------------------------------------------------- 7
pr("=" * 100, "\n7. prevent_01 learned output rows (arm b, 6 rows) applied to (a)/(a1) states (proxy for 'out' vs 'none')\n")
PN = ["P_248046", "P_198", "P_27", "P_4754", "P_13766", "P_10083"]
for s in ("s0", "s1"):
    for h, k in (("v11_plan", "plan_start"), ("v11_plan", "after_plan"), ("v4_format", "continue"), ("v4_format", "stop"),
                 ("v1", "continue"), ("v1", "stop")):
        rs = sel(s, h, k)
        x0 = np.array([primary(r) for r in rs]); x1 = np.array([primary(r, edit_add(r, PN)) for r in rs])
        d = x1 - x0
        pr(f"  {s} {h}/{k}: mean dm {d.mean():+.2f} (SD {d.std():.2f}); % correct side {100*np.mean(x0>0):.1f} -> {100*np.mean(x1>0):.1f}; "
           f"flips +->- {int(((x0>0)&(x1<=0)).sum())}, -->+ {int(((x0<=0)&(x1>0)).sum())}")
    ps = sel(s, "v11_plan", "plan_start")
    pr(f"  {s} plan_start per-row h.delta means: " + ", ".join(f"{n} {np.mean([r['shift'][n] for r in ps]):+.2f}" for n in PN) +
       f"; pairwise '<'-'{{\"' shift {np.mean([r['shift']['P_27']-r['shift']['P_4754'] for r in ps]):+.2f}")
    vc = sel(s, "v4_format", "continue")
    pr(f"  {s} v4 continue per-row h.delta means: " + ", ".join(f"{n} {np.mean([r['shift'][n] for r in vc]):+.2f}" for n in PN))

pr("\n  prevent_01 test outcomes for comparison (204 test items; v11 replies opening '<plan>'; v4 too-few on multi-call):")
for nm in ("prev_b_rows1x_split_none", "prev_b_rows1x_split_out", "prev_b_rows1x", "prev_b_rows1x_s1_split_none", "prev_b_rows1x_s1_split_out", "prev_b_rows1x_s1"):
    p = f"{PV}/items_{nm}.jsonl.gz"
    if not os.path.exists(p):
        continue
    it = list(map(json.loads, jopen(p)))
    v11 = [r for r in it if r["harness"] == "v11_plan"]
    v4 = [r for r in it if r["harness"] == "v4_format" and r["n_gold"] > 1]
    pr(f"   {nm}: v11 open {sum(r['text'].lstrip().startswith('<plan>') for r in v11)}/204, v11 plan-then-stop {sum(r['text'].lstrip().startswith('<plan>') and r['text'].strip().endswith('</plan>') for r in v11)}, "
       f"v4 multi too-few {sum(r['n_pred'] < r['n_gold'] for r in v4)}/{len(v4)}")

# ---------------------------------------------------------------------------------------------------------------- 8
pr("=" * 100, "\n8. Decomposing 24.0 vs 0.5 (test, unpatched) into the two plan decisions\n")
for s, f in (("s0", f"{DW}/stage3/s0/items_Uchk.jsonl.gz"), ("s1", f"{DW}/stage3/s1/items_Uchk.jsonl.gz")):
    v = [r for r in map(json.loads, jopen(f)) if r["harness"] == "v11_plan"]
    op = [r for r in v if r["text"].lstrip().startswith("<plan>")]
    stp = [r for r in op if r["text"].strip().endswith("</plan>")]
    pr(f"  {s} test U: open {len(op)}/204, of which plan-then-stop {len(stp)}, ok {sum(r['ok'] for r in v)}")
    ps = sel(s, "v11_plan", "plan_start"); ap = sel(s, "v11_plan", "after_plan")
    pr(f"     train margins: P(plan_start m>0) {np.mean([primary(r)>0 for r in ps]):.3f}; P(after_plan m_eot>0) {np.mean([primary(r)>0 for r in ap]):.3f}; "
       f"P(both) {np.mean([primary(a)>0 and primary(b)>0 for a,b in zip(ps,ap)]):.3f}")

# ---------------------------------------------------------------------------------------------------------------- 9
pr("=" * 100, "\n9. v4 continue by position; post-hoc after-plan leak of the one-row XML patch (test, aggregate)\n")
for s in ("s0", "s1"):
    for lab, f in (("k=0", lambda r: r["k"] == 0), ("k>=1", lambda r: r["k"] >= 1)):
        v = np.array([primary(r) for r in sel(s, "v4_format", "continue") if f(r)])
        pr(f"  {s} v4 continue {lab}: n {len(v)}, median {np.median(v):+.2f}, min {v.min():+.2f}, max {v.max():+.2f}, "
           f"%<0 {100*(v<0).mean():.0f}, %>4 {100*(v>4).mean():.0f}")
for nm, p in (("s0 U", f"{DW}/stage3/s0/items_Uchk.jsonl.gz"), ("s0 A1", f"{DW}/stage2/items_A.jsonl.gz"), ("s0 D1", f"{DW}/stage2/items_D.jsonl.gz"),
              ("s1 U", f"{DW}/stage3/s1/items_Uchk.jsonl.gz"), ("s1 A1", f"{DW}/stage2_s1/items_A.jsonl.gz"), ("s1 D1", f"{DW}/stage2_s1/items_D.jsonl.gz")):
    v = [r for r in map(json.loads, jopen(p)) if r["harness"] == "v11_plan"]
    op = [r for r in v if r["text"].lstrip().startswith("<plan>")]
    pr(f"  {nm}: test opened {len(op)}, plan-then-stop {sum(r['text'].strip().endswith('</plan>') for r in op)}")
