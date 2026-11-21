"""Analysis of the 3060 reply-start pass (../../results/margins/rs_margins_s{0,1}*.jsonl.gz), matched per item to the
test outcomes in ../../results/. usage: python3 rs_analyze.py > rs_analysis.txt
(committed output: ../../results/margins/rs_analysis.txt)"""
import json
from collections import Counter
import numpy as np
from analyze_lib import RES, jopen, load as load_map, m_gold as map_m_gold, q

DW = str(RES)
J = {4754, 90, 5046}  # JSON openers: '{"', '{', ' {"'
LT, EOS = 27, 248046
NAME = {27: "'<'", 4754: "'{\"'", 90: "'{'", 5046: "' {\"'", 248046: "eos", 198: "'\\n'", 271: "'\\n\\n'", 6994: "'Call'", 10083: "'plan'"}
CHOSEN = {"s0": "B2fte_c8_b-4", "s1": "B2fte_c8_b-8"}


def pr(*a):
    print(*a, flush=True)


def rows(seed):
    return [json.loads(l) for l in jopen(f"{RES}/margins/rs_margins_{seed}.jsonl.gz")]


def zmap(r, add=None):
    z = {int(t): v for t, v in r["top"]}
    for t, v in r["z"].items():
        z.setdefault(int(t), v)
    if add:
        for t, dv in add.items():
            z[t] = z[t] + dv
    return z


def m_target(r, tgt, add=None):
    z = zmap(r, add)
    o = max((k for k in z if k != tgt), key=lambda k: z[k])
    return z[tgt] - z[o], o


def deficit(r, tgt, add=None):  # best other minus target (positive = target loses by this much)
    return -m_target(r, tgt, add)[0]


def m_post(r, add=None):  # best non-JSON-opener minus best JSON opener
    z = zmap(r, add)
    a = max((k for k in z if k not in J), key=lambda k: z[k])
    b = max((k for k in z if k in J), key=lambda k: z[k])
    return z[a] - z[b], a, b


def top1(r):
    return max(zmap(r).items(), key=lambda kv: kv[1])[0]


def items(path):
    return {(r["id"], r["harness"]): r for r in map(json.loads, jopen(path))}


def hist(xs, bins=(-1e9, -12, -8, -4, -2, 0, 2, 4, 8, 12, 16, 1e9)):
    lab = ["<-12", "-12..-8", "-8..-4", "-4..-2", "-2..0", "0..2", "2..4", "4..8", "8..12", "12..16", ">16"]
    c = np.histogram(xs, bins=bins)[0]
    return " ".join(f"{l}:{n}" for l, n in zip(lab, c))


def summ(xs):
    xs = np.array(xs)
    return f"n {len(xs)}, median {np.median(xs):+.2f}, q05 {q(list(xs), .05):+.2f}, q95 {q(list(xs), .95):+.2f}, min {xs.min():+.2f}, %<0 {100*(xs<0).mean():.1f}, %|m|<4 {100*(abs(xs)<4).mean():.1f}"


R = {s: rows(s) for s in ("s0", "s1")}
sel = lambda s, h, sp, pos: [r for r in R[s] if r["harness"] == h and r["split"] == sp and r["pos"] == pos]
A2 = lambda r: {LT: r["shift"]["A2_27"], EOS: r["shift"]["A2_eos"]}
D2 = lambda r: {LT: r["shift"]["D2_27"], EOS: r["shift"]["D2_eos"]}

pr("=" * 100, "\nT0. Merge validation: PC plan-start margins (train) vs H100 map margins (step 1), same items\n")
for s in ("s0", "s1"):
    mp = {r["id"]: map_m_gold(r)[0] for r in load_map(s) if r["kind"] == "plan_start"}
    pc = {r["id"]: m_target(r, LT)[0] for r in sel(s, "v11", "train", "start")}
    d = np.array([pc[i] - mp[i] for i in mp])
    pr(f"  {s}: mean dm {d.mean():+.3f}, mean |dm| {abs(d).mean():.3f}, %|dm|<=0.3 {100*(abs(d)<=0.3).mean():.1f}, max |dm| {abs(d).max():.2f}, "
       f"r {np.corrcoef([pc[i] for i in mp], [mp[i] for i in mp])[0,1]:.4f}")

pr("=" * 100, "\nT1/T2. v1 (JSON) reply start: target '{\"' vs best rival, and the '<' deficit (best non-'<' minus '<')\n")
c95 = {"s0": 4.05, "s1": 11.14}
for s in ("s0", "s1"):
    for sp in ("train", "test"):
        rs = sel(s, "v1", "train" if sp == "train" else "test", "start")
        mj = [m_target(r, 4754)[0] for r in rs]
        dl = [deficit(r, LT) for r in rs]
        pr(f"  {s} {sp}: top token {Counter(NAME.get(top1(r), top1(r)) for r in rs).most_common(3)}")
        pr(f"     '{{\"' - best rival: {summ(mj)}; rival {Counter(NAME.get(m_target(r,4754)[1], m_target(r,4754)[1]) for r in rs).most_common(3)}")
        pr(f"     '<' deficit: {summ(dl)}")
        pr(f"       hist: {hist(dl)}")
        pr(f"     K2: %% items with '<' deficit < c95 ({c95[s]}): {100*np.mean(np.array(dl) < c95[s]):.1f}%; < 8: {100*np.mean(np.array(dl) < 8):.1f}%; < 12: {100*np.mean(np.array(dl) < 12):.1f}%")
        for nm, f in (("A2", A2), ("D2", D2)):
            dd = np.array([deficit(r, LT, f(r)) for r in rs]); sh = np.array([r["shift"][f"{nm}_27"] for r in rs])
            pr(f"     under {nm} (output side): h.delta('<') mean {sh.mean():+.2f} (SD {sh.std():.2f}); deficit median {np.median(dd):+.2f}, min {dd.min():+.2f}, %<0 {100*(dd<0).mean():.1f}")

pr("\n  v1 test: deficit vs observed (no v1 test reply opens '<' at c = 8 in stage 3):")
for s in ("s0", "s1"):
    rs = sel(s, "v1", "test", "start")
    it = items(f"{DW}/stage3/{s}/items_{CHOSEN[s].replace('c8', 'c8')}.jsonl.gz")
    n_pred = sum(deficit(r, LT) < 8 for r in rs)
    n_obs = sum(it[(r["id"], "v1")]["text"].startswith("<") for r in rs)
    pr(f"   {s}: items with deficit < 8 (predicted to open '<' at +8): {n_pred}; observed opening '<' under {CHOSEN[s]}: {n_obs}")

pr("=" * 100, "\nT3. v11 plan start on TEST items ('<' - best rival)\n")
for s in ("s0", "s1"):
    rs = sel(s, "v11", "test", "start")
    m = {r["id"]: m_target(r, LT)[0] for r in rs}
    pr(f"  {s} test: {summ(list(m.values()))}")
    pr(f"       hist: {hist(list(m.values()))}")
    tr = [m_target(r, LT)[0] for r in sel(s, "v11", "train", "start")]
    pr(f"  {s} train (PC): {summ(tr)}")
    # validation vs U and constants (identity-like)
    for arm, c in (("Uchk", 0), ("B2fte_c2_b-4", 2), ("B2fte_c4_b-4", 4), (CHOSEN[s], 8)):
        it = items(f"{DW}/stage3/{s}/items_{arm}.jsonl.gz")
        res = [(m[i] + c, it[(i, "v11_plan")]["text"].startswith("<")) for i in m]
        ok = sum((x > 0) == o for x, o in res if abs(x) >= 0.25); unres = sum(abs(x) < 0.25 for x, _ in res)
        pr(f"     vs {arm} (c={c}): predicted open {sum(x > 0 for x, _ in res)}, observed {sum(o for _, o in res)}, sign agree {ok}/{len(res)-unres} (unresolved {unres})")
    # edits, output side, per test item
    for arm, f in (("A2", A2), ("D2", D2)):
        it = items(f"{DW}/stage3/{s}/items_{arm}.jsonl.gz")
        pred = {r["id"]: m_target(r, LT, f(r))[0] > 0 for r in rs}
        obs = {i: it[(i, "v11_plan")]["text"].startswith("<") for i in pred}
        tp = sum(pred[i] and obs[i] for i in pred); fp = sum(pred[i] and not obs[i] for i in pred); fn = sum(obs[i] and not pred[i] for i in pred)
        pr(f"     {arm} output-side prediction on test: predicted open {sum(pred.values())}, observed {sum(obs.values())}; observed-open but predicted-closed {fn}, predicted-open but closed {fp}")

pr("\n  Same test item, JSON vs plan-first reply start (contrast):")
for s in ("s0", "s1"):
    v1 = {r["id"]: deficit(r, LT) for r in sel(s, "v1", "test", "start")}
    v11 = {r["id"]: deficit(r, LT) for r in sel(s, "v11", "test", "start")}
    ids = sorted(v1)
    a, b = np.array([v1[i] for i in ids]), np.array([v11[i] for i in ids])
    rk = lambda x: np.argsort(np.argsort(x))
    pr(f"   {s}: '<' deficit median JSON {np.median(a):+.2f} vs plan-first {np.median(b):+.2f}; per-item gap (JSON - v11) median {np.median(a-b):+.2f}, min {(a-b).min():+.2f}; "
       f"min JSON deficit {a.min():+.2f} vs v11 q95 {q(list(b), .95):+.2f}; Spearman {np.corrcoef(rk(a), rk(b))[0,1]:.2f}")

pr("=" * 100, "\nT4. Token after '<plan>' (best non-JSON-opener minus best JSON opener; >0 = writes plan text)\n")
for s in ("s0", "s1"):
    for sp in ("train", "test"):
        rs = sel(s, "v11", sp, "post_plan")
        mp_ = [m_post(r)[0] for r in rs]
        pr(f"  {s} {sp}: {summ(mp_)}; top token {Counter(NAME.get(top1(r), top1(r)) for r in rs).most_common(4)}")
        pr(f"       hist: {hist(mp_)}")
a = {r["id"]: m_post(r)[0] for r in sel("s0", "v11", "test", "post_plan")}
b = {r["id"]: m_post(r)[0] for r in sel("s1", "v11", "test", "post_plan")}
d = np.array([b[i] - a[i] for i in a])
pr(f"  seed 1 - seed 0 per test item: mean {d.mean():+.2f}, SD {d.std():.2f}, r {np.corrcoef([a[i] for i in a],[b[i] for i in a])[0,1]:.2f}")
for s in ("s0", "s1"):
    ps = {r["id"]: m_target(r, LT)[0] for r in sel(s, "v11", "test", "start")}
    po = {r["id"]: m_post(r)[0] for r in sel(s, "v11", "test", "post_plan")}
    pr(f"  {s}: correlation plan-start margin vs post-<plan> margin (test, same item): r {np.corrcoef([ps[i] for i in ps],[po[i] for i in ps])[0,1]:.2f}")

pr("\n  Per-item match on test: among replies that begin '<plan>', does the next token go to JSON?")
for s in ("s0", "s1"):
    po = {r["id"]: r for r in sel(s, "v11", "test", "post_plan")}
    for arm, f in (("Uchk", None), (CHOSEN[s], None), (CHOSEN[s].replace("fte", "te"), None), ("A2", A2), ("D2", D2)):
        it = items(f"{DW}/stage3/{s}/items_{arm}.jsonl.gz")
        ids = [i for i in po if it[(i, "v11_plan")]["text"].startswith("<plan>")]
        obs = {i: it[(i, "v11_plan")]["text"][len("<plan>"):].startswith("{") for i in ids}
        pred = {i: m_post(po[i], f(po[i]) if f else None)[0] < 0 for i in ids}
        tp = sum(pred[i] and obs[i] for i in ids); fp = sum(pred[i] and not obs[i] for i in ids); fn = sum(obs[i] and not pred[i] for i in ids)
        unres = sum(abs(m_post(po[i], f(po[i]) if f else None)[0]) < 0.25 for i in ids)
        pr(f"   {s} {arm}: n {len(ids)} opened '<plan>'; observed JSON next {sum(obs.values())}, predicted {sum(pred.values())}; "
           f"agree {len(ids)-fp-fn}/{len(ids)} (fp {fp}, fn {fn}, |m|<0.25: {unres})")

pr("\n  Seed-1 residual failures under A2 / B2-first, by post-<plan> margin (test):")
for s in ("s0", "s1"):
    po = {r["id"]: m_post(r)[0] for r in sel(s, "v11", "test", "post_plan")}
    for arm in ("A2", CHOSEN[s]):
        it = items(f"{DW}/stage3/{s}/items_{arm}.jsonl.gz")
        unc = [i for i in po if it[(i, "v11_plan")]["text"].startswith("<plan>") and "</plan>" not in it[(i, "v11_plan")]["text"]]
        ok = [i for i in po if it[(i, "v11_plan")]["ok"]]
        pr(f"   {s} {arm}: unclosed-plan items n {len(unc)}, their m_post median {np.median([po[i] for i in unc]) if unc else float('nan'):+.2f}; "
           f"correct items n {len(ok)}, m_post median {np.median([po[i] for i in ok]):+.2f}")

# ---------------------------------------------------------------------------------------------------- T4b (Amendment 3)
import os
pr("=" * 100, "\nT4b. Position after '<' 'plan': z('>') minus max(z('>{\"'), z('>{')) (post hoc, Amendment 3)\n")
GT, GJ = 29, (85806, 7082)
NAME.update({29: "'>'", 85806: "'>{\"'", 7082: "'>{'"})
if os.path.exists(f"{RES}/margins/rs_margins_s0_v11.jsonl.gz"):
    RV = {s: [json.loads(l) for l in jopen(f"{RES}/margins/rs_margins_{s}_v11.jsonl.gz")] for s in ("s0", "s1")}
    selv = lambda s, sp, pos: [r for r in RV[s] if r["split"] == sp and r["pos"] == pos]

    def m_gt(r, add=None):  # pre-registered (Amendment 3): '>' vs JSON-merged
        z = zmap(r, add)
        return z[GT] - max(z[t] for t in GJ)

    def m_gt2(r, add=None):  # decision-faithful: best non-JSON-merged token (e.g. '>\n') vs JSON-merged
        z = zmap(r, add)
        return max(v for t, v in z.items() if t not in GJ) - max(z[t] for t in GJ)

    for s in ("s0", "s1"):
        # consistency: the re-run's plan-start states equal the first run's (same script, same hardware)
        a = {r["id"]: m_target(r, LT)[0] for r in sel(s, "v11", "test", "start")}
        b = {r["id"]: m_target(r, LT)[0] for r in selv(s, "test", "start")}
        pr(f"  {s}: re-run vs first 3060 run, plan-start test margins: max |diff| {max(abs(a[i]-b[i]) for i in a):.4f}")
        for sp in ("train", "test"):
            rs = selv(s, sp, "after_plantok")
            m = [m_gt(r) for r in rs]
            pr(f"  {s} {sp}: {summ(m)}; top token {Counter(NAME.get(top1(r), top1(r)) for r in rs).most_common(4)}")
            pr(f"       m_gt2 (best non-JSON vs JSON): {summ([m_gt2(r) for r in rs])}")
            pr(f"       hist: {hist(m)}")
    a = {r["id"]: m_gt(r) for r in selv("s0", "test", "after_plantok")}
    b = {r["id"]: m_gt(r) for r in selv("s1", "test", "after_plantok")}
    d = np.array([b[i] - a[i] for i in a])
    pr(f"  seed 1 - seed 0 per test item: mean {d.mean():+.2f}, SD {d.std():.2f}, r {np.corrcoef([a[i] for i in a],[b[i] for i in a])[0,1]:.2f}")
    for s in ("s0", "s1"):
        ps = {r["id"]: m_target(r, LT)[0] for r in selv(s, "test", "start")}
        pg = {r["id"]: m_gt(r) for r in selv(s, "test", "after_plantok")}
        pr(f"  {s}: r(plan-start margin, after-'plan' margin) on test items {np.corrcoef([ps[i] for i in ps],[pg[i] for i in ps])[0,1]:.2f}")
    pr("\n  Per-item match on test: among replies beginning '<plan', is '>{' emitted (text starts '<plan>{')?")
    for s in ("s0", "s1"):
        pg = {r["id"]: r for r in selv(s, "test", "after_plantok")}
        for arm, f in (("Uchk", None), (CHOSEN[s], None), (CHOSEN[s].replace("fte", "te"), None), ("A2", A2), ("D2", D2), ("C2", None)):
            it = items(f"{DW}/stage3/{s}/items_{arm}.jsonl.gz")
            ids = [i for i in pg if it[(i, "v11_plan")]["text"].startswith("<plan")]
            obs = {i: it[(i, "v11_plan")]["text"].startswith("<plan>{") for i in ids}
            mm = {i: m_gt2(pg[i], f(pg[i]) if f else None) for i in ids}
            pred = {i: mm[i] < 0 for i in ids}
            tp = sum(pred[i] and obs[i] for i in ids); fp = sum(pred[i] and not obs[i] for i in ids); fn = sum(obs[i] and not pred[i] for i in ids)
            unres = sum(abs(mm[i]) < 0.25 for i in ids)
            pr(f"   {s} {arm}: n {len(ids)} begin '<plan'; observed '>{{' {sum(obs.values())}, predicted {sum(pred.values())}; "
               f"tp {tp}, fp {fp}, fn {fn}; |m|<0.25 {unres}; median m of observed-JSON items {np.median([mm[i] for i in ids if obs[i]]) if any(obs.values()) else float('nan'):+.2f}")

# ---------------------------------------------------------------------------------------------------- T5 (after advisor review)
pr("=" * 100, "\nT5. Like-for-like seed control at JSON reply start; eos-only patches at plan start (input-side check)\n")
for sp in ("train", "test"):
    a = {r["id"]: r for r in sel("s0", "v1", sp, "start")}; b = {r["id"]: r for r in sel("s1", "v1", sp, "start")}
    ids = sorted(a)
    for nm, f in (("'<' deficit", lambda r: deficit(r, LT)), ("'{\"' margin", lambda r: m_target(r, 4754)[0])):
        x = np.array([f(a[i]) for i in ids]); y = np.array([f(b[i]) for i in ids]); d = y - x
        pr(f"  JSON reply start {sp} {nm}: s1-s0 mean {d.mean():+.2f}, SD {d.std():.2f}, r {np.corrcoef(x, y)[0,1]:.2f}")
a = {r["id"]: m_target(r, LT)[0] for r in sel("s0", "v11", "test", "start")}; b = {r["id"]: m_target(r, LT)[0] for r in sel("s1", "v11", "test", "start")}
d = np.array([b[i] - a[i] for i in a]); pr(f"  plan start test: s1-s0 mean {d.mean():+.2f}, SD {d.std():.2f}")
for s, u, arms in (("s0", "stage3/s0/items_Uchk.jsonl.gz", [("A1", "stage2/items_A.jsonl.gz"), ("D1", "stage2/items_D.jsonl.gz"), ("A2", "stage3/s0/items_A2.jsonl.gz")]),
                   ("s1", "stage3/s1/items_Uchk.jsonl.gz", [("A1", "stage2_s1/items_A.jsonl.gz"), ("D1", "stage2_s1/items_D.jsonl.gz"), ("A2", "stage3/s1/items_A2.jsonl.gz")])):
    m = {r["id"]: m_target(r, LT)[0] for r in sel(s, "v11", "test", "start")}
    U = items(f"{DW}/{u}")
    for nm, p in arms:
        it = items(f"{DW}/{p}")
        extra = [m[i] for i in m if it[(i, "v11_plan")]["text"].startswith("<") and not U[(i, "v11_plan")]["text"].startswith("<")]
        lost = [i for i in m if U[(i, "v11_plan")]["text"].startswith("<") and not it[(i, "v11_plan")]["text"].startswith("<")]
        pr(f"  {s} {nm}: opens beyond U {len(extra)}, unpatched margins median {np.median(extra):+.2f} [{min(extra):+.2f}, {max(extra):+.2f}]; U-opened lost {len(lost)}")
