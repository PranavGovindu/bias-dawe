"""Per-item scoring with DAWE's own scorer. Usage: python score_items.py preds.jsonl out.jsonl
Run inside a bandr-ai/dawe checkout (commit 8f11c96, BFCL files fetched), e.g. `uv run python <this file> ...`."""
import json, sys
from collections import defaultdict
from dawe import bfcl
from dawe.harness import Harness, parse_calls, render, target, to_bfcl

items = {it.id: it for it in bfcl.load_all()}
cache, out, tally = {}, [], defaultdict(lambda: [0, 0, 0])
for line in open(sys.argv[1]):
    p = json.loads(line)
    it = items.get(p["id"])
    if it is None:
        continue
    h = cache.setdefault(p["harness"], Harness.load(p["harness"]))
    _, names = render(h, it)
    calls = parse_calls(h, p["text"])
    ok = calls is not None and bfcl.check(to_bfcl(calls, names), it)
    gold = parse_calls(h, target(h, it))
    r = {"id": p["id"], "harness": h.name, "cat": it.category, "ok": bool(ok), "fmt_fail": calls is None,
         "n_pred": None if calls is None else len(calls), "n_gold": None if gold is None else len(gold), "text": p["text"]}
    out.append(r)
    t = tally[(h.name, "all")]; t[0] += ok; t[1] += 1; t[2] += calls is None
    t = tally[(h.name, it.category)]; t[0] += ok; t[1] += 1; t[2] += calls is None
with open(sys.argv[2], "w") as f:
    f.write("".join(json.dumps(r) + "\n" for r in out))
for (h, k), (ok, n, ff) in sorted(tally.items()):
    print(f"{h:10s} {k:10s} acc {100*ok/n:5.1f}  n {n:3d}  fmt_fail {100*ff/n:5.1f}")
