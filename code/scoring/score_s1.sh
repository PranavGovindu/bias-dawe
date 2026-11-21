#!/bin/bash
# Seed-1 scoring with DAWE's per-item scorer. usage: score_s1.sh <train|test> <preds dir>
# B is split per split/bias first: train -> Btr0, Btr-2 ... Btr-12 ; test -> B0, B-2 ... B-16. A and D are test.
set -e
# Needs a clone of bandr-ai/dawe at commit 8f11c96 with its BFCL files fetched (`uv sync && uv run dawe fetch`);
# point DAWE_REPO at it. score_items.py (next to this script) imports their scorer.
: "${DAWE_REPO:?set DAWE_REPO to a bandr-ai/dawe checkout (commit 8f11c96)}"
HERE=$(cd "$(dirname "$0")" && pwd)
P=$(cd "${2:?usage: score_s1.sh <train|test> <dir with preds_B.jsonl, preds_A.jsonl, ...>}" && pwd)
cd $P
python3 - <<'PY'
import json
from collections import defaultdict
out = defaultdict(list)
for l in open("preds_B.jsonl"):
    r = json.loads(l)
    tr = r.get("split") == "train"
    name = ("Btr" if tr else "B") + ("0" if r["bias"] is None else str(int(r["bias"])))
    out[name].append(l)
for n, ls in out.items():
    open(f"preds_{n}.jsonl", "w").writelines(ls)
    print(n, len(ls))
PY
cd "$DAWE_REPO"
export DAWE_TELEMETRY_DISABLED=1 DO_NOT_TRACK=1
if [ "$1" = "train" ]; then names=$(cd $P; ls preds_Btr*.jsonl | sed 's/preds_//; s/.jsonl//'); else names=$(cd $P; ls preds_*.jsonl | sed 's/preds_//; s/.jsonl//' | grep -v '^Btr' | grep -vx 'B'); fi
for n in $names; do
  uv run python "$HERE/score_items.py" $P/preds_$n.jsonl $P/items_$n.jsonl > $P/score_$n.txt
  echo "== $n"; grep " all " $P/score_$n.txt
done
