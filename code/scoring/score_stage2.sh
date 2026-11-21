#!/bin/bash
# Score stage-2 preds with DAWE's own per-item scorer. B is split per pass/bias first (B0 = unbiased pass, B-2 ... B-16).
set -e
# Needs a clone of bandr-ai/dawe at commit 8f11c96 with its BFCL files fetched (`uv sync && uv run dawe fetch`);
# point DAWE_REPO at it. score_items.py (next to this script) imports their scorer.
: "${DAWE_REPO:?set DAWE_REPO to a bandr-ai/dawe checkout (commit 8f11c96)}"
HERE=$(cd "$(dirname "$0")" && pwd)
P=$(cd "${1:?usage: score_stage2.sh <dir with preds_*.jsonl>}" && pwd)
cd $P
if [ -f preds_B.jsonl ]; then
python3 - <<'PY'
import json
from collections import defaultdict
out = defaultdict(list)
for l in open("preds_B.jsonl"):
    r = json.loads(l)
    name = "B0" if r["pass"] == "B0" else f"B{int(r['bias'])}"
    out[name].append(l)
for n, ls in out.items():
    open(f"preds_{n}.jsonl", "w").writelines(ls)
    print(n, len(ls))
PY
fi
cd "$DAWE_REPO"
export DAWE_TELEMETRY_DISABLED=1 DO_NOT_TRACK=1
for f in $P/preds_*.jsonl; do
  n=$(basename $f .jsonl); n=${n#preds_}
  [ "$n" = "B" ] && continue
  uv run python "$HERE/score_items.py" $f $P/items_$n.jsonl > $P/score_$n.txt
  echo "== $n"; grep " all " $P/score_$n.txt
done
