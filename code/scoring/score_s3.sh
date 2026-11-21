#!/bin/bash
# Stage-3 scoring with DAWE's per-item scorer. usage: score_s3.sh <seed s0|s1> <train|test> <preds dir>
# preds_S3.jsonl is split per pass and (c, b): B2tr_c4_b-8, B2ftr_..., B2te_..., B2fte_..., A2, D2, C2, A1chk, Uchk, B2fchk_c0_b-8
set -e
# Needs a clone of bandr-ai/dawe at commit 8f11c96 with its BFCL files fetched (`uv sync && uv run dawe fetch`);
# point DAWE_REPO at it. score_items.py (next to this script) imports their scorer.
: "${DAWE_REPO:?set DAWE_REPO to a bandr-ai/dawe checkout (commit 8f11c96)}"
HERE=$(cd "$(dirname "$0")" && pwd)
P=$(cd "${3:?usage: score_s3.sh <s0|s1> <train|test> <dir with preds_S3.jsonl>}" && pwd)
cd $P
python3 - <<'PY'
import json
from collections import defaultdict
out = defaultdict(list)
for l in open("preds_S3.jsonl"):
    r = json.loads(l)
    n = r["pass"] if r.get("c") is None else f"{r['pass']}_c{int(r['c'])}_b{int(r['bias'])}"
    out[n].append(l)
for n, ls in out.items():
    open(f"p_{n}.jsonl", "w").writelines(ls)
print(len(out), "files")
PY
cd "$DAWE_REPO"
export DAWE_TELEMETRY_DISABLED=1 DO_NOT_TRACK=1
if [ "$2" = "train" ]; then pat='^p_B2f?tr_'; else pat='^p_(A1chk|A2|D2|C2|Uchk|B2te_|B2fte_|B2fchk_)'; fi
for f in $(cd $P; ls p_*.jsonl | grep -E "$pat"); do
  n=${f#p_}; n=${n%.jsonl}
  uv run python "$HERE/score_items.py" $P/$f $P/items_$n.jsonl > $P/score_$n.txt
done
echo scored $(cd $P; ls items_*.jsonl | grep -cE "${pat#^p_}")
