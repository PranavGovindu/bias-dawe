"""CPU: logits/margins for the 3060 reply-start states (rs_<tag>.pt). usage: rs_margins.py <tag>
Writes $MARGINS_WORK/rs_margins_<tag>.jsonl: one row per (set, position, item) with z at tokens of interest,
top-10 and h.delta for the stage-3 row edits (A2/D2/C2, bf16-effective), as in margins.py."""
import json, os, sys
from pathlib import Path
import torch
from safetensors import safe_open

torch.set_num_threads(4)
TAG = sys.argv[1]
# Paths (repo-relative; override WORK with MARGINS_WORK and the base model with QWEN_SNAPSHOT=<local snapshot dir>).
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
DELTAS = REPO / "results" / "deltas"
WORK = Path(os.environ.get("MARGINS_WORK", REPO / "runs" / "margins"))  # adapters/, maps/, rs_*.pt in; outputs out
OUT = WORK
if "QWEN_SNAPSHOT" in os.environ:
    SNAP = Path(os.environ["QWEN_SNAPSHOT"])
else:
    from huggingface_hub import snapshot_download
    SNAP = Path(snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
EMB = "model.language_model.embed_tokens.weight"
INTEREST = [27, 4754, 90, 5046, 248046, 198, 271, 10083, 6994, 248044, 29, 85806, 7082]
idx = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
with safe_open(str(SNAP / idx[EMB]), framework="pt") as f:
    E_bf = f.get_tensor(EMB)
E = E_bf.float()
b27, beos = E_bf[27].float(), E_bf[248046].float()
edits = {}
if (DELTAS / f"deltas_s3_{TAG}.pt").exists():  # seeds 0/1 only; step-2 runs have no fitted edits
    s3 = torch.load(DELTAS / f"deltas_s3_{TAG}.pt")
    edits = {"A2_27": s3["A2_27_row"].float() - b27, "A2_eos": s3["A2_eos_row"].float() - beos,
             "D2_27": s3["D2_27_row"].float() - b27, "D2_eos": s3["D2_eos_row"].float() - beos,
             "C2_27": s3["C2_27_row"].float() - b27, "C2_eos": s3["C2_eos_row"].float() - beos}
Dm = torch.stack(list(edits.values())) if edits else torch.zeros(0, E.shape[1])
SRC = sys.argv[2] if len(sys.argv) > 2 else f"rs_{TAG}.pt"
st = torch.load(OUT / SRC)
out = open(OUT / SRC.replace("rs_", "rs_margins_").replace(".pt", ".jsonl"), "w")
for key, X in st.items():
    if not isinstance(X, torch.Tensor):
        continue  # key = "<harness>/<split>/<pos>"
    h_, split, pos = key.split("/")
    ids = st["ids"][f"{h_}/{split}"]
    assert len(ids) == X.shape[0]
    for b0 in range(0, X.shape[0], 64):
        hb = X[b0:b0 + 64].float()
        z = hb @ E.T
        lse = torch.logsumexp(z, -1)
        top = z.topk(10, -1)
        sh = hb @ Dm.T
        for i in range(hb.shape[0]):
            rec = {"seed": TAG, "harness": h_, "split": split, "pos": pos, "id": ids[b0 + i], "lse": round(lse[i].item(), 4),
                   "z": {str(t): round(z[i, t].item(), 4) for t in INTEREST},
                   "top": [[int(t), round(v, 4)] for t, v in zip(top.indices[i].tolist(), top.values[i].tolist())],
                   "shift": {k: round(sh[i, j].item(), 4) for j, k in enumerate(edits)}}
            out.write(json.dumps(rec) + "\n")
    print(key, X.shape[0], flush=True)
out.close()
