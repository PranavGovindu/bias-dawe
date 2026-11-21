"""Read their released .gpatch files: check the 'old' rows equal the base model's rows (LoRA never touches embed_tokens),
and save their deltas (new - old, bf16 values) for later comparison / transplant onto our checkpoint. CPU only.
usage: DAWE_REPO=<bandr-ai/dawe checkout @ 8f11c96> python their_patch_deltas.py   -> ../runs/their_deltas.pt"""
import json, os, sys
from pathlib import Path

REPO = Path(os.environ["DAWE_REPO"])
OUT = Path(__file__).resolve().parent.parent / "runs" / "their_deltas.pt"
OUT.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
import torch
from safetensors import safe_open
from dawe.patch import load_patch

if "QWEN_SNAPSHOT" in os.environ:  # local snapshot dir of Qwen/Qwen3.5-4B @ 851bf6e8
    SNAP = Path(os.environ["QWEN_SNAPSHOT"])
else:
    from huggingface_hub import snapshot_download
    SNAP = Path(snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
idx = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
KEY = "model.language_model.embed_tokens.weight"
with safe_open(str(SNAP / idx[KEY]), "pt") as f:
    E = f.get_slice(KEY)
    def row(r):
        return E[r:r + 1][0]

out = {}
for name in ("s1_eos12", "s2_plan8_eos12"):
    p = load_patch(REPO / "release/patches" / f"{name}.gpatch")
    for e in p.edits:
        assert e.dtype == "BF16" and e.name == KEY, (e.name, e.dtype)
        ii = torch.tensor(e.indices, dtype=torch.int64)
        old = torch.frombuffer(bytearray(e.old), dtype=torch.bfloat16)
        new = torch.frombuffer(bytearray(e.new), dtype=torch.bfloat16)
        rows = sorted(set((ii // e.shape[1]).tolist()))
        for r in rows:
            m = (ii // e.shape[1]) == r
            cols = ii[m] % e.shape[1]
            base = row(r)
            old_match = torch.equal(base[cols], old[m])
            newrow = base.clone()
            newrow[cols] = new[m]
            d = newrow.float() - base.float()
            out[f"{name}/row{r}"] = d
            print(f"{name} row {r}: changed {int(m.sum())}/{e.shape[1]}, old==base {old_match}, "
                  f"|delta| {d.norm():.6f}, |base row| {base.float().norm():.4f}")
torch.save(out, OUT)
a, b = out["s1_eos12/row248046"], out["s2_plan8_eos12/row248046"]
print("cos(s1 eos delta, s2 eos delta) =", float(a @ b / (a.norm() * b.norm())))
