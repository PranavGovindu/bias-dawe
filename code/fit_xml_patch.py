"""Stage 2 fits (fp32, on CPU) on our saved states (weight_map over the 595 train pairs). XML (one-row) patch.
usage: fit_xml_patch.py <maps dir> <tag>   e.g. maps of sft_v1_s0 -> tag stage2 (seed 0), maps of dawe_s1 -> tag s1.
<maps dir> holds states_v1.pt / states_v4_format.pt (`modal volume get <runs volume> <run>/maps`). Needs THEIR_DELTAS
(default ../runs/their_deltas.pt, written by their_patch_deltas.py). Outputs go to ../runs/fits/.
(Originally fit2.py; with the default arguments it is identical to the stage-2 fit.)
A: their exact s1 eos12 fit (sessions/s1/edits/eos12.py): X = [v4 continue; v4 stop; v1 continue; v1 stop] final-layer
   states (fp16 on disk -> fp32), y = -12 on v4 continue else 0, lambda = 0.01*mean(diag(X^T X)), no intercept, on GPU.
C: same-norm random: randn(2560, Generator(cpu).manual_seed(0)) / norm * ||delta_A|| (their control's draw; fp32 add).
D: their released s1 delta (their_deltas.pt), predicted shift h.delta on OUR states.
Writes deltas_stage2.pt and fit_report.json."""
import json
import os
from pathlib import Path
import torch

DEV = "cpu"  # the PC GPU lock was busy; fp32 either way (theirs ran on CUDA)
HERE = Path(__file__).resolve().parent
D = HERE.parent / "runs" / "fits"  # outputs
D.mkdir(parents=True, exist_ok=True)
import sys as _sys
M = Path(_sys.argv[1] if len(_sys.argv) > 1 else HERE.parent / "runs" / "maps" / "sft_v1_s0")
TAG = _sys.argv[2] if len(_sys.argv) > 2 else "stage2"
THEIR = Path(os.environ.get("THEIR_DELTAS", HERE.parent / "runs" / "their_deltas.pt"))
if "QWEN_SNAPSHOT" in os.environ:  # local snapshot dir of Qwen/Qwen3.5-4B @ 851bf6e8
    SNAP = Path(os.environ["QWEN_SNAPSHOT"])
else:
    from huggingface_hub import snapshot_download
    SNAP = Path(snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
import sys; sys.path.insert(0, str(HERE)); import arms as A

st = {h: torch.load(M / f"states_{h}.pt") for h in ("v4_format", "v1")}
rep = {"counts": {h: {k: tuple(v.shape) for k, v in s.items()} for h, s in st.items()}}
xs, ys, groups = [], [], []
for h in ("v4_format", "v1"):
    for kind in ("continue", "stop"):
        x = st[h][kind][:, -1, :].float().to(DEV)
        xs.append(x)
        ys.append(torch.full((len(x),), -12.0 if h == "v4_format" and kind == "continue" else 0.0, device=x.device))
        groups.append(f"{h}/{kind}")
x, y = torch.cat(xs), torch.cat(ys)
gram = x.T @ x
ridge = 0.01 * gram.diag().mean()
delta = torch.linalg.solve(gram + ridge * torch.eye(x.shape[1], device=x.device), x.T @ y)
dA = delta.cpu()
rep["X_shape"] = list(x.shape)
rep["device"] = DEV
rep["lambda"] = ridge.item()
rep["norm_A"] = dA.norm().item()
rep["nonzero_A"] = int((dA != 0).sum())

wm = A.weight_files(SNAP)
base_row = A.read_row(SNAP / wm[A.EMB], A.EMB, A.EOS)
g = torch.Generator(device="cpu").manual_seed(0)
r = torch.randn(2560, generator=g)
dC = r / r.norm() * dA.norm()
their = torch.load(THEIR)
dD = their["s1_eos12/row248046"].float()
rows = {k: (base_row.float() + v).to(torch.bfloat16) for k, v in (("A", dA), ("C", dC), ("D", dD))}
assert torch.equal(rows["D"].float() - base_row.float(), dD)
eff = {k: (rows[k].float() - base_row.float()) for k in rows}  # what the bf16 cast actually adds
rep["norm_C"] = dC.norm().item()
rep["norm_D"] = dD.norm().item()
rep["cos_A_D"] = (dA @ dD / (dA.norm() * dD.norm())).item()
rep["cos_A_C"] = (dA @ dC / (dA.norm() * dC.norm())).item()
rep["row_bits_changed"] = {k: int((rows[k].view(torch.int16) != base_row.view(torch.int16)).sum()) for k in rows}
rep["eff_norm"] = {k: eff[k].norm().item() for k in eff}

def stats(v):
    return {"mean": round(v.mean().item(), 3), "std": round(v.std().item(), 3), "min": round(v.min().item(), 3),
            "max": round(v.max().item(), 3)}

rep["shift"] = {}
for name, vec in (("A_fp32", dA), ("A_bf16", eff["A"]), ("C_bf16", eff["C"]), ("D_bf16", eff["D"])):
    v = vec.to(DEV)
    rep["shift"][name] = {gname: stats(xx @ v) for gname, xx in zip(groups, xs)}
# unpatched <|im_end|> logit at the decision points (h . row), for scale
rep["eos_logit_unpatched"] = {gname: stats(xx @ base_row.float().to(DEV)) for gname, xx in zip(groups, xs)}
torch.save({"A": dA, "C": dC, "D": dD, "D_row": rows["D"], "A_row": rows["A"], "C_row": rows["C"],
            "base_row": base_row}, D / f"deltas_{TAG}.pt")
(D / f"fit_report_{TAG}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
