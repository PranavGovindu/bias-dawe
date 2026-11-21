"""Stage 3 fits for one seed (fp32, CPU). Plan-first (two-row) patch. (Originally fit3.py.)
usage: fit_planfirst_patch.py <maps dir> <tag> <stage-2 deltas file>
  e.g. <maps of sft_v1_s0> s0 ../results/deltas/deltas_stage2.pt ; <maps of dawe_s1> s1 ../results/deltas/deltas_s1.pt
<maps dir> holds states_dp_plan_v11_plan.pt, states_v1.pt, states_v4_format.pt. Needs THEIR_DELTAS (default
../runs/their_deltas.pt, from their_patch_deltas.py). Outputs go to ../runs/fits/.
A2: their sessions/s2/edits/plan8_eos12.py exactly: X = final-layer states of [plan_start, after_plan] (plan map),
    [continue, stop] (v1), [continue, stop] (v4); Y[:,0] = +8 at plan_start; Y[:,1] = -12 at after_plan and v4 continue;
    lambda = 0.01*mean(diag(X^T X)); columns added to rows 27 ('<') and 248046 (<|im_end|>), fp32 add, bf16 cast.
D2: their released s2 delta (their_deltas.pt, from release/patches/s2_plan8_eos12.gpatch; old rows == base, checked).
C2: same-norm random, Generator(cpu).manual_seed(0): first randn(2560) -> row 27, second -> row 248046, each scaled to
    the norm of the matching A2 column.
A1: the stage-2 s1 delta for this seed (for the in-engine swap check)."""
import json, os, sys
from pathlib import Path
import torch

DEV = "cpu"
HERE = Path(__file__).resolve().parent
D = HERE.parent / "runs" / "fits"  # outputs
D.mkdir(parents=True, exist_ok=True)
M = Path(sys.argv[1])
TAG = sys.argv[2]
S2 = torch.load(sys.argv[3])
THEIR = Path(os.environ.get("THEIR_DELTAS", HERE.parent / "runs" / "their_deltas.pt"))
sys.path.insert(0, str(HERE)); import arms as A
if "QWEN_SNAPSHOT" in os.environ:  # local snapshot dir of Qwen/Qwen3.5-4B @ 851bf6e8
    SNAP = Path(os.environ["QWEN_SNAPSHOT"])
else:
    from huggingface_hub import snapshot_download
    SNAP = Path(snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
LT, EOS = 27, 248046

xs, ys, groups = [], [], []
for name in ["states_dp_plan_v11_plan.pt", "states_v1.pt", "states_v4_format.pt"]:
    st = torch.load(M / name)
    for kind, sk in st.items():
        x = sk[:, -1, :].float().to(DEV)
        a = 8.0 if name == "states_dp_plan_v11_plan.pt" and kind == "plan_start" else 0.0
        b = -12.0 if (name == "states_dp_plan_v11_plan.pt" and kind == "after_plan") or \
            (name == "states_v4_format.pt" and kind == "continue") else 0.0
        xs.append(x); ys.append(torch.tensor([a, b]).expand(len(x), 2)); groups.append(f"{name.replace('states_', '').replace('.pt', '')}/{kind}")
x, y = torch.cat(xs), torch.cat(ys)
gram = x.T @ x
ridge = 0.01 * gram.diag().mean()
delta = torch.linalg.solve(gram + ridge * torch.eye(x.shape[1]), x.T @ y)
rep = {"groups": {g: len(v) for g, v in zip(groups, xs)}, "X_shape": list(x.shape), "lambda": ridge.item(),
       "A2_col_norms": delta.norm(dim=0).tolist(), "device": DEV}
wm = A.weight_files(SNAP)
base = {t: A.read_row(SNAP / wm[A.EMB], A.EMB, t) for t in (LT, EOS)}
their = torch.load(THEIR)
d2 = {LT: their["s2_plan8_eos12/row27"].float(), EOS: their["s2_plan8_eos12/row248046"].float()}
g = torch.Generator(device="cpu").manual_seed(0)
r27, reos = torch.randn(2560, generator=g), torch.randn(2560, generator=g)
c2 = {LT: r27 / r27.norm() * delta[:, 0].norm(), EOS: reos / reos.norm() * delta[:, 1].norm()}
a2 = {LT: delta[:, 0].clone(), EOS: delta[:, 1].clone()}
out = {"base_row": base[EOS], "base_row27": base[LT], "A1": S2["A"].float()}
for name, dd in (("A2", a2), ("D2", d2), ("C2", c2)):
    for t, key in ((LT, "27"), (EOS, "eos")):
        out[f"{name}_{key}"] = dd[t].float()
        out[f"{name}_{key}_row"] = (base[t].float() + dd[t].float()).to(torch.bfloat16)
for t, key in ((LT, "27"), (EOS, "eos")):  # D2 rows must equal their released new rows exactly
    assert torch.equal(out[f"D2_{key}_row"].float() - base[t].float(), d2[t]), key
out["A1_row"] = (base[EOS].float() + out["A1"]).to(torch.bfloat16)
assert torch.equal(out["A1_row"].view(torch.int16), S2["A_row"].view(torch.int16)), "A1 row != stage-2 A row"
rep["norms"] = {k: v.norm().item() for k, v in out.items() if k.endswith(("_27", "_eos"))}
rep["row_bits_changed"] = {k: int((out[k + "_row"].view(torch.int16) != base[LT if k.endswith("27") else EOS].view(torch.int16)).sum())
                           for k in ("A2_27", "A2_eos", "D2_27", "D2_eos", "C2_27", "C2_eos")}
rep["cos"] = {f"A2 vs D2 {k}": (a2[t] @ d2[t] / (a2[t].norm() * d2[t].norm())).item() for t, k in ((LT, "27"), (EOS, "eos"))}

def stats(v):
    return [round(v.mean().item(), 3), round(v.std().item(), 3), round(v.min().item(), 3), round(v.max().item(), 3)]

rep["shift"] = {}
for name in ("A2", "D2", "C2"):
    for t, key in ((LT, "27"), (EOS, "eos")):
        eff = out[f"{name}_{key}_row"].float() - base[t].float()
        rep["shift"][f"{name}/{key}"] = {gname: stats(xx @ eff) for gname, xx in zip(groups, xs)}
rep["unpatched_logit"] = {f"{key}": {gname: stats(xx @ base[t].float()) for gname, xx in zip(groups, xs)}
                          for t, key in ((LT, "27"), (EOS, "eos"))}
torch.save(out, D / f"deltas_s3_{TAG}.pt")
(D / f"fit3_{TAG}.json").write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items() if k not in ("shift", "unpatched_logit")}, indent=1))
for k, v in rep["shift"].items():
    print(k, {g: v[g][0] for g in v})
print("unpatched", {k: {g: v[g][0] for g in v} for k, v in rep["unpatched_logit"].items()})
