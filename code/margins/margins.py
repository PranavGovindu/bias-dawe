"""neartie_01 step 1: per-item logit margins at the dawe_01 decision points, from saved final-layer states (CPU only).

Runs on a local machine (the training pins, transformers 5.20.2). usage: margins.py <seed tag s0|s1>
Inputs: dawe_01 maps (train items, teacher-forced, fp16, index -1 = post-norm readout input), base embed_tokens (tied
readout; LoRA never touched it), the SFT/pair files the maps were built from (to recover item ids and gold tokens in map
order), the stage-2/3 deltas and prevent_01's learned rows.
Output: $MARGINS_WORK/margins_<tag>.jsonl (one row per decision point) + check_<tag>.json.
Per row: z at tokens of interest, top-10 (id, logit), gold, fp32 NLL of gold, and h.delta for every edit row, so any
combination of row edits / constant biases can be re-scored exactly offline (top-10 suffices for <= 9 edited rows)."""
import json, os, sys
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

torch.set_num_threads(4)
TAG = sys.argv[1]
MAPS = WORK / ("maps" if TAG == "s0" else "maps_s1")  # modal volume get: sft_v1_s0/maps, dawe_s1/maps
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
LT, LBR, EOS, NL = 27, 4754, 248046, 198

tok = AutoTokenizer.from_pretrained(SNAP)
NN = tok("\n\n", add_special_tokens=False)["input_ids"]
assert len(NN) == 1, NN
NN = NN[0]
INTEREST = [LT, LBR, 90, EOS, NL, NN, 248059, 13766, 10083, 6994, 248044]

idx = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
with safe_open(str(SNAP / idx[EMB]), framework="pt") as f:
    E_bf = f.get_tensor(EMB)  # [V, 2560] bf16
E = E_bf.float()
V = E.shape[0]


# --- replay the map builders to get (item id, kind, gold next token) in state order --------------------------------
def _decision_points(reply, fmt, calls_json=None):
    ends = []
    if fmt == "xml":
        i = 0
        while (j := reply.find("</tool_call>", i)) != -1:
            ends.append(j + len("</tool_call>"))
            i = ends[-1]
    else:
        i = 0
        for cj in calls_json or []:
            j = reply.find(cj, i)
            if j == -1:
                return []
            ends.append(j + len(cj))
            i = ends[-1]
    return [(e, k < len(ends) - 1) for k, e in enumerate(ends)]


def points_for(reply, fmt):
    pts = []
    body, shift = reply, 0
    if reply.startswith("<plan>"):
        end = reply.index("</plan>") + len("</plan>")
        pts += [(0, "plan_start"), (end, "after_plan")]
        rest = reply[end:]
        shift = end + (len(rest) - len(rest.lstrip()))
        body = reply[shift:]
    calls = None if fmt == "xml" else [json.dumps(c, ensure_ascii=False) for c in json.loads(body)["tool_calls"]]
    pts += [(shift + e, "continue" if c else "stop") for e, c in _decision_points(body, fmt, calls)]
    return pts


def encode(msgs, reply):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(prompt + reply + "<|im_end|>", add_special_tokens=False, return_offsets_mapping=True)
    return prompt, enc


meta = {}  # file key -> list of dicts in state order
rows = [json.loads(l) for l in open(DATA / "sft/v11_plan_bfcl_train.jsonl")]
pm = {"plan_start": [], "after_plan": []}
for r in rows:
    msgs, reply = r["messages"][:-1], r["messages"][-1]["content"]
    prompt, enc = encode(msgs, reply)
    offs, ids = enc["offset_mapping"], enc["input_ids"]
    for ce, kind in points_for(reply, "json"):
        if kind in pm:
            d = max(i for i, (s, e) in enumerate(offs) if e <= len(prompt) + ce)
            pm[kind].append({"id": r["id"], "kind": kind, "k": 0, "gold": ids[d + 1], "d_tok": ids[d]})
meta["plan"] = pm
pairs = [json.loads(l) for l in open(DATA / "pairs/v1__v4_format.jsonl")]
for h in ("v1", "v4_format"):
    fmt = "xml" if h == "v4_format" else "json"
    st = {True: [], False: []}
    for pr in pairs:
        msgs = pr[h]
        reply = msgs[-1]["content"]
        prompt, enc = encode(msgs[:-1], reply)
        offs, ids = enc["offset_mapping"], enc["input_ids"]
        cj = [json.dumps(c, ensure_ascii=False) for c in json.loads(reply)["tool_calls"]] if fmt == "json" else None
        for k, (ce, cont) in enumerate(_decision_points(reply, fmt, cj)):
            d = max(i for i, (s, e) in enumerate(offs) if e <= len(prompt) + ce)
            if d + 1 >= len(ids):
                continue
            st[cont].append({"id": pr["id"], "kind": "continue" if cont else "stop", "k": k, "gold": ids[d + 1],
                             "d_tok": ids[d]})
    meta[h] = {"continue": st[True], "stop": st[False]}

# --- edits: effective (bf16-cast) row deltas -----------------------------------------------------------------------
s3 = torch.load(DELTAS / f"deltas_s3_{TAG}.pt")
s2 = torch.load(DELTAS / ("deltas_stage2.pt" if TAG == "s0" else "deltas_s1.pt"))
b27, beos = E_bf[LT].float(), E_bf[EOS].float()
assert torch.equal(s3["base_row27"].float(), b27) and torch.equal(s3["base_row"].float(), beos)
edits = {  # name -> (token id, delta vector)
    "A2_27": (LT, s3["A2_27_row"].float() - b27), "A2_eos": (EOS, s3["A2_eos_row"].float() - beos),
    "D2_27": (LT, s3["D2_27_row"].float() - b27), "D2_eos": (EOS, s3["D2_eos_row"].float() - beos),
    "C2_27": (LT, s3["C2_27_row"].float() - b27), "C2_eos": (EOS, s3["C2_eos_row"].float() - beos),
    "A1_eos": (EOS, s2["A_row"].float() - beos), "D1_eos": (EOS, s2["D_row"].float() - beos),
}
pv = torch.load(REPO / "results" / "prevent" / ("rows_prev_b_rows1x.pt" if TAG == "s0" else "rows_prev_b_rows1x_s1.pt"))
orig_eq = {}
for j, t in enumerate(pv["token_ids"]):
    orig_eq[t] = bool(torch.equal(pv["orig"][j].to(torch.bfloat16).float(), E_bf[t].float()))
    edits[f"P_{t}"] = (t, pv["learned"][j].to(torch.bfloat16).float() - E_bf[t].float())
Dmat = torch.stack([v for _, v in edits.values()])  # [n_edits, 2560]

# --- margins ---------------------------------------------------------------------------------------------------------
check = {"tag": TAG, "nn_token": NN, "prevent_token_ids": pv["token_ids"], "prevent_orig_equals_base_bf16": orig_eq,
         "edit_norms": {k: v.norm().item() for k, (_, v) in edits.items()}, "kinds": {}}
out = open(OUT / f"margins_{TAG}.jsonl", "w")
for fname, key in (("states_dp_plan_v11_plan.pt", "plan"), ("states_v1.pt", "v1"), ("states_v4_format.pt", "v4_format")):
    st = torch.load(MAPS / fname)
    for kind, X in st.items():
        H = X[:, -1, :].float()
        del X
        M = meta[key][kind]
        assert len(M) == H.shape[0], (fname, kind, len(M), H.shape)
        nll_sum = 0.0
        harness = "v11_plan" if key == "plan" else key
        for b0 in range(0, H.shape[0], 64):
            hb = H[b0:b0 + 64]
            z = hb @ E.T  # [B, V] fp32
            lse = torch.logsumexp(z, -1)
            top = z.topk(10, -1)
            sh = hb @ Dmat.T
            for i in range(hb.shape[0]):
                m = M[b0 + i]
                g = m["gold"]
                rec = {"seed": TAG, "harness": harness, "kind": kind, "id": m["id"], "k": m["k"], "gold": g,
                       "d_tok": m["d_tok"], "nll": (lse[i] - z[i, g]).item(), "z_gold": z[i, g].item(),
                       "z": {str(t): round(z[i, t].item(), 4) for t in INTEREST},
                       "top": [[int(t), round(v, 4)] for t, v in zip(top.indices[i].tolist(), top.values[i].tolist())],
                       "hnorm": round(hb[i].norm().item(), 3),
                       "shift": {k: round(sh[i, j].item(), 4) for j, k in enumerate(edits)}}
                nll_sum += rec["nll"]
                out.write(json.dumps(rec) + "\n")
        check["kinds"][f"{harness}/{kind}"] = {"n": H.shape[0], "mean_nll_fp32": nll_sum / H.shape[0]}
        print(TAG, harness, kind, check["kinds"][f"{harness}/{kind}"], flush=True)
out.close()
(OUT / f"check_{TAG}.json").write_text(json.dumps(check, indent=1))
print(json.dumps(check, indent=1))
