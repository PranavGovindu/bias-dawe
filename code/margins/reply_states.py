"""neartie_01 local-GPU pass (an RTX 3060 in our runs): final post-norm states at reply-start positions, on train AND test items.
usage: reply_states.py <tag s0|s1>   (s0 = sft_v1_s0 adapter, s1 = prev_a_s1 adapter = dawe_s1)
Merges the LoRA adapter into the bf16 base on GPU (merge_and_unload, as in stage 1), then one forward per prompt (no
padding), saving h[-1] (fp32) at:
  v1 reply start       (last prompt token)                    train 595 + test 204
  v11 plan start       (last prompt token)                    train 595 + test 204
  v11 post-<plan>      (prompt + tokens '<' 'plan' '>', last)  same prompts, same forward
Output: $MARGINS_WORK/rs_<tag>.pt  {kind/split: [n,2560] fp32, 'ids': {...}, 'check': {...}}"""
import json, os, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel

TAG = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "v11" else None  # smoke: first N prompts per set
V11ONLY = "v11" in sys.argv[2:]  # re-run: v11 sets only, saving the extra positions
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
ADP = OUT / "adapters" / {"s0": "sft_v1_s0", "s1": "prev_a_s1"}.get(TAG, TAG)  # step 2: TAG = run name (adapters/<run>)
PLAN = [27, 10083, 29]  # '<' 'plan' '>'

t0 = time.time()
tok = AutoTokenizer.from_pretrained(SNAP)
assert tok("<plan>", add_special_tokens=False)["input_ids"] == PLAN
model = AutoModelForImageTextToText.from_pretrained(SNAP, dtype=torch.bfloat16, device_map="cuda:0").eval()
E0 = model.get_input_embeddings().weight
rows_before = {t: E0[t].detach().clone() for t in (27, 4754, 248046)}
model = PeftModel.from_pretrained(model, str(ADP)).merge_and_unload().eval()
E1 = model.get_input_embeddings().weight
check = {"embed_rows_unchanged_by_merge": all(torch.equal(rows_before[t], E1[t]) for t in rows_before),
         "tied": E1.data_ptr() == model.get_output_embeddings().weight.data_ptr(),
         "load_merge_s": round(time.time() - t0, 1), "gpu_alloc_GiB_after_merge": round(torch.cuda.memory_allocated() / 2**30, 2)}
print(check, flush=True)


def prompt_ids(msgs):
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(p, add_special_tokens=False)["input_ids"]


pairs = [json.loads(l) for l in open(DATA / "pairs/v1__v4_format.jsonl")]
v11tr = [json.loads(l) for l in open(DATA / "sft/v11_plan_bfcl_train.jsonl")]
test = [json.loads(l) for l in open(DATA / "eval/bfcl_test_3h.jsonl")]
sets = {
    "v1/train": [(p["id"], p["v1"][:-1]) for p in pairs],
    "v1/test": [(r["id"], r["messages"]) for r in test if r["harness"] == "v1"],
    "v11/train": [(r["id"], r["messages"][:-1]) for r in v11tr],
    "v11/test": [(r["id"], r["messages"]) for r in test if r["harness"] == "v11_plan"],
}
if V11ONLY:
    sets = {k: v for k, v in sets.items() if k.startswith("v11")}
out = {"ids": {}, "check": check, "tag": TAG}
logit_check = []
with torch.no_grad():
    for name, items in sets.items():
        items = items[:LIMIT] if LIMIT else items
        is11 = name.startswith("v11")
        H_start, H_post, ids_ = [], [], []
        for k, (iid, msgs) in enumerate(items):
            ids = prompt_ids(msgs)
            seq = ids + PLAN if is11 else ids
            x = torch.tensor([seq], device="cuda:0")
            res = model(input_ids=x, output_hidden_states=True, logits_to_keep=1)
            h = res.hidden_states[-1][0]
            H_start.append(h[len(ids) - 1].float().cpu())
            if is11:
                H_post.append(h[len(ids) - 1:len(ids) + 3].float().cpu())  # [start, after '<', after 'plan', after '>']
            if k < 3:  # last-position logits == h[-1] @ E.T (post-norm readout), in bf16 as the model computes it
                lg = res.logits[0, -1].float()
                lg2 = (h[-1].to(E1.dtype) @ E1.T).float()
                logit_check.append([name, k, (lg - lg2).abs().max().item()])
            ids_.append(iid)
            del res, h, x
        out["ids"][name] = ids_
        out[f"{name}/start"] = torch.stack(H_start)
        if is11:
            P = torch.stack(H_post)
            out[f"{name}/after_lt"] = P[:, 1]
            out[f"{name}/after_plantok"] = P[:, 2]
            out[f"{name}/post_plan"] = P[:, 3]
        print(name, len(ids_), "done", round(time.time() - t0, 1), "s; peak GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
out["check"]["logit_check_maxabs"] = logit_check
out["check"]["peak_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
out["check"]["total_s"] = round(time.time() - t0, 1)
torch.save(out, OUT / (f"rs_{TAG}_smoke.pt" if LIMIT else f"rs_{TAG}_v11.pt" if V11ONLY else f"rs_{TAG}.pt"))
print(out["check"], flush=True)
