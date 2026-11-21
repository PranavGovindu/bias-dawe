"""prevent_01: does letting the v1 LoRA SFT also train a few decision-token rows change the harness break?

In this repo it is the code path that trained the seed-1 plain LoRA (arm a1, run name `prev_a_s1`; see ARMS below), plus the
side result in the README. Adapted from bandr-ai/dawe@8f11c96 gpu/app.py::train (Apache-2.0, see ../NOTICE).

Ephemeral `modal run` only. Same app name and volumes as formatter_exps_gp.py (FEP_APP, FEP_RUNS_VOLUME, FEP_HF_VOLUME);
every path we write on the volume starts with "prev_". Image definitions are copied verbatim from that file so the
cached images are reused.

Arms (all: DAWE's exact v1 recipe, seed 0, same data order; (a) = stage-1 sft_v1_s0, not retrained):
  b  LoRA + 6 decision rows trainable (PEFT trainable_token_indices, tie kept, rows in fp32), rows at lr 1e-4
  c  same rows at 10x lr
  d  LoRA + 6 random rows (parameter-count control)
The only changes from their train(): LoraConfig(trainable_token_indices=...), the row parameter upcast to fp32, a second
optimizer param group for the rows (lr x mult), and assertions (tie, dtype, counts, step-1 loss, merge).

Commands (from this directory, code/; env vars on every command):
  DAWE_TELEMETRY_DISABLED=1 DO_NOT_TRACK=1 modal run prevent.py::arm --name a1   # seed-1 plain LoRA (+ unpatched eval)
  DAWE_TELEMETRY_DISABLED=1 DO_NOT_TRACK=1 modal run prevent.py::arm --name b
  DAWE_TELEMETRY_DISABLED=1 DO_NOT_TRACK=1 modal run prevent.py::arms --names c,d
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent  # code/
REPO = ROOT.parent  # data/ and runs/ live here
MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

PRICE = {"H100": 0.001097, "cpu_core": 0.0000131, "mem_gib": 0.00000222}
GPU, CPU, MEM_GIB = "H100", 2.0, 16
RATE = PRICE["H100"] + CPU * PRICE["cpu_core"] + MEM_GIB * PRICE["mem_gib"]  # ~ $4.29/hr

APP_NAME = os.environ.get("FEP_APP", "formatter-exps-gp")
RUNS_VOLUME = os.environ.get("FEP_RUNS_VOLUME", "formatter-exps-gp-runs")
HF_VOLUME = os.environ.get("FEP_HF_VOLUME", "formatter-exps-gp-hf")
app = modal.App(APP_NAME)
runs = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_VOLUME, create_if_missing=True)
VOLS = {"/runs": runs, "/root/.cache/huggingface": hf_cache}

NO_TELEMETRY = {"DAWE_TELEMETRY_DISABLED": "1", "DO_NOT_TRACK": "1", "VLLM_NO_USAGE_STATS": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "FEP_APP": APP_NAME, "FEP_RUNS_VOLUME": RUNS_VOLUME, "FEP_HF_VOLUME": HF_VOLUME}

# --- images: verbatim from formatter_exps_gp.py ---
IMAGE_MODE = os.environ.get("FEP_IMAGE", "pins")
TRAIN_PINS = ["torch==2.13.0", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0",
              "flash-linear-attention==0.5.2", "hf_transfer"]
VLLM_PINS = ["vllm==0.31.0", "hf_transfer"]


def _pkgs(kind: str) -> list[str]:
    if not modal.is_local():
        return []
    if IMAGE_MODE == "freeze":
        name = f"req_{kind}.txt"
        return [l.strip() for l in (ROOT / name).read_text().splitlines() if l.strip() and not l.startswith("#")]
    return TRAIN_PINS if kind == "train" else VLLM_PINS


train_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*_pkgs("train")).env(NO_TELEMETRY)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*_pkgs("vllm"))
    .env(NO_TELEMETRY | {"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

COMMON = dict(gpu=GPU, cpu=CPU, memory=MEM_GIB * 1024, volumes=VOLS, scaledown_window=2)

# --- arms ---------------------------------------------------------------------------------------
# Decision rows (see PREDICTIONS.md): <|im_end|>, "\n", "<", '{"', "tool", "plan".
DECISION_ROWS = [248046, 198, 27, 4754, 13766, 10083]
# randperm(248044, Generator().manual_seed(0)), first 6 not in DECISION_ROWS; none occurs in the v1 train set.
RANDOM_ROWS = [221956, 86845, 210849, 230425, 55487, 229039]
ARMS = {
    "b": dict(run="prev_b_rows1x", tokens=DECISION_ROWS, mult=1.0),
    "c": dict(run="prev_c_rows10x", tokens=DECISION_ROWS, mult=10.0),
    "d": dict(run="prev_d_rand1x", tokens=RANDOM_ROWS, mult=1.0),
    # seed-1 replicate (2026-11-18): plain LoRA through the same code path (token_ids=[]), and arm b
    "a1": dict(run="prev_a_s1", tokens=[], mult=1.0, seed=1),
    "b1": dict(run="prev_b_rows1x_s1", tokens=DECISION_ROWS, mult=1.0, seed=1),
}
LORA_PARAMS_FULL = 64_929_792  # stage-1 print_trainable_parameters
STAGE1_LOSS1 = 0.0709  # stage-1 sft_v1_s0 step-1 loss (rows start at their originals, so step 1 must match)
STAGE1_MERGED = "/runs/sft_v1_s0/merged"  # read-only: its safetensors key set is the reference


def _cost(t0: float, label: str) -> dict:
    secs = time.time() - t0
    out = {"job": label, "seconds": round(secs), "usd_est": round(secs * RATE, 3)}
    print(f"[cost] {label}: {secs / 60:.1f} min ~ ${out['usd_est']:.2f}")
    return out


# --- core (plain function: the PC smoke test calls it through a modal stub) ----------------------


def _row_param(model):
    """The trainable rows: one nn.Parameter shared by the embedding wrapper and the tied lm_head wrapper, in fp32."""
    import torch
    from peft.utils.other import TrainableTokensWrapper

    inner = model.base_model.model
    emb, head = inner.get_input_embeddings(), inner.get_output_embeddings()
    assert isinstance(emb, TrainableTokensWrapper), f"embedding not wrapped: {type(emb)}"
    assert isinstance(head, TrainableTokensWrapper), f"lm_head not wrapped, tie NOT kept: {type(head)}"
    ta_e, ta_h = emb.token_adapter, head.token_adapter
    assert ta_h.tied_adapter is ta_e, "lm_head token adapter is not tied to the embedding token adapter"
    pd = ta_e.trainable_tokens_delta
    assert ta_h.trainable_tokens_delta is pd
    dtype0 = pd["default"].dtype
    if dtype0 != torch.float32:  # bf16 rows at lr 1e-4 would round most Adam updates away (LoRA adapters are fp32)
        pd["default"] = torch.nn.Parameter(pd["default"].detach().float(), requires_grad=True)
    p = pd["default"]
    assert p.dtype == torch.float32 and p.requires_grad and ta_h.trainable_tokens_delta["default"] is p
    assert not ta_e.base_layer.weight.requires_grad
    print(f"rows param: shape {tuple(p.shape)}, dtype {dtype0} -> {p.dtype}, tied lm_head wrapper OK")
    return p, ta_e


def _train_core(rows: list[dict], run: str, out_root: str, token_ids: list[int], row_lr_mult: float = 1.0,
                epochs: float = 3, lr: float = 1e-4, rank: int = 32, max_steps: int = -1, seed: int = 0,
                n_layers: int | None = None, expect_loss1: float | None = None, save: bool = True) -> dict:
    """Their gpu/app.py::train with trainable decision rows. token_ids=[] gives plain LoRA (for the smoke comparison).
    n_layers truncates the text model (PC smoke only)."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    if n_layers:
        cfg = AutoConfig.from_pretrained(MODEL, revision=REVISION)
        cfg.text_config.num_hidden_layers = n_layers
        cfg.text_config.layer_types = cfg.text_config.layer_types[:n_layers]
        model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, config=cfg, dtype=torch.bfloat16).cuda()
    else:
        model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).cuda()

    # Prompt = chat template with thinking off; loss only on the reply.
    examples = []
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(r["messages"][-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        examples.append((p_ids + c_ids, [-100] * len(p_ids) + c_ids))
    lens = sorted(len(x) for x, _ in examples)
    print(f"{len(examples)} examples, tokens median {lens[len(lens) // 2]}, max {lens[-1]}, total {sum(lens)}")

    # check example: the shortest one; a probe token for the readout test = a row token absent from its input ids
    ex_ids = torch.tensor([min(examples, key=lambda e: len(e[0]))[0]]).cuda()
    probe = next((t for t in token_ids if t not in ex_ids[0].tolist()), None)
    with torch.no_grad():
        ref_logits = model(input_ids=ex_ids).logits[0, -8:].float()

    targets = [
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "language_model" in n and "lm_head" not in n
    ]
    kinds = sorted({n.rsplit(".", 1)[-1] for n in targets})
    print(f"LoRA targets: {len(targets)} layers, kinds {kinds}")
    lcfg = dict(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=targets)
    if token_ids:
        lcfg["trainable_token_indices"] = list(token_ids)
    model = get_peft_model(model, LoraConfig(**lcfg))
    model.print_trainable_parameters()
    lora_A_sum = sum(p.detach().double().sum().item() for n, p in model.named_parameters() if "lora_A" in n)
    print(f"LoRA A init checksum {lora_A_sum:.10e}")

    rows_p, ta = (None, None)
    if token_ids:
        rows_p, ta = _row_param(model)
        orig = ta.trainable_tokens_original["default"].detach().float().clone()
        assert torch.equal(rows_p.detach(), orig), "rows do not start at their original values"
    lora_ps = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    others = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n and p is not rows_p]
    assert not others, f"unexpected trainable params: {others[:5]}"
    n_lora = sum(p.numel() for p in lora_ps)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_rows = 0 if rows_p is None else rows_p.numel()
    assert n_tr == n_lora + n_rows == n_lora + len(token_ids) * rows_p.shape[1] if token_ids else n_tr == n_lora
    if not n_layers:
        assert n_lora == LORA_PARAMS_FULL, n_lora
    print(f"trainable: LoRA {n_lora:,} + rows {n_rows:,} = {n_tr:,}")

    # init equivalence: LoRA B = 0 and rows = originals, so logits must equal the base model's
    with torch.no_grad():
        lg = model(input_ids=ex_ids).logits[0, -8:].float()
    d_init = (lg - ref_logits).abs().max().item()
    print(f"init logits max|diff| vs base: {d_init:.3g}")
    assert d_init < 1e-2, d_init

    # readout/input test: a row change must move that token's logit by h.v (output side) and its embedding (input side)
    if token_ids and probe is not None:
        j = token_ids.index(probe)
        v = torch.randn(rows_p.shape[1], generator=torch.Generator().manual_seed(1)).cuda() * 0.05
        with torch.no_grad():
            o0 = model(input_ids=ex_ids, output_hidden_states=True)
            rows_p[j] += v
            o1 = model(input_ids=ex_ids, output_hidden_states=True)
            emb_out = model.base_model.model.get_input_embeddings()(torch.tensor([[probe]]).cuda())[0, 0].float()
            rows_p[j].copy_(orig[j])
        h = o0.hidden_states[-1][0, -1].float()
        want = (h @ v.to(torch.bfloat16).float()).item()
        got = (o1.logits[0, -1, probe] - o0.logits[0, -1, probe]).float().item()
        same_h = torch.equal(o0.hidden_states[-1], o1.hidden_states[-1])
        in_ok = torch.equal(emb_out, (orig[j] + v).to(torch.bfloat16).float())
        print(f"readout test token {probe}: logit shift {got:.4f} vs h.v {want:.4f}; hidden unchanged {same_h}; "
              f"embedding reads row {in_ok}")
        # bf16 logits: each logit carries ~0.06 rounding at |logit| ~ 16
        assert same_h and in_ok and abs(got - want) < 0.15 + 0.02 * abs(want), "output/input side does not read the rows"
        assert torch.equal(rows_p.detach(), orig)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    groups = [{"params": lora_ps, "lr": lr}]
    if rows_p is not None:
        groups.append({"params": [rows_p], "lr": lr * row_lr_mult})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
    batch, accum = 4, 4  # effective batch 16
    steps_per_epoch = max(1, len(examples) // (batch * accum))
    total = max_steps if max_steps > 0 else int(steps_per_epoch * epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / total))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    g = torch.Generator().manual_seed(seed)
    order: list[int] = []
    model.train()
    log, gnorm, row_log = [], [], []
    for step in range(total):
        loss_sum = 0.0
        for _ in range(accum):
            if len(order) < batch:
                order += torch.randperm(len(examples), generator=g).tolist()
            idx, order = order[:batch], order[batch:]
            b = [examples[i] for i in idx]
            L = max(len(x) for x, _ in b)
            ids = torch.tensor([x + [pad] * (L - len(x)) for x, _ in b]).cuda()
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in b]).cuda()
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in b]).cuda()
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / accum
            loss.backward()
            loss_sum += loss.item()
        if step == 0 and rows_p is not None:
            gr = rows_p.grad.norm(dim=1).tolist()
            print("step-1 row grad norms", {t: f"{x:.2e}" for t, x in zip(token_ids, gr)})
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        if step in (0, 9):
            print(f"step {step + 1} param-group lrs {[g['lr'] for g in opt.param_groups]}")
        opt.step(), sched.step(), opt.zero_grad()
        log.append(round(loss_sum, 4))
        gnorm.append(round(gn, 4))
        if step == 0 and expect_loss1 is not None and abs(loss_sum - expect_loss1) > 0.005:
            raise RuntimeError(f"step-1 loss {loss_sum:.4f} != stage-1 {expect_loss1}: init or data order differs")
        if rows_p is not None and (step % 10 == 0 or step == total - 1):
            with torch.no_grad():
                rel = ((rows_p - orig).norm(dim=1) / orig.norm(dim=1)).tolist()
            row_log.append({"step": step + 1, "rel": [round(x, 4) for x in rel]})
        if step % 5 == 0 or step == total - 1:
            extra = f" rows rel {row_log[-1]['rel']}" if row_log and row_log[-1]["step"] == step + 1 else ""
            print(f"step {step + 1}/{total} loss {loss_sum:.4f} gnorm {gn:.3f}{extra}")

    info = {"run": run, "steps": total, "loss": log, "gnorm": gnorm, "lr": lr, "row_lr_mult": row_lr_mult, "rank": rank,
            "epochs": epochs, "n": len(rows), "seed": seed, "token_ids": list(token_ids), "lora_A_init": lora_A_sum,
            "row_log": row_log, "n_lora": n_lora, "n_rows": n_rows}
    out = Path(out_root) / run
    out.mkdir(parents=True, exist_ok=True)
    if rows_p is not None:
        learned = rows_p.detach().float().cpu()
        torch.save({"token_ids": list(token_ids), "orig": orig.cpu(), "learned": learned, "mult": row_lr_mult}, out / "rows.pt")
        info["row_delta_norm"] = (learned - orig.cpu()).norm(dim=1).tolist()
        info["row_orig_norm"] = orig.cpu().norm(dim=1).tolist()
        E_base = ta.base_layer.weight.detach().clone()
    if save:
        model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()

    # merge checks: plain model, tie intact, rows = learned values, every other row bit-identical to base
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils.other import AuxiliaryTrainingWrapper

    leftovers = [n for n, m in merged.named_modules() if isinstance(m, (BaseTunerLayer, AuxiliaryTrainingWrapper))]
    assert not leftovers, leftovers[:5]
    E, H = merged.get_input_embeddings().weight, merged.get_output_embeddings().weight
    assert E.data_ptr() == H.data_ptr(), "tie broken after merge"
    keys = set(merged.state_dict().keys())
    assert not [k for k in keys if any(s in k for s in ("lora_", "token_adapter", "base_layer", "trainable_tokens"))]
    if rows_p is not None:
        idx = torch.tensor(token_ids, device=E.device)
        assert torch.equal(E[idx], rows_p.detach().to(E.dtype)), "merged rows != learned rows"
        keep = torch.ones(E.shape[0], dtype=torch.bool, device=E.device)
        keep[idx] = False
        assert torch.equal(E[keep], E_base[keep]), "rows outside token_ids changed"
        del E_base
    print("merge checks OK (no wrappers, tied, rows = learned, other rows = base)")

    if save:
        merged.save_pretrained(out / "merged", safe_serialization=True)
        tok.save_pretrained(out / "merged")
        from huggingface_hub import hf_hub_download

        for f in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
            (out / "merged" / f).write_bytes(Path(hf_hub_download(MODEL, f, revision=REVISION)).read_bytes())
        from safetensors import safe_open

        def st_keys(d):
            ks = set()
            for fp in sorted(Path(d).glob("*.safetensors")):
                with safe_open(str(fp), "pt") as fh:
                    ks |= set(fh.keys())
            return ks

        mine = st_keys(out / "merged")
        info["n_saved_tensors"] = len(mine)
        if not n_layers and Path(STAGE1_MERGED).exists():
            ref = st_keys(STAGE1_MERGED)
            assert mine == ref, f"saved keys differ from stage 1: +{sorted(mine - ref)[:5]} -{sorted(ref - mine)[:5]}"
            print(f"saved tensor keys == stage-1 merged ({len(mine)})")
    (out / "train.json").write_text(json.dumps(info, indent=1))
    info["t_secs"] = round(time.time() - t0)
    return info


# --- Modal functions ------------------------------------------------------------------------------


@app.function(image=train_image, timeout=15 * 60, **COMMON)
def train_rows(rows: list[dict], run: str, token_ids: list[int], row_lr_mult: float = 1.0, max_steps: int = -1,
               seed: int = 0) -> dict:
    assert run.startswith("prev_"), run
    t0 = time.time()
    runs.reload()
    info = _train_core(rows, run, "/runs", token_ids, row_lr_mult=row_lr_mult, max_steps=max_steps, seed=seed,
                       expect_loss1=STAGE1_LOSS1 if (seed == 0 and max_steps < 0) else None)
    runs.commit()
    info["cost"] = _cost(t0, f"train {run}")
    return info


@app.function(image=vllm_image, timeout=8 * 60, **COMMON)
def generate(rows: list[dict], model: str, max_tokens: int = 512, enforce_eager: bool = False, tag: str = "") -> dict:
    """Their gpu/app.py::generate; preds also written to /runs/prev_preds/."""
    from vllm import LLM, SamplingParams

    assert model.startswith("prev_"), model
    t0 = time.time()
    runs.reload()
    llm = LLM(
        model=f"/runs/{model}/merged",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outs = llm.chat([r["messages"] for r in rows], params, chat_template_kwargs={"enable_thinking": False})
    preds = [{"id": r["id"], "harness": r["harness"], "text": o.outputs[0].text} for r, o in zip(rows, outs)]
    d = Path("/runs/prev_preds")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"preds_{model}{tag}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    runs.commit()
    return {"preds": preds, "cost": _cost(t0, f"generate {model}{tag} x{len(rows)} eager={enforce_eager}")}


def _split_core(src: str, dst: str, side: str) -> dict:
    """Untied copy of a merged rows-arm checkpoint. side='out': lm_head has the learned rows, embedding has the base rows;
    'in': the reverse; 'both': both learned (should reproduce the tied arm); 'none': both base (the arm's LoRA alone).
    Everything else is copied unchanged."""
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    r = torch.load(f"{src}/rows.pt")
    idx = torch.tensor(r["token_ids"])
    key = "model.language_model.embed_tokens.weight"
    files = sorted(Path(f"{src}/merged").glob("*.safetensors"))
    sd = {}
    for fp in files:
        with safe_open(str(fp), "pt") as fh:
            sd.update({k: fh.get_tensor(k) for k in fh.keys()})
    assert key in sd and "lm_head.weight" not in sd, "expected a tied checkpoint"
    E = sd[key]
    assert torch.equal(E[idx], r["learned"].to(E.dtype)), "merged rows != rows.pt learned"
    E_base = E.clone()
    E_base[idx] = r["orig"].to(E.dtype)
    emb, head = {"out": (E_base, E), "in": (E, E_base), "both": (E, E.clone()), "none": (E_base, E_base.clone())}[side]
    sd[key], sd["lm_head.weight"] = emb.contiguous(), head.contiguous()
    out = Path(f"{dst}/merged")
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    for f in Path(f"{src}/merged").iterdir():
        if f.suffix != ".safetensors" and not f.name.endswith(".index.json") and f.is_file():
            shutil.copy(f, out / f.name)
    cfg = json.loads((out / "config.json").read_text())
    cfg["tie_word_embeddings"] = False
    cfg.setdefault("text_config", {})["tie_word_embeddings"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    with safe_open(str(out / "model.safetensors"), "pt") as fh:
        e2, h2 = fh.get_tensor(key), fh.get_tensor("lm_head.weight")
    info = {"side": side, "emb_rows_learned": bool(torch.equal(e2[idx], r["learned"].to(e2.dtype))),
            "head_rows_learned": bool(torch.equal(h2[idx], r["learned"].to(h2.dtype))),
            "emb_rows_base": bool(torch.equal(e2[idx], r["orig"].to(e2.dtype))),
            "head_rows_base": bool(torch.equal(h2[idx], r["orig"].to(h2.dtype)))}
    print("split", dst, info)
    return info


@app.function(image=train_image, cpu=2.0, memory=40 * 1024, volumes=VOLS, timeout=10 * 60)
def build_split(run: str, side: str) -> dict:
    assert run.startswith("prev_")
    t0 = time.time()
    runs.reload()
    info = _split_core(f"/runs/{run}", f"/runs/{run}_split_{side}", side)
    runs.commit()
    secs = time.time() - t0
    info["cost"] = {"job": f"build_split {run} {side}", "seconds": round(secs),
                    "usd_est": round(secs * (2 * PRICE["cpu_core"] + 40 * PRICE["mem_gib"]), 3)}
    return info


# --- local entry points ------------------------------------------------------------------------


def _read(p: str) -> list[dict]:
    return [json.loads(line) for line in (REPO / p).read_text().splitlines() if line.strip()]


def _log_cost(c: dict) -> None:
    (REPO / "runs").mkdir(exist_ok=True)
    with (REPO / "runs" / "costs.jsonl").open("a") as f:
        f.write(json.dumps(c) + "\n")


def _save_train(info: dict) -> None:
    (REPO / "runs").mkdir(exist_ok=True)
    (REPO / "runs" / f"train_{info['run']}.json").write_text(json.dumps(info, indent=1))
    _log_cost(info["cost"])
    print(json.dumps({k: v for k, v in info.items() if k not in ("loss", "gnorm", "row_log")}, indent=1))
    print("loss first/last", info["loss"][:3], info["loss"][-3:])


def _save_preds(run: str, res: dict) -> None:
    out = REPO / "runs" / f"preds_{run}.jsonl"
    out.write_text("".join(json.dumps(p) + "\n" for p in res["preds"]))
    _log_cost(res["cost"])
    print("saved", out, res["cost"])


EVAL = "data/eval/bfcl_test_3h.jsonl"
TRAIN = "data/sft/v1_bfcl_train.jsonl"


@app.local_entrypoint()
def arm(name: str = "b", gen: bool = True):
    cfg = ARMS[name]
    info = train_rows.remote(_read(TRAIN), cfg["run"], cfg["tokens"], cfg["mult"], seed=cfg.get("seed", 0))
    _save_train(info)
    if gen:
        _save_preds(cfg["run"], generate.remote(_read(EVAL), cfg["run"]))


@app.local_entrypoint()
def arms(names: str = "c,d"):
    cfgs = [ARMS[n] for n in names.split(",")]
    jobs = [train_rows.spawn(_read(TRAIN), c["run"], c["tokens"], c["mult"], seed=c.get("seed", 0)) for c in cfgs]
    gens = []
    for c, j in zip(cfgs, jobs):
        _save_train(j.get())
        gens.append((c["run"], generate.spawn(_read(EVAL), c["run"])))
    for run, j in gens:
        _save_preds(run, j.get())


@app.local_entrypoint()
def split(name: str = "b", sides: str = "out,in"):
    run = ARMS[name]["run"]
    gens = []
    for side in sides.split(","):
        info = build_split.remote(run, side)
        _log_cost(info["cost"])
        print(info)
        gens.append((f"{run}_split_{side}", generate.spawn(_read(EVAL), f"{run}_split_{side}")))
    for r, j in gens:
        _save_preds(r, j.get())
