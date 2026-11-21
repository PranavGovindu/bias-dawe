"""formatter exps gp: an independent replication of bandr-ai/dawe on Modal (ephemeral `modal run` only).

Adapted from bandr-ai/dawe@8f11c96 gpu/app.py (SFT, vLLM generation, weight_map) and gpu/lab.py (decision_map);
Apache-2.0, see ../NOTICE. Self-contained: no `dawe` import; the pins are inlined; data files are prebuilt with
`dawe build-*` and live in ../data/. Their app and volumes are "geodesic*"; ours default to the names below and can be
changed with environment variables in the launching shell (they are also baked into the image env, so the in-container
re-import resolves the same names):
  app     formatter-exps-gp        (FEP_APP)
  volumes formatter-exps-gp-runs   (FEP_RUNS_VOLUME; checkpoints, maps, preds)
          formatter-exps-gp-hf     (FEP_HF_VOLUME; HF cache)
Local outputs (preds, logs, costs) go to ../runs/ (git-ignored); the committed results are in ../results/.

Changes from theirs (everything else is copied line for line):
  - `seed` argument for SFT (theirs hard-codes 0 in torch.manual_seed and the order generator).
  - generate(): `enforce_eager` argument (their app.py used False, their lab.py True; use ONE value for every condition
    you compare), and preds are also written to the volume.
  - weight_map(): `grads=False` skips the backward pass (the saved states are identical either way); True = theirs.
  - plan_map(): their lab.py::decision_map for kinds plan_start/after_plan, reading prebuilt v11_plan SFT rows instead
    of rendering from BFCL files in the container (same rows, same order).
  - Images: their top-level pins by default; FEP_IMAGE=freeze installs their recorded full freezes instead.
  - Timeouts tightened (train 25 min, generate 15, maps 20/15, fetch 20) to cap spend if something hangs.
  - Telemetry off everywhere (DAWE_TELEMETRY_DISABLED, DO_NOT_TRACK, VLLM_NO_USAGE_STATS, HF_HUB_DISABLE_TELEMETRY).

Commands (from this directory, code/; prefix every command with the env vars):
  export DAWE_TELEMETRY_DISABLED=1 DO_NOT_TRACK=1
  modal run formatter_exps_gp.py::prefetch                       # CPU: model into the HF volume
  modal run formatter_exps_gp.py::stage1 --run sft_v1_s0 --seed 0  # SFT + held-out eval of sft and base, one launch
  modal run formatter_exps_gp.py::sft --run sft_v1_s0 --seed 0   # LoRA SFT on v1 -> /runs/sft_v1_s0/merged
  modal run formatter_exps_gp.py::gen --model sft_v1_s0          # 204 x {v1, v4_format, v11_plan} -> runs/preds_sft_v1_s0.jsonl
  modal run formatter_exps_gp.py::state_map --run sft_v1_s0      # s1 states (v1/v4 continue/stop) + s2 states (plan)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent  # code/
REPO = ROOT.parent  # data/, results/ and runs/ live here
MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# Their Modal list prices (gpu/app.py, checked by them 2026-11-11), USD per second. Re-check modal.com/pricing.
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


# Image contents. Default "pins" = their gpu/app.py exactly (top-level pins; transitive versions resolve at build time,
# as theirs did). "freeze" = their recorded repro/freeze_*.txt (req_*.txt here). Those freezes were taken inside Modal
# containers and also list Modal's own runtime packages (grpclib, cbor2/certifi/multidict twice), so "freeze" may clash
# with the Modal runtime; use it only if "pins" drifts. Select with FEP_IMAGE=freeze in the launching shell.
IMAGE_MODE = os.environ.get("FEP_IMAGE", "pins")
TRAIN_PINS = ["torch==2.13.0", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0",
              "flash-linear-attention==0.5.2", "hf_transfer"]
VLLM_PINS = ["vllm==0.31.0", "hf_transfer"]


def _pkgs(kind: str) -> list[str]:
    # The image is built from the local definition; inside a running container this module is re-imported and the
    # requirement files are not mounted, so return nothing there (the image already exists).
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


def _cost(t0: float, label: str) -> dict:
    secs = time.time() - t0
    out = {"job": label, "seconds": round(secs), "usd_est": round(secs * RATE, 3)}
    print(f"[cost] {label}: {secs / 60:.1f} min ~ ${out['usd_est']:.2f}")
    return out


# --- model download (CPU only) -----------------------------------------------------------------


@app.function(image=train_image, cpu=2.0, memory=4096, volumes=VOLS, timeout=20 * 60)
def fetch_model() -> str:
    from huggingface_hub import snapshot_download

    p = snapshot_download(MODEL, revision=REVISION)
    hf_cache.commit()
    return p


# --- training (their gpu/app.py::train, plus `seed`) ---------------------------------------------


@app.function(image=train_image, timeout=25 * 60, **COMMON)
def train(rows: list[dict], run: str, epochs: float = 3, lr: float = 1e-4, rank: int = 32, max_steps: int = -1,
          seed: int = 0) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    t0 = time.time()
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).cuda()

    # LoRA on every Linear layer of the language model (not vision, not lm_head).
    targets = [
        n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "language_model" in n and "lm_head" not in n
    ]
    kinds = sorted({n.rsplit(".", 1)[-1] for n in targets})
    print(f"LoRA targets: {len(targets)} layers, kinds {kinds}")
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=targets))
    model.print_trainable_parameters()

    # Prompt = chat template with thinking off; loss only on the reply.
    examples = []
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(r["messages"][-1]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        examples.append((p_ids + c_ids, [-100] * len(p_ids) + c_ids))
    lens = sorted(len(x) for x, _ in examples)
    print(f"{len(examples)} examples, tokens median {lens[len(lens) // 2]}, max {lens[-1]}, total {sum(lens)}")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    batch, accum = 4, 4  # effective batch 16
    steps_per_epoch = max(1, len(examples) // (batch * accum))
    total = max_steps if max_steps > 0 else int(steps_per_epoch * epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / total))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    g = torch.Generator().manual_seed(seed)
    order: list[int] = []
    model.train()
    log = []
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(), sched.step(), opt.zero_grad()
        log.append(round(loss_sum, 4))
        if step % 5 == 0 or step == total - 1:
            print(f"step {step + 1}/{total} loss {loss_sum:.4f}")

    out = Path(f"/runs/{run}")
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    # vLLM needs the processor files that came with the base model.
    from huggingface_hub import hf_hub_download

    for f in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        (out / "merged" / f).write_bytes(Path(hf_hub_download(MODEL, f, revision=REVISION)).read_bytes())
    info = {"run": run, "steps": total, "loss": log, "lr": lr, "rank": rank, "epochs": epochs, "n": len(rows), "seed": seed}
    (out / "train.json").write_text(json.dumps(info, indent=2))
    runs.commit()
    info["cost"] = _cost(t0, f"train {run}")
    return info


# --- generation (their gpu/app.py::generate, plus enforce_eager and a copy on the volume) --------


@app.function(image=vllm_image, timeout=15 * 60, **COMMON)
def generate(rows: list[dict], model: str = "base", max_tokens: int = 512, enforce_eager: bool = False,
             tag: str = "") -> dict:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    if model == "base":
        path, rev = MODEL, REVISION
    else:
        runs.reload()
        path, rev = f"/runs/{model}/merged", None
    llm = LLM(
        model=path,
        revision=rev,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outs = llm.chat([r["messages"] for r in rows], params, chat_template_kwargs={"enable_thinking": False})
    preds = [{"id": r["id"], "harness": r["harness"], "text": o.outputs[0].text} for r, o in zip(rows, outs)]
    d = Path("/runs/preds")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"preds_{model}{tag}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    runs.commit()
    return {"preds": preds, "cost": _cost(t0, f"generate {model}{tag} x{len(rows)} eager={enforce_eager}")}


# --- hidden-state maps ---------------------------------------------------------------------------


def _decision_points(reply: str, fmt: str, calls_json: list[str] | None = None) -> list[tuple[int, bool]]:
    """(their gpu/app.py) Character offsets right after each tool call in a reply, and whether another call should follow."""
    ends = []
    if fmt == "xml":
        i = 0
        while (j := reply.find("</tool_call>", i)) != -1:
            ends.append(j + len("</tool_call>"))
            i = ends[-1]
    else:  # json: find each serialized call object in order
        i = 0
        for cj in calls_json or []:
            j = reply.find(cj, i)
            if j == -1:
                return []
            ends.append(j + len(cj))
            i = ends[-1]
    return [(e, k < len(ends) - 1) for k, e in enumerate(ends)]


@app.function(image=train_image, timeout=20 * 60, **COMMON)
def weight_map(pairs: list[dict], a: str, b: str, run: str, grads: bool = False) -> dict:
    """Their gpu/app.py::weight_map. Saves /runs/<run>/maps/states_<h>.pt = {'continue': [n, 33, 2560], 'stop': ...} in fp16
    (index 0 = embeddings, -1 = final post-norm state). grads=True also saves their whole-reply gradients."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    t0 = time.time()
    runs.reload()
    path = f"/runs/{run}/merged"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lm_params = {n: p for n, p in model.named_parameters() if "language_model" in n and p.dim() == 2}
    for n, p in model.named_parameters():
        p.requires_grad_(grads and n in lm_params)

    def encode(msgs):
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        reply = msgs[-1]["content"]
        enc = tok(prompt + reply + "<|im_end|>", add_special_tokens=False, return_offsets_mapping=True)
        return prompt, reply, enc

    out: dict = {"a": a, "b": b, "run": run, "n_pairs": len(pairs)}
    Path(f"/runs/{run}/maps").mkdir(parents=True, exist_ok=True)
    for h in (a, b):
        fmt = "xml" if "<tool_call" in pairs[0][h][-1]["content"] else "json"
        model.zero_grad(set_to_none=True)
        states = {True: [], False: []}  # continue? -> list of [layers, hidden]
        nll = {True: [], False: []}
        for pr in pairs:
            msgs = pr[h]
            prompt, reply, enc = encode(msgs)
            ids = torch.tensor([enc["input_ids"]]).cuda()
            offs = enc["offset_mapping"]
            start = next(i for i, (s, e) in enumerate(offs) if e > len(prompt))
            labels = ids.clone()
            labels[0, :start] = -100
            with torch.set_grad_enabled(grads):
                res = model(input_ids=ids, labels=labels, output_hidden_states=True)
                if grads:
                    res.loss.backward()
            calls_json = None
            if fmt == "json":
                calls_json = [json.dumps(c, ensure_ascii=False) for c in json.loads(reply)["tool_calls"]]
            with torch.no_grad():
                logp = torch.log_softmax(res.logits[0].float(), -1)
                for ce, cont in _decision_points(reply, fmt, calls_json):
                    abs_end = len(prompt) + ce
                    d = max(i for i, (s, e) in enumerate(offs) if e <= abs_end)
                    if d + 1 >= ids.shape[1]:
                        continue
                    states[cont].append(torch.stack([hs[0, d].float() for hs in res.hidden_states]).cpu())
                    nll[cont].append(-logp[d, ids[0, d + 1]].item())
        if grads:
            g_ = {n: p.grad.detach().to(torch.bfloat16).cpu().contiguous() for n, p in lm_params.items() if p.grad is not None}
            save_file(g_, f"/runs/{run}/maps/grad_{h}.safetensors")
        out[h] = {"n_continue": len(states[True]), "n_stop": len(states[False]),
                  "nll_continue": sum(nll[True]) / max(1, len(nll[True])), "nll_stop": sum(nll[False]) / max(1, len(nll[False]))}
        if states[True] and states[False]:
            C, S = torch.stack(states[True]), torch.stack(states[False])  # [n, layers, d]
            torch.save({"continue": C.half(), "stop": S.half()}, f"/runs/{run}/maps/states_{h}.pt")
    runs.commit()
    out["cost"] = _cost(t0, f"map {run} {a} vs {b}")
    return out


def points_for(reply: str, fmt: str) -> list[tuple[int, str]]:
    """(their gpu/lab.py) Decision points in a gold reply, as (char offset, kind)."""
    pts: list[tuple[int, str]] = []
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


@app.function(image=train_image, timeout=15 * 60, **COMMON)
def plan_map(rows: list[dict], h: str, run: str, kinds: list[str], tag: str = "plan_") -> dict:
    """Their gpu/lab.py::decision_map (multi_only=False), states only: hidden states at the given decision kinds of
    the gold reply. rows = prebuilt SFT rows for harness h (same items and order as their BFCL loop).
    Saves /runs/<run>/maps/states_dp_<tag><h>.pt = {kind: [n, 33, 2560]} (fp16)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    t0 = time.time()
    runs.reload()
    path = f"/runs/{run}/merged"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    fmt = "xml" if "<tool_call" in rows[0]["messages"][-1]["content"] else "json"
    states: dict = {}
    nll: dict = {}
    with torch.no_grad():
        for r in rows:
            msgs, reply = r["messages"][:-1], r["messages"][-1]["content"]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(prompt + reply + "<|im_end|>", add_special_tokens=False, return_offsets_mapping=True)
            offs = enc["offset_mapping"]
            pts, pk = [], []
            for ce, kind in points_for(reply, fmt):
                if kind in kinds:
                    pts.append(max(i for i, (s, e) in enumerate(offs) if e <= len(prompt) + ce))
                    pk.append(kind)
            if not pts:
                continue
            ids = torch.tensor([enc["input_ids"]]).cuda()
            res = model(input_ids=ids, output_hidden_states=True)
            lp = torch.log_softmax(res.logits[0][pts].float(), -1)
            gold = ids[0, [d + 1 for d in pts]]
            for j, (d, kind) in enumerate(zip(pts, pk)):
                nll.setdefault(kind, []).append(-lp[j, gold[j]].item())
                states.setdefault(kind, []).append(torch.stack([hs[0, d].float() for hs in res.hidden_states]).half().cpu())
    Path(f"/runs/{run}/maps").mkdir(parents=True, exist_ok=True)
    torch.save({k: torch.stack(v) for k, v in states.items()}, f"/runs/{run}/maps/states_dp_{tag}{h}.pt")
    runs.commit()
    out = {"h": h, "n_by_kind": {k: len(v) for k, v in states.items()}, "nll_by_kind": {k: sum(v) / len(v) for k, v in nll.items()}}
    out["cost"] = _cost(t0, f"plan_map {run} {h}")
    return out


# --- local entry points ------------------------------------------------------------------------


def _read(p: str) -> list[dict]:
    return [json.loads(line) for line in (REPO / p).read_text().splitlines() if line.strip()]


def _log(name: str, obj: dict) -> None:
    (REPO / "runs" / name).parent.mkdir(parents=True, exist_ok=True)
    (REPO / "runs" / name).write_text(json.dumps(obj, indent=1))
    with (REPO / "runs" / "costs.jsonl").open("a") as f:
        f.write(json.dumps(obj["cost"]) + "\n")


@app.local_entrypoint()
def prefetch():
    print(fetch_model.remote())


@app.local_entrypoint()
def sft(run: str = "sft_v1_s0", seed: int = 0, epochs: float = 3, max_steps: int = -1, harness: str = "v1"):
    info = train.remote(_read(f"data/sft/{harness}_bfcl_train.jsonl"), run, epochs=epochs, max_steps=max_steps, seed=seed)
    _log(f"train_{run}.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "loss"}, indent=2), "loss first/last", info["loss"][:3], info["loss"][-3:])


@app.local_entrypoint()
def gen(model: str = "sft_v1_s0", eval_file: str = "data/eval/bfcl_test_3h.jsonl", eager: bool = False, tag: str = ""):
    res = generate.remote(_read(eval_file), model, enforce_eager=eager, tag=tag)
    out = REPO / "runs" / f"preds_{model}{tag}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(p) + "\n" for p in res["preds"]))
    with (REPO / "runs" / "costs.jsonl").open("a") as f:
        f.write(json.dumps(res["cost"]) + "\n")
    print("saved", out, res["cost"])


@app.local_entrypoint()
def state_map(run: str = "sft_v1_s0", grads: bool = False):
    """s1 states (weight_map on v1__v4_format pairs) and s2 plan states (decision_map, kinds plan_start/after_plan)."""
    j1 = weight_map.spawn(_read("data/pairs/v1__v4_format.jsonl"), "v1", "v4_format", run, grads)
    j2 = plan_map.spawn(_read("data/sft/v11_plan_bfcl_train.jsonl"), "v11_plan", run, ["plan_start", "after_plan"], "plan_")
    _log(f"map_v1__v4_format_{run}.json", j1.get())
    _log(f"map_plan_{run}.json", j2.get())
    print("maps saved on the volume under /runs/%s/maps (download: modal volume get %s %s/maps)" % (run, RUNS_VOLUME, run))


@app.local_entrypoint()
def stage1(run: str = "sft_v1_s0", seed: int = 0, eval_file: str = "data/eval/bfcl_test_3h.jsonl", eager: bool = False,
           with_base: bool = True):
    """Stage 1 in one launch (their gpu/app.py::full): SFT on v1, then held-out generation for the SFT model and,
    in parallel, the base model (pipeline check against their `base` row: v1 71.6 / v4 81.4 / v11 11.3)."""
    info = train.remote(_read("data/sft/v1_bfcl_train.jsonl"), run, seed=seed)
    _log(f"train_{run}.json", info)
    print("loss first/last", info["loss"][:3], info["loss"][-3:])
    rows = _read(eval_file)
    calls = {run: generate.spawn(rows, run, enforce_eager=eager)}
    if with_base:
        calls["base"] = generate.spawn(rows, "base", enforce_eager=eager)
    for name, c in calls.items():
        res = c.get()
        out = REPO / "runs" / f"preds_{name}.jsonl"
        out.write_text("".join(json.dumps(p) + "\n" for p in res["preds"]))
        with (REPO / "runs" / "costs.jsonl").open("a") as f:
            f.write(json.dumps(res["cost"]) + "\n")
        print("saved", out, res["cost"])


# --- stage 2: all arms in one H100 container (STAGE2_SPEC.md) -----------------------------------------
# arms.py (next to this file) does the work: one working copy of the merged shard on local disk, bytes patched in place
# per arm, a fresh vLLM engine per arm in a subprocess (same LLM settings as generate(), eager=False), edited rows read
# back from the loaded model and compared bitwise before generating. Preds go to /runs/preds/<tag>/ after each arm, and a
# rerun skips arms already there.

arms_image = vllm_image.add_local_file(ROOT / "arms.py", "/root/arms.py")
ARMS_MEM_GIB = 32
ARMS_RATE = PRICE["H100"] + CPU * PRICE["cpu_core"] + ARMS_MEM_GIB * PRICE["mem_gib"]

ARM_SPECS = {
    "B": {"name": "B", "kind": "bias", "passes": [
        {"tag": "B0", "biases": [None]},  # unbiased, same engine: noise floor vs the stage-1 preds
        {"tag": "B", "biases": [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -16.0]}]},
    "A": {"name": "A", "kind": "row", "delta": "A", "expect_row_key": "A_row"},
    "C": {"name": "C", "kind": "row", "delta": "C", "expect_row_key": "C_row"},
    "D": {"name": "D", "kind": "row", "delta": "D", "expect_row_key": "D_row"},
    "E": {"name": "E", "kind": "untied", "delta": "A"},
    "F25": {"name": "F25", "kind": "lora_scale", "alpha": 0.25},
    "F50": {"name": "F50", "kind": "lora_scale", "alpha": 0.5},
    "F75": {"name": "F75", "kind": "lora_scale", "alpha": 0.75},
}


@app.function(image=arms_image, gpu=GPU, cpu=CPU, memory=ARMS_MEM_GIB * 1024, volumes=VOLS, scaledown_window=2,
              timeout=35 * 60)
def gen_arms(rows: list[dict], arms: list[dict], deltas_pt: bytes, run: str = "sft_v1_s0", tag: str = "stage2",
             max_tokens: int = 512, eager: bool = False, budget_min: float = 28, expect_lora: int | None = 248,
             arm_timeout_min: float = 10) -> dict:
    import io
    import sys

    import torch

    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # apply_model read-back of the edited rows
    sys.path.insert(0, "/root")
    import arms as A

    t0 = time.time()
    runs.reload()
    out = Path(f"/runs/preds/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    done = {p.stem[len("preds_"):] for p in out.glob("preds_*.jsonl")}
    rp = Path("/tmp/s2/rows.jsonl")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    deltas = torch.load(io.BytesIO(deltas_pt))
    base = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots") / REVISION
    assert base.exists(), base
    log = A.run_arms(src=Path(f"/runs/{run}/merged"), base=base, work=Path("/tmp/s2/ckpt"), out=out, rows=rp,
                     deltas=deltas, arms=arms, eager=eager, max_tokens=max_tokens, expect_lora=expect_lora,
                     device="cuda", deadline=t0 + budget_min * 60, arm_timeout=int(arm_timeout_min * 60), done=done,
                     on_done=lambda name: runs.commit(), expect_base_row=A.int_bits(deltas["base_row"]))
    runs.commit()
    preds = {p.stem[len("preds_"):]: p.read_text() for p in sorted(out.glob("preds_*.jsonl"))}
    preds |= {p.stem: p.read_text() for p in sorted(out.glob("partial_*.jsonl"))}
    secs = time.time() - t0
    log["cost"] = {"job": f"gen_arms {tag} {','.join(a['name'] for a in arms)}", "seconds": round(secs),
                   "usd_est": round(secs * ARMS_RATE, 3)}
    print(f"[cost] gen_arms: {secs / 60:.1f} min ~ ${log['cost']['usd_est']:.2f}")
    return {"log": log, "preds": preds}


@app.local_entrypoint()
def stage2(arms: str = "B,A,C,D,E,F25,F50,F75", deltas: str = "results/deltas/deltas_stage2.pt", run: str = "sft_v1_s0",
           tag: str = "stage2", budget_min: float = 28, eval_file: str = "data/eval/bfcl_test_3h.jsonl"):
    res = gen_arms.remote(_read(eval_file), [ARM_SPECS[a] for a in arms.split(",")], (REPO / deltas).read_bytes(),
                          run=run, tag=tag, budget_min=budget_min)
    d = REPO / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    for name, text in res["preds"].items():
        (d / f"preds_{name}.jsonl").write_text(text)
    _log(f"{tag}/gen_arms_{time.strftime('%H%M%S')}.json", res["log"])
    print(json.dumps({k: v for k, v in res["log"].items() if k != "arms"}, indent=1))
    for name, m in res["log"]["arms"].items():
        print(name, {k: m.get(k) for k in ("rc", "skipped", "engine_s", "gen_s", "wall_s", "checks")})


# --- seed-1 replicate ----------------------------------------------------------------------------------------------
# The seed-1 plain LoRA (prevent.py arm a1) kept only its adapter at /runs/prev_a_s1/adapter. Re-merge it exactly as train()
# does (bf16 base on GPU, adapter loaded by PEFT in fp32, merge_and_unload, save_pretrained + tokenizer + processor
# files) into /runs/<run>/merged (ours: dawe_s1; deleted when done), then run weight_map on it.

@app.function(image=train_image, timeout=15 * 60, **COMMON)
def merge_adapter(src: str = "prev_a_s1", run: str = "dawe_s1") -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    assert run.startswith("dawe_"), run
    t0 = time.time()
    runs.reload()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(model, f"/runs/{src}/adapter")
    lora_dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n})
    merged = model.merge_and_unload()
    E, H = merged.get_input_embeddings().weight, merged.get_output_embeddings().weight
    assert E.data_ptr() == H.data_ptr(), "tie broken after merge"
    assert not [k for k in merged.state_dict() if "lora_" in k or "base_layer" in k]
    out = Path(f"/runs/{run}")
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    from huggingface_hub import hf_hub_download

    for f in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        (out / "merged" / f).write_bytes(Path(hf_hub_download(MODEL, f, revision=REVISION)).read_bytes())
    from safetensors import safe_open

    def keys(d):
        ks = set()
        for fp in sorted(Path(d).glob("*.safetensors")):
            with safe_open(str(fp), "pt") as fh:
                ks |= set(fh.keys())
        return ks

    mine, ref = keys(out / "merged"), keys("/runs/sft_v1_s0/merged")
    assert mine == ref, (sorted(mine - ref)[:5], sorted(ref - mine)[:5])
    runs.commit()
    info = {"src": src, "run": run, "lora_param_dtypes": lora_dtypes, "n_tensors": len(mine),
            "files": sorted(p.name for p in (out / "merged").iterdir())}
    info["cost"] = _cost(t0, f"merge {src} -> {run}")
    return info


@app.local_entrypoint()
def s1_merge_map(src: str = "prev_a_s1", run: str = "dawe_s1"):
    info = merge_adapter.remote(src, run)
    _log(f"stage2_s1/merge_{run}.json", info)
    print(json.dumps(info, indent=1))
    res = weight_map.remote(_read("data/pairs/v1__v4_format.jsonl"), "v1", "v4_format", run, False)
    _log(f"stage2_s1/map_v1__v4_format_{run}.json", res)
    print(json.dumps(res, indent=1))


ARM_SPECS["B_s1"] = {"name": "B", "kind": "bias", "passes": [
    {"tag": "B0tr", "biases": [None], "split": "train"},  # rule reference: v1 train accuracy at no bias
    {"tag": "Btr", "biases": [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0], "split": "train"},  # b chosen here (PREDICTIONS.md)
    {"tag": "B0", "biases": [None]},  # unbiased test pass: U regenerated (check vs prevent_01's a1 preds)
    {"tag": "B", "biases": [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -16.0], "harnesses": ["v1", "v4_format"]}]}


def _train_rows(harnesses=("v1", "v4_format")) -> list[dict]:
    rows = []
    for h in harnesses:
        for r in _read(f"data/sft/{h}_bfcl_train.jsonl"):
            rows.append({"id": r["id"], "harness": h, "split": "train", "messages": r["messages"][:-1]})
    return rows


@app.local_entrypoint()
def stage2_s1(arms: str = "B_s1,A,D", deltas: str = "results/deltas/deltas_s1.pt", run: str = "dawe_s1",
              tag: str = "stage2_s1", budget_min: float = 20, eval_file: str = "data/eval/bfcl_test_3h.jsonl"):
    rows = _read(eval_file) + _train_rows()
    res = gen_arms.remote(rows, [ARM_SPECS[a] for a in arms.split(",")], (REPO / deltas).read_bytes(),
                          run=run, tag=tag, budget_min=budget_min, expect_lora=None)
    d = REPO / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    for name, text in res["preds"].items():
        (d / f"preds_{name}.jsonl").write_text(text)
    _log(f"{tag}/gen_arms_{time.strftime('%H%M%S')}.json", res["log"])
    for name, m in res["log"]["arms"].items():
        print(name, {k: m.get(k) for k in ("rc", "skipped", "engine_s", "gen_s", "wall_s", "checks")})


# --- stage 3 (STAGE3_SPEC.md): plan-first two-row patch vs constant biases, both seeds ----------------

@app.local_entrypoint()
def s3_merge_planmap(src: str = "prev_a_s1", run: str = "dawe_s1"):
    """Launch 1: re-merge seed-1 plain LoRA (same as stage2_s1) and map its plan_start/after_plan states."""
    info = merge_adapter.remote(src, run)
    _log(f"stage3/merge_{run}.json", info)
    print(json.dumps(info, indent=1))
    res = plan_map.remote(_read("data/sft/v11_plan_bfcl_train.jsonl"), "v11_plan", run, ["plan_start", "after_plan"],
                          "plan_")
    _log(f"stage3/map_plan_{run}.json", res)
    print(json.dumps(res, indent=1))


GRID_ALL = [[c, b] for c in (0, 2, 4, 8) for b in (-4, -8, -12)]  # B2: logit_bias {27: c, 248046: b} at every position
GRID_FIRST = [[c, b] for c in (2, 4, 8) for b in (-4, -8, -12)]  # B2-first (c = 0 is B2 c = 0)


def _swap(tag, r27, reos):
    exp = {k: v + "_row" for k, v in (("27", r27), ("248046", reos)) if v}
    return {"type": "swap", "tag": tag, "rows": {"27": r27, "248046": reos}, "expect": exp}


S3_SPEC = {"name": "S3", "kind": "bias", "steps": [
    # weight arms first, by in-place row swaps in the one engine (+ prefix-cache reset after each swap)
    _swap("A1chk", None, "A1"), {"type": "pass", "tag": "A1chk", "biases": [None]},  # = stage-2 A preds? (swap check)
    _swap("A2", "A2_27", "A2_eos"), {"type": "pass", "tag": "A2", "biases": [None]},
    _swap("D2", "D2_27", "D2_eos"), {"type": "pass", "tag": "D2", "biases": [None]},
    _swap("C2", "C2_27", "C2_eos"), {"type": "pass", "tag": "C2", "biases": [None]},
    _swap("U", None, None), {"type": "pass", "tag": "Uchk", "biases": [None]},  # = U preds? (restore check)
    # constants: train first (the selection rule needs them), then the test grids, then the two-pass check
    {"type": "pass", "tag": "B2tr", "grid": GRID_ALL, "mode": "all", "split": "train"},
    {"type": "pass", "tag": "B2ftr", "grid": GRID_FIRST, "mode": "first", "split": "train"},
    {"type": "pass", "tag": "B2te", "grid": GRID_ALL, "mode": "all"},
    {"type": "pass", "tag": "B2fte", "grid": GRID_FIRST, "mode": "first"},
    {"type": "pass", "tag": "B2fchk", "grid": [[0, -8]], "mode": "first"},  # two-pass at c = 0 == B2te (0, -8)?
]}


@app.local_entrypoint()
def stage3(seeds: str = "s0,s1", arm_timeout_min: float = 15, eval_file: str = "data/eval/bfcl_test_3h.jsonl"):
    """Launch 2: both seeds in parallel, one H100 container and one vLLM engine each."""
    run_of = {"s0": "sft_v1_s0", "s1": "dawe_s1"}
    rows = _read(eval_file) + _train_rows(("v1", "v4_format", "v11_plan"))
    calls = {}
    for s in seeds.split(","):
        calls[s] = gen_arms.spawn(rows, [S3_SPEC], (REPO / f"results/deltas/deltas_s3_{s}.pt").read_bytes(), run=run_of[s],
                                  tag=f"stage3_{s}", budget_min=30, expect_lora=None, arm_timeout_min=arm_timeout_min)
    for s, c in calls.items():
        try:
            res = c.get()
        except Exception as e:  # keep the other seed
            print(f"[stage3 {s}] FAILED: {e!r}")
            continue
        d = REPO / "runs" / "stage3" / s
        d.mkdir(parents=True, exist_ok=True)
        for name, text in res["preds"].items():
            (d / f"{name if name.startswith('partial_') else 'preds_' + name}.jsonl").write_text(text)
        _log(f"stage3/{s}/gen_arms_{time.strftime('%H%M%S')}.json", res["log"])
        print(s, {k: v for k, v in res["log"].items() if k != "arms"})
        for name, m in res["log"]["arms"].items():
            print(s, name, {k: m.get(k) for k in ("rc", "engine_s", "gen_s", "wall_s", "swaps", "complete")})


# --- neartie_01 step 2 (one launch; ../prereg/neartie_PREDICTIONS.md Amendment 4) ---------------------------------
# (1) DAWE's v1 SFT, identical code and config: a seed-0 rerun (primary: seed vs run-to-run numerics) plus seeds 2-4,
#     each followed by the unpatched held-out generation (612 prompts, eager=False, as stage 1). Plan-start / after-plan /
#     JSON reply-start states come afterwards from the adapters on a local RTX 3060 (margins/reply_states.py), not here.
# (2) Seed-1 out-of-sample first-token constant: re-merge prev_a_s1 -> dawe_s1, then one engine runs the stage-3
#     two-pass first-token decode at (c, b) = (8, -8) (control, = stage-3 B2-first) and (12, -8) (c fixed by rule before
#     launch: ceil of seed 1's train c95 = 11.14) on the 204 x 3 test prompts.
# Nothing existing is overwritten: run names must be new sft_v1_s* names; dawe_s1/merged is recreated (deleted after
# stage 3) and can be deleted again afterwards.

S2B_SPEC = {"name": "S2B", "kind": "bias", "steps": [
    {"type": "pass", "tag": "B2fte", "grid": [[8, -8], [12, -8]], "mode": "first"}]}


@app.local_entrypoint()
def step2(runs_spec: str = "sft_v1_s0r:0,sft_v1_s2:2,sft_v1_s3:3,sft_v1_s4:4", s1_arm: bool = True,
          eval_file: str = "data/eval/bfcl_test_3h.jsonl"):
    pairs = [(r, int(s)) for r, s in (x.split(":") for x in runs_spec.split(",") if x)]
    for r, _ in pairs:
        assert r.startswith("sft_v1_s") and r not in ("sft_v1_s0",), f"refusing run name {r}"
    rows = _read(eval_file)
    d = REPO / "runs" / "step2"
    d.mkdir(parents=True, exist_ok=True)
    tr = {r: train.spawn(_read("data/sft/v1_bfcl_train.jsonl"), r, seed=s) for r, s in pairs}
    print("[step2] training spawned:", {r: s for r, s in pairs}, flush=True)
    s1 = None
    if s1_arm:
        try:  # a failure here must not end the entrypoint (that would tear down the four trains in flight)
            info = merge_adapter.remote("prev_a_s1", "dawe_s1")
            _log("step2/merge_dawe_s1.json", info)
            print("[step2] merged dawe_s1:", info.get("n_tensors"), info["cost"], flush=True)
            s1 = gen_arms.spawn(rows, [S2B_SPEC], (REPO / "results/deltas/deltas_s3_s1.pt").read_bytes(), run="dawe_s1",
                                tag="step2_s1", budget_min=20, expect_lora=None, arm_timeout_min=15)
        except Exception as e:
            print(f"[step2] seed-1 merge/arm launch FAILED, arm skipped: {e!r}", flush=True)
            s1 = None
    gens = {}
    for r, c in tr.items():
        try:
            info = c.get()
        except Exception as e:  # keep the other runs
            print(f"[step2] train {r} FAILED: {e!r}", flush=True)
            continue
        try:
            _log(f"step2/train_{r}.json", info)
        except Exception as e:
            print(f"[step2] log train {r} failed: {e!r}", flush=True)
        print(f"[step2] trained {r}: loss first/last {info['loss'][:2]} {info['loss'][-2:]}", info["cost"], flush=True)
        gens[r] = generate.spawn(rows, r)
    for r, c in gens.items():
        try:
            res = c.get()
        except Exception as e:
            print(f"[step2] generate {r} FAILED: {e!r}", flush=True)
            continue
        (d / f"preds_{r}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in res["preds"]))
        with (REPO / "runs" / "costs.jsonl").open("a") as f:
            f.write(json.dumps(res["cost"]) + "\n")
        print(f"[step2] saved preds_{r}.jsonl", res["cost"], flush=True)
    if s1 is not None:
        try:
            res = s1.get()
            ds = d / "s1"
            ds.mkdir(parents=True, exist_ok=True)
            for name, text in res["preds"].items():
                (ds / f"{name if name.startswith('partial_') else 'preds_' + name}.jsonl").write_text(text)
            _log(f"step2/s1/gen_arms_{time.strftime('%H%M%S')}.json", res["log"])
            print("[step2] seed-1 arm:", {k: v for k, v in res["log"].items() if k != "arms"}, flush=True)
        except Exception as e:
            print(f"[step2] seed-1 arm FAILED: {e!r}", flush=True)
