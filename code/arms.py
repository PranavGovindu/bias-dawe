"""Stage-2 arm runner (DAWE replication). Shared by local dry runs and the Modal function gen_arms; no modal import.

Layout: one working copy of the shard(s) any arm edits, every other file a symlink to the source checkpoint. Per arm:
patch bytes in place (safetensors offsets), read them back, start a fresh vLLM engine in a subprocess (clean GPU release),
read the edited rows back out of the loaded model with apply_model and compare bitwise, then generate.

Arms (kinds):
  bias        unpatched source dir; passes of SamplingParams(logit_bias={248046: b}) (b=None -> no bias, stage-1 config)
  row         row 248046 of the tied embedding := bf16(fp32(row) + delta)               (A, C, D)
  untied      config tie_word_embeddings=false (top + text_config), explicit lm_head = embed copy with row 248046 :=
              bf16(fp32(row) + delta); embed_tokens unchanged                          (E)
  lora_scale  W = bf16(base + alpha * (merged - base)) in fp32 for every tensor that differs from base   (F)

Engine subprocess: python arms.py engine <job.json>
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

EOS = 248046
EMB = "model.language_model.embed_tokens.weight"
LMH = "lm_head.weight"


# --- safetensors byte-level helpers ------------------------------------------------------------------

def st_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def weight_files(d: Path) -> dict[str, str]:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    wm = {}
    for p in sorted(d.glob("*.safetensors")):
        for k in st_header(p)[1]:
            wm[k] = p.name
    return wm


def _span(path: Path, name: str, row: int | None = None) -> tuple[int, int, dict]:
    base, h = st_header(path)
    e = h[name]
    assert e["dtype"] == "BF16", (name, e["dtype"])
    s, t = e["data_offsets"]
    if row is None:
        return base + s, t - s, e
    V, H = e["shape"]
    assert 0 <= row < V
    return base + s + row * H * 2, H * 2, e


def read_row(path: Path, name: str, row: int):
    import torch

    off, n, _ = _span(path, name, row)
    with open(path, "rb") as f:
        f.seek(off)
        b = f.read(n)
    return torch.frombuffer(bytearray(b), dtype=torch.bfloat16).clone()


def write_bytes(path: Path, off: int, data: bytes) -> None:
    with open(path, "r+b") as f:
        f.seek(off)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _bits(t) -> bytes:
    import torch

    assert t.dtype == torch.bfloat16
    return t.detach().contiguous().cpu().view(torch.int16).numpy().tobytes()


def write_row(path: Path, name: str, row: int, val) -> None:
    off, n, _ = _span(path, name, row)
    data = _bits(val)
    assert len(data) == n
    write_bytes(path, off, data)
    assert _bits(read_row(path, name, row)) == data, "row read-back mismatch"


def write_tensor(path: Path, name: str, t) -> None:
    off, n, e = _span(path, name)
    assert list(t.shape) == e["shape"], (name, t.shape, e["shape"])
    data = _bits(t)
    assert len(data) == n
    write_bytes(path, off, data)


def read_tensor_bytes(path: Path, name: str) -> bytes:
    off, n, _ = _span(path, name)
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(n)


def int_bits(t) -> list[int]:
    import torch

    return t.detach().cpu().contiguous().view(torch.int16).tolist()


# --- engine (runs in a subprocess) -------------------------------------------------------------------

def select_rows(rows: list[dict], ps: dict) -> list[dict]:
    """Rows for one pass: its split (rows without a split are test rows; default pass split is test) and harnesses."""
    split = ps.get("split", "test")
    rs = [r for r in rows if r.get("split", "test") == split]
    harn = set(ps.get("harnesses") or [r["harness"] for r in rs])
    return [r for r in rs if r["harness"] in harn]


def _probe(model):
    """Inside the vLLM worker: the <|im_end|> row of embed_tokens and lm_head as bf16 bit patterns."""
    import torch

    named = dict(model.named_parameters(remove_duplicate=False))
    emb = [n for n in named if n.endswith("embed_tokens.weight") and "visual" not in n and "mtp" not in n]
    lmh = [n for n in named if n.endswith("lm_head.weight")]
    out = {"emb_names": emb, "lmh_names": lmh}
    E = named[emb[0]]
    out["emb_shape"] = list(E.shape)
    out["emb_dtype"] = str(E.dtype)
    out["emb_row"] = E[EOS].detach().view(torch.int16).cpu().tolist()
    out["emb_row7"] = E[7].detach().view(torch.int16).cpu().tolist()
    if lmh:
        L = named[lmh[0]]
        out["lmh_shape"] = list(L.shape)
        out["lmh_row"] = L[EOS].detach().view(torch.int16).cpu().tolist()
        out["lmh_row7"] = L[7].detach().view(torch.int16).cpu().tolist()
        out["tied"] = L.data_ptr() == E.data_ptr()
    else:  # no separate lm_head parameter: the readout is the embedding itself
        out["tied"] = True
    return out


def engine(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    t0 = time.time()
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(job["rows"]) if l.strip()]
    llm = LLM(model=job["model"], max_model_len=job.get("max_model_len", 8192),
              gpu_memory_utilization=job.get("util", 0.85), enforce_eager=job["eager"],
              limit_mm_per_prompt={"image": 0, "video": 0}, seed=0, **job.get("extra", {}))
    t1 = time.time()
    meta = {"arm": job["arm"], "engine_s": round(t1 - t0, 1)}
    pr = llm.apply_model(_probe)[0]
    exp = job["expect"]
    checks = {
        "emb_row_ok": pr["emb_row"] == exp["emb_row"],
        "tied": pr.get("tied"),
        "tied_ok": pr.get("tied") == exp["tied"],
        "emb_shape": pr["emb_shape"], "emb_dtype": pr["emb_dtype"], "lmh_names": pr["lmh_names"],
    }
    if "lmh_row" in exp:
        checks["lmh_row_ok"] = pr.get("lmh_row") == exp["lmh_row"]
        checks["lmh_row7_eq_emb_row7"] = pr.get("lmh_row7") == pr["emb_row7"]
        checks["lmh_row_ne_emb_row"] = pr.get("lmh_row") != pr["emb_row"]
    meta["checks"] = checks
    ok = checks["emb_row_ok"] and checks["tied_ok"] and checks.get("lmh_row_ok", True) and \
        checks.get("lmh_row7_eq_emb_row7", True) and checks.get("lmh_row_ne_emb_row", True)
    print("[engine] checks", json.dumps(checks), flush=True)
    if not ok:
        meta["error"] = "post-load check failed"
        Path(job["meta"]).write_text(json.dumps(meta, indent=1))
        return 3
    out_f = Path(job["out"])
    out_f.write_text("")
    fin: dict = {}
    steps = job.get("steps") or [dict(ps, type="pass") for ps in job["passes"]]
    for st in steps:
        tp = time.time()
        if st["type"] == "swap":
            rows_bits = {int(k): v for k, v in st["rows"].items()}
            res = llm.apply_model(_make_setter(rows_bits))[0]
            reset = llm.reset_prefix_cache()
            ok = all(res.values())
            meta.setdefault("swaps", {})[st["tag"]] = {"rows_ok": res, "prefix_cache_reset": reset}
            print(f"[engine] swap {st['tag']}: rows_ok={res} reset={reset}", flush=True)
            if not ok:
                meta["error"] = f"swap {st['tag']} failed"
                Path(job["meta"]).write_text(json.dumps(meta, indent=1))
                return 4
            continue
        preds = run_pass(llm, SamplingParams, rows, st, job["max_tokens"])
        with open(out_f, "a") as f:
            f.write("".join(json.dumps(p) + "\n" for p in preds))
        for p in preds:
            k = f"{p['pass']}|{p.get('c')}|{p['bias']}|{p['finish']}"
            fin[k] = fin.get(k, 0) + 1
        meta.setdefault("gen_s", {})[st["tag"]] = round(time.time() - tp, 1)
        meta["finish_counts"] = fin
        meta["total_s"] = round(time.time() - t0, 1)
        Path(job["meta"]).write_text(json.dumps(meta, indent=1))
        print(f"[engine] pass {st['tag']} ({len(preds)} preds) {meta['gen_s'][st['tag']]}s", flush=True)
    meta["total_s"] = round(time.time() - t0, 1)
    meta["complete"] = True
    Path(job["meta"]).write_text(json.dumps(meta, indent=1))
    print("[engine] done", json.dumps({k: v for k, v in meta.items() if k not in ("checks", "finish_counts")}), flush=True)
    return 0


LT = 27  # '<'


def _make_setter(rows_bits: dict):
    """apply_model function: write bf16 bit patterns into embed_tokens rows (tied, so also the readout); read back."""
    def setter(model):
        import torch

        named = dict(model.named_parameters(remove_duplicate=False))
        E = named[[n for n in named if n.endswith("embed_tokens.weight") and "visual" not in n and "mtp" not in n][0]]
        out = {}
        with torch.no_grad():
            for t, bits in rows_bits.items():
                v = torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16).to(E.device)
                E[int(t)].copy_(v)
                out[int(t)] = E[int(t)].view(torch.int16).cpu().tolist() == bits
        return out
    return setter


def _rec(r, tag, c, b, text, finish, stop_reason, n_out, last_tok, mode):
    return {"id": r["id"], "harness": r["harness"], "split": r.get("split", "test"), "pass": tag, "mode": mode,
            "c": c, "bias": b, "text": text, "finish": finish, "stop_reason": stop_reason, "n_out": n_out,
            "last_tok": last_tok}


def run_pass(llm, SamplingParams, rows, ps, max_tokens):
    """One pass. 'biases' (stage 2): constant on <|im_end|> only. 'grid' with mode 'all': logit_bias {27: c, 248046: b}
    at every position (c = 0 -> no '<' entry). mode 'first': two-pass decode, the first reply token with
    {27: c, 248046: b}, then the rest with {248046: b} only."""
    sel = select_rows(rows, ps)
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    preds = []
    if "biases" in ps:
        msgs, params, keys = [], [], []
        for b in ps["biases"]:
            sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                                **({"logit_bias": {EOS: float(b)}} if b is not None else {}))
            for r in sel:
                msgs.append(r["messages"]); params.append(sp); keys.append((r, b))
        outs = llm.chat(msgs, params if len(ps["biases"]) > 1 else params[0], **kw)
        for (r, b), o in zip(keys, outs):
            c = o.outputs[0]
            preds.append(_rec(r, ps["tag"], None, b, c.text, c.finish_reason, c.stop_reason, len(c.token_ids),
                              c.token_ids[-1] if c.token_ids else None, "eos_only"))
        return preds
    grid = [(float(c), float(b)) for c, b in ps["grid"]]
    lb = lambda c, b: ({LT: c, EOS: b} if c else {EOS: b})
    if ps["mode"] == "all":
        msgs, params, keys = [], [], []
        for c, b in grid:
            sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logit_bias=lb(c, b))
            for r in sel:
                msgs.append(r["messages"]); params.append(sp); keys.append((r, c, b))
        outs = llm.chat(msgs, params, **kw)
        for (r, c, b), o in zip(keys, outs):
            x = o.outputs[0]
            preds.append(_rec(r, ps["tag"], c, b, x.text, x.finish_reason, x.stop_reason, len(x.token_ids),
                              x.token_ids[-1] if x.token_ids else None, "all"))
        return preds
    assert ps["mode"] == "first", ps["mode"]
    tok = llm.get_tokenizer()
    first = {}  # (c, b, i) -> (prompt ids, first token)
    for c in sorted({c for c, _ in grid}):
        bs = sorted({b for cc, b in grid if cc == c}, reverse=True)  # least negative first
        outs = llm.chat([r["messages"] for r in sel], SamplingParams(temperature=0.0, max_tokens=1, logit_bias=lb(c, bs[0])), **kw)
        redo = []
        for i, o in enumerate(outs):
            t1 = o.outputs[0].token_ids[0]
            for b in bs:  # b only changes <|im_end|>'s score: if it did not win at the least negative b, it never does
                first[(c, b, i)] = (list(o.prompt_token_ids), t1)
            if t1 == EOS:
                redo.append(i)
        for b in bs[1:]:
            if redo:
                outs2 = llm.chat([sel[i]["messages"] for i in redo],
                                 SamplingParams(temperature=0.0, max_tokens=1, logit_bias=lb(c, b)), **kw)
                for i, o in zip(redo, outs2):
                    first[(c, b, i)] = (list(o.prompt_token_ids), o.outputs[0].token_ids[0])
    prompts, params, keys = [], [], []
    for c, b in grid:
        sp2 = SamplingParams(temperature=0.0, max_tokens=max_tokens - 1, logit_bias={EOS: b})
        for i, r in enumerate(sel):
            pids, t1 = first[(c, b, i)]
            if t1 == EOS:
                preds.append(_rec(r, ps["tag"], c, b, "", "stop", None, 1, EOS, "first"))
                continue
            prompts.append({"prompt_token_ids": pids + [t1]}); params.append(sp2); keys.append((r, c, b, t1))
    outs = llm.generate(prompts, params) if prompts else []
    for (r, c, b, t1), o in zip(keys, outs):
        x = o.outputs[0]
        ids = [t1] + list(x.token_ids)
        preds.append(_rec(r, ps["tag"], c, b, tok.decode(ids, skip_special_tokens=True), x.finish_reason, x.stop_reason,
                          len(ids), ids[-1], "first"))
    return preds


# --- orchestration -----------------------------------------------------------------------------------

def run_arms(*, src: Path, base: Path, work: Path, out: Path, rows: Path, deltas: dict, arms: list[dict], eager: bool,
             max_tokens: int = 512, engine_kw: dict | None = None, expect_lora: int | None = None, device: str = "cpu",
             deadline: float | None = None, arm_timeout: int = 600, done: set | None = None, on_done=None,
             expect_base_row: list[int] | None = None) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    t_start = time.time()
    engine_kw = engine_kw or {}
    done = done or set()
    log: dict = {"arms": {}, "prep": {}}
    src_wm, base_wm = weight_files(src), weight_files(base)
    emb_file = src_wm[EMB]
    orig_row = read_row(src / emb_file, EMB, EOS)
    base_row = read_row(base / base_wm[EMB], EMB, EOS)
    assert torch.equal(orig_row.view(torch.int16), base_row.view(torch.int16)), "merged <|im_end|> row != base row"
    if expect_base_row is not None:
        assert int_bits(orig_row) == expect_base_row, "row differs from the PC-side base row"
    log["prep"]["row_eq_base"] = True
    log["prep"]["src_tensors"] = len(src_wm)

    # every source tensor must exist in base with the same shape/dtype (needed for F; cheap header check)
    base_hdr = {}
    for fn in sorted(set(base_wm.values())):
        base_hdr.update(st_header(base / fn)[1])
    src_hdr = {}
    for fn in sorted(set(src_wm.values())):
        src_hdr.update(st_header(src / fn)[1])
    missing = [k for k in src_hdr if k not in base_hdr]
    bad = [k for k in src_hdr if k in base_hdr and src_hdr[k]["shape"] != base_hdr[k]["shape"]]
    assert not missing and not bad, (missing[:5], bad[:5])
    # base keeps linear_attn.A_log / linear_attn.norm.weight in fp32; the SFT loaded and saved them in bf16 (LoRA never
    # touched them). For those, base is compared after casting to the merged dtype, and F leaves them as merged
    # (bf16(b + a*(bf16(b) - b)) == bf16(b) for any a in [0, 1]).
    dt = sorted(k for k in src_hdr if k in base_hdr and src_hdr[k]["dtype"] != base_hdr[k]["dtype"])
    log["prep"]["dtype_only_diff"] = {"n": len(dt), "kinds": sorted({".".join(k.split(".")[-2:]) for k in dt}),
                                      "pairs": sorted({f"{base_hdr[k]['dtype']}->{src_hdr[k]['dtype']}" for k in dt})}
    print("[prep] dtype-only differences vs base:", log["prep"]["dtype_only_diff"], flush=True)
    log["prep"]["base_only_tensors"] = sorted(k for k in base_hdr if k not in src_hdr)[:3] + ["..."]

    diff: list[str] = []
    if any(a["kind"] == "lora_scale" for a in arms if a["name"] not in done):
        t = time.time()
        fs = {fn: safe_open(str(src / fn), "pt") for fn in set(src_wm.values())}
        fb = {fn: safe_open(str(base / fn), "pt") for fn in set(base_wm.values())}
        for k in sorted(src_wm):
            m = fs[src_wm[k]].get_tensor(k).to(device)
            b = fb[base_wm[k]].get_tensor(k).to(device)
            if not torch.equal(m, b.to(m.dtype)):
                diff.append(k)
        del fs, fb
        bad = [k for k in diff if not ("language_model" in k and k.endswith(".weight") and k != EMB
                                       and len(src_hdr[k]["shape"]) == 2
                                       and src_hdr[k]["dtype"] == base_hdr[k]["dtype"] == "BF16")]
        assert not bad, bad[:5]
        if expect_lora is not None:
            assert len(diff) == expect_lora, f"{len(diff)} tensors differ from base, expected {expect_lora}"
        kinds = sorted({k.rsplit(".", 2)[-2] for k in diff})
        log["prep"]["lora_diff"] = {"n": len(diff), "kinds": kinds, "s": round(time.time() - t, 1)}
        print("[prep] tensors differing from base:", len(diff), kinds, flush=True)

    # working dir: copy the shard(s) any arm edits, symlink everything else
    edit_files = {emb_file} | {src_wm[k] for k in diff}
    t = time.time()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for p in sorted(src.iterdir()):
        if p.name in edit_files:
            shutil.copyfile(p, work / p.name)
        elif p.is_file():
            (work / p.name).symlink_to(p.resolve())
    log["prep"]["copy_s"] = round(time.time() - t, 1)
    log["prep"]["copied"] = sorted(edit_files)
    print("[prep] work dir ready", log["prep"], flush=True)

    state = {"row": "orig", "F": None}

    def set_row(val, tag):
        if state["row"] != tag:
            write_row(work / emb_file, EMB, EOS, val)
            state["row"] = tag
        assert torch.equal(read_row(work / emb_file, EMB, EOS).view(torch.int16), val.view(torch.int16))

    def set_F(alpha):
        if state["F"] == alpha:
            return
        fs = {fn: safe_open(str(src / fn), "pt") for fn in set(src_wm.values())}
        fb = {fn: safe_open(str(base / fn), "pt") for fn in set(base_wm.values())}
        checked = 0
        for k in diff:
            m = fs[src_wm[k]].get_tensor(k).to(device)
            if alpha is None:
                w = m
            else:
                b = fb[base_wm[k]].get_tensor(k).to(device)
                bf, mf = b.float(), m.float()
                w = (bf + alpha * (mf - bf)).to(torch.bfloat16)
                if state.get("F_checked") is None:  # once: alpha=0 -> base, alpha=1 -> merged, bitwise
                    mis0 = int(((bf + 0.0 * (mf - bf)).to(torch.bfloat16).view(torch.int16) != b.view(torch.int16)).sum())
                    mis1 = int(((bf + 1.0 * (mf - bf)).to(torch.bfloat16).view(torch.int16) != m.view(torch.int16)).sum())
                    state["F_mismatch"] = state.get("F_mismatch", 0) + mis0 + mis1
                    checked += 1
            write_tensor(work / src_wm[k], k, w)
            assert read_tensor_bytes(work / src_wm[k], k) == _bits(w), k
        if checked:
            state["F_checked"] = checked
            print(f"[prep] F identity check over {checked} tensors: {state['F_mismatch']} mismatched elements", flush=True)
            assert state["F_mismatch"] <= 1000, "alpha=0/1 do not reproduce base/merged: name mapping or arithmetic bug"
        del fs, fb
        state["F"] = alpha

    def restore_untied():
        for n in ("lm_head.safetensors",):
            if (work / n).exists():
                (work / n).unlink()
        for n in ("config.json", "model.safetensors.index.json"):
            p = work / n
            if p.exists() or p.is_symlink():
                p.unlink()
            if (src / n).exists():
                p.symlink_to((src / n).resolve())

    for a in arms:
        name = a["name"]
        if name in done:
            print(f"[arm {name}] already done, skipped", flush=True)
            log["arms"][name] = {"skipped": "done"}
            continue
        if deadline is not None and time.time() > deadline:
            print(f"[arm {name}] deadline passed, not run", flush=True)
            log["arms"][name] = {"skipped": "deadline"}
            continue
        t = time.time()
        expect = {"emb_row": int_bits(orig_row), "tied": True}
        model_dir = work
        passes = a.get("passes") or [{"tag": name, "biases": [None]}]
        if a["kind"] == "bias":
            model_dir = src
        elif a["kind"] == "row":
            if state["F"] is not None:
                set_F(None)
            restore_untied()
            new = (orig_row.float() + deltas[a["delta"]].float().cpu()).to(torch.bfloat16)
            if a.get("expect_row_key"):
                assert int_bits(new) == int_bits(deltas[a["expect_row_key"]]), f"{name}: row != expected row"
            set_row(new, name)
            expect["emb_row"] = int_bits(new)
        elif a["kind"] == "untied":
            if state["F"] is not None:
                set_F(None)
            set_row(orig_row, "orig")
            restore_untied()
            with safe_open(str(work / emb_file), "pt") as f:
                L = f.get_tensor(EMB).clone()
            new = (orig_row.float() + deltas[a["delta"]].float().cpu()).to(torch.bfloat16)
            L[EOS] = new
            save_file({LMH: L}, str(work / "lm_head.safetensors"), metadata={"format": "pt"})
            del L
            gc.collect()
            wm = dict(src_wm)
            wm[LMH] = "lm_head.safetensors"
            (work / "model.safetensors.index.json").unlink(missing_ok=True)
            (work / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}, indent=1))
            cfg = json.loads((src / "config.json").read_text())
            cfg["tie_word_embeddings"] = False
            cfg.setdefault("text_config", {})["tie_word_embeddings"] = False
            (work / "config.json").unlink()
            (work / "config.json").write_text(json.dumps(cfg, indent=2))
            expect = {"emb_row": int_bits(orig_row), "tied": False, "lmh_row": int_bits(new)}
        elif a["kind"] == "lora_scale":
            restore_untied()
            set_row(orig_row, "orig")
            set_F(float(a["alpha"]))
        else:
            raise ValueError(a["kind"])
        if device != "cpu":
            torch.cuda.empty_cache()
        steps = None
        if a.get("steps"):  # stage 3: passes and in-engine row swaps, in order
            steps = []
            toks = sorted({int(tk) for stp in a["steps"] if stp["type"] == "swap" for tk in stp["rows"]})
            origs = {tk: read_row(src / emb_file, EMB, tk) for tk in toks}
            for tk in toks:
                assert torch.equal(origs[tk].view(torch.int16), read_row(base / base_wm[EMB], EMB, tk).view(torch.int16)), tk
            for stp in a["steps"]:
                if stp["type"] != "swap":
                    steps.append(stp)
                    continue
                rb = {}
                for tk in toks:
                    key = stp["rows"].get(str(tk))
                    new = origs[tk] if key is None else (origs[tk].float() + deltas[key].float().cpu()).to(torch.bfloat16)
                    ek = stp.get("expect", {}).get(str(tk))
                    if ek:
                        assert int_bits(new) == int_bits(deltas[ek]), f"{stp['tag']}: row {tk} != {ek}"
                    rb[str(tk)] = int_bits(new)
                steps.append({"type": "swap", "tag": stp["tag"], "rows": rb})
            log.setdefault("swap_rows_changed", {})
            for stp in steps:
                if stp["type"] == "swap":
                    log["swap_rows_changed"][stp["tag"]] = {k: sum(x != y for x, y in zip(v, int_bits(origs[int(k)])))
                                                            for k, v in stp["rows"].items()}
        prep_s = round(time.time() - t, 1)
        job = {"arm": name, "model": str(model_dir), "rows": str(rows), "eager": eager, "max_tokens": max_tokens,
               "passes": passes, "steps": steps, "expect": expect, "out": str(work.parent / f"preds_{name}.jsonl"),
               "meta": str(work.parent / f"meta_{name}.json"), **engine_kw, **a.get("engine", {})}
        jp = work.parent / f"job_{name}.json"
        jp.write_text(json.dumps(job))
        print(f"[arm {name}] prep {prep_s}s; engine start", flush=True)
        try:
            rc = subprocess.run([sys.executable, __file__, "engine", str(jp)], timeout=arm_timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        meta = json.loads(Path(job["meta"]).read_text()) if Path(job["meta"]).exists() else {}
        meta.update({"rc": rc, "prep_s": prep_s, "wall_s": round(time.time() - t, 1)})
        log["arms"][name] = meta
        print(f"[arm {name}] rc={rc} wall {meta['wall_s']}s", flush=True)
        if rc == 0:
            shutil.copyfile(job["out"], out / f"preds_{name}.jsonl")
            (out / f"meta_{name}.json").write_text(json.dumps(meta, indent=1))
            if on_done:
                on_done(name)
        elif a.get("steps") and Path(job["out"]).exists() and Path(job["out"]).stat().st_size > 0:
            # stage 3: passes are written as they finish; keep what completed (not marked done, so a rerun redoes it)
            shutil.copyfile(job["out"], out / f"partial_{name}.jsonl")
            (out / f"meta_{name}.json").write_text(json.dumps(meta, indent=1))
            if on_done:
                on_done(name)
            log["aborted_at"] = name
            break
        elif a.get("required", True):
            log["aborted_at"] = name
            break
    restore_untied()
    log["F_checked_tensors"] = state.get("F_checked")
    log["F_identity_mismatch"] = state.get("F_mismatch")
    log["total_s"] = round(time.time() - t_start, 1)
    return log


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "engine":
        sys.exit(engine(sys.argv[2]))
    print(__doc__)
