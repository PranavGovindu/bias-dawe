# Stage 2 spec (frozen 2026-11-18, before any stage-2 run)

Checkpoint: sft_v1_s0 (on volume formatter-exps-gp-runs). Eval: 204 held-out × {v1, v4_format, v11_plan}, greedy, vLLM eager=False (same as stage 1), their scorer per item. Stage-1 preds are the unpatched baseline.

## Arms
A. **Ridge row edit (s1 replication), frozen settings:** their exact s1 fit — final-layer states from weight_map over the 595 train pairs (260 v4 "continue" points target −12; all v4 "stop" and all v1 points target 0), λ = 0.01·mean(diag(XᵀX)), no intercept, add δ to row 248046 (`<|im_end|>`) in fp32, cast bf16. No re-selection of targets/λ.
B. **Constant end-of-turn bias, as a frontier:** vLLM `logit_bias={248046: b}` for b ∈ {−2, −4, −6, −8, −10, −12, −16}, all in one engine; score v1 and v4 at each b. Question: does A sit off the (v4 gain, v1 cost) curve traced by B?
C. **Same-norm random row edit** (fp32 add, bf16 cast, same convention as A), 1 draw seed 0.
D. **Their released s1 delta on our checkpoint** (their_deltas.pt; LoRA never touched embedding rows): tests exported patch + sibling transfer. First compute h·δ on our saved states on the PC (mean shift at continue vs stop) to predict whether it works.
E. **Untied ridge:** checkpoint saved with tie_word_embeddings=false, explicit lm_head = embed copy, δ added to lm_head row only. Assert after load that embed_tokens row 248046 is unchanged and lm_head row changed (no silent re-tie). Clean state-dependence comparison: E vs B (both output-only).
F. **LoRA scaling (zero-fit):** W(α) = base + α·(merged − base) in fp32 then bf16, α ∈ {0.25, 0.5, 0.75} (α=0 is base, α=1 is sft; both already scored).
v11 arms (s2) only after A–F are read.

## Analysis (pre-registered)
- Primary: v4 accuracy and v1 accuracy per arm; per-item McNemar vs unpatched and A vs best B.
- Where arms can differ: v4 has 51 failures; 15 of the 20 wrong-argument failures also fail under v1 (unfixable by any readout arm), 31 are too-few-calls (23 of these are correct under v1). So the reachable v4 ceiling is ≈ 87–90%. Discriminating signal: the 48 parallel items, and over-continuation (too many calls / trailing text) on v1 and v4 single-call items.
- Minimum detectable: with n=204 paired, differences under ~4 points (≈ 8 discordant items) are not resolvable; report them as ties.
- Smoke test before launch: on the PC, ~10 items with logit_bias −100 on 248046 must run to max tokens; patch application round-trips (A's row equals base row + δ in bf16).
