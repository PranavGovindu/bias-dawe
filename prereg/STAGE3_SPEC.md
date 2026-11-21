# Stage 3 spec: the plan-first (s2, two-row) patch vs controls (frozen 2026-11-19, before any stage-3 run)

Question: on plan-first (v11), does DAWE's state-dependent two-row patch beat constant biases? The XML result (stage 2) says a tuned constant on <|im_end|> ties the one-row patch. Plan-first needs a context-dependent push on '<' (start <plan> only under v11; v1 replies must still start with '{'), so our prior is that a constant can't do it.

Checkpoints: sft_v1_s0 and seed 1 (re-merge prev_a_s1 → dawe_s1, delete after). Eval: 204 test × {v1, v4_format, v11_plan}, greedy, eager=False, their scorer per item.

Arms (both seeds):
- U: unpatched (reuse existing preds).
- A1: s1 one-row patch (reuse stage-2 / seed-1 preds; v11 already scored).
- D2: their released s2 delta (`<` and <|im_end|> rows) on our checkpoint.
- A2: our own s2 fit with their exact settings (sessions/s2/notes.md): X = final-layer states from plan maps (595 plan_start, 595 after_plan) + v1 (260 continue, 595 stop) + v4 (260 continue, 595 stop); Y col 1 = +8 at plan_start, 0 elsewhere; col 2 = −12 at after_plan and v4 continue, 0 elsewhere; λ = 0.01·mean diag(XᵀX); fp32 add to rows 27 and 248046, cast bf16. No re-selection.
- B2: constant biases, logit_bias {27: c, 248046: b}, every position. Grid c ∈ {0, 2, 4, 8}, b ∈ {−4, −8, −12}.
- B2-first: the strongest fair dumb control — bias '<' by c only at the FIRST reply token (two-pass decode: 1 token with bias, then continue unbiased except the constant b on <|im_end|>). Same grid. Because v1 and v11 prompts differ only in the system message, a first-token push hits v1 too.
- C2: same-norm random edits to both rows (seed 0 draw), same add convention.
Selection rule for B2 / B2-first (fixed now): choose (c, b) on TRAIN items only — maximise mean(v1, v4, v11) train accuracy; ties → smaller |c|, then smaller |b|. Report the full test grid for reference.

Analysis: per-arm v1 / v4 / v4 parallel / v11; McNemar per item for A2 vs best B2-first, D2 vs A2, A2 vs A1 on v11; seed by seed (no pooling across seeds as independent evidence: same 204 items). Read 20 raw v11 outputs of A2 and best B2-first before aggregates. Differences under ~4 points on 204 items are ties.
