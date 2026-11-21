# Predictions (written 2026-11-18, before any run)

Stage 1 (our SFT seed 0, 204 held-out, greedy):
- v1 84–90%. v4_format 65–80% (break reproduces: v4 ≥ 7 points below v1). v11_plan 10–45%.
- v4 multi-call items much worse than single-call (most v4 failures are early end-of-turn after the first call).
Stage 2 (v4):
- Ridge row edit: v4 80–88%, v1 within ±2.
- Constant end-of-turn bias at matched end-of-turn rate: recovers 40–80% of the edit's gain; a bias applied only right after </tool_call> recovers ≥ 80%. My honest prior: most of DAWE's v4 gain is "stop less after a call"; the state-dependence adds a few points mainly on single-call items.
- Same-norm random edit: below unpatched (as they report).
- Untied output row: within 2 points of the tied edit.
Second seed: break and repair both reproduce, magnitudes within ±6.
- LoRA scaling (best alpha chosen on dev): recovers ≥ half of the row edit's v4 gain but costs ≥ 3 points on v1; no help on v11.

## Stage-2 additions (written after stage 1, before stage 2; see STAGE2_SPEC.md)
- Constant bias frontier: some b in −6…−12 reaches v4 within 3 points of arm A; at that b, v1 drops ≤ 2 points. I.e. A lies on or near the B curve (state-dependence adds < 4 points on v4).
- D (their delta on our checkpoint): works (v4 ≥ 82) because rows and base are identical and the SFT is the same recipe.
- E (untied) within 2 points of A on v4 and v1.
- F: α=0.5 gives v4 ≥ 80 with v1 ≤ 84.

## Seed-1 replicate of stage 2 (written 2026-11-18 after STAGE2.md, before any seed-1 stage-2 run)
- Rule: b for the constant bias is chosen on TRAIN items only (595 v4 train prompts): the b in {−2…−12} with the highest v4 train accuracy at no v1 train loss; ties → smaller |b|. Fixed now.
- Prediction: on seed 1, A (ridge, frozen s1 settings) and B (train-chosen b) are within 4 points on v4 and v1 (McNemar not significant); D (their δ) works again (v4 within 3 of A).

## Stage 3: plan-first two-row patch (written 2026-11-19, before any stage-3 run; see STAGE3_SPEC.md)
- D2 and A2 on seed 0: v11 80–90, v1 within ±2 of U, v4 ≈ A1's.
- Best B2 (all-position constants): v11 ≤ 65 at no v1 cost; any c large enough to fix plan-start costs ≥ 10 points on v1. So A2 beats best B2 on v11 by ≥ 15 points (significant) on seed 0.
- Best B2-first: v1 drops ≥ 10 points at any c that lifts v11 above 70, so under the train rule it lands near A1 on v11. A2 beats it by ≥ 10 on v11.
- Seed 1 (deep-break seed, v11 at 0.5): A2 lifts v11 a lot less than on seed 0 (≤ 60), D2 similar to A2.
- If a constant (either kind) gets within 4 points of A2 on v11 with ≤ 2 points v1 cost on both seeds, then DAWE's state-dependence matters nowhere we tested.
