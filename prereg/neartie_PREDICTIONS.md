# neartie_01 predictions (written 2026-11-19, before any margin was computed)

## Definitions (fixed before computing)
- States: final-layer (index −1, post-norm, fp16 on disk) states from dawe_01 maps. Seed 0 = `sft_v1_s0`, seed 1 =
  `dawe_s1` (prevent_01's a1 LoRA, re-merged). All are **train** items (595), with teacher forcing on the gold reply.
- Logits: `z = h @ E.T` with E = base `embed_tokens` (tied; LoRA does not touch it). Margin `m = z(target) − max over
  t ≠ target of z(t)` (target minus best rival). I also report the pairwise margin against the named rival.
  - plan_start (v11, last prompt token): target `<` (27); expected rival `{"` (4754).
  - after_plan (v11, `>` of the gold `</plan>`): target `\n` (198); expected rival `<|im_end|>` (248046).
  - v4 continue (after `</tool_call>`): target `\n` (198); rival `<|im_end|>`. v4 stop: target `<|im_end|>`.
  - v1 continue / stop: target = the gold next token at their decision offset (often a merged token such as `},`).
- **Near tie: |m| < 4 logits.** Scale reasons: (i) the JSON reply-start side has only a lower bound, m_JSON(`{"` over
  `<`) > 8. +8 on `<` never opened a v1 reply on 799 items × 2 seeds, so 4 is under half of what JSON is known to hold.
  (ii) The behavioural train CDF at plan start already implies a median between-seed shift of about 5 logits (below).
  "Near zero" for a distribution means median |m| < 4 and ≥ 50% of items with |m| < 4. "Far" means median |m| ≥ 7.
- **Already known before computing** (dawe_01 stage-3 train generations, vLLM, first token = the plan_start argmax): the
  fraction of v11 train replies opening with `<` at c = 0 / +2 / +4 / +8 is 54.8 / 83.9 / 94.6 / 99.8% at seed 0 and
  10.6 / 22.4 / 39.2 / 79.2% at seed 1. So P(m > −c) at plan start is known at four points per seed. **Predictions that
  only restate this CDF are consistency checks, not evidence.** The same holds for "bias c flips iff m < c" (arithmetic).
- **Falsifiable content** (supervisor): (a) margins concentrated near zero under v11/v4 and far under v1; (b) a seed shift
  that is small relative to the JSON margin and mostly an offset; (c) teacher-forced margins predicting free-running
  outcomes at continue/stop points; (d) the sign and size of prevent_01's row shift.

## Predictions

P1. **Alignment check (gate).** Per-kind mean NLL recomputed from the fp16 states matches the map logs within 3% relative
(seed 0: plan_start 1.143, after_plan 2.279, v4 continue 0.770, v1 continue 0.00068; seed 1: 5.422 / 3.181 / 1.42 /
0.003). The sign of each item's plan-start margin agrees with its vLLM train first token at c = 0 on ≥ 97% of items per
seed, and every disagreement has |m| < 1. If this fails, stop: the states cannot be matched to items.

P2. **Plan start, seed 0 (check plus shape).** Median m in [−0.5, +1.0]. Fraction with |m| < 4 is ≥ 75% (near zero). The
fraction with m > +4 is in [5%, 25%]. The top rival is `{"` on ≥ 95% of items with m < 0.

P3. **Plan start, seed 1.** Median m in [−7, −4]. Fraction with |m| < 4 is in [25%, 50%]. Top rival `{"` on ≥ 95% of items
with m < 0. The constant needed to open 95% of train items (c95 = −5th percentile of m) is in [4, 5] at seed 0 and
[10, 16] at seed 1.

P4. **Seed shift at plan start (claim b).** The median shift (seed 1 − seed 0) is in [−7, −3], so it is smaller than the
JSON lower bound of 8 (90%). The per-item correlation between seeds (same 595 items) is r in [0.3, 0.8]. The shift is
mostly an offset: |mean of per-item difference| > SD of per-item difference (60%; this is the part that could fail).

P5. **After plan (teacher-forced on gold "Call f." plans; off-policy).** Median m: seed 0 in [−2, +2], seed 1 in [−3, +1].
Seed 1 lies below seed 0 (median shift in [−3, 0]). Top rival `<|im_end|>` on ≥ 90% of points with m < 0. The |b| needed
to clear 95% is in [5, 12].

P6. **Continue decision, v4 vs v1 (claim a, the one contrast we can measure in step 1).** v4 continue: seed 0 median m
in [+1, +4] with 12–30% negative; seed 1 median in [−1, +2.5] with 25–50% negative; top rival `<|im_end|>` on ≥ 90% of
points. v1 continue at the same items: median m ≥ 7, ≤ 1% negative, on both seeds. So v4 continue is "near zero" and v1
"far" on both seeds.

P7. **v4 stop.** Median m (`<|im_end|>` over best rival) in [8, 14]; 5th percentile ≥ 5; top rival `\n` on ≥ 80%.

P8. **Constant size vs margins at v4 (claim on the −6…−10 window).** The |b| that clears 95% of continue points (pairwise
vs `<|im_end|>`) is in [3, 8] at seed 0 and [5, 10] at seed 1. The |b| at which 2% of stop points cross is in [7, 12] on
both seeds. So the margin-implied clean window overlaps the observed −6…−10.

P9. **Teacher-forced → free-running (claim c).** On the 152 multi-call train items, at b ∈ {−4, −8, −12} (seed 0:
stage-3 `B2tr_c0_b*`; seed 1: `Btr*`, also b = 0, −2, −6, −10), "some continue point has z(`\n`) + |b| < z(`<|im_end|>`)"
matches the observed "too few calls" (n_pred < n_gold) on ≥ 85% of items at each b. "Some stop point has
z(`<|im_end|>`) − |b| < best non-eos rival" matches "too many calls" on ≥ 85% of all 595 items at each b. The likely failure
mode is that the model's own first-call text differs from the gold prefix.

P10. **Fix edits at plan start.** Per-item shift h·δ_A2(`<`) at plan start has mean +8.0 (in-map, by construction) and
SD ≤ 0.6 on both seeds. So A2 acts at plan start like a constant +8: the predicted open fraction under A2 is within 2
points of c = 8 (99.8% / 79.2%). D2 is similar, with a larger SD at seed 1. A2's effect at v1 reply start cannot be computed.

P11. **prevent_01 rows (claim c of the brief), on (a)/(a1) states.** Swapping in (b)'s learned output rows (6 rows) moves
margins the wrong way. Mean Δm at v4 continue (pairwise `\n` vs `<|im_end|>`) is in [−1.5, −0.2] at both seeds. Mean
Δm at plan start (pairwise `<` vs `{"`) is in [−4, −1] at seed 0. That is the size needed to explain "out" moving v11
plan-skips from 82 to 168 of 204 if the near-tie story holds; seed 1 is negative too. Caveat fixed in advance: these are
(a)'s states, not (b)'s own LoRA states; prevent_01 found (b)'s LoRA without rows ≈ (a).

## How the hypothesis is scored
- (a) is supported for the continue decision if P6 holds. At reply start it is **untestable in step 1** (no v1 states);
  only "v11 near zero at seed 0" can be checked.
- (b) is supported if the plan-start shift is < 8 and offset-dominated (P4). It is weakened if the shift is mostly item
  reshuffling (|mean| < SD), and refuted as "small noise" if the shift is ≥ 8.
- (c) is supported if P11 has the right sign and at least 1 logit at plan start (seed 0).
- Seed 1 coverage: of A2's 87 seed-1 v11 failures, 26 skip the plan and 48 write `<plan>` and then JSON (computed before
  this file from stage-3 items). The second class sits at a token after `<plan>` that no map covers. Whatever the
  margins show, step 1 can speak only to the first class.

## Amendment 1 (2026-11-19, still before any margin was computed; after an advisor review)

A1. **End-of-turn decisions use an eos margin, not a gold-token margin.** The model's own replies write `</plan>\n\n{`
where gold has `</plan>\n{`, so a gold-`\n` margin can be negative with `\n\n` as the rival, which is not a format failure.
For after_plan and v4 continue the primary margin is **m_eot = max over t ≠ eos of z(t) − z(`<|im_end|>`)** (positive
means the model continues). For v4 stop (and v1 stop) it is **z(`<|im_end|>`) − max over t ≠ eos of z(t)**. P5, P6
(v4 side), P7, P8 and P9 are scored on m_eot with the same numeric ranges. Gold-minus-best-rival is kept as a second
column. At v1 continue/stop the real contenders are merged tokens (`},` against `}]}`-type), not eos. So P6's v1-vs-v4
contrast compares different decisions; I report v1 on both margins and say so.

A2. **Noise yardstick for claim (b).** Compute the per-item seed-1-minus-seed-0 difference of the margin (same 595 items) at
the in-distribution points (v1 continue and v1 stop, gold-minus-best-rival) and at v4 stop (m_eot). Ratio
R = |mean diff at plan start| / max(|mean diff|, SD of diff) at the in-distribution points.
- R ≤ 2: the seeds differ by the same noise everywhere, and v11 merely sits nearer a boundary. This supports (b).
- R ≥ 3: seed variance is larger off-distribution than in-distribution. This is a **miss for (b)** as "small SFT noise
  near a tie".
- In between: unresolved.
- **Prediction: R ≥ 3 (65%).** Off-distribution decisions are less constrained by training.

A3. **P1 gate tolerances.** The map NLLs came from bf16 logits; I recompute in fp32 from fp16 states. The 3% relative rule
applies to plan_start, after_plan and v4 continue. v1 continue and v4 stop are checked in absolute terms (≤ 0.005).
Items with |m| < 0.25 are counted as unresolved in every sign comparison.

A4. **Scope fixed now.**
- **A2 and A are in-sample on these states** (they were fit on them), so P10's mean +8 and small SD are by construction.
  D2 (their release, fit on their checkpoint) is the only out-of-sample edit check.
- **h·δ on saved states gives only the output side** of a tied row. Stage 2's A vs E showed that the input side matters on
  v11.
- **P11 is the analogue of prevent_01's "out" vs "none"** on (a)'s states. It is a weaker proxy at seed 1.
- **Per-item reply-start checks against B2tr at c = 0/2/4/8 are pipeline validation, not evidence.**

## Step 2 draft predictions (written 2026-11-19 after step 1, before any new seed or rerun exists)

These come from seeds 0 and 1 only, plus prevent_01's arm (d) as a behavioural near-replicate. Scope: same recipe, three
new SFT seeds and one same-seed rerun of seed 0, with maps extended to v1 and v4 reply start, the token after `<plan>`,
and on-policy after-plan states.

S1. **Same-seed rerun of seed 0.**
- Plan-start median within ±1.5 logits of +0.12; item-rank correlation with seed 0 ≥ 0.9.
- v1 continue/stop medians within ±0.5 of seed 0's.
- If the median moves by more than 1.5, numerics alone are enough to swing v11. Arm (d) suggests about −1 logit
  behaviourally: 89 vs 125 opens.

S2. **Each new seed.**
- Plan-start median in [−7, +2], with at least one of the three within ±1.5 of 0 (60%).
- Item-rank correlation of plan-start margins with seed 0 in [0.7, 0.9].
- Per-item v1 continue/stop shift vs seed 0: |mean| ≤ 1.0 and SD ≤ 1.5. v4 stop median in [13, 17].
- First XML continue (k = 0) median in [−3, +1].
- The spread of plan-start medians across the five seeds is at least 3× the spread of v1 stop medians.

S3. **JSON reply start (the missing contrast)** on every seed. `{"` is the top token on ≥ 99% of items. The `<` deficit
(max over t ≠ `<` of z(t) minus z(`<`)) is > 8 on 100% of items, which behaviour already implies, with median ≥ 14. If
the median is < 10, then "JSON is far, v11 is near" holds only weakly.

S4. **The token after `<plan>`** (gold plan word vs `{"`) at seed 1: median in [−3, +2]. Seed 0's median is above
seed 1's. (Position corrected by Amendment 3: the JSON choice is `>` vs `>{"` at the token after `plan`.)

S5. ~~Kill rule: plan-start medians uncorrelated with v11 baseline accuracy.~~ **Withdrawn after audit (2026-11-19).** A
seed's open rate follows from its own plan-start margins by arithmetic, so this cannot fail informatively.

**Kill rules for step 2 (rewritten after audit, 2026-11-19, before the 3060 pass below):**
- **K1, same-seed rerun (S1), primary.**
  - If the rerun's plan-start median moves by more than 1.5 logits from seed 0's, or its item-rank correlation with
    seed 0 is < 0.9, then "seed" differences at plan start are numerics-sized. Every seed claim becomes a numerics claim
    and the near-tie-vs-offset reading is dropped.
  - If it moves by ≤ 0.5 while new seeds spread by ≥ 3, the seed (not numerics) sets the offset.
- **K2, JSON reply-start margin (S3).** "JSON far, v11 near" is killed if the v1 `<` deficit (max over t ≠ `<` of
  z(t) minus z(`<`)) has median < 10 on any seed, or if more than 5% of v1 items have a deficit below the c needed to
  open 95% of v11 items on that seed. In the second case no single first-token constant serves both.
- **K3, data requirement.** Step 2 must save states for **test** items as well as train: reply start for v1/v4/v11, the
  token after `<plan>`, and after-plan on the model's own plans. Margins can then be matched per item to the test
  outcomes, not only to memorised train items.

## Amendment 2: predictions for the free 3060 pass (2026-11-19, written before it ran)

Setup: the PC 3060 merges each LoRA adapter into the base (sft_v1_s0 adapter = seed 0, prev_a_s1 adapter = seed 1; bf16,
`merge_and_unload` on GPU as in stage 1). It then runs one forward per prompt and saves the final post-norm state at:
- v1 reply start (train 595 + test 204);
- v11 plan start, and the position right after the tokens `<` `plan` `>` (train 595 + test 204).

Margins are computed on CPU from base E. The post-`<plan>` margin is m_post = max over t ∉ {`{"`, `{`} of z(t) minus
max(z(`{"`), z(`{`)); positive means the model writes plan text rather than JSON.

- **T0 (merge validation, gate).** On the 595 train items, PC plan-start margins match the H100 map margins with
  |Δm| ≤ 0.3 on ≥ 95% of items and mean |Δm| ≤ 0.15. If not, stop and report.
- **T1 (S3 on seeds 0/1).** v1 reply start:
  - `{"` is the top token on ≥ 99% of train and test items;
  - the `<` deficit is > 8 on 100% of items (behaviour requires this, up to the ~0.2 HF/vLLM offset), with median ≥ 14;
  - `{"` minus best rival has median ≥ 8.
- **T2 (K2 on seeds 0/1).** The fraction of v1 items with `<` deficit below that seed's c95 (4.1 at seed 0, 11.1 at
  seed 1) is 0% at seed 0 and ≤ 5% at seed 1.
- **T3 (test plan start).**
  - Median at seed 0 in [−0.5, +1.5], at seed 1 in [−6, −3.5] (test opens more often than train: 61 / 13% vs 55 / 11%).
  - Sign vs the U first token agrees on ≥ 97% of items (validation).
- **T4 (S4, post-`<plan>` on test).**
  - Seed 1 median m_post in [−3, +2]; seed 0 median above seed 1's.
  - Among test items whose B2-first (8, b) reply begins `<plan>`, "m_post < 0" matches "`<plan>` followed by JSON with no
    `</plan>`" on ≥ 90% of items at seed 1. That is near-identity (same prefix), so it is validation.
  - The finding is where the m_post distribution sits.

## Amendment 3 (2026-11-19, after T4 missed and before the re-run): the decision after `<plan>` sits one token earlier

`<plan>{"` tokenizes as `<` `plan` `>{"` (85806). The JSON choice is therefore made at the position after `plan`: `>` (29)
against `>{"` / `>{` (85806 / 7082). The pre-registered T4 measured the position after `>`, which a free-running JSON
reply never reaches. That is a design miss, reported as such. Post-hoc re-measurement (same 3060 setup, v11 only; the
states after `<` and after `plan` are saved too):
- m_gt = z(`>`) − max(z(`>{"`), z(`>{`)) at the position after `<` `plan`.
- Prediction: seed-1 test median m_gt in [−2, +4], and seed 0 above seed 1.
- Among test replies that begin `<plan`, the sign of m_gt matches "`>{"` emitted" on ≥ 90% of items under the
  B2-first and U arms (validation, same prefix).
- For A2/D2, output-side m_gt under-predicts JSON-after-`<plan` less well (input side of the `<` row).

## Amendment 4: step-2 predictions (2026-11-19, written before any smoke test or launch)

**Question.** Is the plan-start offset set by the seed or by run-to-run numerics, and how wide is it across runs?

**Runs.** DAWE's v1 recipe on Modal H100, identical code and config:
- `sft_v1_s0r`: seed 0 rerun, the **primary test**;
- `sft_v1_s2`, `sft_v1_s3`, `sft_v1_s4`: seeds 2–4.

Each run is scored unpatched on v1/v4/v11 test (612 prompts, vLLM eager=False, as in stage 1). Each also gets states
from the free PC 3060 pass (`reply_states.py` on its adapter, the same path as seeds 0/1): v1 reply start, plan start,
after `<`, after `plan` and after `>`, on train and test. All margins across runs therefore come from one deterministic
extraction path; its cross-hardware noise is SD 0.3–0.4 logit against the H100 maps.

Seed 0/1 references (3060, test):
- plan start: +0.35 / −5.05;
- after `plan`: +2.82 / +0.80;
- JSON `<` deficit: 26.8 / 26.4.

**K1, rerun (primary).**
- The rerun's adapter is bitwise identical to `sft_v1_s0`'s: 35%. If it is, run-to-run numerics on this pipeline are
  zero, and the seed (init plus data order) sets the offset.
- If not identical: the rerun's plan-start test median lies within ±1.0 of +0.35, with per-item r ≥ 0.9 against seed 0
  (60%).
  - **Kill:** a move of more than 1.5, or r < 0.9, means numerics alone produce seed-scale shifts. Every "seed" claim
    then becomes a numerics claim.
- Rerun v11 unpatched accuracy in [10, 35]. It sits at a tie, so it is sensitive: arm (d) went 24.0 → 3.9.

**Width (S2, updated).**
- Seeds 2–4: each plan-start test median in [−7, +2]. Across the five seeds, max minus min of the medians ≥ 3 (60%).
  At least one new seed within ±1.5 of 0 (55%).
- **K2:** every run's JSON `<` deficit median is in [24, 29] (85%). Kill if any run's is < 10.
- After-`plan` test median in [−0.5, +4] for every run.
- v1 test accuracy in [86, 91] for every run.

**Out-of-sample constant on seed 1** (Fable's test; same engine as stage 3, first-token two-pass, test items only).
- **c is fixed now by rule:** c = ceil(c95 of seed 1's H100 train plan-start margins) = ceil(11.14) = **12**. The 3060
  train c95 is 11.25, so the rule gives the same value.
- **b = −8**, stage 3's train-chosen end-of-turn bias for seed 1. So the only change from stage 3's B2-first (8, −8)
  arm is c.
- A c = 8 control runs in the same engine.
- Predictions on 204 test items per harness:
  - **v11 opens** (reply begins `<plan`): **193** (94.6%), range [186, 198]. Per-item test margins give 192–193.
  - **JSON flips:** **0** of 204 v1 replies begin with `<` (minimum test deficit 21.3 > 12). Pass only if 0.
  - **Residual moves to the `>{"` position:**
    - `<plan>{` count **44**, range [34, 56]. Per-item margins give 50, calibrated by c = 8's 38 predicted vs 32
      observed.
    - This is the largest v11 failure class and ≥ 50% of v11 failures; no-plan failures ≤ 18 (stage-3 c = 8: no-plan
      38, `<plan>{` 31–32).
  - **v11 accuracy:** **64**, range [57, 71]. v1 in [88.7, 90.2]; v4 within ±2 of the c = 8 control.
  - **The c = 8 control** reproduces stage-3 B2-first (8, −8) v11 = 58.3 within ±2 points.
- **Refuted out of sample if:** opens < 186, or any JSON flip, or `<plan>{` ≤ 33.

## Amendment 4b (2026-11-19, after an advisor review, still before any launch): the out-of-sample claim uses train margins only

The test-item numbers in Amendment 4 (193 opens, 50 `<plan>{`) come from the same 204 test items whose margins we already
have. That is near-identity (sign agreement 97–100%), so they are **pipeline validation, not evidence**. The test-margin
forecast for opens is 94.6%, **under** the supervisor's "≥ ~95%".

**Out-of-sample forecast for seed 1 at (c, b) = (12, −8), from train margins only** (3060 states, 595 train items):
- **Opens:** 96.5% of plan-first replies (574/595 on train). On test: 197/204, range [188, 202].
- **`>{"` share:** among openers, train after-`plan` margins give 24.6% (141/574) negative. On test: 48 `<plan>{`
  replies, range [36, 60]. Calibration at c = 8 on train: 86 predicted against 87 observed.
- **JSON flips:** 0. The train minimum v1 `<` deficit is 18.9 > 12.
- **v11 accuracy:** 65, range [57, 72], assuming c = 8's 11% other-failure rate among proper openers.
- **Fails out of sample if:** test opens < 188, or any v1 reply opens with `<`, or the `<plan>{` share among openers
  lies outside [15%, 35%].

**Code-path note (recorded before launch).**
- Seed 1 (`prev_a_s1`) was trained by prevent_01's `_train_core(token_ids=[])`, not by `formatter_exps_gp.train`.
- It is the same recipe, init RNG, data order and optimiser parameter set.
- It adds no-grad check forwards before and after `get_peft_model`. prevent_01 logged identical step-1 loss across both
  paths at seed 0 (0.0709) and equal LoRA-A checksums across arms.
- The new runs (`s0r`, `s2`–`s4`) use `train`, like seed 0. So "width across five seeds" mixes two numerically
  near-identical code paths for one of the five.
