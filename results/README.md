# Results

The README's table cells are recomputed from these files by `tools/tables.py`, and the figures by `tools/make_figures.py`. The full stage analyses (McNemar tests, identical-text counts, failure classes, token-cap counts) come from `code/analyze_*.py`, whose outputs match the committed `analysis*.json` exactly, and the margin and rerun numbers from `code/margins/*.py`, whose outputs match `margins/*.txt` and `step2/analysis.txt`. A few figures in the README bullets (the paired confidence intervals, and counts such as the ~35k JSON replies with no `<` start) are quoted from my stage write-ups, which are not included; the per-item files here are enough to recompute them.

## Formats

- `items_<arm>.jsonl.gz`: one row per item and harness, written by DAWE's own scorer (`code/scoring/score_items.py`): `id`, `harness` (`v1` = JSON, `v4_format` = XML, `v11_plan` = plan-first), `cat` (simple / multiple / parallel; "XML multi-call" = `parallel`), `ok`, `fmt_fail`, `n_pred`, `n_gold`, and the generated `text`. Test files have 612 rows (204 × 3); train files (`tr` in the name) have 595 per harness.
- `finish.jsonl.gz`: the generator's finish reason and output length per item (`name` = the arm), used for the token-cap counts. Stage-1 runs did not record it.
- Arm names: `U` unpatched; `A` one-row ridge edit (my fit, their settings); `D` their released one-row patch; `B<b>` constant bias b on `<|im_end|>`; `C` same-norm random row; `E` untied (output-only) version of `A`; `F25/F50/F75` LoRA scaled by 0.25/0.5/0.75. Stage 3: `A2` two-row edit (my fit), `D2` their released two-row patch, `C2` same-norm random rows, `B2te_c<c>_b<b>` constants at every position, `B2fte_c<c>_b<b>` first-token constants, `A1chk` / `Uchk` / `B2fchk` engine checks; `tr` instead of `te` marks train items.

## Files and tables

| Path | What | README table |
|---|---|---|
| `stage1/items_sft_v1_s0.jsonl.gz` | seed 0, unpatched | all "Unpatched" rows, seed 0 |
| `stage1/items_prev_a_s1.jsonl.gz` | seed 1, unpatched (trained by `code/prevent.py`, arm `a1`) | all "Unpatched" rows, seed 1 |
| `stage1/items_base.jsonl.gz` | base model, unpatched | baseline table, "Base model" |
| `stage1/train_*.json`, `stage1/map_*.json` | training loss logs; state-map summaries (teacher-forced NLL at decision points) | |
| `stage2/` | XML arms on seed 0: `A`, `D`, `B0`, `B-2`…`B-16`, `C`, `E`, `F*` | XML table and figure, seed 0 |
| `stage2/fit_report.json` | the seed-0 one-row fit: λ, norms, in-sample shifts | |
| `stage2_s1/` | XML arms on seed 1: `A`, `D`, `B0`, `B-2`…`B-16` (test), `Btr0`, `Btr-2`…`Btr-12` (train) | XML table and figure, seed 1 |
| `stage2_s1/b_choice.json` | b = −8 chosen on train, before any seed-1 test score | |
| `stage3/s0/`, `stage3/s1/` | plan-first arms: `A2`, `D2`, `C2`, the full constant grids on train and test, checks | plan-first table |
| `stage3/s*/choice_recorded_before_test.json` | (c, b) chosen on train, recorded before any stage-3 test file was scored (`choice.json` is identical) | |
| `stage3/fit3_s*.json` | the two-row fits per seed | |
| `step2/items_sft_v1_s0r.jsonl.gz`, `step2/items_sft_v1_s2.jsonl.gz` | seed-0 identical-config rerun and seed 2, unpatched | baseline table and figure |
| `step2/items_s1_c12.jsonl.gz`, `step2/items_s1_c8.jsonl.gz` | seed 1, first-token constant at c = +12 and the c = +8 control in the same engine (b = −8); the pass is `B2fte` in the generator's output | plan-first, the +12 bullet |
| `step2/analysis.txt` | output of `code/margins/step2_analyze.py` | baseline bullets |
| `margins/` | per-item logit margins at the reply-start and plan decisions (logit summaries, not hidden states) and the outputs of `code/margins/analyze.py` and `rs_analyze.py` | the margin bullets |
| `deltas/` | the row edits used by the Modal runs: my fits and DAWE's released deltas (`deltas_stage2.pt` seed 0 XML, `deltas_s1.pt` seed 1 XML, `deltas_s3_s*.pt` plan-first) | |
| `prevent/` | side experiment (decision rows trained during the fine-tune), and its learned rows | side result |
| `costs_dawe.jsonl`, `costs_prevent.jsonl` | Modal function-timer cost estimates per job ($6.71 and $4.44) | cost |

Not included: model checkpoints, LoRA adapters, hidden-state dumps and raw generator logs.
