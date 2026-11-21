# Draft thread

Each post is under 280 characters as written.

---

1/
I re-ran DAWE (bandr-ai) on Qwen3.5-4B with their exact recipe, adding the controls they listed as not yet done: a constant logit bias, a second fine-tune seed, a plan-first random control. In short: the repairs replicate, a tuned bias gets close, and the baseline is fragile.

2/
What replicates: the XML break and its fix. Their ridge edit, fit on my checkpoints, takes XML tool calls from 75.0 to 85.3 (seed 0) and 72.5 to 86.3 (seed 1). Their released patch, applied to my checkpoints, gives the same XML accuracy on both.

3/
The control: a constant bias on <|im_end|> ("end the turn less often") gets XML 85.8 and 85.3, with JSON accuracy unchanged. Per item, edit vs bias: 0/1 and 4/2. But it must be tuned: only b = −6 to −10 ties on both seeds; at −2 or −12 the edit wins (p ≤ 0.03).

4/
Plan-first harness: their two-row edit, refit here, replicates on seed 0 (24.0 → 88.7; theirs 20.1 → 88.2). On seed 1 it goes 0.5 → 57.4. Same recipe, different seed.

5/
A +8 bias on "<" at the first reply token only (serving code supplies that position), plus an end-of-turn bias, both chosen on train, matches it on both seeds: 88.2 vs 88.7 and 58.3 vs 57.4. No JSON reply opened with "<". At every position it loses on seed 1 (50.0 vs 57.4).

6/
On seed 1, a first-token constant of +12 (set from that seed's train margins, forecast written beforehand) reached 65.2 vs the edit's 57.4. Not like for like: the constant was re-tuned for this seed and the edit kept DAWE's fixed +8 target. One seed, one constant.

7/
The bigger issue for users is the unpatched plan-first baseline. Seed 0: 24.0. Identical-config rerun of seed 0: 10.3. Seed 1: 0.5. Seed 2: 0.0. DAWE reports 20.1 from one run. XML multi-call spans 21–56 across my runs.

8/
Where the edit wins: one weight patch covers both harnesses, with no serving code. It also looks less sensitive to strength: on train, DAWE's 6 to 16 gives 93.3 to 93.9 (their checkpoint), my seed-1 bias 87.9 to 95.0 over -6 to -12. Its XML edge (+1.5 to +2.9) is within noise.

9/
Caveats: one model, one recipe, the same 204 test items per harness for both seeds, edits on two seeds. Arms and selection rules were written down before the runs, except the seed-0 XML bias strength, picked with test visible. Compute: about $11 of Modal H100.

10/
Code, per-item results and the notes I wrote before each run: github.com/PranavGovindu/bias-dawe. Thanks to the DAWE authors for releasing their code, their patches and a clear list of controls not yet run. That made this easy to check.
