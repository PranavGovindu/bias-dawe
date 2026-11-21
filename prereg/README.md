# Pre-registration notes

The notes mention "Fable", "advisor" and "supervisor". These are internal reviewer and coordinator roles (Claude models I used to check designs and claims). They are unrelated to the "Astra/Fable" editing agent in DAWE's documentation.

These are my internal working notes, copied byte for byte. They are the evidence that the arms, grids and selection rules were fixed before the runs they govern. They use my working names (`v1`, `v4`/`v4_format`, `v11`/`v11_plan` for JSON, XML and plan-first; arms A–F, A2/D2/B2/C2; "s1"/"s2" for DAWE's one-row and two-row sessions), refer to files that are not in this repository, and were written for ourselves, not for publication.

| File | Written | Governs |
|---|---|---|
| `PREDICTIONS.md` | 2026-11-18, before any run; sections added before stage 2, before the seed-1 replicate and before stage 3 (dates in the headings) | predictions for every stage; the seed-1 rule for choosing b on train |
| `STAGE2_SPEC.md` | 2026-11-18, before any stage-2 run | XML arms A–F and their analysis |
| `STAGE3_SPEC.md` | 2026-11-19, before any stage-3 run | plan-first arms, the constant grids and the train-only selection rule |
| `neartie_PREDICTIONS.md` | 2026-11-19, in dated amendments | the margin analysis; Amendments 4 and 4b fix c = +12 for seed 1 and its forecast (opens, JSON flips, accuracy 65) before launch |

The dates are the ones written in the files. The working directory was not under version control, so there is no independent timestamp. Two choices were also recorded as files before the matching test scores existed: `results/stage2_s1/b_choice.json` and `results/stage3/s*/choice_recorded_before_test.json`.

The one comparison that was not pre-registered in this way is the seed-0 XML bias strength (b = −6): the spec gave no selection rule, and the rule I added before the run (in my stage-2 notes, not included here) selects on test items.

sha256:

```
81eb9c01451c20f4680be38f280e31f6b63f7923ffe83bcc428bfb2c1dce5e8f  PREDICTIONS.md
226a915fc39944976147dee4b0cab80bb2aa9dfa13559fb8b3a0e512f3aec000  STAGE2_SPEC.md
508a948c2bfb0116a749c0c3bcdddd015e33eeb5e283fa576784bca939f699e7  STAGE3_SPEC.md
58b6b1ad2c6cacd5a20c6e6056a946bac39e3d024ec80d21b593d6413d0d10df  neartie_PREDICTIONS.md
```
