# Data

Prebuilt with DAWE's own builders (`dawe build-sft`, `build-eval`, `build-pairs` at commit `8f11c96`) from the Berkeley Function Calling Leaderboard v3 (simple, multiple and parallel; dataset revision `61fc0608`). DAWE's hash split puts 595 items in train and 204 in test (48 of them parallel). The Modal app reads these files directly.

| File | Rows | What | sha256 |
|---|---:|---|---|
| `eval/bfcl_test_3h.jsonl` | 612 | the 204 test prompts under JSON (`v1`), XML (`v4_format`) and plan-first (`v11_plan`) | `52a85b757482360f…` |
| `sft/v1_bfcl_train.jsonl` | 595 | JSON training conversations (the fine-tune data) | `e00b336f04df459a…` |
| `sft/v4_format_bfcl_train.jsonl` | 595 | the same items rendered for XML (train-side bias selection) | `f4192e7655fc3035…` |
| `sft/v11_plan_bfcl_train.jsonl` | 595 | the same items rendered for plan-first (plan states, train-side selection) | `93a9389ed90c91ce…` |
| `pairs/v1__v4_format.jsonl` | 595 | JSON/XML pairs of each train item (hidden states for the one-row fit) | `4f5094e211e84d2b…` |

The questions and function schemas come from BFCL, which is released under Apache-2.0 (https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard). Please cite it if you use these files:

```bibtex
@misc{berkeley-function-calling-leaderboard,
  title={Berkeley Function Calling Leaderboard},
  author={Fanjia Yan and Huanzhi Mao and Charlie Cheng-Jie Ji and Tianjun Zhang and Shishir G. Patil and Ion Stoica and Joseph E. Gonzalez},
  howpublished={\url{https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html}},
  year={2024},
}
```
