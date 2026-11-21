"""Print the README's headline tables from the per-item files in results/ (stdlib only).

usage: python3 tools/tables.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from analyze_stage2 import acc, load_items, mcnemar  # noqa: E402

R = ROOT / "results"
V1, V4, V11 = "v1", "v4_format", "v11_plan"


def it(p):
    return load_items(R / p)


def same(a, b, h):
    return sum(a[k]["text"] == b[k]["text"] for k in a if k[1] == h and k in b)


def a(x, h, cat=None):
    return acc(x, h, cat)[0]


def mc(x, y, h):
    w, l, p = mcnemar(x, y, h)
    return f"{w}/{l}, p = {p:.2g}"


S = {  # seed -> arms
    0: {"U": it("stage1/items_sft_v1_s0.jsonl.gz"), "A": it("stage2/items_A.jsonl.gz"), "D": it("stage2/items_D.jsonl.gz"),
        "B": it("stage2/items_B-6.jsonl.gz"), "C": it("stage2/items_C.jsonl.gz"),
        "A2": it("stage3/s0/items_A2.jsonl.gz"), "D2": it("stage3/s0/items_D2.jsonl.gz"),
        "F": it("stage3/s0/items_B2fte_c8_b-4.jsonl.gz"), "B2": it("stage3/s0/items_B2te_c8_b-4.jsonl.gz"),
        "C2": it("stage3/s0/items_C2.jsonl.gz")},
    1: {"U": it("stage1/items_prev_a_s1.jsonl.gz"), "A": it("stage2_s1/items_A.jsonl.gz"),
        "D": it("stage2_s1/items_D.jsonl.gz"), "B": it("stage2_s1/items_B-8.jsonl.gz"),
        "A2": it("stage3/s1/items_A2.jsonl.gz"), "D2": it("stage3/s1/items_D2.jsonl.gz"),
        "F": it("stage3/s1/items_B2fte_c8_b-8.jsonl.gz"), "B2": it("stage3/s1/items_B2te_c8_b-8.jsonl.gz"),
        "C2": it("stage3/s1/items_C2.jsonl.gz"),
        "F12": it("step2/items_s1_c12.jsonl.gz"), "F8e": it("step2/items_s1_c8.jsonl.gz")},
}

print("XML / JSON (seed 0 | seed 1)")
for name, k in (("Unpatched", "U"), ("Ridge edit, their settings", "A"), ("Their released patch", "D"),
                ("Constant bias on <|im_end|> (b = -6 | b = -8)", "B"), ("Same-norm random row edit", "C")):
    cells = [f"{a(S[s][k], V4):.1f} / {a(S[s][k], V1):.1f}" if k in S[s] else "-" for s in (0, 1)]
    print(f"  {name:48s} {cells[0]:14s} | {cells[1]}")
for s in (0, 1):
    x = S[s]
    print(f"  seed {s}: edit vs constant on XML {mc(x['A'], x['B'], V4)}; identical XML texts {same(x['A'], x['B'], V4)}/204; "
          f"their patch vs edit identical XML texts {same(x['D'], x['A'], V4)}/204")

print("\nXML gain over unpatched, by constant bias b (JSON accuracy in brackets)")
for s, d in ((0, "stage2"), (1, "stage2_s1")):
    u = a(S[s]["U"], V4)
    row = []
    for b in (-2, -4, -6, -8, -10, -12, -16):
        x = it(f"{d}/items_B{b}.jsonl.gz")
        row.append(f"{b}: {a(x, V4) - u:+.1f} [{a(x, V1):.1f}]")
    print(f"  seed {s}: " + "  ".join(row))
    for b in (-2, -12):
        x = it(f"{d}/items_B{b}.jsonl.gz")
        print(f"    edit vs b = {b} on XML: {mc(S[s]['A'], x, V4)}")

print("\nPlan-first / JSON (seed 0 | seed 1)")
for name, k in (("Unpatched", "U"), ("Two-row edit, their settings", "A2"), ("Their released two-row patch", "D2"),
                ("First-token constant (c = +8)", "F"), ("Constant on '<' at every position", "B2"),
                ("Same-norm random two-row edit", "C2")):
    cells = [f"{a(S[s][k], V11):.1f} / {a(S[s][k], V1):.1f}" for s in (0, 1)]
    print(f"  {name:48s} {cells[0]:14s} | {cells[1]}")
for s in (0, 1):
    x = S[s]
    print(f"  seed {s}: first-token vs edit {mc(x['F'], x['A2'], V11)}; all-position vs edit {mc(x['B2'], x['A2'], V11)}; "
          f"their patch vs our fit {mc(x['D2'], x['A2'], V11)}; edit vs first-token on XML {mc(x['A2'], x['F'], V4)}; "
          f"edit vs all-position on XML {mc(x['A2'], x['B2'], V4)}")
    print(f"          JSON replies starting '<': first-token {sum(r['text'].lstrip().startswith('<') for k, r in x['F'].items() if k[1] == V1)}, "
          f"all-position {sum(r['text'].lstrip().startswith('<') for k, r in x['B2'].items() if k[1] == V1)}")
x = S[1]
print(f"  seed 1, first-token c = +12 (re-tuned): plan-first {a(x['F12'], V11):.1f}, JSON {a(x['F12'], V1):.1f}, XML {a(x['F12'], V4):.1f}; "
      f"vs edit {mc(x['F12'], x['A2'], V11)}; vs their patch {mc(x['F12'], x['D2'], V11)}; XML vs edit {mc(x['F12'], x['A2'], V4)}; "
      f"c = +8 in the same engine {a(x['F8e'], V11):.1f}")
print(f"    opens '<plan': {sum(r['text'].startswith('<plan') for k, r in x['F12'].items() if k[1] == V11)}/204; "
      f"JSON replies starting '<': {sum(r['text'].lstrip().startswith('<') for k, r in x['F12'].items() if k[1] == V1)}")

print("\nUnpatched baselines: JSON / XML / XML multi-call (n=48) / plan-first")
runs = (("Seed 0", "stage1/items_sft_v1_s0.jsonl.gz"), ("Seed 0 rerun (identical config)", "step2/items_sft_v1_s0r.jsonl.gz"),
        ("Seed 1", "stage1/items_prev_a_s1.jsonl.gz"), ("Seed 2", "step2/items_sft_v1_s2.jsonl.gz"),
        ("Base model", "stage1/items_base.jsonl.gz"))
for name, p in runs:
    x = it(p)
    opened = sum(r["text"].lstrip().startswith("<plan>") for k, r in x.items() if k[1] == V11)
    print(f"  {name:32s} {a(x, V1):5.1f} / {a(x, V4):5.1f} / {a(x, V4, 'parallel'):5.1f} / {a(x, V11):5.1f}   (plans opened {opened}/204)")
print("  DAWE, reported (one run)          87.7 /  73.5 /  25.0 /  20.1")
