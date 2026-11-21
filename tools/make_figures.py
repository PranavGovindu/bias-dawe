"""Render figures/*.png from the per-item files in results/ (matplotlib). Every plotted number is computed here from
results/; the only constant is DAWE's reported plan-first baseline (20.1), drawn as "reported".

usage: python3 tools/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from analyze_stage2 import acc, load_items  # noqa: E402

R = ROOT / "results"
OUT = ROOT / "figures"
DAWE_PLANFIRST = 20.1  # DAWE's reported unpatched plan-first accuracy (their single run)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  wash="#f0efec", s1="#2a78d6", s2="#eb6834", other="#c3c2b7"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 wash="#262624", s1="#3987e5", s2="#d95926", other="#5c5b57"),
}
plt.rcParams.update({"font.family": ["Inter", "Noto Sans", "DejaVu Sans"], "font.size": 10})


def a(path, h, cat=None):
    return acc(load_items(R / path), h, cat)[0]


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.grid(axis="y", color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def fig_xml_bias(theme):
    t = THEMES[theme]
    bs = [0, -2, -4, -6, -8, -10, -12, -16]
    seeds = {
        0: dict(dir="stage2", u="stage1/items_sft_v1_s0.jsonl.gz", chosen=-6, note="b = −6, chosen with test visible"),
        1: dict(dir="stage2_s1", u="stage1/items_prev_a_s1.jsonl.gz", chosen=-8, note="b = −8, chosen on train"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, facecolor=t["surface"])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.14, wspace=0.08)
    for ax, (s, cfg) in zip(axes, seeds.items()):
        style_axes(ax, t)
        xml, js = [], []
        for b in bs:
            p = cfg["u"] if b == 0 else f"{cfg['dir']}/items_B{b}.jsonl.gz"  # b = 0 is the unpatched checkpoint
            xml.append(a(p, "v4_format"))
            js.append(a(p, "v1"))
        edit = a(f"{cfg['dir']}/items_A.jsonl.gz", "v4_format")
        theirs = a(f"{cfg['dir']}/items_D.jsonl.gz", "v4_format")
        assert edit == theirs, (s, edit, theirs)  # identical on both seeds, so one reference line
        ax.axvspan(-10.5, -5.5, color=t["wash"], zorder=0, linewidth=0)
        ax.text(-8, 4, "ties the edit\non both seeds", ha="center", va="bottom", fontsize=8.5, color=t["ink2"])
        ax.axhline(edit, color=t["ink2"], linewidth=1.25, zorder=2, label="ridge edit = their patch (XML)")
        kw = dict(linewidth=2, solid_capstyle="round", solid_joinstyle="round", marker="o", markersize=7,
                  markeredgewidth=2, markeredgecolor=t["surface"], zorder=3)
        ax.plot(bs, js, color=t["s2"], label="JSON (v1)", **kw)
        ax.plot(bs, xml, color=t["s1"], label="XML (v4)", **kw)
        i = bs.index(cfg["chosen"])
        ax.annotate(cfg["note"], xy=(cfg["chosen"], xml[i]), xytext=(cfg["chosen"] + 1.2, 62), fontsize=8.5,
                    color=t["ink2"], ha="left", arrowprops=dict(arrowstyle="-", color=t["muted"], linewidth=1))
        ax.text(-16.3, js[-1] + 2.5, f"JSON {js[-1]:.1f}", ha="right", va="bottom", fontsize=8.5, color=t["ink"])
        ax.text(-15.5, xml[-1] - 1, f"XML {xml[-1]:.1f}", ha="right", va="center", fontsize=8.5, color=t["ink"])
        ax.text(0.2, xml[0] - 3, f"unpatched\nXML {xml[0]:.1f}", ha="left", va="top", fontsize=8.5, color=t["ink"])
        ax.set_xlim(0.6, -16.6)
        ax.set_xticks(bs)
        ax.set_xticklabels([str(b).replace("-", "−") for b in bs])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(f"Seed {s}", loc="left", fontsize=10.5, color=t["ink"], fontweight="semibold", pad=6)
        ax.set_title(f"edit and their patch: XML {edit:.1f}", loc="right", fontsize=9, color=t["ink2"], pad=6)
        ax.set_xlabel("constant bias b on <|im_end|> (logits)", color=t["ink2"], fontsize=9)
    axes[0].set_ylabel("accuracy, % of 204 held-out items", color=t["ink2"], fontsize=9)
    fig.text(0.07, 0.95, "XML: a tuned constant end-of-turn bias matches DAWE's edit", fontsize=13, color=t["ink"],
             fontweight="semibold", ha="left", va="top")
    fig.text(0.07, 0.885, "JSON accuracy doesn't move for b = −2…−12. Only b = −6…−10 ties the edit on both seeds; "
             "at −2 or −12 the edit is better (p ≤ 0.03).", fontsize=9.5, color=t["ink2"], ha="left", va="top")
    h, lab = axes[0].get_legend_handles_labels()
    order = [lab.index(x) for x in ("XML (v4)", "JSON (v1)", "ridge edit = their patch (XML)")]
    leg = fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper left", bbox_to_anchor=(0.065, 0.85), ncol=3, frameon=False,
                     fontsize=9, labelcolor=t["ink"], handlelength=1.6)
    for line in leg.get_lines():
        line.set_markersize(6)
    out = OUT / ("xml_bias.png" if theme == "light" else "xml_bias_dark.png")
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return out


def rounded_hbar(ax, y, w, h, color, radius_px=8):
    """Horizontal bar, square at the baseline, rounded data end (radius_px in output pixels; 8 px = 4 px at 1x)."""
    if w <= 0:
        return
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    rx, ry = radius_px / abs(x1 - x0), radius_px / abs(y1 - y0)  # radius in data units along x and y
    rx = min(rx, w)
    ax.add_patch(FancyBboxPatch((0, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=ry / rx, facecolor=color, edgecolor="none", zorder=3))
    if w > rx:
        ax.add_patch(Rectangle((0, y - h / 2), w - rx, h, facecolor=color, edgecolor="none", zorder=3))


def fig_planfirst_baseline(theme):
    t = THEMES[theme]
    runs = [("Seed 0", "stage1/items_sft_v1_s0.jsonl.gz"),
            ("Seed 0 rerun, identical config", "step2/items_sft_v1_s0r.jsonl.gz"),
            ("Seed 1", "stage1/items_prev_a_s1.jsonl.gz"),
            ("Seed 2", "step2/items_sft_v1_s2.jsonl.gz")]
    vals = [(name, a(p, "v11_plan"), True) for name, p in runs] + [("DAWE, reported (one run)", DAWE_PLANFIRST, False)]
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=t["surface"])
    fig.subplots_adjust(left=0.27, right=0.95, top=0.74, bottom=0.14)
    style_axes(ax, t)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=1, linestyle="-")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(t["axis"])
    n = len(vals)
    h = 0.42
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.6, n - 0.4)
    fig.canvas.draw()  # fix the transforms before converting the corner radius to data units
    for i, (name, v, ours) in enumerate(vals):
        y = n - 1 - i
        rounded_hbar(ax, y, v, h, t["s1"] if ours else t["other"])
        label = f"{v:.1f}" + ("  (no plans opened)" if ours and v == 0 else "")
        ax.text(v + 0.5, y, label, va="center", ha="left", fontsize=9, color=t["ink"])
    ax.set_yticks(range(n))
    ax.set_yticklabels([v[0] for v in vals][::-1], color=t["ink"], fontsize=9)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("unpatched plan-first accuracy, % of 204 held-out items", color=t["ink2"], fontsize=9)
    fig.text(0.03, 0.95, "Plan-first: the unpatched baseline moves a lot between runs", fontsize=13, color=t["ink"],
             fontweight="semibold", ha="left", va="top")
    fig.text(0.03, 0.865, "Same recipe; only the seed changes, or only run-to-run numerics (identical-config rerun). DAWE reports one run.",
             fontsize=9.5, color=t["ink2"], ha="left", va="top")
    out = OUT / ("planfirst_baseline.png" if theme == "light" else "planfirst_baseline_dark.png")
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for theme in THEMES:
        print(fig_xml_bias(theme))
        print(fig_planfirst_baseline(theme))
