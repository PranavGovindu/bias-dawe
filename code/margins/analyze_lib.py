"""Helpers: load margin rows, derive margins (optionally under row edits / constant biases)."""
import gzip
import json
from pathlib import Path
EOS, LT, LBR, NL, NN = 248046, 27, 4754, 198, 271
RES = Path(__file__).resolve().parents[2] / "results"  # committed per-item results and margin files


def jopen(p):
    p = str(p)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load(seed):
    with jopen(RES / "margins" / f"margins_{seed}.jsonl.gz") as f:
        return [json.loads(l) for l in f]


def logits(r, add=None):
    """dict token->logit for top-10 plus interest tokens, with additive per-token changes `add` {tok: delta}."""
    z = {int(t): v for t, v in r["top"]}
    for t, v in r["z"].items():
        z.setdefault(int(t), v)
    z.setdefault(r["gold"], r["z_gold"])
    if add:
        for t, dv in add.items():
            if t not in z:
                raise KeyError(t)
            z[t] = z[t] + dv
    return z


def best_other(z, excl):
    """max logit over tokens not in excl. Exact when z holds the top-10 and <= 9 tokens were edited."""
    t = max((k for k in z if k not in excl), key=lambda k: z[k])
    return t, z[t]


def m_gold(r, add=None):
    z = logits(r, add)
    t, v = best_other(z, {r["gold"]})
    return z[r["gold"]] - v, t


def m_eot(r, add=None):
    """max non-eos minus eos (positive = continues). For stop points use -m_eot."""
    z = logits(r, add)
    t, v = best_other(z, {EOS})
    return v - z[EOS], t


def pair(r, a, b, add=None):
    z = logits(r, add)
    return z[a] - z[b]


def edit_add(r, names):
    """per-token logit change from a set of edit names in r['shift'] (e.g. A2_27 -> token 27)."""
    tokmap = {"27": LT, "eos": EOS}
    add = {}
    for n in names:
        if n.startswith("P_"):
            t = int(n[2:])
        else:
            t = tokmap[n.split("_")[1]]
        add[t] = add.get(t, 0.0) + r["shift"][n]
    return add


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
