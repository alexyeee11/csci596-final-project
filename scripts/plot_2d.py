"""Usage: plot_2d.py <dir with dp<dp>_pp<pp>[_m<M>].log> [1-GPU log]. Writes summary.csv and layouts.png there.
Per run: median of steps 5..29 of compute_s, comm_s, step_s. Logs without _m<M> are M = 8 for pp > 1 (M = 1 for
pp = 1). For pp > 1, comm_s is everything but the mean kernel time: bubble, activation transfers and the per-stage
gradient all-reduce. The optional 1-GPU log (e.g. results/pipeline/p1_m1.log) is drawn as a reference line."""
import csv
import glob
import os
import re
import statistics
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else None


def medians(path):
    # step lines: step i loss L compute_s c comm_s m step_s s
    st = [l.split() for l in open(path) if l.startswith("step ")]
    st = [(float(w[5]), float(w[7]), float(w[9])) for w in st if 5 <= int(w[1]) <= 29]
    assert st, f"no steps 5..29 in {path}"
    return [statistics.median(x[j] for x in st) for j in range(3)]


rows = []
for path in glob.glob(f"{d}/dp*_pp*.log"):
    g = re.fullmatch(r"dp(\d+)_pp(\d+)(?:_m(\d+))?\.log", os.path.basename(path))
    if not g:
        continue
    dp, pp = int(g[1]), int(g[2])
    M = int(g[3]) if g[3] else (8 if pp > 1 else 1)
    c, m, s = medians(path)
    rows.append(dict(dp=dp, pp=pp, M=M, compute_s=c, comm_s=m, step_s=s))
assert rows, f"no dp*_pp*.log in {d}"
rows.sort(key=lambda r: (r["pp"], -r["M"]))
with open(f"{d}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

# one group per layout, one stacked bar (kernels + the rest) per micro-batch count
layouts = sorted({(r["dp"], r["pp"]) for r in rows}, key=lambda l: l[1])
Ms = sorted({r["M"] for r in rows if r["pp"] > 1}, reverse=True) or [1]
width = 0.8 / len(Ms)
fig, ax = plt.subplots(figsize=(7, 4.2))
for gi, (dp, pp) in enumerate(layouts):
    group = [r for r in rows if (r["dp"], r["pp"]) == (dp, pp)]
    for r in group:
        j = Ms.index(r["M"]) if pp > 1 else (len(Ms) - 1) / 2
        x = gi + (j - (len(Ms) - 1) / 2) * width
        hatch = "//" if pp > 1 and r["M"] != Ms[0] else None
        ax.bar(x, 1000 * r["compute_s"], width * 0.95, color="C0", hatch=hatch, edgecolor="white")
        ax.bar(x, 1000 * r["comm_s"], width * 0.95, bottom=1000 * r["compute_s"], color="C1", hatch=hatch,
               edgecolor="white")
        ax.text(x, 1000 * r["step_s"], f"{1000 * r['step_s']:.0f}" + (f"\nM={r['M']}" if pp > 1 else ""),
                ha="center", va="bottom", fontsize=8)
ax.bar(0, 0, color="C0", label="compute (mean kernel time)")
ax.bar(0, 0, color="C1", label="the rest: all-reduce, bubble, transfers")
if base:
    b = medians(base)[2] * 1000
    ax.axhline(b, color="k", ls=":", label=f"1 GPU ({b:.0f} ms)")
ax.set_xticks(range(len(layouts)), [f"dp={dp} x pp={pp}" for dp, pp in layouts])
ax.set(ylabel="ms per step (median of steps 5-29)",
       title=f"{layouts[0][0] * layouts[0][1]} GPUs, batch 512: data x pipeline layouts")
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(f"{d}/layouts.png", dpi=150)
