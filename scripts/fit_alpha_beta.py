"""Usage: fit_alpha_beta.py <dir>. Reads the bench_comm CSVs in <dir> (from scripts/slurm/bench_comm_sweep.sh),
fits latency alpha and inverse bandwidth beta from ping-pong, writes <dir>/alpha_beta.txt and
<dir>/allreduce_model.png (measured vs alpha-beta model), and prints where model and measurement disagree."""
import csv
import math
import os
import statistics
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
MB = 1 << 20


def load(name):
    path = os.path.join(d, name + ".csv")
    rows = list(csv.DictReader(open(path)))
    if len(rows) != 70:
        print(f"warning: {path} has {len(rows) + 1} lines, expected 71")
    return rows


def fit(rows):
    """alpha = median one-way time for messages <= 1 KB; beta = least-squares slope of (t - alpha) vs bytes
    through the origin, for messages >= 1 MB."""
    pp = [(int(r["bytes"]), float(r["seconds"])) for r in rows if r["kind"] == "pingpong"]
    alpha = statistics.median(t for b, t in pp if b <= 1024)
    big = [(b, t - alpha) for b, t in pp if b >= MB]
    beta = sum(b * y for b, y in big) / sum(b * b for b, _ in big)
    return alpha, beta


# n = message bytes, p = ranks. Each step sends one message; the reduction arithmetic is ignored.
MODEL = {
    "naive": lambda p, n, a, b: 2 * (p - 1) * (a + n * b),  # root receives p-1 and sends p-1 full messages
    "tree": lambda p, n, a, b: math.log2(p) * (a + n * b),  # recursive doubling, p a power of two
    "ring": lambda p, n, a, b: 2 * (p - 1) * (a + n * b / p),  # reduce-scatter + all-gather of n/p chunks
}

fits = {k: fit(load(f"{k}_np2")) for k in ("intra", "inter")}
with open(os.path.join(d, "alpha_beta.txt"), "w") as f:
    for k, (a, b) in fits.items():
        f.write(f"{k} alpha_s={a:.4e} beta_s_per_byte={b:.4e}\n")
        print(f"{k}: alpha = {a * 1e6:.2f} us, bandwidth = {1 / b / 1e9:.2f} GB/s")

# 4+ ranks at 2 per node always cross InfiniBand, and the slowest link sets the pace: use the inter-node fit.
a, b = fits["inter"]
nps = (4, 8, 16)
fig, axes = plt.subplots(1, len(nps), figsize=(15, 4.5), sharey=True)
print("\nmeasured / model, messages >= 1 MB (1.0 = exact):")
for ax, p in zip(axes, nps):
    rows = [r for r in load(f"allreduce_np{p}") if r["kind"] == "allreduce"]
    for algo in ("naive", "tree", "ring", "mpi"):
        pts = [(int(r["bytes"]), float(r["seconds"])) for r in rows if r["algo"] == algo]
        xs = [x for x, _ in pts]
        line, = ax.plot(xs, [t for _, t in pts], marker="o", ms=3, label=f"{algo} measured")
        if algo in MODEL:
            ax.plot(xs, [MODEL[algo](p, x, a, b) for x in xs], ls="--", color=line.get_color(), label=f"{algo} model")
            ratios = [t / MODEL[algo](p, x, a, b) for x, t in pts if x >= MB]
            print(f"  np={p:<2} {algo:<5} " + " ".join(f"{r:5.2f}" for r in ratios)
                  + f"   (max {max(ratios, key=lambda r: max(r, 1 / r)):.2f})")
    ax.set(xscale="log", yscale="log", xlabel="message bytes", title=f"{p} ranks (2 per node)")
    ax.grid(True, which="major", alpha=0.3)
axes[0].set_ylabel("all-reduce time (s)")
axes[0].legend(fontsize=7)
fig.suptitle(f"All-reduce: measured (solid) vs alpha-beta model (dashed), "
             f"inter-node alpha = {a * 1e6:.1f} us, bandwidth = {1 / b / 1e9:.2f} GB/s")
fig.tight_layout()
fig.savefig(os.path.join(d, "allreduce_model.png"), dpi=150)
print(f"\nwrote {os.path.join(d, 'alpha_beta.txt')} and {os.path.join(d, 'allreduce_model.png')}")
