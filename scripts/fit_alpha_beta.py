"""Usage: fit_alpha_beta.py <dir>. Reads the bench_comm CSVs in <dir> (from scripts/slurm/bench_comm_sweep.sh),
fits latency alpha and inverse bandwidth beta from ping-pong, then the reduction cost gamma from the 2-rank
inter-node all-reduce. Writes <dir>/alpha_beta.txt and <dir>/allreduce_model.png (measured vs model), and prints
where model and measurement disagree."""
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


def through_origin(pts):
    """Least-squares slope of y against x with no intercept."""
    return sum(x * y for x, y in pts) / sum(x * x for x, _ in pts)


def fit(rows):
    """alpha = median one-way time for messages <= 1 KB; beta = slope of (t - alpha) vs bytes for messages >= 1 MB."""
    pp = [(int(r["bytes"]), float(r["seconds"])) for r in rows if r["kind"] == "pingpong"]
    alpha = statistics.median(t for b, t in pp if b <= 1024)
    return alpha, through_origin([(b, t - alpha) for b, t in pp if b >= MB])


# n = message bytes, p = ranks. COMM: time on the wire, one message per step.
# REDUCED: bytes the slowest rank adds up, which the alpha-beta model ignores; gamma is the cost per byte added.
COMM = {
    "naive": lambda p, n, a, b: 2 * (p - 1) * (a + n * b),  # root receives p-1 and sends p-1 full messages
    "tree": lambda p, n, a, b: math.log2(p) * (a + n * b),  # recursive doubling, p a power of two
    "ring": lambda p, n, a, b: 2 * (p - 1) * (a + n * b / p),  # reduce-scatter + all-gather of n/p chunks
}
REDUCED = {
    "naive": lambda p, n: (p - 1) * n,  # root adds every other rank's buffer
    "tree": lambda p, n: math.log2(p) * n,  # one full add per round
    "ring": lambda p, n: (p - 1) * n / p,  # one chunk per reduce-scatter step
}


def allreduce_pts(rows, algo, min_bytes=0):
    return [(int(r["bytes"]), float(r["seconds"])) for r in rows
            if r["kind"] == "allreduce" and r["algo"] == algo and int(r["bytes"]) >= min_bytes]


fits = {k: fit(load(f"{k}_np2")) for k in ("intra", "inter")}
a, b = fits["inter"]
# gamma: what the 2-rank inter-node all-reduce spends beyond the wire time, per byte added (messages >= 1 MB)
inter2 = load("inter_np2")
g = through_origin([(REDUCED[algo](2, n), t - COMM[algo](2, n, a, b))
                    for algo in COMM for n, t in allreduce_pts(inter2, algo, MB)])
with open(os.path.join(d, "alpha_beta.txt"), "w") as f:
    for k, (fa, fb) in fits.items():
        f.write(f"{k} alpha_s={fa:.4e} beta_s_per_byte={fb:.4e}\n")
        print(f"{k}: alpha = {fa * 1e6:.2f} us, bandwidth = {1 / fb / 1e9:.2f} GB/s")
    f.write(f"reduce gamma_s_per_byte={g:.4e}\n")
    print(f"reduce: gamma = {g:.3e} s/byte = {1 / g / 1e9:.2f} GB/s added, {g / b:.1f}x beta")

# 4+ ranks at 2 per node always cross InfiniBand, and the slowest link sets the pace: use the inter-node fit.
nps = (4, 8, 16)
fig, axes = plt.subplots(1, len(nps), figsize=(15, 4.5), sharey=True)
print("\nmeasured / model for messages >= 1 MB (1.0 = exact), alpha-beta | alpha-beta-gamma:")
for ax, p in zip(axes, nps):
    rows = load(f"allreduce_np{p}")
    for algo in ("naive", "tree", "ring", "mpi"):
        pts = allreduce_pts(rows, algo)
        xs = [x for x, _ in pts]
        line, = ax.plot(xs, [t for _, t in pts], marker="o", ms=3, label=f"{algo} measured")
        if algo not in COMM:
            continue
        ab = lambda n: COMM[algo](p, n, a, b)
        abg = lambda n: ab(n) + g * REDUCED[algo](p, n)
        c = line.get_color()
        ax.plot(xs, [ab(x) for x in xs], ls="--", color=c, alpha=0.6, label=f"{algo} α-β")
        ax.plot(xs, [abg(x) for x in xs], ls=":", color=c, lw=2, label=f"{algo} α-β-γ")
        big = [(x, t) for x, t in pts if x >= MB]
        print(f"  np={p:<2} {algo:<5} " + " ".join(f"{t / ab(x):5.2f}" for x, t in big)
              + "  | " + " ".join(f"{t / abg(x):5.2f}" for x, t in big))
    ax.set(xscale="log", yscale="log", xlabel="message bytes", title=f"{p} ranks (2 per node)")
    ax.grid(True, which="major", alpha=0.3)
axes[0].set_ylabel("all-reduce time (s)")
axes[0].legend(fontsize=6, ncol=2)
fig.suptitle(f"All-reduce measured (solid) vs α-β (dashed) and α-β-γ (dotted) models: inter-node α = {a * 1e6:.1f} µs, "
             f"{1 / b / 1e9:.1f} GB/s; reduction {1 / g / 1e9:.1f} GB/s")
fig.tight_layout()
fig.savefig(os.path.join(d, "allreduce_model.png"), dpi=150)
print(f"\nwrote {os.path.join(d, 'alpha_beta.txt')} and {os.path.join(d, 'allreduce_model.png')}")
