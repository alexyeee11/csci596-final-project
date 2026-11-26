"""Usage: plot_bucket.py <dir with bucket_off.log and bucket_<B>.log>. Writes bucket_sweep.csv and bucket_sweep.png there.
Per run: median of steps 5..29. With --overlap, comm_s is the communication still outstanding once backward is done
(exposed comm); with overlap off it is the whole all-reduce."""
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


def medians(path):
    # step lines: step i loss L compute_s c comm_s m step_s s
    st = [l.split() for l in open(path) if l.startswith("step ")]
    st = [(float(w[5]), float(w[7]), float(w[9])) for w in st if 5 <= int(w[1]) <= 29]
    assert st, f"no steps 5..29 in {path}"
    return [statistics.median(x[j] for x in st) for j in range(3)]


off = medians(f"{d}/bucket_off.log")
rows = []
for path in glob.glob(f"{d}/bucket_*.log"):
    g = re.fullmatch(r"bucket_([0-9.]+)\.log", os.path.basename(path))
    if g:
        rows.append((float(g[1]), *medians(path)))
assert rows, f"no bucket_<B>.log in {d}"
rows.sort()
with open(f"{d}/bucket_sweep.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["bucket_mb", "compute_s", "comm_s", "step_s"])
    w.writerow(["off"] + [f"{v:.6f}" for v in off])
    for B, c, m, s in rows:
        w.writerow([f"{B:g}"] + [f"{v:.6f}" for v in (c, m, s)])

Bs = [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(Bs, [1000 * r[3] for r in rows], marker="o", color="C0", label="step, overlap on")
ax.plot(Bs, [1000 * r[2] for r in rows], marker="s", color="C1", label="exposed comm, overlap on")
ax.axhline(1000 * off[2], color="C0", ls=":", label=f"step, overlap off ({1000 * off[2]:.0f} ms)")
ax.axhline(1000 * off[1], color="C1", ls=":", label=f"comm, overlap off ({1000 * off[1]:.0f} ms)")
best = min(rows, key=lambda r: r[3])
ax.annotate(f"best: {best[0]:g} MB, {1000 * best[3]:.0f} ms", (best[0], 1000 * best[3]), textcoords="offset points",
            xytext=(0, 9), ha="center", fontsize=8)
ax.set_xscale("log")
ax.set_xticks(Bs, [f"{B:g}" for B in Bs])
ax.set_xticks([], minor=True)
ax.set(xlabel="bucket size (MB)", ylabel="ms per step (median of steps 5-29)",
       title="Overlap bucket size, 8 GPUs, batch 512, ring")
ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(f"{d}/bucket_sweep.png", dpi=150)
