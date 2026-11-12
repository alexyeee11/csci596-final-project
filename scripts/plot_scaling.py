"""Usage: plot_scaling.py <dir with {strong,weak}_n{N}_{off,on}.log>. Writes summary.csv and scaling.png there.
Per run: median of steps 5..29. speedup = step_s(1)/step_s(N); efficiency = speedup/N for strong, speedup for weak
(weak efficiency = t(1)/t(N) since the work per GPU is constant)."""
import csv
import os
import statistics
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
NS = (1, 2, 4, 8)
rows = []
for kind in ("strong", "weak"):
    for ov in ("off", "on"):
        med = {}
        for n in NS:
            # step lines: step i loss L compute_s c comm_s m step_s s
            st = [l.split() for l in open(f"{d}/{kind}_n{n}_{ov}.log") if l.startswith("step ")]
            st = [(float(w[5]), float(w[7]), float(w[9])) for w in st if 5 <= int(w[1]) <= 29]
            assert st, f"no steps 5..29 in {kind}_n{n}_{ov}.log"
            med[n] = [statistics.median(x[j] for x in st) for j in range(3)]
        for n in NS:
            sp = med[1][2] / med[n][2]
            rows.append(dict(kind=kind, overlap=ov, np=n, compute_s=med[n][0], comm_s=med[n][1], step_s=med[n][2],
                             speedup=sp, efficiency=sp / n if kind == "strong" else sp))
with open(f"{d}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

get = lambda kind, ov: [r for r in rows if r["kind"] == kind and r["overlap"] == ov]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))
a.plot(NS, NS, "k--", label="ideal")
for ov in ("off", "on"):
    a.plot(NS, [r["speedup"] for r in get("strong", ov)], marker="o", label=f"overlap {ov}")
    b.plot(NS, [r["efficiency"] for r in get("weak", ov)], marker="o", label=f"overlap {ov}")
b.axhline(1, color="k", ls="--", label="ideal")
a.set(xlabel="GPUs", ylabel="speedup", title="Strong scaling (batch 512)", xscale="log", yscale="log")
b.set(xlabel="GPUs", ylabel="t(1) / t(N)", title="Weak scaling (128 per GPU)", xscale="log", ylim=(0, 1.1))
for ax in (a, b):
    ax.set_xticks(NS)
    ax.set_xticklabels(NS)
    ax.legend()
w = 0.38
for j, ov in enumerate(("off", "on")):
    r = get("strong", ov)
    x = [i + (j - 0.5) * w for i in range(len(NS))]
    comp = [1000 * q["compute_s"] for q in r]
    c.bar(x, comp, w, color="C0", label="compute" if j == 0 else None)
    c.bar(x, [1000 * q["comm_s"] for q in r], w, bottom=comp, color="C3", label="exposed comm" if j == 0 else None)
    for xi, q in zip(x, r):
        c.text(xi, 1000 * (q["compute_s"] + q["comm_s"]), ov, ha="center", va="bottom", fontsize=8)
c.set_xticks(range(len(NS)))
c.set_xticklabels(NS)
c.set(xlabel="GPUs (left bar overlap off, right on)", ylabel="ms per step", title="Strong scaling step time")
c.legend()
fig.tight_layout()
fig.savefig(f"{d}/scaling.png", dpi=150)
