"""Usage: plot_pipeline.py <dir with p1_m1.log and p{2,4}_m{1,2,4,8,16}.log>. Writes summary.csv and bubble.png there.
Per run: median of steps 5..29. bubble_measured = 1 - compute_s / step_s (compute_s = mean busy time over stages);
bubble_model = (P-1) / (M+P-1), the GPipe bubble with equal stages and free transfers."""
import csv
import statistics
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
PS, MS = (2, 4), (1, 2, 4, 8, 16)


def medians(name):
    # step lines: step i loss L compute_s c comm_s m step_s s
    st = [l.split() for l in open(f"{d}/{name}.log") if l.startswith("step ")]
    st = [(float(w[5]), float(w[7]), float(w[9])) for w in st if 5 <= int(w[1]) <= 29]
    assert st, f"no steps 5..29 in {name}.log"
    return [statistics.median(x[j] for x in st) for j in range(3)]


rows = []
for P, M in [(1, 1)] + [(P, M) for P in PS for M in MS]:
    c, m, s = medians(f"p{P}_m{M}")
    rows.append(dict(P=P, M=M, compute_s=c, comm_s=m, step_s=s, bubble_measured=1 - c / s,
                     bubble_model=(P - 1) / (M + P - 1)))
with open(f"{d}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
for i, P in enumerate(PS):
    r = [q for q in rows if q["P"] == P]
    a.plot(MS, [q["bubble_measured"] for q in r], marker="o", color=f"C{i}", label=f"P={P} measured")
    a.plot(MS, [q["bubble_model"] for q in r], ls="--", color=f"C{i}", label=f"P={P} model (P-1)/(M+P-1)")
    b.plot(MS, [1000 * q["step_s"] for q in r], marker="o", color=f"C{i}", label=f"P={P}")
base = rows[0]["step_s"] * 1000
b.axhline(base, color="k", ls=":", label="1 GPU")
a.set(xlabel="micro-batches M", ylabel="bubble fraction", title="Pipeline bubble (batch 512)", ylim=(0, 1))
b.set(xlabel="micro-batches M", ylabel="ms per step", title="Step time")
for ax in (a, b):
    ax.set_xscale("log", base=2)
    ax.set_xticks(MS, [str(m) for m in MS])
    ax.set_xticks([], minor=True)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{d}/bubble.png", dpi=150)
