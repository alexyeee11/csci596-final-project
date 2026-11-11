"""Usage: plot_gemm.py <csv> <png>. Left: GFLOP/s vs size (square). Right: training shapes."""
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open(sys.argv[1])))
impls = list(dict.fromkeys(r["impl"] for r in rows))
square = lambda r: r["M"] == r["N"] == r["K"]

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
for i in impls:
    d = [r for r in rows if r["impl"] == i and square(r)]
    a.plot([int(r["M"]) for r in d], [float(r["gflops"]) for r in d], marker="o", label=i)
a.set_xscale("log", base=2)
a.set_xlabel("N (square N x N x N)")
a.set_ylabel("GFLOP/s")
a.set_title("Square GEMM")
a.legend()

tr = {i: [r for r in rows if r["impl"] == i and not square(r)] for i in impls}
labels = [f"{r['M']}x{r['N']}x{r['K']}" + ("\nA^T" if r["transA"] == "1" else "\nB^T" if r["transB"] == "1" else "")
          for r in tr[impls[0]]]
w = 0.8 / len(impls)
for j, i in enumerate(impls):
    b.bar([x + j * w for x in range(len(labels))], [float(r["gflops"]) for r in tr[i]], w, label=i)
b.set_xticks([x + 0.4 - w / 2 for x in range(len(labels))])
b.set_xticklabels(labels, fontsize=8)
b.set_ylabel("GFLOP/s")
b.set_title("Training GEMMs (batch 512, hidden 4096)")
b.legend()
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150)
