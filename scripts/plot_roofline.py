"""Usage: plot_roofline.py <csv> <png>. Roofline: x = FLOP/DRAM byte, y = FLOP/s (bench time), log-log."""
import csv
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOFS = {"PCIE": (14e12, 112e12), "SXM2": (15.7e12, 125e12)}  # FP32, tensor; 900 GB/s HBM2 for both
BW = 900e9

rows = {}  # latest row wins per (label, M, N, K)
for r in csv.DictReader(open(sys.argv[1])):
    rows[(r["label"], r["M"], r["N"], r["K"])] = r
rows = list(rows.values())
for g in {r["gpu"] for r in rows}:
    if not any(v in g for v in ROOFS):
        sys.exit(f"plot_roofline: unknown GPU '{g}' (want PCIE or SXM2 in the name)")
variants = sorted({v for r in rows for v in ROOFS if v in r["gpu"]})
if len(variants) != 1:
    sys.exit(f"plot_roofline: CSV mixes GPU variants {variants}; split it first")
variant = variants[0]
fp32, tensor = ROOFS[variant]

fig, ax = plt.subplots(figsize=(8, 5.5))
xs = [10 ** (k / 20) for k in range(0, 81)]  # 1 .. 1e4
ax.plot(xs, [min(BW * x, tensor) for x in xs], "k-", label=f"HBM2 900 GB/s / tensor {tensor / 1e12:g} TFLOP/s")
ax.plot(xs, [min(BW * x, fp32) for x in xs], "k--", label=f"FP32 {fp32 / 1e12:g} TFLOP/s")

impls = list(dict.fromkeys(r["impl"] for r in rows))
shapes = list(dict.fromkeys((r["M"], r["N"], r["K"]) for r in rows))
for r in rows:
    f = float(r["flops"])
    ax.scatter(f / float(r["dram_bytes"]), f / float(r["bench_time_s"]),
               marker="osD^vP*X"[impls.index(r["impl"]) % 8], color=f"C{shapes.index((r['M'], r['N'], r['K']))}", s=50, zorder=3)
    ax.annotate(r["label"], (f / float(r["dram_bytes"]), f / float(r["bench_time_s"])), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
for j, s in enumerate(shapes):
    ax.scatter([], [], color=f"C{j}", label="x".join(s))
for k, i in enumerate(impls):
    ax.scatter([], [], color="gray", marker="osD^vP*X"[k % 8], label=i)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 1e4)
ax.set_ylim(1e11, 2e14)
ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte)")
ax.set_ylabel("FLOP/s")
ax.set_title(f"Roofline, V100-{variant}" + (" (analytical bytes)" if any(r["source"] == "analytic" for r in rows) else ""))
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150)
