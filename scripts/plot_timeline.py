"""Usage: plot_timeline.py <report.sqlite> <out.png> [step] [title]
Draws one training step (default step 5) of an Nsight Systems report exported with
`nsys export --type sqlite` (from scripts/slurm/nsys_overlap.sh, one rank's report): GPU kernels split into forward,
backward and SGD; host<->device copies; and the MPI data movement of the main thread and of the overlap comm thread.
Open MPI moves intra-node messages with process_vm_readv (CMA single copy), which the osrt trace records per thread, so
those calls stand for the all-reduce. Steps are delimited by the loss MPI_Allreduce that ends every step."""
import sqlite3
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

db, out = sys.argv[1], sys.argv[2]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 5
title = sys.argv[4] if len(sys.argv) > 4 else ""
c = sqlite3.connect(db)
q = lambda sql, *a: c.execute(sql, a).fetchall()

# step k runs from the end of loss all-reduce k-1 to the end of loss all-reduce k
ends = [e for (e,) in q("SELECT end FROM MPI_COLLECTIVES_EVENTS ORDER BY start")]
assert 1 <= step < len(ends), f"step {step} not in 1..{len(ends) - 1}"
t0, t1 = ends[step - 1], ends[step]
# the step itself starts with the input copy, after any evaluation that follows the previous step
#   (the first host-to-device copy of at least 1 MB on the kernels' stream; overlap's bucket copies use their own stream)
first = q("SELECT min(start) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 1 AND start > ? AND start < ? "
          "AND bytes >= 1048576 AND streamId = (SELECT streamId FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY streamId "
          "ORDER BY count(*) DESC LIMIT 1)", t0, t1)[0][0]
t0 = first or t0
ms = lambda t: (t - t0) / 1e6

kernels = q("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName "
            "WHERE k.start >= ? AND k.start < ? ORDER BY k.start", t0, t1)
phase, spans = "forward", {"forward": [], "backward": [], "SGD": []}
for s, e, name in kernels:
    if name.startswith("sgd"):
        phase = "SGD"
    spans[phase].append((ms(s), ms(e) - ms(s)))
    if name.startswith("softmax_xent"):
        phase = "backward"
copies = {k: [(ms(s), ms(e) - ms(s)) for s, e in q(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = ? AND start >= ? AND start < ?", kind, t0, t1)]
    for k, kind in (("device to host", 2), ("host to device", 1))}

main = q("SELECT globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId "
         "WHERE s.value LIKE 'cudaLaunchKernel%' GROUP BY globalTid ORDER BY count(*) DESC LIMIT 1")[0][0]
mpi = q("SELECT o.globalTid, o.start, o.end FROM OSRT_API o JOIN StringIds s ON s.id = o.nameId "
        "WHERE s.value = 'process_vm_readv' AND o.start >= ? AND o.start < ?", t0, t1)
mpi_main = [(ms(s), ms(e) - ms(s)) for tid, s, e in mpi if tid == main]
mpi_comm = [(ms(s), ms(e) - ms(s)) for tid, s, e in mpi if tid != main]

rows = [("GPU kernels", [(spans[k], col) for k, col in (("forward", "C0"), ("backward", "C1"), ("SGD", "C2"))]),
        ("GPU copies", [(copies["device to host"], "C4"), (copies["host to device"], "C5")]),
        ("main thread: MPI", [(mpi_main, "C3")]),
        ("comm thread: MPI", [(mpi_comm, "C3")])]
fig, ax = plt.subplots(figsize=(10, 2.8))
for i, (label, series) in enumerate(rows):
    y = len(rows) - 1 - i
    for bars, col in series:
        # stretch very short events to 0.15 ms so they stay visible
        ax.broken_barh([(s, max(w, 0.15)) for s, w in bars], (y - 0.35, 0.7), color=col, linewidth=0)
ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
ax.set_xlim(0, ms(t1))
ax.set_xlabel(f"ms since the start of step {step}")
ax.set_title(title or f"One training step ({ms(t1):.0f} ms)")
ax.legend(handles=[Patch(color="C0", label="forward"), Patch(color="C1", label="backward"),
                   Patch(color="C2", label="SGD"), Patch(color="C4", label="copy to host"),
                   Patch(color="C5", label="copy to device"), Patch(color="C3", label="MPI transfer")],
          fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"{out}: step {step}, {ms(t1):.1f} ms, backward ends at {max((s + w for s, w in spans['backward']), default=0):.1f} ms,"
      f" {len(mpi_main)} main-thread / {len(mpi_comm)} comm-thread MPI transfers")
