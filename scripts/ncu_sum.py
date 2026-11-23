"""Usage: ncu_sum.py <ncu csv log>. Prints 'dram_bytes ncu_time_s', summed over every kernel row."""
import csv
import sys

s = {}
for r in csv.DictReader(l for l in open(sys.argv[1]) if l.startswith('"')):
    s[r["Metric Name"]] = s.get(r["Metric Name"], 0) + float(r["Metric Value"].replace(",", ""))
if "gpu__time_duration.sum" not in s:
    sys.exit("ncu_sum: no kernels profiled")
print(int(s["dram__bytes_read.sum"] + s["dram__bytes_write.sum"]), s["gpu__time_duration.sum"] / 1e9)
