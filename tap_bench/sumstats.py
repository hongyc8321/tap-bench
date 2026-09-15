#!/usr/bin/env python3
import csv
from collections import defaultdict

with open("/home/rhap/nkst/tap_bench/probe_set/gt_curves.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

stats = defaultdict(lambda: defaultdict(int))
for r in rows:
    c = r["category"]
    stats[c]["total"] += 1
    if r["reversal_pred"] == "True":
        stats[c]["reversal"] += 1
    ft = float(r["f_tex_hz"])
    if ft <= 10.5:
        stats[c]["sub_nyq"] += 1
    elif ft <= 13.5:
        stats[c]["critical"] += 1
    else:
        stats[c]["super_nyq"] += 1

print("cat\ttotal\treversal\tsub_nyq\tcritical\tsuper_nyq")
for c, s in stats.items():
    print(f"{c}\t{s['total']}\t{s['reversal']}\t{s['sub_nyq']}\t"
          f"{s['critical']}\t{s['super_nyq']}")
