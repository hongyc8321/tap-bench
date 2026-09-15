import csv
from collections import defaultdict

ORIG = "/home/rhap/nkst/phase2_results_final/nyquist_rows.csv"
A0 = "/home/rhap/nkst/phase3_results_A0/nyquist_rows.csv"

BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0),
        (1.0, 1.5), (1.5, 2.0), (2.0, 99.0)]


def load(path, model=None):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["low_conf"] == "False"]
    if model:
        rows = [r for r in rows if r["model"].startswith(model)]
    return rows


def agg(rows):
    out = {}
    for lo, hi in BINS:
        sub = [r for r in rows
               if lo <= float(r["f_ratio"]) < hi and r["motion"] == "True"]
        n = len(sub)
        rev = sum(0 if r["sign_match"] == "True" else 1 for r in sub)
        sr = [float(r["speed_ratio"]) for r in sub]
        out[(lo, hi)] = (n, rev / n if n else None,
                         sum(sr) / len(sr) if sr else None)
    n = len(rows)
    rev = sum(0 if r["sign_match"] == "True" else 1 for r in rows)
    return out, n, rev / n if n else 0


orig, n_o, rr_o = agg(load(ORIG, "cogvideox2b"))
a0, n_a, rr_a = agg(load(A0))

print(f"{'bin':10s} {'n_orig':>6s} {'RR_orig':>7s} {'SR_orig':>7s}   "
      f"{'n_A0':>5s} {'RR_A0':>6s} {'SR_A0':>6s}")
for k in BINS:
    no, ro, so = orig[k]
    na, ra, sa = a0[k]
    print(f"{k[0]:.2f}-{k[1]:<6.2f} {no:6d} "
          f"{ro if ro is not None else float('nan'):7.2f} "
          f"{so if so is not None else float('nan'):7.2f}   "
          f"{na:5d} {ra if ra is not None else float('nan'):6.2f} "
          f"{sa if sa is not None else float('nan'):6.2f}")
print(f"\noverall: orig n={n_o} RR={rr_o:.2f} | A0 n={n_a} RR={rr_a:.2f}")
