import csv
from collections import defaultdict

with open("/home/rhap/nkst/phase2_results/nyquist_rows.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"total rows: {len(rows)}")
stats = defaultdict(lambda: [0, 0, 0])
for r in rows:
    b = "low" if float(r["f_ratio"]) < 0.5 else (
        "mid" if float(r["f_ratio"]) < 2 else "high")
    s = stats[(r["category"], b)]
    s[0] += 1
    s[1] += 0 if r["low_conf"] == "True" else 0
    if r["low_conf"] == "True":
        s[2] += 1

print("\ncategory x ratio-band: n_total / n_lowconf")
for (cat, b), (n, _, lc) in sorted(stats.items()):
    print(f"{cat:8s} {b:5s}  n={n:3d}  lowconf={lc:3d}")

conf = [r for r in rows if r["low_conf"] == "False"]
print(f"\nconfident: {len(conf)}")
by_cat = defaultdict(lambda: [0, 0])
for r in conf:
    by_cat[r["category"]][0] += 1
    by_cat[r["category"]][1] += 0 if r["sign_match"] == "True" else 1
for c, (n, rev) in sorted(by_cat.items()):
    print(f"{c:8s} n={n} reversal={rev} ({rev / n:.0%})")

print("\nspeed_ratio by band (confident):")
bands = defaultdict(list)
for r in conf:
    b = "low" if float(r["f_ratio"]) < 0.5 else (
        "mid" if float(r["f_ratio"]) < 2 else "high")
    bands[b].append(float(r["speed_ratio"]))
for b, v in sorted(bands.items()):
    print(f"{b:5s} mean={sum(v) / len(v):.2f}  median={sorted(v)[len(v) // 2]:.2f}")

print("\nstationary outputs (motion=False, confident):")
st = [r for r in conf if r["motion"] == "False"]
print(f"n={len(st)}")
for r in st[:8]:
    print(f"  {r['cell_key']} f_ratio={float(r['f_ratio']):.2f} "
          f"est={float(r['est']):.2f} gt={float(r['omega_gt'])}")
