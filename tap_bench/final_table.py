import csv

BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0),
        (1.0, 1.5), (1.5, 2.0), (2.0, 99.0)]
SOURCES = [
    ("cog-orig", "/home/rhap/nkst/phase2_results_final/nyquist_rows.csv",
     "cogvideox2b"),
    ("cog-A0", "/home/rhap/nkst/phase3_results_A0/nyquist_rows.csv",
     "cogvideox2b"),
    ("cog-A3", "/home/rhap/nkst/phase3_results_A3/nyquist_rows.csv",
     "cogvideox2b"),
    ("seedance20", "/home/rhap/nkst/phase3_results_seedance/nyquist_rows.csv",
     "seedance20"),
    ("ltxvideo", "/home/rhap/nkst/phase2_results_final/nyquist_rows.csv",
     "ltxvideo"),
    ("wan13b", "/home/rhap/nkst/phase2_results_final/nyquist_rows.csv",
     "wan13b"),
]


def load(path, model_prefix):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["low_conf"] == "False"]
    rows = [r for r in rows if r["model"].startswith(model_prefix)]
    return [r for r in rows if r.get("mode", "t2v") == "t2v"]


print(f"{'model':11s} {'n':>4s} {'RR':>5s}  "
      f"0-.25 | .25-.5 | .5-.75 | .75-1 | 1-1.5 | 1.5-2 | 2+   (moving-cond)")
for name, path, pref in SOURCES:
    rows = load(path, pref)
    n = len(rows)
    rev = sum(0 if r["sign_match"] == "True" else 1 for r in rows)
    rr = rev / n if n else float("nan")
    cells = []
    for lo, hi in BINS:
        sub = [r for r in rows
               if lo <= float(r["f_ratio"]) < hi and r["motion"] == "True"]
        m = len(sub)
        rv = sum(0 if r["sign_match"] == "True" else 1 for r in sub)
        cells.append(f"{rv / m:.2f}({m})" if m else "  -  ")
    print(f"{name:11s} {n:4d} {rr:5.2f}  " + " | ".join(cells))
