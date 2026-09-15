#!/usr/bin/env python3
"""Phase 2 eval: measure generated probe videos and aggregate Nyquist curves.

Reads generation manifest(s) from gen_out, measures each mp4 with the
TAP-Bench pipeline, joins with GT, and produces:
  eval_out/nyquist_rows.csv    per-video measurements
  eval_out/nyquist_bins.csv    reversal rate per f_tex/fps_model bin per model
  eval_out/nyquist_curves.png  the empirical temporal-Nyquist curves
  eval_out/summary.txt         headline table

Usage:
  python eval_gen.py --gen-root gen_out --gt gt_curves.csv
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import measure_angle as ma

BINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 99.0]
BIN_LABELS = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1", "1-1.5",
              "1.5-2", "2+"]


def bin_of(ratio):
    for i in range(len(BINS) - 1):
        if BINS[i] <= ratio < BINS[i + 1]:
            return BIN_LABELS[i]
    return BIN_LABELS[-1]


def gt_lookup(gt_csv):
    lut = {}
    with open(gt_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = row["clip_id"].split("/")[1]
            stem = "_".join(stem.split("_")[:-1])
            if row["variant"] != "point" and stem in lut:
                continue
            lut[stem] = row
    return lut


def build_meta(row, stem, fps_model):
    meta = {"fps": fps_model}
    if row["category"] == "wheel":
        parts = stem.split("_")
        meta["k"] = int(parts[0][1:])
        meta["band_rmin"] = float(row["band_rmin"])
        meta["band_rmax"] = float(row["band_rmax"])
        cx, cy = row["center"].split(";")
        meta["center"] = [float(cx), float(cy)]
        meta["omega_gt"] = float(row["omega_gt"])
    elif row["category"] == "grating":
        parts = stem.split("_")
        meta["fs_cyc_px"] = int(parts[1][1:]) / 1000.0
        meta["v_gt"] = float(parts[0][1:])
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", default="gen_out")
    ap.add_argument("--gt", default="gt_curves.csv")
    ap.add_argument("--out", default="eval_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lut = gt_lookup(args.gt)

    rows = []
    for name in sorted(os.listdir(args.gen_root)):
        mpath = os.path.join(args.gen_root, name)
        if not name.endswith("_manifest.csv") or not os.path.isfile(mpath):
            continue
        with open(mpath, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)

    out_rows = []
    for r in rows:
        if not os.path.isfile(r["path"]):
            continue
        gt = lut.get(r["stem"])
        if gt is None:
            continue
        fps_model = float(r["fps_model"])
        meta = build_meta(gt, r["stem"], fps_model)
        cat = r["category"]
        if cat == "wheel":
            m = ma.measure_wheel(r["path"], meta=meta, fps=fps_model)
            if "error" in m:
                continue
            est = m["omega_est"]
            gt_val = float(r["omega_gt"])
        elif cat == "grating":
            m = ma.measure_grating(r["path"], meta=meta, fps=fps_model)
            if "error" in m:
                continue
            est = m["v_est"] / 100.0
            gt_val = float(r["v_px_s"]) / 100.0
        else:
            continue
        low_conf = m.get("snr", 0) < ma.SNR_MIN
        out_rows.append({
            "model": r["model"], "mode": r.get("mode", "t2v"),
            "cell_key": r["cell_key"],
            "category": cat, "k": r.get("k", ""),
            "omega_gt": gt_val, "est": est,
            "f_tex_hz": float(r["f_tex_hz"]), "fps_model": fps_model,
            "f_ratio": float(r["f_tex_hz"]) / fps_model,
            "sign_match": bool(est * gt_val > 0),
            "speed_ratio": abs(est) / max(abs(gt_val), 1e-6),
            "motion": abs(est) >= 0.15,
            "snr": round(m.get("snr", 0), 1), "low_conf": low_conf,
        })

    if not out_rows:
        print("no measurements; check gen_out")
        return
    fields = list(out_rows[0])
    with open(os.path.join(args.out, "nyquist_rows.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    conf = [r for r in out_rows if not r["low_conf"]]
    for r in out_rows:
        r["model"] = f'{r["model"]}/{r["mode"]}'
    agg = defaultdict(lambda: {"n": 0, "rev": 0, "nomove": 0, "sr": []})
    for r in conf:
        a = agg[(r["model"], bin_of(r["f_ratio"]))]
        a["n"] += 1
        a["rev"] += 0 if r["sign_match"] else 1
        a["nomove"] += 0 if r["motion"] else 1
        a["sr"].append(min(r["speed_ratio"], 3.0))

    models = sorted({r["model"] for r in conf})
    bin_order = BIN_LABELS
    summary = ["model\tbin\tn\treversal_rate\tstationary_rate\tmean_speed_ratio"]
    for mdl in models:
        for b in bin_order:
            a = agg.get((mdl, b))
            if not a or a["n"] == 0:
                continue
            summary.append(
                f"{mdl}\t{b}\t{a['n']}\t{a['rev'] / a['n']:.2f}\t"
                f"{a['nomove'] / a['n']:.2f}\t"
                f"{sum(a['sr']) / len(a['sr']):.2f}")
    summary.append("")
    summary.append("model\tbin\tn_moving\treversal_rate_given_moving")
    for mdl in models:
        for b in bin_order:
            sub = [r for r in conf if r["model"] == mdl
                   and bin_of(r["f_ratio"]) == b and r["motion"]]
            if len(sub) >= 3:
                rev = sum(0 if r["sign_match"] else 1 for r in sub)
                summary.append(f"{mdl}\t{b}\t{len(sub)}\t{rev / len(sub):.2f}")
    n_all = defaultdict(lambda: [0, 0])
    for r in conf:
        n_all[r["model"]][0] += 1
        n_all[r["model"]][1] += 0 if r["sign_match"] else 1
    summary.append("")
    summary.append("model\tn_conf\toverall_reversal_rate")
    for mdl in models:
        n, rev = n_all[mdl]
        summary.append(f"{mdl}\t{n}\t{rev / n:.2f}")
    with open(os.path.join(args.out, "summary.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))

    with open(os.path.join(args.out, "nyquist_bins.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "bin", "n", "reversal_rate", "stationary_rate",
                    "mean_speed_ratio"])
        for mdl in models:
            for b in bin_order:
                a = agg.get((mdl, b))
                if a and a["n"]:
                    w.writerow([mdl, b, a["n"], round(a["rev"] / a["n"], 3),
                                round(a["nomove"] / a["n"], 3),
                                round(sum(a["sr"]) / len(a["sr"]), 3)])

    plt.figure(figsize=(7, 4.5))
    for mdl in models:
        xs, ys = [], []
        for b in bin_order:
            a = agg.get((mdl, b))
            if a and a["n"]:
                mid = (BINS[bin_order.index(b)] + BINS[bin_order.index(b) + 1]) / 2
                xs.append(min(mid, 2.4))
                ys.append(a["rev"] / a["n"])
        plt.plot(xs, ys, marker="o", label=mdl)
    plt.axvline(0.5, ls="--", c="gray", lw=1)
    plt.text(0.51, 0.02, "Nyquist limit", rotation=90, fontsize=8, c="gray")
    plt.xlabel("texture temporal frequency / model fps  (f_tex / fps)")
    plt.ylabel("reversal rate (sign error vs GT)")
    plt.ylim(-0.03, 1.0)
    plt.legend()
    plt.title("TAP-Bench empirical temporal-Nyquist curves")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "nyquist_curves.png"), dpi=150)
    print(f"plot: {args.out}/nyquist_curves.png")


if __name__ == "__main__":
    main()
