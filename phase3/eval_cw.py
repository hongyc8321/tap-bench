#!/usr/bin/env python3
"""Eval clockwise-compliance of generated CW probe videos.

Reads a generation manifest (gen_cw_<tag>/cogvideox2b_t2v_manifest.csv),
measures each mp4, and reports direction compliance (sign match vs the
negative-omega GT), speed ratio and motion rate.
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tap_bench"))
sys.path.insert(0, os.path.expanduser("~/tap_bench"))
sys.path.insert(0, os.path.expanduser("~/phase2_bundle/phase2"))
sys.path.insert(0, HERE)

import measure_angle as ma  # noqa: E402


def build_meta(gt_row, fps_model):
    return {
        "fps": fps_model,
        "k": int(gt_row["k"]),
        "band_rmin": float(gt_row["band_rmin"]),
        "band_rmax": float(gt_row["band_rmax"]),
        "center": [float(v) for v in gt_row["center"].split(";")],
        "omega_gt": float(gt_row["omega_gt"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gen-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lut = {}
    with open(args.gt, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = row["clip_id"].split("/")[1]
            stem = "_".join(stem.split("_")[:-1])
            lut[stem] = row

    rows = []
    for name in sorted(os.listdir(args.gen_root)):
        if not name.endswith("_manifest.csv"):
            continue
        with open(os.path.join(args.gen_root, name), encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))

    out_rows = []
    for r in rows:
        if not os.path.isfile(r["path"]):
            continue
        gt = lut.get(r["stem"])
        if gt is None:
            continue
        fps_model = float(r["fps_model"])
        meta = build_meta(gt, fps_model)
        m = ma.measure_wheel(r["path"], meta=meta, fps=fps_model)
        if "error" in m:
            continue
        est = m["omega_est"]
        gt_val = float(r["omega_gt"])
        out_rows.append({
            "gen_root": os.path.basename(args.gen_root),
            "stem": r["stem"], "k": r.get("k", ""),
            "omega_gt": gt_val, "est": round(est, 4),
            "sign_match": bool(est * gt_val > 0),
            "speed_ratio": round(abs(est) / max(abs(gt_val), 1e-6), 3),
            "motion": bool(abs(est) >= 0.15),
            "snr": round(m.get("snr", 0), 1),
        })

    out_csv = os.path.join(args.out, "cw_rows.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    n = len(out_rows)
    conf = [r for r in out_rows if r["snr"] >= ma.SNR_MIN]
    n_conf = len(conf)
    comp = sum(1 for r in conf if r["sign_match"]) / max(n_conf, 1)
    sr = sum(r["speed_ratio"] for r in conf) / max(n_conf, 1)
    mov = sum(1 for r in conf if r["motion"]) / max(n_conf, 1)
    with open(os.path.join(args.out, "cw_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"tag\t{os.path.basename(args.gen_root)}\n")
        f.write(f"n_total\t{n}\nn_conf\t{n_conf}\n")
        f.write(f"cw_compliance\t{comp:.2f}\n")
        f.write(f"mean_speed_ratio\t{sr:.2f}\n")
        f.write(f"motion_rate\t{mov:.2f}\n")
    print(f"[{os.path.basename(args.gen_root)}] n={n} conf={n_conf} "
          f"cw_compliance={comp:.2f} speed_ratio={sr:.2f} motion={mov:.2f}")


if __name__ == "__main__":
    main()
