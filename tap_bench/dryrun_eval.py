#!/usr/bin/env python3
import csv
import os
import shutil
import subprocess
import sys

import cv2

ROOT = "/home/rhap/nkst"
TAP = os.path.join(ROOT, "tap_bench", "probe_set")
TEST = os.path.join(ROOT, "tap_bench", "dryrun")

if os.path.isdir(TEST):
    shutil.rmtree(TEST)
gen_dir = os.path.join(TEST, "gen_out", "fakemodel", "t2v")
os.makedirs(gen_dir)

cells = [
    ("wheel", "k02_w0.5_s01", 2, 0.5, 1.0, 24),
    ("wheel", "k08_w2.0_s01", 8, 2.0, 16.0, 24),
    ("wheel", "k16_w4.0_s01", 16, 4.0, 64.0, 24),
    ("grating", "v512_f100_s01", 512, 0.1, 51.2, 24),
]
rows = []
for cat, stem, k_or_v, w_or_fs, f_tex, fps in cells:
    src = os.path.join(TAP, cat, stem + "_point")
    dst = os.path.join(gen_dir, stem + "_rep0.mp4")
    frames = sorted(os.path.join(src, f) for f in os.listdir(src)
                    if f.endswith(".png"))
    img0 = cv2.imread(frames[0])
    h, w = img0.shape[:2]
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fp in frames:
        vw.write(cv2.imread(fp))
    vw.release()
    rows.append({
        "model": "fakemodel", "mode": "t2v", "cell_key": stem + "_rep0",
        "category": cat, "stem": stem,
        "k": k_or_v if cat == "wheel" else "",
        "omega_gt": w_or_fs if cat == "wheel" else "",
        "v_px_s": k_or_v if cat == "grating" else "",
        "fs_cyc_px": w_or_fs if cat == "grating" else "",
        "f_tex_hz": f_tex, "fps_model": fps, "path": dst,
    })
    print("encoded", dst)

mpath = os.path.join(TEST, "gen_out", "fakemodel_manifest.csv")
with open(mpath, "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
    wcsv.writeheader()
    wcsv.writerows(rows)

r = subprocess.run([sys.executable, "eval_gen.py",
                    "--gen-root", os.path.join(TEST, "gen_out"),
                    "--gt", os.path.join(TAP, "gt_curves.csv"),
                    "--out", os.path.join(TEST, "eval_out")],
                   cwd=os.path.join(ROOT, "phase2"),
                   capture_output=True, text=True)
print(r.stdout[-3000:])
print(r.stderr[-2000:] if r.returncode else "eval OK")
