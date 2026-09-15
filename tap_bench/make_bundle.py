#!/usr/bin/env python3
import os
import shutil
import subprocess
import csv

ROOT = "/home/rhap/nkst"
STAGE = os.path.join(ROOT, "phase2_bundle")
PROBE = os.path.join(ROOT, "tap_bench", "probe_set")

if os.path.isdir(STAGE):
    shutil.rmtree(STAGE)
shutil.copytree(os.path.join(ROOT, "phase2"), os.path.join(STAGE, "phase2"))
shutil.copy(os.path.join(PROBE, "gt_curves.csv"),
            os.path.join(STAGE, "phase2", "gt_curves.csv"))

ff_root = os.path.join(STAGE, "phase2", "first_frames")
seen = set()
for cat in ("wheel", "grating"):
    cat_dir = os.path.join(PROBE, cat)
    if not os.path.isdir(cat_dir):
        continue
    for name in sorted(os.listdir(cat_dir)):
        if "_point" not in name:
            continue
        stem = "_".join(name.split("_")[:-1])
        if stem in seen:
            continue
        src = os.path.join(cat_dir, name, "frame_0000.png")
        if os.path.isfile(src):
            os.makedirs(os.path.join(ff_root, cat), exist_ok=True)
            shutil.copy(src, os.path.join(ff_root, cat, stem + ".png"))
            seen.add(stem)
print("first frames:", len(seen))

out = os.path.join(ROOT, "phase2_bundle.tar.gz")
subprocess.run(["tar", "czf", out, "-C", ROOT, "phase2_bundle"], check=True)
print("bundle:", out, os.path.getsize(out) // (1024 * 1024), "MB")
