#!/usr/bin/env python3
"""Phase 3: build LoRA training sets for the two-arm ablation.

Renders randomized wheel clips (continuous omega in [0.5, 8.5], excluding the
probe grid values so the probe stays held-out), encodes to mp4 at 24 fps, and
assembles per-arm training directories following the diffusers
CogVideoX LoRA script conventions (videos/ + prompts.txt).

Arms:
  A0  naive point-sampled frames, uniform sampling        (baseline)
  A1  naive point-sampled frames, high-omega 4x duplicated
  A2  shutter180 temporally filtered frames, uniform
  A3  shutter180 temporally filtered, high-omega 4x duplicated

Usage:
  python make_trainset.py --arm A0 --n 500 --out train_A0 --workers 10
  python make_trainset.py --all --n 500 --out-root . --workers 10
"""

import argparse
import csv
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tap_bench"))
import render_probe as rp  # noqa: E402

# Render at CogVideoX native grid: 49 frames @ 8 fps (6.125 s), so the
# training script consumes the clips without any further temporal decimation.
# point      = what a naive 8 fps camera sees (temporal aliasing included)
# shutter180 = the same camera with a 180-degree shutter (temporally filtered)
rp.FPS = 8
rp.N_FRAMES = 49
rp.T_EXP = 0.5 / 8

K_GRID = [2, 4, 6, 8, 12, 16]
OMEGA_LO, OMEGA_HI = 0.5, 8.5
PROBE_OMEGAS = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
HIGH_OMEGA = 4.0
DUP = 4


def sample_omega(rng):
    while True:
        w = round(rng.uniform(OMEGA_LO, OMEGA_HI), 2)
        if all(abs(w - p) > 0.1 for p in PROBE_OMEGAS):
            return w


def encode_clip(clip_dir, mp4_path):
    import imageio.v2 as imageio
    frames = sorted(f for f in os.listdir(clip_dir) if f.endswith(".png"))
    with imageio.get_writer(mp4_path, fps=rp.FPS, codec="libx264",
                            quality=7, macro_block_size=1) as w:
        for f in frames:
            w.append_data(imageio.imread(os.path.join(clip_dir, f)))


def render_and_encode(job):
    variant, k, omega, seed, tmp_dir, mp4_path = job
    rp.render_wheel_clip((k, omega, seed, variant, tmp_dir))
    with open(os.path.join(tmp_dir, "meta.json"), encoding="utf-8") as f:
        import json
        prompt = json.load(f)["prompt"]
    encode_clip(tmp_dir, mp4_path)
    shutil.rmtree(tmp_dir)
    return mp4_path, prompt, omega, k, seed


def build_arm(arm, n, out_root, workers):
    arm_seed = {"A0": 1, "A1": 2, "A2": 3, "A3": 4}[arm]
    rng = random.Random(20260830 + arm_seed)
    variant = "point" if arm in ("A0", "A1") else "shutter180"
    cells = []
    for i in range(n):
        k = rng.choice(K_GRID)
        omega = sample_omega(rng)
        cells.append((k, omega, i + 1))
    out_dir = os.path.join(out_root, f"train_{arm}")
    vid_dir = os.path.join(out_dir, "videos")
    tmp_root = os.path.join(out_root, "_tmp")
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)

    jobs = []
    for k, omega, seed in cells:
        stem = f"k{k:02d}_w{omega:.2f}_s{seed:03d}"
        jobs.append((variant, k, omega, seed,
                     os.path.join(tmp_root, stem),
                     os.path.join(vid_dir, stem + ".mp4")))
    print(f"[{arm}] rendering {len(jobs)} clips ({variant})...", flush=True)
    rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for mp4, prompt, omega, k, seed in ex.map(render_and_encode, jobs,
                                                  chunksize=1):
            rows.append({"video": os.path.basename(mp4), "prompt": prompt,
                         "k": k, "omega": omega, "seed": seed})
            done += 1
            if done % 50 == 0:
                print(f"[{arm}] {done}/{len(jobs)}", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)

    if arm in ("A1", "A3"):
        extra = []
        for r in rows:
            if r["omega"] >= HIGH_OMEGA:
                for d in range(1, DUP):
                    dup_name = r["video"].replace(".mp4", f"_dup{d}.mp4")
                    shutil.copy(os.path.join(vid_dir, r["video"]),
                                os.path.join(vid_dir, dup_name))
                    extra.append({**r, "video": dup_name})
        rows.extend(extra)
        print(f"[{arm}] duplicated high-omega clips: +{len(extra)}", flush=True)

    with open(os.path.join(out_dir, "prompts.txt"), "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["video"]):
            f.write(r["prompt"] + "\n")
    with open(os.path.join(out_dir, "videos.txt"), "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["video"]):
            f.write("videos/" + r["video"] + "\n")
    with open(os.path.join(out_dir, "metadata.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "prompt", "k", "omega",
                                          "seed"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: x["video"]))
    print(f"[{arm}] done: {len(rows)} videos -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A0", "A1", "A2", "A3"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    arms = ["A0", "A1", "A2", "A3"] if args.all else [args.arm]
    for arm in arms:
        build_arm(arm, args.n, args.out_root, args.workers)


if __name__ == "__main__":
    main()
