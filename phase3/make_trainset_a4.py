#!/usr/bin/env python3
"""A4: camera-response-family training set.

Reuses the A0 base cells (point-variant mp4s already rendered) and adds the
shutter180 variant for the same cells. Prompts are conditioned on the shutter
angle so the model learns the parameterized camera response family:

  point      -> "... exposed with a 0-degree (instantaneous) shutter"
  shutter180 -> "... exposed with a 180-degree shutter causing motion blur"

Output: train_A4/{videos, prompts.txt, videos.txt, metadata.csv}

Usage (on cloud):
  python3 make_trainset_a4.py --a0-root train_A0 --out-root . --workers 8
"""

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tap_bench"))
import render_probe as rp  # noqa: E402

rp.FPS = 8
rp.N_FRAMES = 49
rp.T_EXP = 0.5 / 8

SHUTTER_PHRASE = "no motion blur"


def conditioned_prompt(base_prompt, variant):
    p = base_prompt.replace(", " + SHUTTER_PHRASE, "")
    if variant == "point":
        return p + ", exposed with a 0-degree (instantaneous) shutter"
    return p + ", exposed with a 180-degree shutter causing motion blur"


def render_shutter(job):
    k, omega, seed, out_dir = job
    rp.render_wheel_clip((k, omega, seed, "shutter180", out_dir))
    return out_dir


def encode_clip(clip_dir, mp4_path):
    import imageio.v2 as imageio
    frames = sorted(f for f in os.listdir(clip_dir) if f.endswith(".png"))
    with imageio.get_writer(mp4_path, fps=rp.FPS, codec="libx264",
                            quality=7, macro_block_size=1) as w:
        for f in frames:
            w.append_data(imageio.imread(os.path.join(clip_dir, f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-root", default="train_A0")
    ap.add_argument("--out-root", default=".")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.a0_root, "metadata.csv"), encoding="utf-8") as f:
        cells = [r for r in csv.DictReader(f)]

    n_unique = len({(r["k"], r["omega"]) for r in cells})
    if args.limit:
        cells = cells[: args.limit]
    print(f"{len(cells)} A0 cells loaded ({n_unique} unique k-omega)",
          flush=True)

    out_dir = os.path.join(args.out_root, "train_A4")
    vid_dir = os.path.join(out_dir, "videos")
    tmp_root = os.path.join(args.out_root, "train_A4", "_tmp")
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)

    jobs = []
    for r in cells:
        k, omega, seed = int(r["k"]), float(r["omega"]), int(r["seed"])
        stem = f"k{k:02d}_w{omega:.2f}_s{seed:03d}"
        jobs.append((k, omega, seed, os.path.join(tmp_root, stem)))

    print(f"rendering {len(jobs)} shutter180 clips...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(render_shutter, jobs, chunksize=1)):
            if (i + 1) % 50 == 0:
                print(f"rendered {i + 1}/{len(jobs)}", flush=True)

    rows = []
    for r in cells:
        k, omega, seed = int(r["k"]), float(r["omega"]), int(r["seed"])
        stem = f"k{k:02d}_w{omega:.2f}_s{seed:03d}"
        for variant in ("point", "shutter180"):
            name = f"{stem}_{variant}.mp4"
            dst = os.path.join(vid_dir, name)
            if variant == "point":
                src = os.path.join(args.a0_root, "videos", stem + ".mp4")
                shutil.copy(src, dst)
            else:
                encode_clip(os.path.join(tmp_root, stem), dst)
            rows.append({"video": name,
                         "prompt": conditioned_prompt(r["prompt"], variant),
                         "k": k, "omega": omega, "seed": seed,
                         "variant": variant})
    shutil.rmtree(tmp_root, ignore_errors=True)

    rows.sort(key=lambda x: x["video"])
    with open(os.path.join(out_dir, "prompts.txt"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["prompt"] + "\n")
    with open(os.path.join(out_dir, "videos.txt"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write("videos/" + r["video"] + "\n")
    with open(os.path.join(out_dir, "metadata.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "prompt", "k", "omega",
                                          "seed", "variant"])
        w.writeheader()
        w.writerows(rows)
    print(f"[A4] done: {len(rows)} videos -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
