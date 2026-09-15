#!/usr/bin/env python3
"""A5: direction minimal-pair training set.

Builds train_A5 = A0's 500 CCW clips (copied verbatim) + 500 CW twins of the
same cells (point variant, same k/|omega|/seed, mirrored rotation sign).
The only delta vs A0 is the clockwise twins -> clean attribution.

Prompts: CW twins get "clockwise at exactly |w| rev/s" (renderer's own prompt
would embed the negative number, so we construct it here).
"""

import argparse
import csv
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tap_bench"))
sys.path.insert(0, os.path.expanduser("~/tap_bench"))

import render_probe as rp  # noqa: E402

rp.FPS = 8
rp.N_FRAMES = 49
rp.T_EXP = 0.5 / 8


def encode_clip(clip_dir, mp4_path):
    import imageio.v2 as imageio
    frames = sorted(f for f in os.listdir(clip_dir) if f.endswith(".png"))
    with imageio.get_writer(mp4_path, fps=rp.FPS, codec="libx264",
                            quality=7, macro_block_size=1) as w:
        for f in frames:
            w.append_data(imageio.imread(os.path.join(clip_dir, f)))


def render_cw_twin(job):
    k, omega, seed, tmp_dir, mp4_path = job
    rp.render_wheel_clip((k, -omega, seed, "point", tmp_dir))
    prompt = (f"a grayscale wheel with {k} spokes rotating clockwise at "
              f"exactly {omega} revolutions per second, static camera, "
              f"sharp focus, no motion blur, white background")
    encode_clip(tmp_dir, mp4_path)
    shutil.rmtree(tmp_dir)
    return mp4_path, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-root", default=os.path.join(HERE, "train_A0"))
    ap.add_argument("--out-root", default=HERE)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    a0_meta = os.path.join(args.a0_root, "metadata.csv")
    out_dir = os.path.join(args.out_root, "train_A5")
    vid_dir = os.path.join(out_dir, "videos")
    tmp_root = os.path.join(out_dir, "_tmp")
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)

    with open(a0_meta, encoding="utf-8") as f:
        a0_rows = list(csv.DictReader(f))
    assert len(a0_rows) == 500, f"expected 500 A0 rows, got {len(a0_rows)}"

    rows = []
    for r in a0_rows:
        shutil.copy(os.path.join(args.a0_root, "videos", r["video"]),
                    os.path.join(vid_dir, r["video"]))
        rows.append({"video": r["video"], "prompt": r["prompt"],
                     "k": r["k"], "omega": r["omega"], "seed": r["seed"],
                     "dir": "ccw"})

    jobs = []
    for r in a0_rows:
        k, omega, seed = int(r["k"]), float(r["omega"]), int(r["seed"])
        stem = r["video"].replace(".mp4", "")
        cw_name = stem + "_cw.mp4"
        jobs.append((k, omega, seed,
                     os.path.join(tmp_root, stem + "_cw"),
                     os.path.join(vid_dir, cw_name)))
    print(f"[A5] rendering {len(jobs)} CW twins...", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for mp4, prompt in ex.map(render_cw_twin, jobs, chunksize=1):
            stem = os.path.basename(mp4).replace(".mp4", "")
            base = stem.replace("_cw", "")
            src = next(x for x in a0_rows if x["video"].replace(".mp4", "") == base)
            rows.append({"video": os.path.basename(mp4), "prompt": prompt,
                         "k": src["k"], "omega": -float(src["omega"]),
                         "seed": src["seed"], "dir": "cw"})
            done += 1
            if done % 50 == 0:
                print(f"[A5] {done}/{len(jobs)}", flush=True)
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
                                          "seed", "dir"])
        w.writeheader()
        w.writerows(rows)
    n_cw = sum(1 for r in rows if r["dir"] == "cw")
    print(f"[A5] done: {len(rows)} videos ({500 - n_cw} ccw + {n_cw} cw) "
          f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
