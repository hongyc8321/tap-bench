#!/usr/bin/env python3
"""TAP-Bench driver for closed-source models via fal.ai queue API.

Submits probe prompts to a fal endpoint (e.g. Seedance 2.0), downloads the
mp4s, probes the real output fps, and writes a manifest compatible with
eval_gen.py.

Usage:
  FAL_KEY=$(cat ~/fal_key) python3 gen_closed.py \
    --endpoint bytedance/seedance-2.0/text-to-video --name seedance20 \
    --subset closed-signal --gt gt_curves.csv
"""

import argparse
import csv
import hashlib
import json
import os
import time

import cv2
import requests

QUEUE = "https://queue.fal.run"


def cell_seed(stem):
    h = hashlib.sha256(stem.encode()).digest()
    return int.from_bytes(h[:4], "little")


def load_cells(gt_csv, subset):
    cells = {}
    with open(gt_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            if cat == "ball":
                continue
            stem = row["clip_id"].split("/")[1]
            stem = "_".join(stem.split("_")[:-1])
            if stem in cells:
                continue
            parts = stem.split("_")
            if cat == "wheel":
                k, omega, seed = int(parts[0][1:]), float(parts[1][1:]), int(parts[2][1:])
                f_tex = k * omega
            else:
                v, fs, seed = float(parts[0][1:]), int(parts[1][1:]) / 1000.0, int(parts[2][1:])
                omega, k = None, None
                f_tex = v * fs
            if subset == "closed-signal":
                take = seed == 1 and (
                    (cat == "wheel" and (f_tex >= 12 or f_tex <= 4))
                    or cat == "grating")
            else:
                take = True
            if take:
                cells[stem] = {"category": cat, "stem": stem,
                               "prompt": row["prompt"], "k": k,
                               "omega": omega, "v": v if cat == "grating" else "",
                               "fs": fs if cat == "grating" else "",
                               "f_tex": f_tex}
    return dict(sorted(cells.items()))


def fal_submit(endpoint, key, payload):
    last = None
    for attempt in range(3):
        try:
            r = requests.post(f"{QUEUE}/{endpoint}", json=payload,
                              headers={"Authorization": f"Key {key}"},
                              timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = e
            print(f"  submit retry {attempt + 1}: {e}", flush=True)
            time.sleep(15 * (attempt + 1))
    raise last


def fal_wait(job, key, timeout_s=900):
    t0 = time.time()
    status_url, response_url = job["status_url"], job["response_url"]
    while time.time() - t0 < timeout_s:
        try:
            st = requests.get(status_url,
                              headers={"Authorization": f"Key {key}"},
                              timeout=30).json()
        except requests.RequestException:
            time.sleep(8)
            continue
        if st.get("status") == "COMPLETED":
            last = None
            for attempt in range(3):
                try:
                    return requests.get(response_url, headers={
                        "Authorization": f"Key {key}"}, timeout=60).json()
                except requests.RequestException as e:
                    last = e
                    time.sleep(10 * (attempt + 1))
            raise last
        if st.get("status") in ("FAILED", "ERROR"):
            raise RuntimeError(f"fal job failed: {st}")
        time.sleep(6)
    raise TimeoutError("fal job timeout")


def probe_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return round(fps, 3), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="bytedance/seedance-2.0/text-to-video")
    ap.add_argument("--name", default="seedance20")
    ap.add_argument("--subset", default="closed-signal",
                    choices=["closed-signal", "full"])
    ap.add_argument("--gt", default="gt_curves.csv")
    ap.add_argument("--out", default="gen_closed")
    ap.add_argument("--resolution", default="720p")
    ap.add_argument("--duration", default="5")
    ap.add_argument("--limit", type=int, default=0, help="only first N cells")
    args = ap.parse_args()

    key = os.environ.get("FAL_KEY") or open(os.path.expanduser("~/fal_key")).read().strip()
    cells = load_cells(args.gt, args.subset)
    if args.limit:
        cells = dict(list(cells.items())[: args.limit])
    print(f"{len(cells)} cells for {args.name} ({args.endpoint})", flush=True)

    out_dir = os.path.join(args.out, args.name, "t2v")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, f"{args.name}_manifest.csv")
    done = set()
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if os.path.isfile(r["path"]) and os.path.getsize(r["path"]) > 0:
                    done.add(r["cell_key"])

    new_rows = []
    for i, (stem, c) in enumerate(cells.items()):
        key_i = f"{stem}_rep0"
        if key_i in done:
            continue
        path = os.path.join(out_dir, f"{key_i}.mp4")
        payload = {"prompt": c["prompt"], "resolution": args.resolution,
                   "duration": args.duration, "aspect_ratio": "16:9"}
        try:
            job = fal_submit(args.endpoint, key, payload)
            result = fal_wait(job, key)
            if "video" not in result:
                print(f"  response keys: {list(result.keys())} "
                      f"body={json.dumps(result)[:300]}", flush=True)
                raise KeyError("video")
            url = result["video"]["url"]
            for attempt in range(4):
                try:
                    with requests.get(url, stream=True,
                                      timeout=(30, 600)) as r:
                        r.raise_for_status()
                        with open(path, "wb") as f:
                            for chunk in r.iter_content(1 << 20):
                                f.write(chunk)
                    break
                except Exception as de:
                    if attempt == 3:
                        raise
                    print(f"  download retry {attempt + 1}: {de}", flush=True)
                    time.sleep(20 * (attempt + 1))
        except Exception as e:
            print(f"[{i + 1}/{len(cells)}] {key_i} FAILED: {e}", flush=True)
            continue
        fps, frames = probe_fps(path)
        row = {"model": args.name, "mode": "t2v", "cell_key": key_i,
               "category": c["category"], "stem": stem,
               "k": c["k"], "omega_gt": c["omega"],
               "v_px_s": c["v"], "fs_cyc_px": c["fs"],
               "f_tex_hz": round(c["f_tex"], 3), "fps_model": fps,
               "frames": frames, "path": path}
        new_rows.append(row)
        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                w.writeheader()
            w.writerows([row])
        print(f"[{i + 1}/{len(cells)}] {key_i} fps={fps} frames={frames}", flush=True)
    print(f"done; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
