#!/usr/bin/env python3
"""P0 audit for manifests and generated videos.

The script never modifies existing results. It records missing paths, optional
OpenCV metadata, and a stratified blind-review queue.
"""
import argparse
import csv
import hashlib
import os
import random
from collections import Counter, defaultdict


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=".", help="directory used to resolve relative paths")
    ap.add_argument("--out", default="p0_audit_results")
    ap.add_argument("--sample-per-stratum", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        import cv2
    except ImportError:
        cv2 = None

    records = []
    with open(args.manifest, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("path", "")
            candidates = [
                raw,
                os.path.join(os.path.dirname(args.manifest), raw),
            ]
            # Seedance manifests were created from the remote run root; map
            # their gen_closed prefix to the checked-in local asset folder.
            if raw.startswith("gen_closed/seedance20/"):
                candidates.append(os.path.join(
                    os.path.dirname(args.manifest),
                    "seedance20", raw[len("gen_closed/seedance20/"):]))
            path = next((os.path.abspath(p) for p in candidates if os.path.isfile(p)), "")
            rec = dict(row)
            rec.update({"resolved_path": path, "path_status": "missing"})
            if path:
                rec["path_status"] = "found"
                rec["bytes"] = os.path.getsize(path)
                rec["sha256"] = sha256(path)
                if cv2 is not None:
                    cap = cv2.VideoCapture(path)
                    rec["decode_open"] = bool(cap.isOpened())
                    rec["fps_video"] = cap.get(cv2.CAP_PROP_FPS)
                    rec["frames_video"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    rec["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    rec["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    rec["fps_delta"] = rec["fps_video"] - float(row.get("fps_model") or 0)
                    cap.release()
                    rec["path_status"] = "decodable" if rec["decode_open"] else "decode_failed"
                else:
                    rec["decode_open"] = "not_checked_no_opencv"
            records.append(rec)

    fields = sorted({k for r in records for k in r})
    with open(os.path.join(args.out, "manifest_audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(records)

    rng = random.Random(20260905)
    groups = defaultdict(list)
    for r in records:
        ratio = float(r.get("f_tex_hz") or 0) / max(float(r.get("fps_model") or 0), 1e-9)
        band = "0-0.5" if ratio < .5 else "0.5-1" if ratio < 1 else "1-2" if ratio < 2 else "2+"
        groups[(r.get("model", ""), band)].append(r)
    review = []
    for (model, band), items in sorted(groups.items()):
        for r in rng.sample(items, min(args.sample_per_stratum, len(items))):
            review.append({"model": model, "frequency_band": band, "cell_key": r.get("cell_key", ""),
                           "video_path": r.get("resolved_path", ""), "path_status": r.get("path_status", ""),
                           "exists": "", "object_present": "", "is_rotating": "", "direction": "",
                           "spokes_correct": "", "camera_stable": "", "measurement_trust": "", "notes": ""})
    with open(os.path.join(args.out, "blind_review_queue.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0]) if review else ["model"])
        w.writeheader(); w.writerows(review)

    counts = Counter(r["path_status"] for r in records)
    with open(os.path.join(args.out, "audit_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"manifest={args.manifest}\nrecords={len(records)}\n")
        for key, value in sorted(counts.items()): f.write(f"{key}={value}\n")
        f.write(f"opencv_available={cv2 is not None}\n")
        f.write("manual_review_file=blind_review_queue.csv\n")
    print(f"records={len(records)} status={dict(counts)} out={args.out}")


if __name__ == "__main__":
    main()
