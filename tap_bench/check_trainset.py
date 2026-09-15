#!/usr/bin/env python3
import cv2
import glob
import os

root = "/home/rhap/nkst/phase3/dryrun8/train_A0"
for mp4 in sorted(glob.glob(os.path.join(root, "videos", "*.mp4"))):
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()
    print(os.path.basename(mp4), "frames:", n, "fps:", fps, "read_ok:", ok)
    cap.release()
print(open(os.path.join(root, "prompts.txt")).read().splitlines()[:2])
