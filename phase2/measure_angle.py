#!/usr/bin/env python3
"""TAP-Bench angle/velocity measurement pipeline.

Estimates apparent rotation rate (rev/s) of spoke-wheel clips, drift rate
(px/s) of grating clips, and effective gravity (m/s^2) of ball clips from
PNG frame sequences or videos, fully automatically.

Method (wheel):
  1. sample luminance on rings inside the spoke band -> angular signal s(theta, t)
  2. first-harmonic demodulation z(t) = mean_theta s(theta,t) * exp(-i*k*theta)
  3. global estimate: temporal FFT peak f* of z(t) -> omega_est = -f*/k  (fft)
     per-frame estimate: unwrapped phase slope -> omega_est(t)        (phase)

Usage:
  python measure_angle.py --root probe_set --results results.csv --selftest
  python measure_angle.py --root generated_wan --results wan_results.csv
  python measure_angle.py --single path/to/frames_or.mp4 --k 8
"""

import argparse
import csv
import glob
import json
import math
import os

import cv2
import numpy as np

M_ANG = 720
R_SAMPLES = 24
FFT_LEN = 8192
SNR_MIN = 5.0
FA_MIN = 1.5


def list_frames(path):
    if os.path.isdir(path):
        frames = sorted(glob.glob(os.path.join(path, "frame_*.png")))
        if not frames:
            frames = sorted(glob.glob(os.path.join(path, "*.png")))
        return frames
    return None


def load_frames_gray(path, max_frames=None):
    files = list_frames(path)
    out = []
    if files is not None:
        for fp in files:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            out.append(img.astype(np.float32) / 255.0)
            if max_frames and len(out) >= max_frames:
                break
    else:
        cap = cv2.VideoCapture(path)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
            if max_frames and len(out) >= max_frames:
                break
        cap.release()
    return out


def find_center(gray):
    blur = cv2.GaussianBlur(gray, (9, 9), 2.0)
    circles = cv2.HoughCircles((blur * 255).astype(np.uint8), cv2.HOUGH_GRADIENT,
                               dp=1.5, minDist=200, param1=120, param2=60,
                               minRadius=80, maxRadius=0)
    if circles is not None:
        c = circles[0][0]
        return float(c[0]), float(c[1])
    h, w = gray.shape
    return w / 2.0, h / 2.0


def angular_signals(frames, cx, cy, rmin, rmax):
    thetas = np.linspace(0.0, 2.0 * np.pi, M_ANG, endpoint=False)
    radii = np.linspace(rmin, rmax, R_SAMPLES)
    tt, rr = np.meshgrid(thetas, radii, indexing="ij")
    map_x = (cx + rr * np.cos(tt)).astype(np.float32)
    map_y = (cy - rr * np.sin(tt)).astype(np.float32)
    sig = np.empty((len(frames), M_ANG), dtype=np.float64)
    for i, f in enumerate(frames):
        s = cv2.remap(f, map_x, map_y, cv2.INTER_LINEAR)
        sig[i] = s.mean(axis=1)
    return sig


def demodulate(sig, k):
    theta = np.linspace(0.0, 2.0 * np.pi, M_ANG, endpoint=False)
    basis = np.exp(-1j * k * theta)
    return sig @ basis / M_ANG


def fft_estimate(z, fps):
    n = len(z)
    win = np.hanning(n)
    zw = (z - z.mean()) * win
    complex_in = np.iscomplexobj(zw)
    spec = np.fft.fft(zw, FFT_LEN) if complex_in else np.fft.rfft(zw, FFT_LEN)
    freqs = (np.fft.fftfreq(FFT_LEN, d=1.0 / fps) if complex_in
             else np.fft.rfftfreq(FFT_LEN, d=1.0 / fps))
    band = np.abs(freqs) <= fps / 2.0 + 1.0
    mag = np.abs(spec)
    mag[~band] = 0.0
    peak_i = int(np.argmax(mag))
    f_star = freqs[peak_i]
    noise = np.median(mag[band]) + 1e-12
    snr = mag[peak_i] / noise
    return f_star, snr, freqs, mag


def phase_estimate(z, fps, k):
    phi = np.unwrap(np.angle(z))
    omega_t = -np.diff(phi) * fps / (2.0 * np.pi * k)
    return omega_t


def measure_wheel(path, meta=None, fps=24.0):
    frames = load_frames_gray(path)
    if len(frames) < 8:
        return {"error": "too_few_frames"}
    if meta:
        cx, cy = meta.get("center", [None, None])
        if cx is None:
            cx, cy = find_center(frames[0])
        rmin = meta["band_rmin"]
        rmax = meta["band_rmax"]
        ref_h = float(meta.get("band_ref_h", frames[0].shape[0]))
        s = frames[0].shape[0] / ref_h if ref_h else 1.0
        rmin, rmax = rmin * s, rmax * s
        k = int(meta["k"])
        fps = meta.get("fps", fps)
    else:
        cx, cy = find_center(frames[0])
        h, w = frames[0].shape
        rmin = 0.18 * min(h, w) / 2
        rmax = 0.62 * min(h, w) / 2
        k = 8
    sig = angular_signals(frames, cx, cy, rmin, rmax)
    z = demodulate(sig, k)
    f_star, snr, _, _ = fft_estimate(z, fps)
    omega_fft = -f_star / k
    omega_t = phase_estimate(z, fps, k)
    out = {
        "omega_est": omega_fft,
        "omega_phase_mean": float(omega_t.mean()),
        "omega_phase_std": float(omega_t.std()),
        "phase_jitter_cv": float(omega_t.std() / max(abs(omega_t.mean()), 1e-6)),
        "snr": float(snr),
        "k": k,
        "center": [float(cx), float(cy)],
        "n_frames": len(frames),
    }
    return out


def measure_grating(path, meta=None, fps=24.0):
    frames = load_frames_gray(path)
    if len(frames) < 8:
        return {"error": "too_few_frames"}
    fs = float(meta["fs_cyc_px"]) if meta else 0.1
    fps = meta.get("fps", fps) if meta else fps
    h, w = frames[0].shape
    rows = slice(int(h * 0.35), int(h * 0.65))
    xs = np.arange(w, dtype=np.float64)
    basis = np.exp(-1j * 2.0 * np.pi * fs * xs)
    z = np.empty(len(frames), dtype=np.complex128)
    for i, f in enumerate(frames):
        b = f[rows].mean(axis=0)
        z[i] = b @ basis / w
    f_star, snr, _, _ = fft_estimate(z, fps)
    out = {
        "v_est": -f_star / fs,
        "snr": float(snr),
        "fs": fs,
        "n_frames": len(frames),
    }
    return out


def measure_ball(path, meta=None, fps=24.0):
    frames = load_frames_gray(path)
    if len(frames) < 8:
        return {"error": "too_few_frames"}
    fps = meta.get("fps", fps) if meta else fps
    h, w = frames[0].shape
    x_lo, x_hi = int(w * 0.30), int(w * 0.70)
    ys = []
    for f in frames:
        col = f[:, x_lo:x_hi]
        dark = col < 0.5
        if dark.sum() < 10:
            ys.append(np.nan)
            continue
        yy, _ = np.nonzero(dark)
        ys.append(yy.mean())
    ys = np.array(ys)
    ok = ~np.isnan(ys)
    t = np.arange(len(ys))[ok] / fps
    px_per_m = float(meta["px_per_m"]) if meta else 100.0
    y_m = ys[ok] / px_per_m
    A = np.stack([t * t, np.ones_like(t)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y_m, rcond=None)
    g_eff = 2.0 * coef[0]
    out = {
        "g_eff": float(g_eff),
        "n_used": int(ok.sum()),
        "n_frames": len(frames),
    }
    if meta and "g_ms2" in meta:
        out["g_gt"] = float(meta["g_ms2"])
        out["g_err_pct"] = 100.0 * abs(g_eff - meta["g_ms2"]) / meta["g_ms2"]
    return out


MEASURERS = {"wheel": measure_wheel, "grating": measure_grating, "ball": measure_ball}


def iter_clips(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        if "meta.json" in filenames:
            yield dirpath


def run_batch(root, results_path, selftest=False):
    rows = []
    for clip in sorted(iter_clips(root)):
        with open(os.path.join(clip, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        cat = meta["category"]
        m = MEASURERS[cat](clip, meta=meta)
        row = {
            "clip_id": os.path.relpath(clip, root),
            "category": cat,
            "variant": meta.get("variant", ""),
            "omega_gt": meta.get("omega_gt", ""),
            "f_tex_hz": round(meta.get("f_tex_hz", float("nan")), 3),
            "alias_f_hz": round(meta.get("alias_f_hz", float("nan")), 3),
            "pred": round(meta.get("omega_app_pred",
                                   meta.get("v_app_pred", float("nan"))), 4),
            "reversal_pred": meta.get("reversal_pred", ""),
            "prompt": meta.get("prompt", ""),
        }
        if "error" in m:
            row["error"] = m["error"]
            rows.append(row)
            print(f"FAIL {row['clip_id']}: {m['error']}", flush=True)
            continue
        if cat == "wheel":
            est = m["omega_est"]
            row.update(omega_est=round(est, 4),
                       phase_mean=round(m["omega_phase_mean"], 4),
                       jitter_cv=round(m["phase_jitter_cv"], 3),
                       snr=round(m["snr"], 1))
            row["err_vs_pred"] = round(est - row["pred"], 4)
            row["sign_match_gt"] = bool(np.sign(est) == np.sign(row["omega_gt"])) \
                if row["omega_gt"] != "" else ""
            row["low_conf"] = bool(m["snr"] < SNR_MIN
                                   or abs(row["alias_f_hz"]) < FA_MIN
                                   or abs(abs(row["alias_f_hz"]) - 12.0) < 0.75)
        elif cat == "grating":
            est = m["v_est"]
            row.update(v_est=round(est, 2), snr=round(m["snr"], 1))
            row["err_vs_pred"] = round(est - row["pred"], 2)
            row["low_conf"] = bool(m["snr"] < SNR_MIN
                                   or abs(row["alias_f_hz"]) < FA_MIN
                                   or abs(abs(row["alias_f_hz"]) - 12.0) < 0.75)
        else:
            row.update(g_eff=round(m["g_eff"], 3),
                       g_gt=m.get("g_gt", ""),
                       g_err_pct=round(m.get("g_err_pct", float("nan")), 2))
        rows.append(row)
        print(f"OK   {row['clip_id']}: "
              f"{row.get('omega_est', row.get('v_est', row.get('g_eff')))}", flush=True)

    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=fields)
            wcsv.writeheader()
            wcsv.writerows(rows)
        print(f"results: {results_path} ({len(rows)} rows)")

    if selftest:
        report_selftest(rows)


def report_selftest(rows):
    print("\n===== SELFTEST =====")
    for variant in ("point", "shutter180"):
        sub = [r for r in rows if r.get("variant") == variant
               and r.get("omega_est") is not None and "err_vs_pred" in r
               and not r.get("low_conf", False)]
        if not sub:
            continue
        errs = [abs(r["err_vs_pred"]) for r in sub]
        sign_ok = [abs(r["err_vs_pred"]) <= max(0.3, 0.15 * abs(r["pred"]))
                   for r in sub]
        rev_rows = [r for r in sub if r["reversal_pred"]]
        rev_ok = [abs(r["err_vs_pred"]) <= max(0.3, 0.15 * abs(r["pred"]))
                  for r in rev_rows]
        print(f"[{variant}] n={len(sub)} "
              f"MAE(vs analytic pred)={np.mean(errs):.3f} rev/s  "
              f"pred-match={100 * np.mean(sign_ok):.0f}%  "
              f"(reversal cells n={len(rev_rows)}, "
              f"match={100 * np.mean(rev_ok) if rev_ok else float('nan'):.0f}%)")
    balls = [r for r in rows if r.get("g_eff") is not None]
    if balls:
        print(f"[ball] n={len(balls)} "
              f"mean |g_err| = {np.mean([r['g_err_pct'] for r in balls]):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="directory containing clip subdirs with meta.json")
    ap.add_argument("--single", help="one clip dir or video file (wheel)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.single:
        meta = None
        mp = os.path.join(args.single, "meta.json")
        if os.path.isfile(mp):
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        print(json.dumps(measure_wheel(args.single, meta=meta), indent=1))
    else:
        run_batch(args.root, args.results, selftest=args.selftest)


if __name__ == "__main__":
    main()
