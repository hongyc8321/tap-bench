#!/usr/bin/env python3
"""TAP-Bench probe renderer.

Renders temporally-parametric probe clips (rotating spoke wheels, drifting
gratings, falling balls) at 24 fps with two camera variants:

  point      : per-frame point sampling of the continuous trajectory
               (the natural naive 24fps capture; exhibits wagon-wheel aliasing)
  shutter180 : temporal box filter over a 180-degree shutter window (T=1/48 s)
               before sampling (physically anti-aliased reference)

Spatial anti-aliasing: analytic rendering on a 2x supersampled canvas followed
by a box downsample to 1280x720. All aliasing predictions in meta.json are
analytic (first-harmonic temporal frequency f0 = k*omega wrapped into
[-fps/2, fps/2]; the shutter variant additionally carries the sign of
sinc(f0*T_exp), which can flip apparent direction).

Usage:
  python render_probe.py --grid smoke --out probe_set
  python render_probe.py --grid full  --out probe_set --workers 8
"""

import argparse
import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

FPS = 24
DUR_S = 5.0
N_FRAMES = int(FPS * DUR_S)
W, H = 1280, 720
SUP = 2
T_EXP = 0.5 / FPS
N_SHUTTER_TAPS = 5

OMEGA_GRID = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
K_GRID = [2, 4, 6, 8, 12, 16]
SEEDS = 5
V_GRID = [64.0, 128.0, 256.0, 512.0]
FS_GRID = [0.05, 0.1, 0.2]
G_SEEDS = 3

OUT_W_F, OUT_H_F = float(W), float(H)
CX, CY = OUT_W_F / 2.0, OUT_H_F / 2.0


def wrap_hz(f0, fps=FPS):
    """Alias (apparent) temporal frequency of f0 sampled at fps, in [-fps/2, fps/2]."""
    return f0 - fps * round(f0 / fps)


def shutter_sign(f0, t_exp=T_EXP):
    """Sign of the shutter transfer function sinc(f0*T) = sin(pi f0 T)/(pi f0 T)."""
    x = math.pi * f0 * t_exp
    if abs(x) < 1e-9:
        return 1.0
    return 1.0 if math.sin(x) / x > 0 else -1.0


def seed_rng(*parts):
    key = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "little"))


class Canvas:
    """Supersampled pixel grid in OUTPUT-pixel units, cached per worker process."""

    def __init__(self):
        ys, xs = np.mgrid[0:H * SUP, 0:W * SUP]
        self.x = (xs + 0.5) / SUP
        self.y = (ys + 0.5) / SUP
        self.dx = self.x - CX
        self.dy = self.y - CY


_CANVAS = None


def get_canvas():
    global _CANVAS
    if _CANVAS is None:
        _CANVAS = Canvas()
    return _CANVAS


def save_frame(arr01, path):
    img = Image.fromarray((np.clip(arr01, 0.0, 1.0) * 255.0).astype(np.uint8))
    img = img.resize((W, H), Image.BOX)
    img.save(path, compress_level=1)


def render_wheel_clip(job):
    k, omega, seed, variant, out_dir = job
    rng = seed_rng("wheel", k, omega, seed, variant)
    duty = float(rng.uniform(0.38, 0.62))
    fg = float(rng.uniform(0.08, 0.20))
    bg = float(rng.uniform(0.80, 0.92))
    hub_lum = float(rng.uniform(0.40, 0.50))
    rim_lum = float(rng.uniform(0.25, 0.35))
    r_rim_out = float(rng.uniform(275.0, 305.0))
    rim_w = float(rng.uniform(8.0, 14.0))
    r_rim_in = r_rim_out - rim_w
    r_hub = float(rng.uniform(36.0, 44.0))
    band_rmin, band_rmax = r_hub + 8.0, r_rim_in - 8.0

    c = get_canvas()
    r = np.hypot(c.dx, c.dy)
    phi = np.arctan2(-c.dy, c.dx)
    wheel_m = (r > r_hub) & (r < r_rim_in)
    rim_m = (r >= r_rim_in) & (r <= r_rim_out)
    hub_m = r <= r_hub
    static = np.where(rim_m, rim_lum, np.where(hub_m, hub_lum, bg)).astype(np.float32)
    base = np.where(wheel_m, bg, static).astype(np.float32)

    n_taps = 1 if variant == "point" else N_SHUTTER_TAPS
    taus = np.linspace(-T_EXP / 2, T_EXP / 2, n_taps)
    w2pi = 2.0 * math.pi * omega
    os.makedirs(out_dir, exist_ok=True)
    for j in range(N_FRAMES):
        t0 = j / FPS
        acc = np.zeros_like(base)
        for tau in taus:
            u = ((phi - w2pi * (t0 + tau)) * k / (2.0 * math.pi)) % 1.0
            prof = (u < duty).astype(np.float32)
            acc += np.where(wheel_m, fg + (bg - fg) * prof, static)
        frame = acc / n_taps
        save_frame(frame, os.path.join(out_dir, f"frame_{j:04d}.png"))

    f0 = k * omega
    fa = wrap_hz(f0)
    omega_app = fa / k
    atten = abs(shutter_sign(f0))
    meta = {
        "category": "wheel",
        "variant": variant,
        "k": k,
        "omega_gt": omega,
        "seed": seed,
        "duty": duty,
        "fg_lum": fg,
        "bg_lum": bg,
        "r_hub": r_hub,
        "r_rim_in": r_rim_in,
        "r_rim_out": r_rim_out,
        "band_rmin": band_rmin,
        "band_rmax": band_rmax,
        "center": [CX, CY],
        "fps": FPS,
        "n_frames": N_FRAMES,
        "f_tex_hz": f0,
        "f_nyquist_hz": FPS / 2.0,
        "alias_f_hz": fa,
        "omega_app_pred": omega_app,
        "harm_atten": atten,
        "reversal_pred": bool(omega_app < -1e-6 and abs(fa) > 1.5),
        "direction_word": "counterclockwise" if omega > 0 else "clockwise",
        "prompt": (
            f"a grayscale wheel with {k} spokes rotating "
            f"{'counterclockwise' if omega > 0 else 'clockwise'} at exactly "
            f"{omega} revolutions per second, static camera, sharp focus, "
            f"no motion blur, white background"
        ),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return out_dir


def render_grating_clip(job):
    v, fs, seed, variant, out_dir = job
    rng = seed_rng("grating", v, fs, seed, variant)
    duty = float(rng.uniform(0.4, 0.6))
    fg = float(rng.uniform(0.08, 0.20))
    bg = float(rng.uniform(0.80, 0.92))
    period = 1.0 / fs

    c = get_canvas()
    n_taps = 1 if variant == "point" else N_SHUTTER_TAPS
    taus = np.linspace(-T_EXP / 2, T_EXP / 2, n_taps)
    os.makedirs(out_dir, exist_ok=True)
    for j in range(N_FRAMES):
        t0 = j / FPS
        acc = np.zeros((H * SUP, W * SUP), dtype=np.float32)
        for tau in taus:
            u = (((c.x - v * (t0 + tau)) / period) % 1.0)
            prof = (u < duty).astype(np.float32)
            acc += bg + (fg - bg) * prof
        save_frame(acc / n_taps, os.path.join(out_dir, f"frame_{j:04d}.png"))

    f0 = v * fs
    fa = wrap_hz(f0)
    v_app = fa / fs
    atten = abs(shutter_sign(f0))
    meta = {
        "category": "grating",
        "variant": variant,
        "v_px_s": v,
        "fs_cyc_px": fs,
        "seed": seed,
        "duty": duty,
        "fps": FPS,
        "n_frames": N_FRAMES,
        "f_tex_hz": f0,
        "f_nyquist_hz": FPS / 2.0,
        "alias_f_hz": fa,
        "v_app_pred": v_app,
        "harm_atten": atten,
        "reversal_pred": bool(v_app < -1e-6 and abs(fa) > 1.5),
        "direction_word": "right" if v > 0 else "left",
        "prompt": (
            f"vertical black and white stripes drifting to the right continuously, "
            f"static camera, sharp focus, no motion blur"
        ),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return out_dir


def render_ball_clip(job):
    g, seed, variant, out_dir = job
    rng = seed_rng("ball", g, seed, variant)
    px_per_m = 100.0
    ball_r = 16.0
    x0 = CX + float(rng.uniform(-60.0, 60.0))
    y_top = 80.0
    lum = float(rng.uniform(0.08, 0.18))
    bg = float(rng.uniform(0.85, 0.95))

    c = get_canvas()
    n_taps = 1 if variant == "point" else N_SHUTTER_TAPS
    taus = np.linspace(-T_EXP / 2, T_EXP / 2, n_taps)
    os.makedirs(out_dir, exist_ok=True)
    y_traj = []
    for j in range(N_FRAMES):
        t0 = j / FPS
        acc = None
        for tau in taus:
            t = t0 + tau
            y = y_top + 0.5 * g * px_per_m * t * t
            y_traj.append(y / px_per_m)
            d = np.hypot(c.x - x0, c.y - y)
            frame = np.where(d <= ball_r, lum, bg).astype(np.float32)
            acc = frame if acc is None else acc + frame
        save_frame(acc / n_taps, os.path.join(out_dir, f"frame_{j:04d}.png"))
    y_traj = np.array(y_traj[:N_FRAMES])

    meta = {
        "category": "ball",
        "variant": variant,
        "g_ms2": g,
        "seed": seed,
        "px_per_m": px_per_m,
        "ball_r": ball_r,
        "x0": x0,
        "y_top_m": y_top / px_per_m,
        "fps": FPS,
        "n_frames": N_FRAMES,
        "y_traj_m": [round(float(v), 4) for v in y_traj],
        "prompt": (
            f"a black ball falling straight down under gravity from rest, "
            f"static camera, sharp focus, no motion blur, white background"
        ),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return out_dir


def clip_done(out_dir):
    return (os.path.isfile(os.path.join(out_dir, "meta.json"))
            and os.path.isfile(os.path.join(out_dir, f"frame_{N_FRAMES - 1:04d}.png")))


def build_jobs(grid, out_root, skip_existing=True):
    jobs = []
    if grid == "smoke":
        wheel = [(6, 2.0), (8, 2.0), (16, 4.0)]
        for k, w in wheel:
            for variant in ("point", "shutter180"):
                jobs.append(("wheel", (k, w, 1, variant,
                            os.path.join(out_root, "wheel",
                                         f"k{k:02d}_w{w:.1f}_s01_{variant}"))))
        for variant in ("point", "shutter180"):
            jobs.append(("grating", (256.0, 0.1, 1, variant,
                        os.path.join(out_root, "grating",
                                     f"v256_f010_s01_{variant}"))))
        jobs.append(("ball", (9.8, 1, "point",
                     os.path.join(out_root, "ball", "g9.8_s01_point"))))
        return jobs
    for k in K_GRID:
        for w in OMEGA_GRID:
            for s in range(1, SEEDS + 1):
                for variant in ("point", "shutter180"):
                    name = f"k{k:02d}_w{w:.1f}_s{s:02d}_{variant}"
                    jobs.append(("wheel", (k, w, s, variant,
                                os.path.join(out_root, "wheel", name))))
    for v in V_GRID:
        for fs in FS_GRID:
            for s in range(1, G_SEEDS + 1):
                for variant in ("point", "shutter180"):
                    name = f"v{int(v)}_f{int(fs*1000):03d}_s{s:02d}_{variant}"
                    jobs.append(("grating", (v, fs, s, variant,
                                os.path.join(out_root, "grating", name))))
    for g in (9.8, 19.6):
        for s in range(1, G_SEEDS + 1):
            for variant in ("point", "shutter180"):
                name = f"g{g}_s{s:02d}_{variant}"
                jobs.append(("ball", (g, s, variant,
                            os.path.join(out_root, "ball", name))))
    if skip_existing:
        jobs = [(c, j) for c, j in jobs if not clip_done(j[-1])]
    return jobs


RENDERERS = {"wheel": render_wheel_clip,
             "grating": render_grating_clip,
             "ball": render_ball_clip}


def run_job(item):
    cat, job = item
    return cat, RENDERERS[cat](job)


MANIFEST_FIELDS = [
    "clip_id", "category", "variant", "k", "omega_gt", "seed",
    "f_tex_hz", "f_nyquist_hz", "alias_f_hz", "omega_app_pred",
    "reversal_pred", "band_rmin", "band_rmax", "center", "n_frames",
    "meta_path", "prompt",
]


def write_manifest(out_root):
    rows = []
    for cat in ("wheel", "grating", "ball"):
        cat_dir = os.path.join(out_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for name in sorted(os.listdir(cat_dir)):
            meta_path = os.path.join(cat_dir, name, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path, encoding="utf-8") as f:
                m = json.load(f)
            rows.append({
                "clip_id": f"{cat}/{name}",
                "category": cat,
                "variant": m["variant"],
                "k": m.get("k", ""),
                "omega_gt": m.get("omega_gt", ""),
                "seed": m.get("seed", ""),
                "f_tex_hz": round(m.get("f_tex_hz", float("nan")), 3),
                "f_nyquist_hz": m.get("f_nyquist_hz", ""),
                "alias_f_hz": round(m.get("alias_f_hz", float("nan")), 3),
                "omega_app_pred": round(m.get("omega_app_pred",
                                              m.get("v_app_pred",
                                                    float("nan"))), 4),
                "reversal_pred": m.get("reversal_pred", ""),
                "band_rmin": m.get("band_rmin", ""),
                "band_rmax": m.get("band_rmax", ""),
                "center": ";".join(str(v) for v in m.get("center", [])),
                "n_frames": m.get("n_frames", ""),
                "meta_path": os.path.relpath(meta_path, out_root),
                "prompt": m.get("prompt", ""),
            })
    csv_path = os.path.join(out_root, "gt_curves.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--out", default="probe_set")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    jobs = build_jobs(args.grid, args.out)
    print(f"{len(jobs)} clips to render, workers={args.workers}")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for cat, path in ex.map(run_job, jobs, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {path}", flush=True)
    csv_path, n = write_manifest(args.out)
    print(f"manifest: {csv_path} ({n} rows)")


if __name__ == "__main__":
    main()
