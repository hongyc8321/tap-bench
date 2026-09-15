#!/usr/bin/env python3
"""CW capability probe: render clockwise GT cells + manifest + self-check.

Cells are restricted to |k*omega| < 4 Hz (strictly sub-Nyquist at 8 fps) so
that a compliant CW output has an unambiguous CW apparent direction.
Self-check compares est against the ALIASED apparent prediction
(omega_app_pred), not the true omega.
"""

import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tap_bench"))
sys.path.insert(0, os.path.expanduser("~/tap_bench"))
sys.path.insert(0, os.path.expanduser("~/phase2_bundle/phase2"))
sys.path.insert(0, HERE)

import render_probe as rp  # noqa: E402
import measure_angle as ma  # noqa: E402

rp.FPS = 8
rp.N_FRAMES = 49
rp.T_EXP = 0.5 / 8

# (k, |omega|) with k*omega < 4 -> apparent direction = true direction = CW
CELLS = [(2, 0.5), (2, 1.0), (2, 1.5), (4, 0.5), (4, 0.75), (6, 0.5)]
SEEDS = [1, 2, 3]

OUT = os.path.join(HERE, "gt_cw")


def render_one(job):
    k, omega, seed, out_dir = job
    rp.render_wheel_clip((k, omega, seed, "point", out_dir))
    return out_dir


def fix_prompt(path):
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        m = json.load(f)
    m["prompt"] = (f"a grayscale wheel with {m['k']} spokes rotating "
                   f"clockwise at exactly {abs(m['omega_gt'])} revolutions "
                   f"per second, static camera, sharp focus, no motion "
                   f"blur, white background")
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=1)
    return m


def self_check():
    print("=== CW GT self-check (vs aliased apparent prediction) ===",
          flush=True)
    errs = []
    ok = True
    for name in sorted(os.listdir(os.path.join(OUT, "wheel"))):
        cdir = os.path.join(OUT, "wheel", name)
        with open(os.path.join(cdir, "meta.json"), encoding="utf-8") as f:
            m = json.load(f)
        res = ma.measure_wheel(cdir, meta=m, fps=m["fps"])
        if "error" in res:
            print(f"{name}: ERROR {res['error']}")
            ok = False
            continue
        est = res["omega_est"]
        pred = m["omega_app_pred"]
        snr = res.get("snr", 0)
        errs.append(abs(est - pred))
        status = "OK" if (abs(est - pred) < 0.02 and snr >= ma.SNR_MIN) \
            else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{name}: est={est:+.4f} pred={pred:+.4f} snr={snr:.0f} "
              f"{status}")
    mae = sum(errs) / len(errs)
    print(f"MAE vs apparent pred = {mae:.4f} rev/s over {len(errs)} clips")
    return mae, ok


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "wheel"))

    jobs = []
    for k, w in CELLS:
        for s in SEEDS:
            name = f"k{k:02d}_w-{w}_s{s:02d}_point"
            jobs.append((k, -w, s, os.path.join(OUT, "wheel", name)))
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, out_dir in enumerate(ex.map(render_one, jobs)):
            fix_prompt(out_dir)
            print(f"[{i + 1}/{len(jobs)}] rendered", flush=True)

    csv_path, n = rp.write_manifest(OUT)
    print(f"manifest: {csv_path} ({n} rows)")
    mae, ok = self_check()
    if mae > 0.02 or not ok:
        print("SELF-CHECK FAILED — do not use for generation", flush=True)
        sys.exit(1)
    print("CW GT ready.", flush=True)


if __name__ == "__main__":
    main()
