"""Decode local assets and rerun the legacy estimator without changing labels."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'phase2'))
import measure_angle as ma


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def main():
    out = ROOT / 'p0_audit_results' / 'verification_v1'
    out.mkdir(exist_ok=False)
    fps = 24.0
    y, x = np.mgrid[:256, :256]
    # Screen y increases down: this angle increases counterclockwise.
    angle = np.arctan2(128 - y, x - 128)
    tests = []
    for category in ('wheel', 'grating'):
        for sign in (-1, 1):
            expected = sign * (0.5 if category == 'wheel' else 8.0)
            frames = []
            for t in np.arange(120) / fps:
                phase = (4 * (angle - 2 * np.pi * expected * t) if category == 'wheel'
                         else 2 * np.pi * 0.05 * (x - expected * t))
                frames.append(((1 + np.cos(phase)) / 2).astype(np.float32))
            with patch.object(ma, 'load_frames_gray', return_value=frames):
                if category == 'wheel':
                    result = ma.measure_wheel('', {'center': [128, 128], 'band_rmin': 40,
                        'band_rmax': 100, 'k': 4, 'fps': fps})
                    estimate = result['omega_est']
                else:
                    result = ma.measure_grating('', {'fs_cyc_px': 0.05, 'fps': fps})
                    estimate = result['v_est']
            passed = abs(estimate - expected) < abs(expected) * 0.03
            tests.append(dict(category=category, expected=expected, estimate=estimate, passed=passed))
    write_csv(out / 'direction_selftest.csv', tests)
    assert all(r['passed'] for r in tests), tests

    with (ROOT / 'phase2_bundle/phase2/gt_curves.csv').open(encoding='utf-8') as f:
        gt = {r['clip_id'].split('/')[1].removesuffix('_point'): r
              for r in csv.DictReader(f) if r['variant'] == 'point'}
    with (ROOT / 'assets_seedance_videos/seedance20_manifest.csv').open(encoding='utf-8') as f:
        manifest = list(csv.DictReader(f))
    with (ROOT / 'phase3_results_seedance/nyquist_rows.csv').open(encoding='utf-8') as f:
        old = {r['cell_key']: r for r in csv.DictReader(f)}
    rows = []
    for item in manifest:
        path = ROOT / 'assets_seedance_videos' / 'seedance20' / item['mode'] / (item['cell_key'] + '.mp4')
        row = dict(cell_key=item['cell_key'], category=item['category'], path=str(path),
                   fps_manifest=float(item['fps_model']), status='pending')
        try:
            cap = cv2.VideoCapture(str(path))
            frames = []
            try:
                row['fps_video'] = cap.get(cv2.CAP_PROP_FPS)
                row['container_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255)
            finally:
                cap.release()
            row['decoded_frames'] = len(frames)
            row['full_decode_count_match'] = len(frames) == row['container_frames'] == int(item['frames'])
            if not row['full_decode_count_match'] or len(frames) < 8 or row['fps_video'] <= 0:
                raise ValueError('Incomplete decode or invalid media metadata')
            row['fps_delta'] = row['fps_video'] - row['fps_manifest']
            row['duration_seconds'] = len(frames) / row['fps_video']
            g = gt[item['stem']]
            row['gt_prompt'] = g['prompt']
            with patch.object(ma, 'load_frames_gray', return_value=frames):
                if item['category'] == 'wheel':
                    meta = dict(center=[float(v) for v in g['center'].split(';')],
                        band_rmin=float(g['band_rmin']), band_rmax=float(g['band_rmax']),
                        k=int(item['k']), fps=row['fps_video'])
                    m = ma.measure_wheel(str(path), meta)
                    estimate = m['omega_est']
                    target = float(item['omega_gt'])
                else:
                    meta = dict(fs_cyc_px=float(item['fs_cyc_px']), fps=row['fps_video'])
                    m = ma.measure_grating(str(path), meta)
                    estimate = m['v_est'] / 100
                    target = float(item['v_px_s']) / 100
            row.update(status='measured', estimate=estimate, snr=m['snr'],
                low_conf=m['snr'] < ma.SNR_MIN, sign_match=estimate * target > 0,
                motion=abs(estimate) >= 0.15, speed_ratio=abs(estimate / target),
                historical_estimate=float(old[item['cell_key']]['est']))
            row['estimate_delta'] = estimate - row['historical_estimate']
        except Exception as e:
            row.update(status='failed', error=f'{type(e).__name__}: {e}')
        rows.append(row)
        print(item['cell_key'], row['status'], flush=True)
    write_csv(out / 'remeasured.csv', rows)
    stats = {'total': len(rows), 'selftest_passed': len(tests), 'categories': {}}
    for cat in ('wheel', 'grating'):
        measured = [r for r in rows if r['category'] == cat and r['status'] == 'measured']
        conf = [r for r in measured if not r['low_conf']]
        stats['categories'][cat] = dict(total=sum(r['category'] == cat for r in rows),
            measured=len(measured), low_conf=len(measured)-len(conf),
            rr_all=sum(not r['sign_match'] for r in measured)/len(measured) if measured else None,
            rr_conf=sum(not r['sign_match'] for r in conf)/len(conf) if conf else None,
            stationary_all=sum(not r['motion'] for r in measured)/len(measured) if measured else None)
    stats.update(numpy=np.__version__, opencv=cv2.__version__,
        estimator_sha256=hashlib.sha256((ROOT/'phase2/measure_angle.py').read_bytes()).hexdigest())
    (out / 'summary.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
