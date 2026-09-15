"""Exploratory wheel direction windows; positive means screen counterclockwise."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'phase2'))
import measure_angle as ma


def windows(z, fps, k, seconds):
    size = round(seconds * fps)
    rows = []
    for start in range(0, len(z) - size + 1, max(1, round(fps * .25))):
        segment = z[start:start + size]
        f, snr, freqs, mag = ma.fft_estimate(segment, fps)
        positive = float(mag[freqs > 0].max())
        negative = float(mag[freqs < 0].max())
        dominance = max(positive, negative) / max(min(positive, negative), 1e-12)
        amplitude = float(np.std(segment))
        # One cycle per window is a conservative resolution guard, not a
        # calibrated confidence interval. SNR alone cannot reject deformations.
        reason = ('flat_signal' if amplitude < 1e-5 else
                  'low_snr' if snr < ma.SNR_MIN else
                  'near_zero_frequency' if abs(f) < 1 / seconds else
                  'near_nyquist' if abs(f) > fps / 2 - 1 / seconds else
                  'competing_directions' if dominance < 1.5 else '')
        rows.append(dict(start_seconds=start/fps, end_seconds=(start+size)/fps,
            center_seconds=(start+size/2)/fps, window_seconds=seconds,
            omega_est=-float(f)/k, snr=float(snr), dominance=dominance,
            signal_std=amplitude, reason=reason,
            direction='uncertain' if reason else 'ccw' if f < 0 else 'cw'))
    return rows


def events(rows):
    result = []
    last = None
    for row in rows:
        if row['direction'] == 'uncertain':
            continue
        if last and row['direction'] != last['direction']:
            result.append(dict(from_direction=last['direction'], to_direction=row['direction'],
                bracket_start_seconds=last['start_seconds'], bracket_end_seconds=row['end_seconds'],
                previous_window_center=last['center_seconds'], next_window_center=row['center_seconds']))
        last = row
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}) or ['cell_key'])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'p0_audit_results/windowed_v1')
    args = ap.parse_args()
    args.out.mkdir(exist_ok=False)
    fps, k = 24., 4
    t = np.arange(121)/fps
    tests = []
    rng = np.random.default_rng(17)
    signals = dict(ccw=np.exp(-2j*np.pi*2*t), cw=np.exp(2j*np.pi*2*t),
        stopped=np.ones(len(t), complex),
        reversal=np.exp(-2j*np.pi*2*np.where(t <= 2.5, t, 5-t)))
    signals['noisy_ccw'] = signals['ccw'] + .05*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
    for name, z in signals.items():
        for seconds in (1., 1.5, 2.):
            r = windows(z, fps, k, seconds)
            directions = {x['direction'] for x in r} - {'uncertain'}
            expected = {'ccw','cw'} if name == 'reversal' else set() if name == 'stopped' else {'cw'} if name == 'cw' else {'ccw'}
            passed = directions == expected and (len(events(r)) == 1 if name == 'reversal' else len(events(r)) == 0)
            tests.append(dict(case=name, window_seconds=seconds, passed=passed))
    write_csv(args.out/'selftests.csv', tests)
    assert all(r['passed'] for r in tests), tests
    with (ROOT/'phase2_bundle/phase2/gt_curves.csv').open(encoding='utf-8') as f:
        gt = {r['clip_id'].split('/')[1].removesuffix('_point'):r for r in csv.DictReader(f) if r['variant']=='point'}
    with (ROOT/'assets_seedance_videos/seedance20_manifest.csv').open(encoding='utf-8') as f:
        manifest = [r for r in csv.DictReader(f) if r['category']=='wheel']
    with (ROOT/'p0_audit_results/blind_review_queue.csv').open(encoding='utf-8-sig') as f:
        labels = {r['cell_key']:r for r in csv.DictReader(f)}
    all_rows, all_events, summary = [], [], []
    for item in manifest:
        key = item['cell_key']
        path = ROOT/'assets_seedance_videos/seedance20'/item['mode']/(key+'.mp4')
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        frames = ma.load_frames_gray(str(path))
        if len(frames) != int(item['frames']) or fps <= 0:
            raise ValueError(f'Invalid media: {key}')
        g = gt[item['stem']]
        cx, cy = map(float, g['center'].split(';'))
        z = ma.demodulate(ma.angular_signals(frames, cx, cy, float(g['band_rmin']), float(g['band_rmax'])), int(item['k']))
        del frames
        full, _, _, _ = ma.fft_estimate(z, fps)
        label = labels.get(key, {})
        for seconds in (1., 1.5, 2.):
            rows = windows(z, fps, int(item['k']), seconds)
            ev = events(rows)
            all_rows.extend(dict(cell_key=key, **r) for r in rows)
            all_events.extend(dict(cell_key=key, window_seconds=seconds, **e) for e in ev)
            summary.append(dict(cell_key=key, window_seconds=seconds, windows=len(rows),
                uncertain=sum(r['direction']=='uncertain' for r in rows),
                ccw=sum(r['direction']=='ccw' for r in rows), cw=sum(r['direction']=='cw' for r in rows),
                candidate_changes=len(ev), full_omega=-float(full)/int(item['k']),
                human_direction=label.get('direction',''), human_spokes=label.get('spokes_correct',''),
                human_notes=label.get('notes','')))
        print(key, 'complete', flush=True)
    write_csv(args.out/'windows.csv', all_rows)
    write_csv(args.out/'candidate_intervals.csv', all_events)
    write_csv(args.out/'summary.csv', summary)
    (args.out/'config.json').write_text(json.dumps(dict(window_seconds=[1,1.5,2], step_seconds=.25,
        snr_min=ma.SNR_MIN, dominance_min=1.5, signal_std_min=1e-5,
        guard='1/window_seconds <= abs(f) <= fps/2 - 1/window_seconds',
        note='Events bridge uncertain windows; brackets are review intervals, not change timestamps. Overlapping windows are not independent.',
        numpy=np.__version__, opencv=cv2.__version__,
        estimator_sha256=hashlib.sha256((ROOT/'phase2/measure_angle.py').read_bytes()).hexdigest()), indent=2), encoding='utf-8')
    print('selftests',len(tests),'windows',len(all_rows),'candidate_events',len(all_events))


if __name__ == '__main__':
    main()
