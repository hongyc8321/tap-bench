"""Audit snapshot media and remeasure clockwise controls locally."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from p0_verify import ma, write_csv

ROOT = Path(__file__).resolve().parent


def main():
    snapshot = ROOT/'p0_audit_results/cloud_snapshot_v1'
    out = ROOT/'p0_audit_results/snapshot_audit_v2'
    out.mkdir(exist_ok=False)
    with (snapshot/'phase3/gt_cw/gt_curves.csv').open() as f:
        gt = {'_'.join(r['clip_id'].split('/')[1].split('_')[:-1]):r for r in csv.DictReader(f)}
    media, controls, sets = [], [], {}
    for manifest in sorted(snapshot.rglob('*manifest*.csv')):
        rel = str(manifest.relative_to(snapshot))
        with manifest.open() as f:
            items = list(csv.DictReader(f))
        sets[rel] = {r['cell_key'] for r in items}
        for item in items:
            raw = Path(item['path'])
            base = snapshot/'phase3' if rel.startswith('phase3/') else snapshot/'phase2_bundle/phase2'
            candidates = [snapshot/raw.relative_to('/home/ubuntu')] if raw.is_absolute() else [base/raw, manifest.parent/raw, snapshot/raw]
            row = dict(manifest=rel, cell_key=item['cell_key'], status='failed')
            try:
                path = next(p for p in candidates if p.is_file())
                cap = cv2.VideoCapture(str(path))
                frames, n = [], 0
                cw = '/gen_cw_' in rel
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        n += 1
                        if cw:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255)
                finally:
                    cap.release()
                row.update(decoded_frames=n, container_frames=declared,
                    manifest_frames=int(item['frames']) if item.get('frames') else '', fps_video=fps,
                    fps_manifest=float(item['fps_model']))
                if n != declared or (item.get('frames') and n != int(item['frames'])) or n < 8 or fps <= 0:
                    raise ValueError('frame count or fps invalid')
                row['status'] = 'decoded'
                if cw:
                    g = gt[item['stem']]
                    meta = dict(fps=fps, k=int(g['k']),
                        center=[float(v) for v in g['center'].split(';')],
                        band_rmin=float(g['band_rmin']), band_rmax=float(g['band_rmax']))
                    with patch.object(ma, 'load_frames_gray', return_value=frames):
                        m = ma.measure_wheel(str(path), meta)
                    controls.append(dict(manifest=rel, cell_key=item['cell_key'],
                        gt=float(item['omega_gt']), est=m['omega_est'], snr=m['snr'],
                        sign_match=m['omega_est']*float(item['omega_gt'])>0,
                        motion=abs(m['omega_est'])>=.15, low_conf=m['snr']<ma.SNR_MIN))
            except Exception as e:
                row.update(status='failed', error=f'{type(e).__name__}: {e}')
            media.append(row)
        print(rel, 'processed', len(items), flush=True)
    write_csv(out/'media.csv', media)
    write_csv(out/'cw_remeasured.csv', controls)
    summary = dict(total=len(media), decoded=sum(r['status']=='decoded' for r in media),
        fps_mismatches=sum(r.get('fps_video')!=r.get('fps_manifest') for r in media), cw={})
    for name in sorted({r['manifest'] for r in controls}):
        rows = [r for r in controls if r['manifest']==name]
        conf = [r for r in rows if not r['low_conf']]
        summary['cw'][name] = dict(n=len(rows), n_conf=len(conf),
            compliance_all=sum(r['sign_match'] for r in rows)/len(rows),
            compliance_conf=sum(r['sign_match'] for r in conf)/len(conf) if conf else None)
    baseline = sets['phase3/gen_A0/cogvideox2b_t2v_manifest.csv']
    summary['cell_set_vs_A0'] = {name:dict(n=len(s), shared=len(s & baseline),
        missing=len(baseline-s), extra=len(s-baseline)) for name,s in sets.items()}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
