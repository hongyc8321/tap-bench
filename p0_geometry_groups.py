"""Compare candidate geometries on local CogVideoX arms; retain every failure."""
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from p0_paired_remeasure import ARMS, SNAP
from p0_verify import ma, write_csv


def main():
    out = Path(__file__).resolve().parent/'p0_audit_results/geometry_groups_v1'
    out.mkdir(exist_ok=False)
    arms = dict(ARMS)
    for arm in ('ORIG', 'A0', 'A5'):
        arms['cw_'+arm] = f'phase3/gen_cw_{arm}/cogvideox2b_t2v_manifest.csv'
    tables = {}
    for protocol, rel in [('standard','phase2_bundle/phase2/gt_curves.csv'), ('cw','phase3/gt_cw/gt_curves.csv')]:
        with (SNAP/rel).open() as f:
            tables[protocol] = {r['clip_id'].split('/')[1].removesuffix('_point'):r
                for r in csv.DictReader(f) if r['variant']=='point'}
    rows, disagreements = [], []
    methods = ('scaled_circle', 'hough_or_fallback')
    for arm, rel in arms.items():
        protocol = 'cw' if arm.startswith('cw_') else 'standard'
        with (SNAP/rel).open() as f:
            items = [r for r in csv.DictReader(f) if r['category']=='wheel']
        if len({r['cell_key'] for r in items}) != len(items):
            raise ValueError('Duplicate cell identity')
        for item in items:
            outputs = []
            try:
                raw = Path(item['path'])
                base = SNAP/('phase3' if rel.startswith('phase3/') else 'phase2_bundle/phase2')
                candidates = [SNAP/raw.relative_to('/home/ubuntu')] if raw.is_absolute() else [base/raw, (SNAP/rel).parent/raw, SNAP/raw]
                path = next(p for p in candidates if p.is_file())
                cap = cv2.VideoCapture(str(path))
                fps, declared = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                frames = ma.load_frames_gray(str(path))
                if len(frames)<8 or len(frames)!=declared or not np.isfinite(fps) or fps<=0:
                    raise ValueError('Invalid media metadata or incomplete decode')
                h,w = frames[0].shape
                g = tables[protocol][item['stem']]
                x,y = map(float,g['center'].split(';'))
                lo,hi = float(g['band_rmin'])*h/720, float(g['band_rmax'])*h/720
                centers = {'scaled_circle':(x*w/1280,y*h/720), 'hough_or_fallback':ma.find_center(frames[0])}
                for method in methods:
                    row = dict(arm=arm, protocol=protocol, cell_key=item['cell_key'], method=method, status='failed')
                    try:
                        cx,cy = centers[method]
                        theta = np.linspace(0,2*np.pi,ma.M_ANG,endpoint=False)[:,None]
                        radius = np.linspace(lo,hi,ma.R_SAMPLES)[None,:]
                        mx,my = cx+radius*np.cos(theta),cy-radius*np.sin(theta)
                        outside = float(np.mean((mx<0)|(mx>w-1)|(my<0)|(my>h-1)))
                        z = ma.demodulate(ma.angular_signals(frames,cx,cy,lo,hi),int(g['k']))
                        frequency,snr,_,_ = ma.fft_estimate(z,fps)
                        est = -float(frequency)/int(g['k'])
                        if not np.isfinite(est) or not np.isfinite(snr):
                            raise ValueError('Nonfinite estimate')
                        row.update(status='measured',cx=cx,cy=cy,rmin=lo,rmax=hi,
                            fps=fps,frames=len(frames),out_of_frame_fraction=outside,
                            estimate=est,snr=float(snr),low_conf=bool(snr<ma.SNR_MIN),
                            geometry_valid=outside==0,sign_match=est*float(item['omega_gt'])>0,
                            motion=abs(est)>=.15,signal_std=float(np.std(z)))
                    except Exception as e:
                        row['error']=f'{type(e).__name__}: {e}'
                    outputs.append(row)
                del frames
            except Exception as e:
                outputs = [dict(arm=arm,protocol=protocol,cell_key=item['cell_key'],method=m,
                    status='failed',error=f'{type(e).__name__}: {e}') for m in methods]
            rows.extend(outputs)
            if all(r['status']=='measured' for r in outputs):
                a,b = outputs
                if np.sign(a['estimate']) != np.sign(b['estimate']):
                    disagreements.append(dict(arm=arm,cell_key=item['cell_key'],scaled_estimate=a['estimate'],
                        detected_estimate=b['estimate'],scaled_low_conf=a['low_conf'],detected_low_conf=b['low_conf'],
                        both_geometry_valid=a['geometry_valid'] and b['geometry_valid']))
        write_csv(out/f'{arm}_rows.csv',[r for r in rows if r['arm']==arm])
        print(arm,len(items),'videos processed',flush=True)
    write_csv(out/'rows.csv',rows)
    write_csv(out/'direction_disagreements.csv',disagreements or [dict(arm='',cell_key='')])
    summary=[]
    for arm in arms:
        for method in methods:
            all_rows=[r for r in rows if r['arm']==arm and r['method']==method]
            measured=[r for r in all_rows if r['status']=='measured']
            usable=[r for r in measured if r['geometry_valid'] and not r['low_conf']]
            summary.append(dict(arm=arm,method=method,n=len(all_rows),measured=len(measured),
                low_snr=sum(r['low_conf'] for r in measured),out_of_frame=sum(not r['geometry_valid'] for r in measured),
                usable=len(usable),rr_all=sum(not r['sign_match'] for r in measured)/len(measured) if measured else None,
                rr_usable=sum(not r['sign_match'] for r in usable)/len(usable) if usable else None))
    write_csv(out/'summary.csv',summary)
    (out/'config.json').write_text(json.dumps(dict(reference_width=1280,reference_height=720,
        radial_scale='height_only_circle',snr_min=ma.SNR_MIN,methods=methods,
        estimator_sha256=hashlib.sha256(Path(ma.__file__).read_bytes()).hexdigest(),
        numpy=np.__version__,opencv=cv2.__version__,
        warning='Hough helper may silently fall back to frame center; zero border violations do not validate target geometry.'),indent=2))
    print(json.dumps(summary,indent=2))
    print('direction_disagreements',len(disagreements))


if __name__=='__main__':
    main()
