"""R12/R14 temporal diagnostics with fixed, corrected candidate geometries."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from p0_verify import ma, write_csv
from p0_windowed import windows

ROOT=Path(__file__).resolve().parent


def phase_intervals(z,fps,k):
    increment=np.angle(z[1:]*np.conj(z[:-1]))
    floor=max(1e-5,.2*float(np.median(abs(z))))
    result=[]
    for i,delta in enumerate(increment):
        reason=('low_amplitude' if min(abs(z[i]),abs(z[i+1]))<floor else
                'near_pi_ambiguity' if abs(delta)>.8*np.pi else
                'near_zero_increment' if abs(delta)<.02 else '')
        result.append(dict(start=i/fps,end=(i+1)/fps,omega=-float(delta)*fps/(2*np.pi*k),
            phase_increment=float(delta),amplitude_floor=floor,reason=reason,
            direction='uncertain' if reason else 'cw' if delta>0 else 'ccw'))
    return result


def main():
    out=ROOT/'p0_audit_results/temporal_followup_v1'
    out.mkdir(exist_ok=False)
    # Synthetic per-interval directions at the actual 8 FPS; no unwrapping
    # across uncertain intervals and no inference of subframe motion.
    tests=[]
    for name,velocity in [('cw',np.full(48,-.5)),('reversal',np.r_[np.full(32,-.5),np.full(16,.5)]),
                          ('jitter',np.array([.5 if i in (16,17) else -.5 for i in range(48)]))]:
        z=np.exp(-2j*np.pi*2*np.r_[0,np.cumsum(velocity)/8])
        r=phase_intervals(z,8,2)
        passed=all(x['direction']==('ccw' if v>0 else 'cw') for x,v in zip(r,velocity))
        tests.append(dict(case=name,passed=passed))
    tests.append(dict(case='zero_amplitude',passed=all(r['direction']=='uncertain' for r in phase_intervals(np.zeros(49,complex),8,2))))
    assert all(t['passed'] for t in tests)
    review=ROOT/'p0_audit_results/anonymous_review_v1'
    with (review/'mapping_DO_NOT_OPEN_BEFORE_REVIEW.csv').open(encoding='utf-8-sig') as f:
        mapping={r['sample_id']:r for r in csv.DictReader(f)}
    with (ROOT/'p0_audit_results/cloud_snapshot_v1/phase2_bundle/phase2/gt_curves.csv').open() as f:
        gt={r['clip_id'].split('/')[1].removesuffix('_point'):r for r in csv.DictReader(f) if r['variant']=='point'}
    intervals=[];window_rows=[];consensus=[];summaries=[]
    for sid in ('R12','R14'):
        m=mapping[sid];g=gt[m['cell_key'].removesuffix('_rep0')]
        path=review/'videos'/f'{sid}.mp4'
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        frames=ma.load_frames_gray(str(path));h,w=frames[0].shape
        k=int(g['k']);x,y=map(float,g['center'].split(';'))
        geometry={'scaled':(x*w/1280,y*h/720),'detected':ma.find_center(frames[0])}
        phases={}
        for method,(cx,cy) in geometry.items():
            z=ma.demodulate(ma.angular_signals(frames,cx,cy,float(g['band_rmin'])*h/720,float(g['band_rmax'])*h/720),k)
            phases[method]=phase_intervals(z,fps,k)
            intervals.extend(dict(sample_id=sid,method=method,**r) for r in phases[method])
            for seconds in (1.,1.5,2.):
                wr=windows(z,fps,k,seconds)
                window_rows.extend(dict(sample_id=sid,method=method,**r) for r in wr)
                summaries.append(dict(sample_id=sid,method=method,window=seconds,
                    cw=sum(r['direction']=='cw' for r in wr),ccw=sum(r['direction']=='ccw' for r in wr),
                    uncertain=sum(r['direction']=='uncertain' for r in wr)))
        for a,b in zip(phases['scaled'],phases['detected']):
            direction=a['direction'] if a['direction']==b['direction'] else 'uncertain'
            consensus.append(dict(sample_id=sid,start=a['start'],end=a['end'],direction=direction,
                methods_agree=a['direction']==b['direction']))
    write_csv(out/'phase_intervals.csv',intervals)
    write_csv(out/'windows.csv',window_rows)
    write_csv(out/'phase_consensus.csv',consensus)
    (out/'selftests.json').write_text(json.dumps(tests,indent=2))
    (out/'window_summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))
    for sid in ('R12','R14'):
        print(sid,'phase CCW intervals',[(r['start'],r['end']) for r in consensus if r['sample_id']==sid and r['direction']=='ccw'])
        print(sid,'uncertain',sum(r['sample_id']==sid and r['direction']=='uncertain' for r in consensus))


if __name__=='__main__':main()
