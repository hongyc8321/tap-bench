"""Separate dominant direction from local temporal complexity."""
import csv,json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from p0_verify import ma,write_csv
from p0_windowed import windows
from p0_localization_exceptions import inspect_contours,sample

ROOT=Path(__file__).resolve().parent/'p0_audit_results'

def consensus(zs,fps,k,seconds):
    sets=[windows(z,fps,k,seconds) for z in zs];directions=[]
    for pair in zip(*sets):directions.append(pair[0]['direction'] if pair[0]['direction']==pair[1]['direction'] else 'uncertain')
    c=Counter(directions);coverage=1-c['uncertain']/len(directions)
    dominant='uncertain'
    if coverage>=.5 and bool(c['cw']) != bool(c['ccw']):dominant='cw' if c['cw'] else 'ccw'
    return dict(seconds=seconds,windows=len(directions),coverage=coverage,cw=c['cw'],ccw=c['ccw'],uncertain=c['uncertain'],dominant=dominant)

def main():
    out=ROOT/'dual_timescale_v2';out.mkdir(exist_ok=False)
    # Test decision logic using ideal complex harmonic signals.
    tests=[];fps=8;k=4;t=np.arange(49)/fps
    for name,phase in [('steady',-.25*t),('reverse',np.where(t<3,-.25*t,-.75+.25*(t-3))),
        ('jitter',-.25*t+np.where((t>=3)&(t<3.25),.5*(t-3),np.where(t>=3.25,.125,0)))]:
        z=np.exp(-2j*np.pi*k*phase); short=consensus([z,z],fps,k,1.0);long3=consensus([z,z],fps,k,3.0)
        tests.append(dict(case=name,short=short,long=long3))
    selected={}
    with (ROOT/'adaptive_harmonic_v1/video_results.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):selected[(r['arm'],r['cell_key'])]=r
    results=[];scales=[]
    for arm in ('A0','A3'):
        with (ROOT/f'formal_{arm}_eval_download_v1/local_manifest.csv').open(encoding='utf-8-sig') as f:
            items=[r for r in csv.DictReader(f) if r['category']=='wheel']
        for item in items:
            choice=selected[(arm,item['cell_key'])]
            if choice['status']!='selected':
                results.append(dict(arm=arm,cell_key=item['cell_key'],status='harmonic_unresolved'));continue
            frames=ma.load_frames_gray(item['path']);_,contours=inspect_contours((frames[0]*255).astype('uint8'),190);valid=[r for r in contours if r['valid']]
            if len(valid)!=1:results.append(dict(arm=arm,cell_key=item['cell_key'],status='localization_rejected'));continue
            c=valid[0];zs=[];k=int(choice['selected_k']);fps=float(item['fps_model'])
            for a,b,angle in [(2*c['radius'],2*c['radius'],0),(c['a'],c['b'],c['angle'])]:
                zs.append(ma.demodulate(sample(frames,c['cx'],c['cy'],a,b,angle)[0],k))
            values={seconds:consensus(zs,fps,k,seconds) for seconds in (1.,3.,4.)}
            for value in values.values():scales.append(dict(arm=arm,cell_key=item['cell_key'],**value))
            long=[values[3.]['dominant'],values[4.]['dominant']]
            dominant=long[0] if long[0]==long[1] and long[0]!='uncertain' else 'uncertain'
            short=values[1.];local=('bidirectional_candidate' if short['cw']>=2 and short['ccw']>=2 else
                'one_opposite_candidate' if short['cw'] and short['ccw'] else
                'stable_candidate' if short['dominant']!='uncertain' else 'unresolved')
            results.append(dict(arm=arm,cell_key=item['cell_key'],status='dominant_resolved' if dominant!='uncertain' else 'dominant_unresolved',
                selected_k=k,dominant_direction=dominant,local_behavior=local,
                direction_error=(dominant!='ccw') if dominant!='uncertain' else '',
                short_coverage=short['coverage'],long3_coverage=values[3.]['coverage'],long4_coverage=values[4.]['coverage']))
        print(arm,'complete',flush=True)
    write_csv(out/'video_results.csv',results);write_csv(out/'scale_results.csv',scales)
    summary=[]
    for arm in ('A0','A3'):
        rows=[r for r in results if r['arm']==arm];ok=[r for r in rows if r['status']=='dominant_resolved']
        summary.append(dict(arm=arm,n=len(rows),dominant_resolved=len(ok),coverage=len(ok)/len(rows),
            conditional_candidate_error=sum(r['direction_error'] is True for r in ok)/len(ok) if ok else None,
            local_behavior=dict(Counter(r.get('local_behavior','not_run') for r in rows))))
    (out/'summary.json').write_text(json.dumps(dict(groups=summary,synthetic_diagnostics=tests,
        caveat='Long-window dominant and short-window local behavior are separate candidate outputs; neither calibrated as physical truth.'),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
