"""Fixed-algorithm synthetic frame benchmark; no threshold fitting."""
import json
from pathlib import Path
import cv2
import numpy as np
from p0_verify import ma, write_csv
from p0_temporal_followup import phase_intervals
from p0_spatial_match import match

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'p0_audit_results/temporal_benchmark_v1'
    out.mkdir(exist_ok=False)
    rows=[];catalog=[]
    y,x=np.mgrid[:256,:256]
    theta=np.arctan2(128-y,x-128)
    radius=np.hypot(x-128,y-128)
    roi=(radius>=35)&(radius<=100)
    for fps in (8,24):
        n=round(6*fps)
        t=np.arange(n)/fps
        for speed in (.25,1.5):
            for texture in ('symmetric','asymmetric'):
                for motion in ('steady','reverse','jitter','stopped'):
                    for seed in (11,29):
                        name=f'f{fps}_v{speed}_{texture}_{motion}_s{seed}'
                        velocity=np.full(n,-speed)
                        if motion=='reverse':velocity[t>=3]=speed
                        if motion=='jitter':velocity[(t>=3)&(t<3.25)]=speed
                        if motion=='stopped':velocity[:]=0
                        phase=2*np.pi*np.r_[0,np.cumsum(velocity)/fps]
                        rng=np.random.default_rng(seed)
                        frames=[]
                        for a in phase:
                            u=theta-a
                            signal=.5+.30*np.cos(4*u)
                            # A broad unique marker breaks exact rotational symmetry.
                            if texture=='asymmetric':signal+=.16*np.cos(u+.3)+.03*np.sin(2*u)
                            frame=np.where(roi,signal,1.)
                            frame=np.clip(frame+rng.normal(0,.008,frame.shape),0,1)
                            frames.append((np.round(frame*255)/255).astype('float32'))
                        cv2.imwrite(str(out/(name+'.png')),(frames[0]*255).astype('uint8'))
                        sig=ma.angular_signals(frames,128,128,40,90)
                        z=ma.demodulate(sig,4)
                        phase_rows=phase_intervals(z,fps,4)
                        spatial_rows=[match(a,b) for a,b in zip(sig[:-1],sig[1:])]
                        for method,results in [('phase_k4',phase_rows),('full_circle_match',spatial_rows)]:
                            assert len(results)==n
                            for i,(v,r) in enumerate(zip(velocity,results)):
                                physical='none' if v==0 else 'ccw' if v>0 else 'cw'
                                alias=((4*v+fps/2)%fps-fps/2)/4
                                observable='none' if alias==0 else 'ccw' if alias>0 else 'cw'
                                rows.append(dict(case=name,fps=fps,speed=speed,texture=texture,motion=motion,seed=seed,
                                    method=method,interval=i,start=i/fps,end=(i+1)/fps,
                                    target_velocity=float(v),physical_direction=physical,
                                    k4_sampled_direction=observable,predicted=r['direction'],
                                    reason=r.get('reason',''),resolved=r['direction']!='uncertain',
                                    physical_match=r['direction']==physical,
                                    regime='static' if motion=='stopped' else 'k4_subnyquist' if 4*speed<fps/2 else 'k4_supernyquist'))
                        catalog.append(dict(case=name,frames=n+1,fps=fps,k=4,texture=texture,motion=motion,
                            reverse_intervals=int(np.sum(velocity>0)),seed=seed))
    write_csv(out/'intervals.csv',rows)
    write_csv(out/'cases.csv',catalog)
    summary=[]
    keys=sorted({(r['method'],r['texture'],r['regime'],r['motion']) for r in rows})
    for method,texture,regime,motion in keys:
        subset=[r for r in rows if (r['method'],r['texture'],r['regime'],r['motion'])==(method,texture,regime,motion)]
        resolved=[r for r in subset if r['resolved']]
        positive=[r for r in subset if r['physical_direction']=='ccw']
        negative=[r for r in subset if r['physical_direction']=='cw']
        summary.append(dict(method=method,texture=texture,regime=regime,motion=motion,n=len(subset),
            resolved=len(resolved),coverage=len(resolved)/len(subset),
            accuracy_given_resolved=sum(r['physical_match'] for r in resolved)/len(resolved) if resolved else None,
            reverse_total=len(positive),reverse_detected=sum(r['predicted']=='ccw' for r in positive),
            reverse_wrong=sum(r['predicted']=='cw' for r in positive),reverse_rejected=sum(not r['resolved'] for r in positive),
            false_reverse=sum(r['predicted']=='ccw' for r in negative),clockwise_total=len(negative),
            false_motion_static=sum(r['resolved'] for r in subset) if motion=='stopped' else 0))
    write_csv(out/'summary.csv',summary)
    (out/'protocol.json').write_text(json.dumps(dict(cases=len(catalog),interval_rows=len(rows),
        seed_values=[11,29],noise_std=.008,geometry='known center and ring; no localization errors',
        label='velocity on interval [i/fps,(i+1)/fps); negative clockwise',
        jitter_seconds=[3,3.25],reverse_start_seconds=3,
        warning='Correlated synthetic intervals, no real-world CI. k4 alias label applies to harmonic, not entire asymmetric texture. Thresholds unchanged.'),indent=2))
    print('cases',len(catalog),'interval_rows',len(rows))
    for r in summary:
        if r['motion'] in ('jitter','stopped'):
            print(json.dumps(r))


if __name__=='__main__':main()
