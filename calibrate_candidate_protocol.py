"""Holdout signal calibration generated after candidate rules were frozen."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from p0_verify import ma,write_csv
from p0_adaptive_harmonic import choose
from p0_multiring import decide,RINGS
from p0_dual_timescale import consensus

ROOT=Path(__file__).resolve().parent/'p0_audit_results'

def main():
    out=ROOT/'candidate_calibration_holdout_v2';out.mkdir(exist_ok=False)
    rng=np.random.default_rng(20260909);fps=8;n=49;rows=[]
    actual_ks=[2,3,5,7,11,13]
    combinations=[(regime,motion,k,rep) for regime in ('subnyquist','supernyquist')
        for motion in ('steady','reverse','jitter') for k in actual_ks for rep in range(3)]
    for case,(regime,motion,actual_k,rep) in enumerate(combinations):
        target_k=actual_k if rep<2 else actual_ks[(actual_ks.index(actual_k)+1)%len(actual_ks)]
        speed=(.28*fps/actual_k) if regime=='subnyquist' else (.72*fps/actual_k)
        velocity=speed if case%2 else -speed
        v=np.full(n,velocity)
        if motion=='reverse':v[np.arange(n)/fps>=3]*=-1
        if motion=='jitter':v[(np.arange(n)/fps>=3)&(np.arange(n)/fps<3.25)]*=-1
        phase=2*np.pi*np.r_[0,np.cumsum(v)/fps]
        channels={};measurements=[]
        for shape in ('circle','ellipse'):
            for ring in RINGS:
                amplitude=rng.uniform(.18,.42);noise=rng.uniform(.005,.035)
                z=amplitude*np.exp(-1j*actual_k*phase[:-1])
                z+=noise*(rng.normal(size=n)+1j*rng.normal(size=n))
                channels[(shape,ring)]=z
                for k in range(1,25):
                    f,snr,_,_=ma.fft_estimate(z if k==actual_k else noise*(rng.normal(size=n)+1j*rng.normal(size=n)),fps)
                    measurements.append(dict(shape=shape,ring=ring,k=k,estimate=-float(f)/k,snr=float(snr)))
        adaptive=choose(measurements)
        method=[]
        for shape in ('circle','ellipse'):
            method.append(decide([r for r in measurements if r['shape']==shape and r['k']==target_k]))
        target_direction=method[0] if method[0]==method[1] else 'uncertain'
        full=[np.mean([channels[(shape,ring)] for ring in RINGS],axis=0) for shape in ('circle','ellipse')]
        long3=consensus(full,fps,actual_k,3.);long4=consensus(full,fps,actual_k,4.);short=consensus(full,fps,actual_k,1.)
        dominant=long3['dominant'] if long3['dominant']==long4['dominant'] else 'uncertain'
        physical='ccw' if velocity>0 else 'cw'
        rows.append(dict(case=case,regime=regime,motion=motion,actual_k=actual_k,target_k=target_k,target_matches_actual=target_k==actual_k,
            physical_initial=physical,adaptive_status=adaptive['status'],adaptive_k=adaptive.get('selected_k',''),adaptive_direction=adaptive.get('direction',''),
            target_multiring_direction=target_direction,dominant_direction=dominant,
            short_cw=short['cw'],short_ccw=short['ccw'],short_coverage=short['coverage']))
    write_csv(out/'cases.csv',rows)
    sub=[r for r in rows if r['regime']=='subnyquist' and r['target_matches_actual']]
    substeady=[r for r in sub if r['motion']=='steady']
    selected=[r for r in substeady if r['adaptive_status']=='selected'];multi=[r for r in substeady if r['target_multiring_direction']!='uncertain']
    steady=[r for r in substeady if r['dominant_direction']!='uncertain']
    reverse=[r for r in sub if r['motion']=='reverse']
    jitter=[r for r in sub if r['motion']=='jitter']
    summary=dict(cases=len(rows),subnyquist_matching_target=len(sub),
        steady_subnyquist_matching_target=len(substeady),
        adaptive=dict(coverage=len(selected)/len(substeady),k_accuracy=sum(r['adaptive_k']==r['actual_k'] for r in selected)/len(selected),
            direction_accuracy=sum(r['adaptive_direction']==r['physical_initial'] for r in selected)/len(selected)),
        target_multiring=dict(coverage=len(multi)/len(substeady),direction_accuracy=sum(r['target_multiring_direction']==r['physical_initial'] for r in multi)/len(multi)),
        dual_steady=dict(n=len(steady),direction_accuracy=sum(r['dominant_direction']==r['physical_initial'] for r in steady)/len(steady) if steady else None),
        reverse_short_both_directions=sum(r['short_cw']>0 and r['short_ccw']>0 for r in reverse)/len(reverse),
        jitter_short_both_directions=sum(r['short_cw']>0 and r['short_ccw']>0 for r in jitter)/len(jitter),
        supernyquist_physical_direction_scored=False,
        note='Holdout is analytic noisy complex-signal simulation with known geometry; not generated-video calibration.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
