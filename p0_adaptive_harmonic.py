"""Select a visible harmonic from image evidence only; no GT k/direction input."""
import csv,json
from pathlib import Path
import cv2
import numpy as np
from p0_verify import ma,write_csv
from p0_localization_exceptions import inspect_contours
from p0_multiring import RINGS,signal

ROOT=Path(__file__).resolve().parent/'p0_audit_results'

def choose(measurements):
    candidates=[]
    for k in range(1,25):
        rows=[r for r in measurements if r['k']==k and r['snr']>=5 and r['estimate']!=0]
        if len(rows)<4:continue
        signs=[np.sign(r['estimate']) for r in rows]
        majority=1 if signs.count(1)>signs.count(-1) else -1
        agreement=sum(s==majority for s in signs)
        if agreement<4:continue
        score=float(np.median([np.log1p(r['snr']) for r in rows if np.sign(r['estimate'])==majority]))
        candidates.append(dict(k=k,direction='ccw' if majority>0 else 'cw',score=score,agreement=agreement,usable=len(rows)))
    if not candidates:return dict(status='no_harmonic',selected_k='',direction='')
    candidates.sort(key=lambda r:(-r['score'],r['k']))
    best=candidates[0];runner=candidates[1]['score'] if len(candidates)>1 else 0
    # Keep selection ambiguity explicit; 0.10 is a frozen development margin,
    # not a calibrated probability threshold.
    if best['score']-runner<.10:return dict(status='harmonic_ambiguous',selected_k=best['k'],direction='',score=best['score'],margin=best['score']-runner)
    return dict(status='selected',selected_k=best['k'],direction=best['direction'],score=best['score'],margin=best['score']-runner)

def main():
    out=ROOT/'adaptive_harmonic_v1';out.mkdir(exist_ok=False)
    # Known visible harmonic tests, including a different caller-side target.
    yy,xx=np.mgrid[:256,:256];rad=np.hypot(xx-128,yy-128);theta=np.arctan2(128-yy,xx-128)
    tests=[]
    for actual_k in (3,7,13,21):
        for velocity in (-.25,.25):
            frames=[]
            for t in np.arange(120)/24:
                image=np.ones((256,256),float);mask=(rad>=25)&(rad<=90)
                image[mask]=.5+.4*np.cos(actual_k*(theta[mask]-2*np.pi*velocity*t))
                frames.append(image.astype('float32'))
            measurements=[]
            for shape in ('circle','ellipse'):
                for ring,(lo,hi) in RINGS.items():
                    sig=signal(frames,128,128,200,200,0,lo,hi)
                    for k in range(1,25):
                        f,snr,_,_=ma.fft_estimate(ma.demodulate(sig,k),24)
                        measurements.append(dict(shape=shape,ring=ring,k=k,estimate=-float(f)/k,snr=float(snr)))
            selected=choose(measurements)
            tests.append(dict(actual_k=actual_k,velocity=velocity,selected_k=selected.get('selected_k'),direction=selected.get('direction'),
                status=selected['status'],passed=selected['status']=='selected' and selected['selected_k']==actual_k and selected['direction']==('ccw' if velocity>0 else 'cw')))
    assert all(t['passed'] for t in tests),tests
    all_measurements=[];videos=[]
    for arm in ('A0','A3'):
        with (ROOT/f'formal_{arm}_eval_download_v1/local_manifest.csv').open(encoding='utf-8-sig') as f:
            items=[r for r in csv.DictReader(f) if r['category']=='wheel']
        for item in items:
            frames=ma.load_frames_gray(item['path']);_,contours=inspect_contours((frames[0]*255).astype('uint8'),190)
            valid=[r for r in contours if r['valid']]
            if len(valid)!=1:
                videos.append(dict(arm=arm,cell_key=item['cell_key'],status='localization_rejected',target_k=item['k']));continue
            c=valid[0];measurements=[]
            for shape,a,b,angle in [('circle',2*c['radius'],2*c['radius'],0),('ellipse',c['a'],c['b'],c['angle'])]:
                for ring,(lo,hi) in RINGS.items():
                    sig=signal(frames,c['cx'],c['cy'],a,b,angle,lo,hi)
                    for k in range(1,25):
                        f,snr,_,_=ma.fft_estimate(ma.demodulate(sig,k),float(item['fps_model']))
                        row=dict(arm=arm,cell_key=item['cell_key'],shape=shape,ring=ring,k=k,estimate=-float(f)/k,snr=float(snr))
                        measurements.append(row);all_measurements.append(row)
            result=choose(measurements)
            videos.append(dict(arm=arm,cell_key=item['cell_key'],target_k=int(item['k']),**result,
                selected_matches_target_k=(result.get('selected_k')==int(item['k'])) if result['status']=='selected' else '',
                candidate_direction_error=(result.get('direction')!='ccw') if result['status']=='selected' else ''))
        print(arm,'complete',flush=True)
    write_csv(out/'harmonic_rows.csv',all_measurements);write_csv(out/'video_results.csv',videos)
    summary=[]
    for arm in ('A0','A3'):
        rs=[r for r in videos if r['arm']==arm];ok=[r for r in rs if r['status']=='selected']
        summary.append(dict(arm=arm,n=len(rs),selected=len(ok),coverage=len(ok)/len(rs),
            selected_matches_target_k=sum(r['selected_matches_target_k'] is True for r in ok)/len(ok) if ok else None,
            conditional_candidate_error=sum(r['candidate_direction_error'] is True for r in ok)/len(ok) if ok else None,
            statuses=dict(__import__('collections').Counter(r['status'] for r in rs))))
    (out/'summary.json').write_text(json.dumps(dict(groups=summary,synthetic_tests=tests,
        selection='max median log1p(SNR), >=4 usable ring-shape measurements, >=4 sign agreement, top margin >=0.10; no GT'),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
