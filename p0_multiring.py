"""Three-ring direction consensus, selected without GT direction."""
import csv,json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from p0_verify import ma,write_csv
from p0_localization_exceptions import inspect_contours

ROOT=Path(__file__).resolve().parent/'p0_audit_results'
RINGS={'inner':(.25,.45),'middle':(.45,.65),'outer':(.65,.85)}

def signal(frames,cx,cy,a,b,angle,lo,hi):
    theta=np.linspace(0,2*np.pi,ma.M_ANG,endpoint=False)[:,None]
    radius=np.linspace(lo,hi,ma.R_SAMPLES)[None,:]
    phi=np.deg2rad(angle);u=a/2*radius*np.cos(theta);v=-b/2*radius*np.sin(theta)
    mx=(cx+np.cos(phi)*u-np.sin(phi)*v).astype('float32')
    my=(cy+np.sin(phi)*u+np.cos(phi)*v).astype('float32')
    return np.array([cv2.remap(f,mx,my,cv2.INTER_LINEAR).mean(axis=1) for f in frames])

def decide(items):
    usable=[r for r in items if r['snr']>=5 and r['estimate']!=0]
    counts=Counter('ccw' if r['estimate']>0 else 'cw' for r in usable)
    winners=[d for d,n in counts.items() if n>=2]
    return winners[0] if len(winners)==1 else 'uncertain'

def main():
    out=ROOT/'multiring_v1';out.mkdir(exist_ok=False)
    # Analytic rendered test: corrupt one ring with opposite motion. Majority
    # should survive; two corrupted rings must follow the majority, not GT.
    yy,xx=np.mgrid[:256,:256];r=np.hypot(xx-128,yy-128);theta=np.arctan2(128-yy,xx-128)
    tests=[]
    for target in (-.25,.25):
        for corrupt in (0,1,2):
            frames=[]
            for t in np.arange(120)/24:
                image=np.ones((256,256),float)
                for i,(lo,hi) in enumerate(RINGS.values()):
                    velocity=-target if i<corrupt else target
                    mask=(r>=lo*100)&(r<=hi*100)
                    image[mask]=.5+.4*np.cos(4*(theta[mask]-2*np.pi*velocity*t))
                frames.append(image.astype('float32'))
            measurements=[]
            for name,(lo,hi) in RINGS.items():
                z=ma.demodulate(signal(frames,128,128,200,200,0,lo,hi),4)
                f,snr,_,_=ma.fft_estimate(z,24);measurements.append(dict(ring=name,estimate=-float(f)/4,snr=float(snr)))
            expected=target if corrupt<2 else -target
            observed=decide(measurements)
            tests.append(dict(target=target,corrupted_rings=corrupt,expected='ccw' if expected>0 else 'cw',observed=observed,passed=observed==('ccw' if expected>0 else 'cw')))
    assert all(t['passed'] for t in tests),tests
    rows=[];videos=[]
    for arm in ('A0','A3'):
        with (ROOT/f'formal_{arm}_eval_download_v1/local_manifest.csv').open(encoding='utf-8-sig') as f:
            items=[r for r in csv.DictReader(f) if r['category']=='wheel']
        for item in items:
            frames=ma.load_frames_gray(item['path']);_,contours=inspect_contours((frames[0]*255).astype('uint8'),190)
            valid=[r for r in contours if r['valid']]
            if len(valid)!=1:
                videos.append(dict(arm=arm,cell_key=item['cell_key'],status='localization_rejected',direction='uncertain'));continue
            c=valid[0];k=int(item['k']);methods=[]
            for shape,a,b,angle in [('circle',2*c['radius'],2*c['radius'],0),('ellipse',c['a'],c['b'],c['angle'])]:
                ring_rows=[]
                for ring,(lo,hi) in RINGS.items():
                    z=ma.demodulate(signal(frames,c['cx'],c['cy'],a,b,angle,lo,hi),k)
                    frequency,snr,_,_=ma.fft_estimate(z,float(item['fps_model']))
                    row=dict(arm=arm,cell_key=item['cell_key'],shape=shape,ring=ring,
                        estimate=-float(frequency)/k,snr=float(snr),signal_std=float(np.std(z)))
                    rows.append(row);ring_rows.append(row)
                methods.append(decide(ring_rows))
            direction=methods[0] if methods[0]==methods[1] else 'uncertain'
            usable=sum(r['snr']>=5 and r['estimate']!=0 for r in rows if r['arm']==arm and r['cell_key']==item['cell_key'])
            videos.append(dict(arm=arm,cell_key=item['cell_key'],status='consensus' if direction!='uncertain' else 'unresolved',
                circle_direction=methods[0],ellipse_direction=methods[1],direction=direction,usable_ring_methods=usable,
                direction_error=(direction!='ccw') if direction!='uncertain' else ''))
        print(arm,'complete',flush=True)
    write_csv(out/'ring_rows.csv',rows);write_csv(out/'video_results.csv',videos)
    summary=[]
    for arm in ('A0','A3'):
        subset=[r for r in videos if r['arm']==arm];ok=[r for r in subset if r['status']=='consensus']
        summary.append(dict(arm=arm,n=len(subset),consensus=len(ok),coverage=len(ok)/len(subset),
            conditional_candidate_error=sum(r['direction_error'] is True for r in ok)/len(ok) if ok else None,
            localization_rejected=sum(r['status']=='localization_rejected' for r in subset)))
    (out/'summary.json').write_text(json.dumps(dict(groups=summary,synthetic_tests=tests,
        caveat='Majority consensus is deterministic but not calibrated against independent generated-video truth.'),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
