"""Full-circle adjacent-frame correlation; no target harmonic or direction prior."""
import csv
import json
from pathlib import Path
import numpy as np
from p0_verify import ma, write_csv

ROOT=Path(__file__).resolve().parent


def match(a,b):
    a=a-a.mean();b=b-b.mean()
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    if norm<1e-8:return dict(direction='uncertain',reason='weak_texture')
    corr=np.fft.ifft(np.fft.fft(b)*np.conj(np.fft.fft(a))).real/norm
    best=int(np.argmax(corr));n=len(a)
    shift=best if best<=n//2 else best-n
    separation=np.minimum((np.arange(n)-best)%n,(best-np.arange(n))%n)
    second=float(corr[separation>5].max())
    reverse=np.fft.ifft(np.fft.fft(a)*np.conj(np.fft.fft(b))).real/norm
    back=int(np.argmax(reverse))
    error=min((best+back)%n,(-(best+back))%n)
    reason=('weak_match' if corr[best]<.5 else 'ambiguous_peak' if corr[best]-second<.05 else
            'roundtrip_failure' if error>2 else 'near_zero' if abs(shift)<=1 else '')
    return dict(direction='uncertain' if reason else 'ccw' if shift>0 else 'cw',reason=reason,
        shift_degrees=shift*360/n,correlation=float(corr[best]),peak_margin=float(corr[best]-second),roundtrip_bins=int(error))


def main():
    out=ROOT/'p0_audit_results/spatial_match_v1';out.mkdir(exist_ok=False)
    rng=np.random.default_rng(82);a=rng.normal(size=720)
    tests=[dict(name='ccw_shift',passed=match(a,np.roll(a,12))['direction']=='ccw'),
           dict(name='cw_shift',passed=match(a,np.roll(a,-12))['direction']=='cw'),
           dict(name='blank',passed=match(np.zeros(720),np.zeros(720))['direction']=='uncertain')]
    periodic=np.cos(np.arange(720)*2*np.pi*8/720)
    tests.append(dict(name='periodic_ambiguity',passed=match(periodic,np.roll(periodic,12))['reason']=='ambiguous_peak'))
    assert all(r['passed'] for r in tests)
    review=ROOT/'p0_audit_results/anonymous_review_v1'
    with (ROOT/'p0_audit_results/geometry_all_v1/geometry.csv').open(encoding='utf-8-sig') as f:
        geometry=[r for r in csv.DictReader(f) if r['sample_id'] in {'R12','R14'}]
    rows=[]
    for g in geometry:
        if g['method']=='legacy':continue
        frames=ma.load_frames_gray(str(review/'videos'/(g['sample_id']+'.mp4')))
        signals=ma.angular_signals(frames,float(g['cx']),float(g['cy']),float(g['rmin']),float(g['rmax']))
        for i,(a,b) in enumerate(zip(signals[:-1],signals[1:])):
            rows.append(dict(sample_id=g['sample_id'],method=g['method'],start=i/float(g['fps']),end=(i+1)/float(g['fps']),**match(a,b)))
    write_csv(out/'intervals.csv',rows)
    summary=[]
    for sid in ('R12','R14'):
        for method in ('scaled_circle','hough_or_fallback'):
            subset=[r for r in rows if r['sample_id']==sid and r['method']==method]
            summary.append(dict(sample_id=sid,method=method,counts={d:sum(r['direction']==d for r in subset) for d in ('cw','ccw','uncertain')},
                ccw_intervals=[(r['start'],r['end']) for r in subset if r['direction']=='ccw']))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'tests.json').write_text(json.dumps(tests,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
