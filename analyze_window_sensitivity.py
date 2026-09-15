"""Measure window-size sensitivity on new A0/A3 outputs; no score replacement."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from p0_verify import ma
from p0_localization_exceptions import inspect_contours,sample
from p0_windowed import windows

ROOT=Path(__file__).resolve().parent/'p0_audit_results'


def main():
    out=ROOT/'window_sensitivity_A0_A3_v2';out.mkdir(exist_ok=False)
    result_rows=[];summary=[]
    for arm in ('A0','A3'):
        with (ROOT/f'formal_{arm}_eval_download_v1/local_manifest.csv').open(encoding='utf-8-sig') as f:
            items=[r for r in csv.DictReader(f) if r['category']=='wheel']
        for item in items:
            frames=ma.load_frames_gray(item['path'])
            _,contours=inspect_contours((frames[0]*255).astype('uint8'),190)
            valid=[r for r in contours if r['valid']]
            if len(valid)!=1:
                summary.append(dict(arm=arm,cell_key=item['cell_key'],stage='localization_rejected'))
                continue
            c=valid[0]; fps=float(item['fps_model']); k=int(item['k'])
            variants=[]
            for shape,a,b,angle in [('circle',2*c['radius'],2*c['radius'],0),('ellipse',c['a'],c['b'],c['angle'])]:
                sig,_=sample(frames,c['cx'],c['cy'],a,b,angle)
                z=ma.demodulate(sig,k); variants.append((shape,z))
            for seconds in (1.,1.5,2.,3.,4.):
                per_method=[]
                for shape,z in variants:
                    ws=windows(z,fps,k,seconds)
                    per_method.append((shape,ws))
                n=min(len(x[1]) for x in per_method)
                directions=[]
                for i in range(n):
                    a=per_method[0][1][i];b=per_method[1][1][i]
                    direction=a['direction'] if a['direction']==b['direction'] else 'uncertain'
                    directions.append(direction)
                    result_rows.append(dict(arm=arm,cell_key=item['cell_key'],window_seconds=seconds,
                        start_seconds=a['start_seconds'],end_seconds=a['end_seconds'],direction=direction,
                        circle_reason=a['reason'],ellipse_reason=b['reason'],
                        circle_snr=a['snr'],ellipse_snr=b['snr'],
                        circle_estimate=a['omega_est'],ellipse_estimate=b['omega_est']))
                counts=Counter(directions);coverage=1-counts['uncertain']/len(directions)
                summary.append(dict(arm=arm,cell_key=item['cell_key'],window_seconds=seconds,
                    windows=len(directions),cw=counts['cw'],ccw=counts['ccw'],uncertain=counts['uncertain'],
                    coverage=coverage,bidirectional=int(counts['cw']>=2 and counts['ccw']>=2),
                    mixed=int(bool(counts['cw'] and counts['ccw']))))
            del frames
        print(arm,'complete',flush=True)
    with (out/'window_rows.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in result_rows for k in r}));w.writeheader();w.writerows(result_rows)
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in summary for k in r}));w.writeheader();w.writerows(summary)
    compact=[]
    for arm in ('A0','A3'):
        for seconds in (1.,1.5,2.,3.,4.):
            rows=[r for r in summary if r.get('arm')==arm and abs(float(r.get('window_seconds',-1))-seconds)<1e-9]
            compact.append(dict(arm=arm,window_seconds=seconds,videos=len(rows),mean_coverage=float(np.mean([float(r['coverage']) for r in rows])) if rows else None,
                videos_any_bidirectional=sum(r['bidirectional']=='1' for r in rows),videos_mixed=sum(r['mixed']=='1' for r in rows)))
    (out/'compact.json').write_text(json.dumps(compact,indent=2))
    print(json.dumps(compact,indent=2))


if __name__=='__main__':main()
