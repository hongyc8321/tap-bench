"""Recompute unchanged window gates for lost A3 cases, retaining reasons."""
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from investigate_coverage import read,ROOT
from p0_verify import ma,write_csv
from p0_localization_exceptions import inspect_contours,sample
from p0_windowed import windows

def main():
    out=ROOT/'coverage_A0_A3_v1'
    cases=read(out/'lost_cases.csv')
    manifests={a:{r['cell_key']:r for r in read(ROOT/f'formal_{a}_eval_download_v1/local_manifest.csv')} for a in ('A0','A3')}
    records=[];summaries=[]
    for case in cases:
        if case['A3_status']!='temporal_unresolved':continue
        for arm in ('A0','A3'):
            r=manifests[arm][case['cell_key']];frames=ma.load_frames_gray(r['path'])
            _,contours=inspect_contours((frames[0]*255).astype('uint8'),190)
            c=next(x for x in contours if x['valid']);reasons=Counter()
            for shape,a,b,angle in [('circle',2*c['radius'],2*c['radius'],0),('ellipse',c['a'],c['b'],c['angle'])]:
                sig,_=sample(frames,c['cx'],c['cy'],a,b,angle)
                z=ma.demodulate(sig,int(r['k']))
                for w in windows(z,float(r['fps_model']),int(r['k']),1.5):
                    reason=w['reason'] or 'accepted';reasons[reason]+=1
                    records.append(dict(id=case['id'],arm=arm,shape=shape,**w))
            summaries.append(dict(id=case['id'],arm=arm,reasons=dict(reasons)))
    write_csv(out/'window_reasons.csv',records)
    (out/'window_causes.json').write_text(json.dumps(summaries,indent=2))
    for arm in ('A0','A3'):
        print(arm,dict(Counter(r['reason'] or 'accepted' for r in records if r['arm']==arm)))

if __name__=='__main__':main()
