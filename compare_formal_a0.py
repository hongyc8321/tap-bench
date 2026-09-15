import csv,json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent/'p0_audit_results'

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    out=ROOT/'formal_A0_scored_v1'
    new={r['cell_key']:r for r in read(out/'rows.csv')}
    old=read(ROOT/'evaluation_v1/rows.csv')
    media=[]
    for r in read(ROOT/'formal_A0_eval_download_v1/local_manifest.csv'):
        cap=cv2.VideoCapture(r['path']);fps=cap.get(cv2.CAP_PROP_FPS);n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(480,720);n+=1
        cap.release();assert n==49 and fps==8
        media.append(dict(cell_key=r['cell_key'],frames=n,fps=fps))
    results=[]
    for baseline in ('orig','A0'):
        previous={r['cell_key']:r for r in old if r['arm']==baseline}
        assert set(new)==set(previous)
        for band in ('all','2+'):
            keys=[k for k in new if band=='all' or float(new[k]['f_ratio'])>=2]
            paired=[k for k in keys if new[k]['status']==previous[k]['status']=='accepted_apparent_direction']
            diff=np.array([int(new[k]['sign_match']=='False')-int(previous[k]['sign_match']=='False') for k in paired])
            ci=None
            if len(diff)>1 and len(set(diff))>1:
                rng=np.random.default_rng(1931)
                ci=np.quantile(rng.choice(diff,(10000,len(diff))).mean(axis=1),[.025,.975]).tolist()
            results.append(dict(baseline=baseline,band=band,input_pairs=len(keys),accepted_pairs=len(paired),
                delta=float(diff.mean()) if len(diff) else None,ci95=ci))
    (out/'comparison.json').write_text(json.dumps(results,indent=2))
    (out/'media_verified.json').write_text(json.dumps(dict(videos=len(media),all_49_frames_8fps_720x480=True)))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
