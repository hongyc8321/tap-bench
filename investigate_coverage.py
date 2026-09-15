"""Paired coverage transitions, descriptive sensitivity and matched frame panels."""
import csv,json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent/'p0_audit_results'

def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    out=ROOT/'coverage_A0_A3_v1';out.mkdir(exist_ok=False)
    groups={a:{r['cell_key']:r for r in read(ROOT/f'formal_{a}_scored_v1/rows.csv')} for a in ('A0','A3')}
    a,b=groups['A0'],groups['A3'];assert set(a)==set(b)
    transitions=Counter((a[k]['status'],b[k]['status']) for k in a)
    lost=[k for k in sorted(a) if a[k]['status']=='accepted_apparent_direction' and b[k]['status']!='accepted_apparent_direction']
    gained=[k for k in a if b[k]['status']=='accepted_apparent_direction' and a[k]['status']!='accepted_apparent_direction']
    files={arm:{r['cell_key']:r['path'] for r in read(ROOT/f'formal_{arm}_eval_download_v1/local_manifest.csv')} for arm in groups}
    rows=[]
    for i,key in enumerate(lost,1):
        images=[];row=dict(id=f'C{i:02}',cell_key=key,A0_status=a[key]['status'],A3_status=b[key]['status'])
        for arm in groups:
            r=groups[arm][key]
            for field in ('snr_min','snr_max','time_coverage','cw_windows','ccw_windows','center_drift','radius_drift'):
                row[arm+'_'+field]=r.get(field,'')
            cap=cv2.VideoCapture(files[arm][key]);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));sharp=[]
            for frame_index in (0,count//2,count-1):
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,img=cap.read();assert ok
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                sharp.append(float(cv2.Laplacian(gray,cv2.CV_64F).var()))
                cv2.putText(img,f'C{i:02} {arm} frame {frame_index}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
                images.append(cv2.resize(img,(360,240)))
            cap.release();row[arm+'_laplacian_variance_3frame_mean']=float(np.mean(sharp))
        panel=np.concatenate([np.concatenate(images[:3],axis=1),np.concatenate(images[3:],axis=1)],axis=0)
        cv2.imwrite(str(out/f'C{i:02}.png'),panel);rows.append(row)
    with (out/'lost_cases.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    sensitivity={}
    for arm in groups:
        rs=list(groups[arm].values())
        sensitivity[arm]=dict(statuses=dict(Counter(r['status'] for r in rs)),
            temporal_reject_coverage=[float(r['time_coverage']) for r in rs if r['status']=='temporal_unresolved'],
            low_signal_min_snrs=[float(r['snr_min']) for r in rs if r['status']=='low_signal'])
    result=dict(lost=len(lost),gained=len(gained),lost_reason=dict(Counter(b[k]['status'] for k in lost)),
        gained_previous_reason=dict(Counter(a[k]['status'] for k in gained)),
        transitions=[dict(A0=x,A3=y,n=n) for (x,y),n in transitions.items()],
        sensitivity=sensitivity)
    (out/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
