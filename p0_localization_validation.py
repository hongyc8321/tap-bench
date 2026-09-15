"""Frozen-detector stress tests and cell-disjoint visual validation queue."""
import csv
import hashlib
import json
import random
from pathlib import Path
import cv2
import numpy as np
from p0_localization import locate, render
from p0_verify import write_csv
from p0_paired_remeasure import ARMS, SNAP

ROOT=Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    out=ROOT/'p0_audit_results/localization_validation_v1'
    out.mkdir(exist_ok=False)
    tests=[]
    for bg in (210,230,255):
        for fg in (40,140,200):
            for radius in (25,50,85):
                for offset in (False,True):
                    cx,cy=(100,100) if offset else (160,120)
                    base=render(cx,cy,radius)
                    image=np.where(base<190,fg,bg).astype('uint8')
                    r=locate(image)
                    tests.append(dict(background=bg,foreground=fg,radius_gt=radius,
                        offset=offset,kind='wheel',center_error=float(np.hypot(r['cx']-cx,r['cy']-cy)) if r['accepted'] else '',
                        radius_error=abs(r['radius']-radius) if r['accepted'] else '',**r))
    # A shape detector cannot distinguish solid disks and plain rings from wheels.
    for kind in ('solid_disk','plain_ring'):
        image=np.full((240,320),255,np.uint8)
        cv2.circle(image,(160,120),75,70,-1 if kind=='solid_disk' else 4)
        tests.append(dict(kind=kind,**locate(image)))
    write_csv(out/'stress_tests.csv',tests)
    excluded=set()
    for path in [ROOT/'p0_audit_results/disagreement_diagnostic_v1/cases.csv',
                 ROOT/'p0_audit_results/anonymous_review_v1/mapping_DO_NOT_OPEN_BEFORE_REVIEW.csv']:
        excluded.update(r['cell_key'] for r in read(path))
    # Also exclude cells reviewed earlier with Seedance, regardless of model.
    excluded.update(r['cell_key'] for r in read(ROOT/'p0_audit_results/blind_review_queue.csv'))
    rng=random.Random(41029)
    queue=[]; predictions=[]; mapping=[]
    for index,(arm,rel) in enumerate(ARMS.items(),1):
        pool=[r for r in read(SNAP/rel) if r['category']=='wheel' and r['cell_key'] not in excluded]
        item=rng.choice(sorted(pool,key=lambda r:r['cell_key']))
        excluded.add(item['cell_key'])
        path=(SNAP/('phase3' if rel.startswith('phase3/') else 'phase2_bundle/phase2'))/item['path']
        cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        strips=[]
        try:
            for number,frame_index in enumerate((0,count//2,count-1),1):
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,image=cap.read()
                if not ok:raise ValueError('Unable to decode validation frame')
                sample=f'V{index:02}_{number}'
                cv2.imwrite(str(out/f'{sample}_raw.png'),image)
                result=locate(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
                predictions.append(dict(sample_id=sample,**result))
                overlay=image.copy()
                if result['accepted']:
                    c=(round(result['cx']),round(result['cy']))
                    cv2.circle(overlay,c,round(result['radius']),(0,180,0),2)
                    cv2.drawMarker(overlay,c,(0,0,255),cv2.MARKER_CROSS,14,2)
                cv2.putText(overlay,sample+' '+('accepted' if result['accepted'] else 'rejected'),(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
                cv2.imwrite(str(out/f'{sample}_overlay.png'),overlay)
                strips.append(cv2.resize(overlay,(360,240)))
                queue.append(dict(sample_id=sample,raw_image=f'{sample}_raw.png',overlay_image=f'{sample}_overlay.png',
                    visible_wheel='',near_circular='',center_alignment='',outer_radius_fit='',notes=''))
                mapping.append(dict(sample_id=sample,arm=arm,cell_key=item['cell_key'],source=str(path),frame=frame_index,time_seconds=frame_index/fps))
        finally:cap.release()
        cv2.imwrite(str(out/f'V{index:02}_contact.png'),np.concatenate(strips,axis=1))
    write_csv(out/'labels.csv',queue)
    write_csv(out/'predictions.csv',predictions)
    write_csv(out/'mapping.csv',mapping)
    summary=dict(wheel_stress_cases=54,wheel_accepted=sum(r['accepted'] for r in tests if r['kind']=='wheel'),
        distractors=[dict(kind=r['kind'],accepted=r['accepted']) for r in tests if r['kind']!='wheel'],
        validation_frames=len(predictions),validation_accepted=sum(r['accepted'] for r in predictions),
        detector_sha256=hashlib.sha256((ROOT/'p0_localization.py').read_bytes()).hexdigest(),seed=41029,
        warning='Disjoint from manually reviewed/diagnostic cells, but drawn from previously automatically measured pool. Not an unseen dataset.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
