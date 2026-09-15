"""Conservative silhouette localization for dark wheels on light backgrounds.

No fallback center, no motion signs used in selection. Not a general detector.
"""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from p0_verify import write_csv

ROOT = Path(__file__).resolve().parent


def locate(gray):
    h, w = gray.shape
    border = np.concatenate((gray[0], gray[-1], gray[:,0], gray[:,-1]))
    if np.median(border) < 220:
        return {'accepted':False, 'reason':'unsupported_background'}
    mask = (gray < 190).astype('uint8')*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < .015*h*w:
            continue
        x,y,bw,bh = cv2.boundingRect(contour)
        if x<=1 or y<=1 or x+bw>=w-1 or y+bh>=h-1:
            continue
        if len(contour)<20:
            continue
        (cx,cy),(a,b),_ = cv2.fitEllipse(contour)
        radius = (a+b)/4
        aspect = min(a,b)/max(a,b)
        points = contour[:,0,:]
        distances = np.hypot(points[:,0]-cx,points[:,1]-cy)
        residual = float(np.quantile(abs(distances-radius),.9)/radius)
        fill = area/(np.pi*radius**2)
        if aspect < .90 or residual > .06 or not .87 < fill < 1.12:
            continue
        candidates.append(dict(cx=float(cx),cy=float(cy),radius=float(radius),
            aspect=float(aspect),radial_residual=residual,fill=float(fill)))
    if len(candidates)!=1:
        return dict(accepted=False,reason='no_valid_circle' if not candidates else 'multiple_circles',candidates=len(candidates))
    return dict(accepted=True,reason='circle_geometry_only',**candidates[0])


def render(cx=160, cy=120, radius=75, kind='wheel'):
    image = np.full((240,320),255,np.uint8)
    if kind=='blank':return image
    if kind=='rectangle':
        cv2.rectangle(image,(70,40),(250,200),70,-1)
        return image
    if kind=='ellipse':
        cv2.ellipse(image,(cx,cy),(radius,round(radius*.7)),0,0,360,70,5)
        return image
    cv2.circle(image,(cx,cy),radius,70,4)
    for theta in np.arange(8)*np.pi/4:
        cv2.line(image,(cx,cy),(round(cx+(radius-2)*np.cos(theta)),round(cy+(radius-2)*np.sin(theta))),70,6)
    cv2.circle(image,(cx,cy),12,70,-1)
    if kind=='occluded':cv2.rectangle(image,(cx,0),(319,239),255,-1)
    if kind=='blur':image=cv2.GaussianBlur(image,(7,7),1.4)
    if kind=='two':cv2.circle(image,(270,190),25,70,4)
    return image


def main():
    out=ROOT/'p0_audit_results/localization_v1'
    out.mkdir(exist_ok=False)
    tests=[]
    cases=[('center',160,120,75,'wheel',True),('offset',95,95,60,'wheel',True),
        ('small',170,120,35,'wheel',True),('large',160,120,100,'wheel',True),
        ('blur',160,120,75,'blur',True),('blank',160,120,75,'blank',False),
        ('rectangle',160,120,75,'rectangle',False),('ellipse',160,120,75,'ellipse',False),
        ('occluded',160,120,75,'occluded',False),('cropped',20,120,75,'wheel',False),
        ('two',95,95,60,'two',False)]
    for name,cx,cy,radius,kind,expected in cases:
        image=render(cx,cy,radius,kind)
        result=locate(image)
        error=np.hypot(result.get('cx',cx)-cx,result.get('cy',cy)-cy)
        radius_error=abs(result.get('radius',radius)-radius)
        passed=result['accepted']==expected and (not expected or error<3 and radius_error<4)
        tests.append(dict(case=name,expected=expected,passed=bool(passed),center_error=float(error),radius_error=float(radius_error),**result))
        cv2.imwrite(str(out/f'test_{name}.png'),image)
    write_csv(out/'synthetic_tests.csv',tests)
    if not all(r['passed'] for r in tests):
        print(json.dumps(tests,indent=2));raise RuntimeError('Synthetic checks failed; do not promote detector')
    with (ROOT/'p0_audit_results/disagreement_diagnostic_v1/cases.csv').open(encoding='utf-8-sig') as f:
        cases=list(csv.DictReader(f))
    rows=[]
    for case in cases:
        cap=cv2.VideoCapture(case['source'])
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        panels=[]
        try:
            for index in (0,count//2,count-1):
                cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok,image=cap.read()
                if not ok:raise ValueError('Frame decoding failed')
                result=locate(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
                rows.append(dict(id=case['id'],arm=case['arm'],cell_key=case['cell_key'],frame=index,**result))
                if result['accepted']:
                    center=(round(result['cx']),round(result['cy']))
                    cv2.circle(image,center,round(result['radius']),(0,180,0),2)
                    for fraction in (.25,.85):cv2.circle(image,center,round(result['radius']*fraction),(255,120,0),1)
                cv2.putText(image,case['id']+' '+result['reason'],(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
                panels.append(cv2.resize(image,(360,240)))
        finally:cap.release()
        cv2.imwrite(str(out/(case['id']+'.png')),np.concatenate(panels,axis=1))
    write_csv(out/'problem_frames.csv',rows)
    summary=dict(synthetic_passed=sum(r['passed'] for r in tests),synthetic_n=len(tests),
        problem_frames=len(rows),accepted_frames=sum(r['accepted'] for r in rows),
        all_three_accepted=[c['id'] for c in cases if all(r['accepted'] for r in rows if r['id']==c['id'])])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
