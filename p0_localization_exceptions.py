"""Frozen-protocol diagnostics: contour rejection and ellipse/center sensitivity."""
import csv
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from p0_localization import locate
from p0_verify import ma, write_csv

ROOT = Path(__file__).resolve().parent


def inspect_contours(gray, threshold):
    h,w = gray.shape
    mask = (gray < threshold).astype('uint8')*255
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    rows=[]
    for index,c in enumerate(sorted(contours,key=cv2.contourArea,reverse=True)):
        area=cv2.contourArea(c)
        x,y,bw,bh=cv2.boundingRect(c)
        reasons=[]
        if area<.015*h*w: reasons.append('area_too_small')
        if x<=1 or y<=1 or x+bw>=w-1 or y+bh>=h-1: reasons.append('border_touch')
        row=dict(contour=index,area=area,points=len(c),threshold=threshold)
        if len(c)<20:
            reasons.append('too_few_points')
        else:
            (cx,cy),(a,b),angle=cv2.fitEllipse(c)
            radius=(a+b)/4
            distances=np.hypot(c[:,0,0]-cx,c[:,0,1]-cy)
            residual=float(np.quantile(abs(distances-radius),.9)/radius)
            aspect=min(a,b)/max(a,b)
            fill=area/(np.pi*radius**2)
            if aspect<.90:reasons.append('axis_ratio')
            if residual>.06:reasons.append('radial_residual')
            if not .87<fill<1.12:reasons.append('fill_ratio')
            row.update(cx=cx,cy=cy,a=a,b=b,angle=angle,radius=radius,
                aspect=aspect,radial_residual=residual,fill=fill)
        row.update(reasons=';'.join(reasons),valid=not reasons)
        rows.append(row)
    return mask,rows


def sample(frames,cx,cy,a,b,angle):
    # Local y is down, theta increases counterclockwise. The mapping has
    # positive determinant, so ellipse orientation cannot reverse the sign.
    theta=np.linspace(0,2*np.pi,ma.M_ANG,endpoint=False)[:,None]
    radius=np.linspace(.25,.85,ma.R_SAMPLES)[None,:]
    u=a/2*radius*np.cos(theta);v=-b/2*radius*np.sin(theta)
    phi=np.deg2rad(angle)
    mx=(cx+np.cos(phi)*u-np.sin(phi)*v).astype('float32')
    my=(cy+np.sin(phi)*u+np.cos(phi)*v).astype('float32')
    h,w=frames[0].shape
    outside=float(np.mean((mx<0)|(mx>w-1)|(my<0)|(my>h-1)))
    signals=np.array([cv2.remap(f,mx,my,cv2.INTER_LINEAR).mean(axis=1) for f in frames])
    return signals,outside


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'p0_audit_results/localization_exceptions_v2')
    out=parser.parse_args().out
    out.mkdir(exist_ok=False)
    # Independent analytic affine-ellipse rotations validate remap handedness.
    yy,xx=np.mgrid[:240,:320]
    phi=np.deg2rad(28)
    u=np.cos(phi)*(xx-160)+np.sin(phi)*(yy-120)
    v=-np.sin(phi)*(xx-160)+np.cos(phi)*(yy-120)
    angle=np.arctan2(-v/70,u/95)
    tests=[]
    for sign in (-1,1):
        frames=[((1+np.cos(4*(angle-sign*2*np.pi*.25*t/24)))/2).astype('float32') for t in range(120)]
        sig,_=sample(frames,160,120,190,140,28)
        f,_,_,_=ma.fft_estimate(ma.demodulate(sig,4),24)
        est=-float(f)/4
        tests.append(dict(expected=sign*.25,estimate=est,passed=abs(est-sign*.25)<.01))
    assert all(r['passed'] for r in tests),tests
    # Circle remap must reduce to the existing sampling convention.
    sig,_=sample(frames,160,120,140,140,0)
    reference=ma.angular_signals(frames,160,120,17.5,59.5)
    difference=float(np.max(abs(sig-reference)))
    assert difference<1e-6,difference
    validation=ROOT/'p0_audit_results/localization_validation_v1'
    with (validation/'mapping.csv').open(encoding='utf-8-sig') as f:
        mapping=list(csv.DictReader(f))
    all_contours=[];results=[];diagnostics=[]
    for group in ('V01','V04','V07'):
        item=next(r for r in mapping if r['sample_id']==group+'_1')
        cap=cv2.VideoCapture(item['source']);fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        frames=ma.load_frames_gray(item['source'])
        k=int(item['cell_key'].split('_')[0][1:])
        for idx in (0,len(frames)//2,len(frames)-1):
            gray=(frames[idx]*255).astype('uint8')
            for threshold in ((170,190,210,230) if group=='V01' else (190,)):
                mask,contours=inspect_contours(gray,threshold)
                cv2.imwrite(str(out/f'{group}_frame{idx}_threshold{threshold}.png'),mask)
                all_contours.extend(dict(group=group,frame=idx,**r) for r in contours)
                diagnostics.append(dict(group=group,frame=idx,threshold=threshold,
                    valid_candidates=sum(r['valid'] for r in contours),
                    largest_reasons=contours[0]['reasons'] if contours else 'no_contours'))
        if group=='V01':continue  # No forced motion estimate after rejection.
        _,contours=inspect_contours((frames[0]*255).astype('uint8'),190)
        candidates=[r for r in contours if r['valid']]
        assert len(candidates)==1
        c=candidates[0]
        assert locate((frames[0]*255).astype('uint8'))['accepted']
        shapes=[('circle',2*c['radius'],2*c['radius'],0),('ellipse',c['a'],c['b'],c['angle'])]
        for shape,a,b,rotation in shapes:
            for ox,oy in ((0,0),(-5,0),(-2,0),(2,0),(5,0),(0,-5),(0,5)):
                sig,outside=sample(frames,c['cx']+ox,c['cy']+oy,a,b,rotation)
                frequency,snr,freqs,mag=ma.fft_estimate(ma.demodulate(sig,k),fps)
                positive=float(mag[freqs>0].max());negative=float(mag[freqs<0].max())
                results.append(dict(group=group,shape=shape,offset_x=ox,offset_y=oy,
                    cx=c['cx']+ox,cy=c['cy']+oy,a=a,b=b,angle=rotation,
                    estimate=-float(frequency)/k,snr=float(snr),low_snr=bool(snr<ma.SNR_MIN),
                    peak_ratio=max(positive,negative)/max(min(positive,negative),1e-12),
                    out_of_frame_fraction=outside))
    write_csv(out/'contour_gates.csv',all_contours)
    write_csv(out/'rejection_summary.csv',diagnostics)
    write_csv(out/'motion_sensitivity.csv',results)
    (out/'tests.json').write_text(json.dumps(dict(ellipse_tests=tests,circle_max_difference=difference,
        locator_sha256=hashlib.sha256((ROOT/'p0_localization.py').read_bytes()).hexdigest()),indent=2))
    print(json.dumps(diagnostics,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
