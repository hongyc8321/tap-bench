"""Diagnostic geometry ablation, not a replacement benchmark estimator."""
import csv
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from p0_verify import ma, write_csv

ROOT = Path(__file__).resolve().parent


def flow_direction(frames, cx, cy, rmin, rmax, fps):
    y, x = np.mgrid[:frames[0].shape[0], :frames[0].shape[1]]
    dx, dy = x-cx, y-cy
    r2 = dx*dx+dy*dy
    mask = (r2>rmin*rmin)&(r2<rmax*rmax)
    speeds = []
    for a,b in zip(frames[:-1],frames[1:]):
        flow = cv2.calcOpticalFlowFarneback((a*255).astype('uint8'),
            (b*255).astype('uint8'), None, .5, 3, 21, 3, 5, 1.2, 0)
        # Screen y increases downward: positive cross product is clockwise.
        omega = -(dx*flow[:,:,1]-dy*flow[:,:,0])/np.maximum(r2,1)*fps/(2*np.pi)
        speeds.append(float(np.median(omega[mask])))
    return speeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    out=ROOT/'p0_audit_results'/('geometry_all_v1' if args.all else 'geometry_v1')
    out.mkdir(exist_ok=False)
    review=ROOT/'p0_audit_results/anonymous_review_v1'
    snap=ROOT/'p0_audit_results/cloud_snapshot_v1'
    with (review/'mapping_DO_NOT_OPEN_BEFORE_REVIEW.csv').open(encoding='utf-8-sig') as f:
        mapping={r['sample_id']:r for r in csv.DictReader(f)}
    rows, flows=[] , []
    for sid in (sorted(mapping) if args.all else ('R01','R09','R10')):
        m=mapping[sid]
        gtpath=snap/('phase3/gt_cw/gt_curves.csv' if m['protocol']=='cw' else 'phase2_bundle/phase2/gt_curves.csv')
        with gtpath.open() as f:
            gt=next(r for r in csv.DictReader(f) if r['clip_id']=='wheel/'+m['cell_key'].removesuffix('_rep0')+'_point')
        path=review/'videos'/f'{sid}.mp4'
        cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS); cap.release()
        frames=ma.load_frames_gray(str(path)); h,w=frames[0].shape
        if len(frames) < 8 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid media: {sid}')
        cx,cy=map(float,gt['center'].split(';'))
        rmin,rmax=float(gt['band_rmin']),float(gt['band_rmax'])
        detected=ma.find_center(frames[0])
        geometries={'legacy':(cx,cy,rmin,rmax),
            'scaled_circle':(cx*w/1280,cy*h/720,rmin*h/720,rmax*h/720),
            'hough_or_fallback':(*detected,rmin*h/720,rmax*h/720)}
        canvas=cv2.cvtColor((frames[0]*255).astype('uint8'),cv2.COLOR_GRAY2BGR)
        for idx,(method,(x,y,lo,hi)) in enumerate(geometries.items()):
            theta=np.linspace(0,2*np.pi,ma.M_ANG,endpoint=False)[:,None]
            radius=np.linspace(lo,hi,ma.R_SAMPLES)[None,:]
            mx=x+radius*np.cos(theta); my=y-radius*np.sin(theta)
            invalid=float(np.mean((mx<0)|(mx>w-1)|(my<0)|(my>h-1)))
            z=ma.demodulate(ma.angular_signals(frames,x,y,lo,hi),int(gt['k']))
            frequency,snr,_,_=ma.fft_estimate(z,fps)
            speed=flow_direction(frames,x,y,lo,hi,fps)
            rows.append(dict(sample_id=sid,method=method,cx=x,cy=y,rmin=lo,rmax=hi,
                out_of_frame_fraction=invalid,fft_omega=-float(frequency)/int(gt['k']),
                snr=float(snr),flow_median=float(np.median(speed)),fps=fps))
            flows.extend(dict(sample_id=sid,method=method,start_seconds=i/fps,omega=v) for i,v in enumerate(speed))
            color=[(0,0,255),(0,200,0),(255,160,0)][idx]
            for radius in (lo,hi):cv2.circle(canvas,(round(x),round(y)),round(radius),color,2)
            cv2.drawMarker(canvas,(round(x),round(y)),color,cv2.MARKER_CROSS,15,2)
            cv2.putText(canvas,method,(10,25+idx*25),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        cv2.imwrite(str(out/f'{sid}_rings.png'),canvas)
        print(sid, 'completed', flush=True)
    # Optical-flow sign sanity checks using rendered low-speed rotation.
    y,x=np.mgrid[:256,:256]; angle=np.arctan2(128-y,x-128)
    tests=[]
    for sign in (-1,1):
        frames=[((1+np.cos(4*(angle-2*np.pi*sign*.25*t/24)))/2).astype('float32') for t in range(24)]
        estimate=float(np.median(flow_direction(frames,128,128,40,100,24)))
        tests.append(dict(expected_sign=sign,estimate=estimate,passed=estimate*sign>0))
    assert all(r['passed'] for r in tests),tests
    write_csv(out/'geometry.csv',rows);write_csv(out/'flow_intervals.csv',flows)
    (out/'selftest.json').write_text(json.dumps(tests,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
