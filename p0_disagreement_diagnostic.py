"""Inspect geometry sensitivity without choosing the expected motion sign."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from p0_verify import ma, write_csv
from p0_paired_remeasure import ARMS, SNAP

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT/'p0_audit_results/geometry_groups_v1'
    with (source/'rows.csv').open(encoding='utf-8-sig') as f:
        all_rows = list(csv.DictReader(f))
    with (source/'direction_disagreements.csv').open(encoding='utf-8-sig') as f:
        disagreements = {(r['arm'],r['cell_key']) for r in csv.DictReader(f)}
    outside = {(r['arm'],r['cell_key']) for r in all_rows if float(r.get('out_of_frame_fraction') or 0)>0}
    selected = sorted(disagreements | outside)
    out = ROOT/'p0_audit_results/disagreement_diagnostic_v1'
    out.mkdir(exist_ok=False)
    summaries, measurements, panels = [], [], []
    for number,(arm,key) in enumerate(selected,1):
        tag = f'D{number:02}'
        rel = ARMS[arm] if not arm.startswith('cw_') else f'phase3/gen_cw_{arm[3:]}/cogvideox2b_t2v_manifest.csv'
        with (SNAP/rel).open() as f:
            item = next(r for r in csv.DictReader(f) if r['cell_key']==key)
        base = SNAP/('phase3' if rel.startswith('phase3/') else 'phase2_bundle/phase2')
        path = base/item['path']
        frames = ma.load_frames_gray(str(path))
        pair = {r['method']:r for r in all_rows if r['arm']==arm and r['cell_key']==key}
        scaled, detected = pair['scaled_circle'], pair['hough_or_fallback']
        fps,k = float(scaled['fps']),int(item['k'])
        cx,cy = float(scaled['cx']),float(scaled['cy'])
        dx,dy = float(detected['cx']),float(detected['cy'])
        lo,hi = float(scaled['rmin']),float(scaled['rmax'])
        h,w=frames[0].shape
        # Check whether the existing Hough helper had candidates on frames
        # spread across the clip; a fallback must not masquerade as detection.
        centers=[]
        canvas=[]
        for index in (0,len(frames)//2,len(frames)-1):
            gray=frames[index]
            blur=cv2.GaussianBlur(gray,(9,9),2.)
            circles=cv2.HoughCircles((blur*255).astype('uint8'),cv2.HOUGH_GRADIENT,
                dp=1.5,minDist=200,param1=120,param2=60,minRadius=80,maxRadius=0)
            centers.append(None if circles is None else [float(v) for v in circles[0][0]])
            image=cv2.cvtColor((gray*255).astype('uint8'),cv2.COLOR_GRAY2BGR)
            for x,y,color in [(cx,cy,(0,180,0)),(dx,dy,(255,130,0))]:
                for radius in (lo,hi):cv2.circle(image,(round(x),round(y)),round(radius),color,2)
                cv2.drawMarker(image,(round(x),round(y)),color,cv2.MARKER_CROSS,16,2)
            cv2.putText(image,f'{tag} t={index/fps:.2f}s',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
            canvas.append(cv2.resize(image,(360,240)))
        strip=np.concatenate(canvas,axis=1)
        cv2.imwrite(str(out/f'{tag}.png'),strip)
        panels.append(strip)
        perturb_signs=[]
        for name,x,y in [('scaled',cx,cy),('detected',dx,dy),
                         ('left5',cx-5,cy),('right5',cx+5,cy),('up5',cx,cy-5),('down5',cx,cy+5)]:
            z=ma.demodulate(ma.angular_signals(frames,x,y,lo,hi),k)
            frequency,snr,freqs,mag=ma.fft_estimate(z,fps)
            pos=float(mag[freqs>0].max()); neg=float(mag[freqs<0].max())
            dominance=max(pos,neg)/max(min(pos,neg),1e-12)
            est=-float(frequency)/k
            measurements.append(dict(id=tag,arm=arm,cell_key=key,geometry=name,cx=x,cy=y,
                estimate=est,snr=float(snr),direction_peak_ratio=dominance,
                texture_frequency=float(frequency),nyquist_distance=fps/2-abs(float(frequency))))
            if name not in ('scaled','detected'):perturb_signs.append(int(np.sign(est)))
        summaries.append(dict(id=tag,arm=arm,cell_key=key,source=str(path),
            direction_disagreement=(arm,key) in disagreements,out_of_frame=(arm,key) in outside,
            both_low_snr=scaled['low_conf']=='True' and detected['low_conf']=='True',
            either_low_snr=scaled['low_conf']=='True' or detected['low_conf']=='True',
            center_distance_px=float(np.hypot(dx-cx,dy-cy)),
            first_frame_hough_found=centers[0] is not None,
            hough_samples=json.dumps(centers),five_px_perturbation_changes_sign=len(set(perturb_signs))>1))
        print(tag,arm,key,flush=True)
    write_csv(out/'cases.csv',summaries)
    write_csv(out/'sensitivity.csv',measurements)
    for start in range(0,len(panels),7):
        cv2.imwrite(str(out/f'contact_{start//7+1}.png'),np.concatenate(panels[start:start+7],axis=0))
    stats=dict(selected=len(selected),disagreements=len(disagreements),out_of_frame=len(outside),
        both_low=sum(r['both_low_snr'] for r in summaries),either_low=sum(r['either_low_snr'] for r in summaries),
        perturbation_sensitive=sum(r['five_px_perturbation_changes_sign'] for r in summaries),
        first_frame_fallback=sum(not r['first_frame_hough_found'] for r in summaries))
    (out/'summary.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
