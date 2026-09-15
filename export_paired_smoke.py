"""Export four smoke clips and multiplicity-weighted lists for the existing loader."""
import csv
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from p0_verify import ma, write_csv

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'p0_audit_results/paired_render_smoke_v1')
    parser.add_argument('--out',type=Path,default=ROOT/'p0_audit_results/paired_video_smoke_v1')
    parser.add_argument('--lossless',action='store_true',help='FFV1 AVI with exact decoded grayscale pixel verification')
    args=parser.parse_args()
    source=args.source
    out=args.out;out.mkdir(exist_ok=False)
    (out/'videos').mkdir()
    with (source/'manifest.csv').open() as f:manifest=list(csv.DictReader(f))
    exports={};audit=[]
    for relative in sorted({r['frames_path'] for r in manifest}):
        folder=source/relative;meta=json.loads((folder/'meta.json').read_text())
        b=meta['base'];fps=int(b['fps']);w=int(b['width']);h=int(b['height'])
        images=[cv2.imread(str(p),cv2.IMREAD_GRAYSCALE) for p in sorted(folder.glob('frame_*.png'))]
        assert len(images)==int(b['frames']) and all(i is not None and i.shape==(h,w) for i in images)
        digest=hashlib.sha256()
        for image in images:digest.update(image.tobytes())
        assert digest.hexdigest()==meta['frame_pixels_sha256']
        path=out/'videos'/(relative.replace('/','_')+('.avi' if args.lossless else '.mp4'))
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*('FFV1' if args.lossless else 'mp4v')),fps,(w,h))
        if not writer.isOpened():raise RuntimeError('MP4 encoder unavailable')
        try:
            for image in images:writer.write(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
        finally:writer.release()
        cap=cv2.VideoCapture(str(path));decoded=[]
        try:
            actual_fps=cap.get(cv2.CAP_PROP_FPS)
            while True:
                ok,frame=cap.read()
                if not ok:break
                decoded.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        finally:cap.release()
        assert len(decoded)==len(images) and actual_fps==fps
        assert all(i.shape==(h,w) for i in decoded)
        appearance=meta['appearance'];cx,cy=appearance['center']
        estimates=[];snrs=[]
        for frames in (images,decoded):
            sig=ma.angular_signals([i.astype('float32')/255 for i in frames],cx,cy,appearance['hub']+8,appearance['inner']-8)
            freq,snr,_,_=ma.fft_estimate(ma.demodulate(sig,int(b['k'])),fps)
            estimates.append(-float(freq)/int(b['k']));snrs.append(float(snr))
        error=float(np.mean([np.abs(a.astype(float)-d).mean() for a,d in zip(images,decoded)]))
        if args.lossless and error!=0:raise RuntimeError('Lossless decoded pixels differ')
        same_sign=np.sign(estimates[0])==np.sign(estimates[1])
        audit.append(dict(frames_path=relative,video=str(path.relative_to(out)),frames=len(decoded),fps=fps,
            width=w,height=h,pixel_mae_255=error,source_estimate=estimates[0],encoded_estimate=estimates[1],
            source_snr=snrs[0],encoded_snr=snrs[1],apparent_direction_preserved=bool(same_sign),
            video_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        exports[relative]=(path,b)
        if len(audit)%50==0:print(f'{len(audit)} videos decoded and audited',flush=True)
    write_csv(out/'encoding_audit.csv',audit)
    if any(not row['apparent_direction_preserved'] for row in audit):
        raise RuntimeError('Encoding direction audit failed; full audit saved, training lists not published')
    schedules={}
    for arm in ('A0','A1','A2','A3'):
        folder=out/arm;folder.mkdir()
        paths=[];prompts=[];ids=[]
        for row in manifest:
            if row['arm']!=arm:continue
            path,b=exports[row['frames_path']]
            prompt=f"a grayscale wheel with {b['k']} spokes rotating counterclockwise at exactly {b['omega_rev_s']} revolutions per second, static camera, white background"
            for _ in range(int(row['sampling_weight'])):
                paths.append('../'+str(path.relative_to(out)));prompts.append(prompt);ids.append(row['base_id'])
        (folder/'videos.txt').write_text('\n'.join(paths)+'\n')
        (folder/'prompts.txt').write_text('\n'.join(prompts)+'\n')
        assert len(paths)==len(prompts) and all((folder/p).is_file() for p in paths)
        schedules[arm]=ids
    assert schedules['A0']==schedules['A2'] and schedules['A1']==schedules['A3']
    # Existing trainer uses shuffle=True: repeats express multiplicity over
    # an epoch, not independent weighted-with-replacement sampling.
    tests=[]
    for left,right in [('A0','A2'),('A1','A3')]:
        a=list(schedules[left]);b=list(schedules[right])
        random.Random(42).shuffle(a);random.Random(42).shuffle(b)
        assert a==b
        tests.append(dict(pair=[left,right],epoch_rows=len(a),multiplicity=dict(Counter(a)),same_seed_order=True))
    write_csv(out/'encoding_audit.csv',audit)
    (out/'preflight.json').write_text(json.dumps(dict(sampling_tests=tests,codec='FFV1' if args.lossless else 'mp4v',
        actual_torch_loader_executed=False,training_started=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(json.dumps(audit,indent=2));print(json.dumps(tests,indent=2))


if __name__=='__main__':main()
