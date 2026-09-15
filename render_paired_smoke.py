"""CPU paired render preflight. Produces lossless frames, not training MP4s."""
import csv
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent


def render_cell(row, plan, out):
        cv2.setNumThreads(1)
        manifest=[]
        w,h,fps,n=map(int,(row['width'],row['height'],row['fps'],row['frames']))
        rng=np.random.default_rng(int(row['appearance_seed']))
        outer=float(rng.uniform(.36,.42)*h);hub=float(rng.uniform(.05,.065)*h)
        inner=outer-.015*h;duty=float(rng.uniform(.38,.62))
        fg=float(rng.uniform(.08,.20));bg=float(rng.uniform(.80,.92))
        y,x=np.mgrid[:h*2,:w*2];dx=(x+.5)/2-w/2;dy=(y+.5)/2-h/2
        radius=np.hypot(dx,dy);phi=np.arctan2(-dy,dx)
        band=(radius>hub)&(radius<inner)
        static=np.where(radius<=hub,.45,np.where((radius>=inner)&(radius<=outer),.3,bg))
        k=int(row['k']);omega=float(row['omega_rev_s'])
        appearance=dict(center=[w/2,h/2],outer=outer,inner=inner,hub=hub,duty=duty,fg=fg,bg=bg)
        appearance_hash=hashlib.sha256(json.dumps(appearance,sort_keys=True).encode()).hexdigest()
        reference=[];hashes={};different=0
        for variant in ('point','shutter180'):
            folder=out/row['base_id']/variant;folder.mkdir(parents=True)
            exposure=.5/fps
            # Midpoint quadrature approximates a centered box exposure.
            offsets=np.array([0.]) if variant=='point' else ((np.arange(16)+.5)/16-.5)*exposure
            assert variant!='point' or offsets[0]==0
            digest=hashlib.sha256()
            for j in range(n):
                acc=np.zeros(radius.shape,dtype=np.float64)
                for offset in offsets:
                    u=((phi-2*np.pi*omega*(j/fps+offset))*k/(2*np.pi))%1
                    acc+=np.where(band,fg+(bg-fg)*(u<duty),static)
                image=np.round(cv2.resize(acc/len(offsets),(w,h),interpolation=cv2.INTER_AREA)*255).astype('uint8')
                if not cv2.imwrite(str(folder/f'frame_{j:04}.png'),image):raise IOError('PNG export failed')
                digest.update(image.tobytes())
                if variant=='point':reference.append(image)
                else:
                    different+=int(not np.array_equal(image,reference[j]))
                    # Away from band boundaries, temporal filtering must not
                    # change the hub, rim or background.
                    yy,xx=np.mgrid[:h,:w];rr=np.hypot(xx+.5-w/2,yy+.5-h/2)
                    fixed=(rr<hub-2)|(rr>inner+2)
                    assert np.array_equal(image[fixed],reference[j][fixed])
            hashes[variant]=digest.hexdigest()
            meta=dict(base=row,appearance=appearance,appearance_sha256=appearance_hash,
                variant=variant,exposure_seconds=0 if variant=='point' else exposure,
                offsets_seconds=offsets.tolist(),frame_pixels_sha256=hashes[variant],
                renderer='paired_smoke_v1',temporal_filter='finite_16_tap_box_not_ideal_antialias')
            (folder/'meta.json').write_text(json.dumps(meta,indent=2))
        assert different>0
        for arm,variant in [('A0','point'),('A1','point'),('A2','shutter180'),('A3','shutter180')]:
            with (plan/f'{arm}.csv').open() as f:planned=next(r for r in csv.DictReader(f) if r['base_id']==row['base_id'])
            assert planned['temporal_variant']==variant
            manifest.append(dict(arm=arm,base_id=row['base_id'],frames_path=f"{row['base_id']}/{variant}",
                frame_count=n,fps=fps,sampling_weight=planned['sampling_weight'],appearance_sha256=appearance_hash,pixels_sha256=hashes[variant]))
        check=dict(base_id=row['base_id'],frames_per_variant=n,frames_differing=different,
            static_regions_identical=True,paired_appearance_identical=True)
        return manifest,check


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,default=ROOT/'p0_audit_results/paired_training_plan_v1')
    parser.add_argument('--out',type=Path,default=ROOT/'p0_audit_results/paired_render_smoke_v1')
    parser.add_argument('--full',action='store_true',help='Render every base cell; CPU-intensive')
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.workers < 1:parser.error('--workers must be positive')
    plan=args.plan
    with (plan/'base_cells.csv').open() as f:base=list(csv.DictReader(f))
    if len({r['base_id'] for r in base}) != len(base):raise ValueError('Duplicate base IDs')
    selected=base if args.full else [next(r for r in base if float(r['omega_rev_s'])<4),next(r for r in base if float(r['omega_rev_s'])>=4)]
    out=args.out;out.mkdir(exist_ok=False)
    manifest=[];checks=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows,check in pool.map(render_cell,selected,repeat(plan),repeat(out),chunksize=1):
            manifest.extend(rows);checks.append(check)
            print(f"{len(checks)}/{len(selected)} {check['base_id']} paired render verified",flush=True)
    with (out/'manifest.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    (out/'checks.json').write_text(json.dumps(dict(cases=checks,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_plan_sha256=hashlib.sha256((plan/'base_cells.csv').read_bytes()).hexdigest(),
        workers=args.workers,
        status='full_frames_no_training_video_export' if args.full else 'smoke_only_no_training_video_export'),indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
