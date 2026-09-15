"""Frozen conservative apparent-direction evaluation. CPU, local snapshot only."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from p0_paired_remeasure import ARMS, SNAP
from p0_localization import locate
from p0_localization_exceptions import inspect_contours, sample
from p0_verify import ma, write_csv
from p0_windowed import windows

ROOT=Path(__file__).resolve().parent
CONFIG=dict(version='apparent_direction_v1',snr_min=5,signal_std_min=1e-5,
    ring_fraction=[.25,.85],center_offsets=[[0,0],[-5,0],[5,0],[0,-5],[0,5]],
    center_drift_radius_fraction=.05,radius_drift_fraction=.05,
    geometry_shapes=['circle','ellipse'],localization_frames='all',
    temporal_window_seconds=1.5,window_step_seconds=.25,
    temporal_min_coverage=.5,temporal_min_each_direction_windows=2,
    seed=1931,bootstrap_repetitions=10000)


def adjudicate(estimates,snrs):
    signs=np.sign(estimates)
    if np.any(signs==0) or len(set(signs))!=1:return 'geometry_direction_unstable'
    if min(snrs)<5:return 'low_signal'
    return 'candidate'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'p0_audit_results/evaluation_v1')
    parser.add_argument('--manifest',type=Path,help='Local-path manifest for one additional group')
    parser.add_argument('--group-name',default='formal_A0',help='Label for single-manifest evaluation')
    args=parser.parse_args();out=args.out
    out.mkdir(exist_ok=False)
    sources=['p0_evaluate_v1.py','p0_localization.py','p0_localization_exceptions.py','phase2/measure_angle.py','p0_windowed.py']
    provenance={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    (out/'protocol.json').write_text(json.dumps(dict(config=CONFIG,code_hashes=provenance,
        numpy=np.__version__,opencv=cv2.__version__,snapshot_index_sha256=hashlib.sha256((SNAP/'snapshot_index.json').read_bytes()).hexdigest()),indent=2))
    assert adjudicate([1,-1],[10,10])=='geometry_direction_unstable'
    assert adjudicate([1,1],[4.9,10])=='low_signal'
    assert adjudicate([1,1],[5,5])=='candidate'
    arms=dict(ARMS)
    for a in ('ORIG','A0','A5'):arms['cw_'+a]=f'phase3/gen_cw_{a}/cogvideox2b_t2v_manifest.csv'
    if args.manifest:
        arms={args.group_name:str(args.manifest.resolve())}
        (out/'input_manifest_sha256.txt').write_text(hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    all_rows=[];details=[];temporal=[]
    identities={}
    for arm,rel in arms.items():
        with (SNAP/rel).open() as f:items=[r for r in csv.DictReader(f) if r['category']=='wheel']
        if len({r['cell_key'] for r in items})!=len(items):raise ValueError('Duplicate manifest identities')
        identities[arm]={r['cell_key'] for r in items}
        for index,item in enumerate(items):
            row=dict(arm=arm,cell_key=item['cell_key'],status='execution_failed',protocol='cw' if arm.startswith('cw_') else 'standard',
                gt=float(item['omega_gt']),f_ratio=float(item['f_tex_hz'])/float(item['fps_model']))
            try:
                raw=Path(item['path']);base=SNAP/('phase3' if rel.startswith('phase3') else 'phase2_bundle/phase2')
                path=raw if args.manifest else SNAP/raw.relative_to('/home/ubuntu') if raw.is_absolute() else base/raw
                row['video_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
                frames=ma.load_frames_gray(str(path))
                if len(frames)!=count or count<8 or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid decode')
                row.update(fps_video=fps,decoded_frames=count,fps_manifest=float(item['fps_model']))
                if abs(fps-float(item['fps_model']))>1e-6:
                    row['status']='fps_mismatch';all_rows.append(row);continue
                found=[locate((frame*255).astype('uint8')) for frame in frames]
                row['localization_rejected_frames']=sum(not r['accepted'] for r in found)
                if row['localization_rejected_frames']:
                    row['status']='localization_rejected';all_rows.append(row);continue
                median=np.median([[r['cx'],r['cy'],r['radius']] for r in found],axis=0)
                drift=max(np.hypot(r['cx']-median[0],r['cy']-median[1]) for r in found)/median[2]
                radial=max(abs(r['radius']-median[2]) for r in found)/median[2]
                row.update(center_drift=drift,radius_drift=radial)
                if drift>.05 or radial>.05:
                    row['status']='geometry_temporal_unstable';all_rows.append(row);continue
                _,contours=inspect_contours((frames[0]*255).astype('uint8'),190)
                c=next(r for r in contours if r['valid']);k=int(item['k'])
                estimates=[];snrs=[];amplitudes=[];oob=[];central={}
                for shape,a,b,angle in [('circle',2*c['radius'],2*c['radius'],0),('ellipse',c['a'],c['b'],c['angle'])]:
                    for dx,dy in CONFIG['center_offsets']:
                        sig,outside=sample(frames,c['cx']+dx,c['cy']+dy,a,b,angle)
                        z=ma.demodulate(sig,k);frequency,snr,_,_=ma.fft_estimate(z,fps)
                        estimate=-float(frequency)/k;amplitude=float(np.std(z))
                        if not np.isfinite(estimate) or not np.isfinite(snr):raise ValueError('Nonfinite signal')
                        estimates.append(estimate);snrs.append(float(snr));amplitudes.append(amplitude);oob.append(outside)
                        details.append(dict(arm=arm,cell_key=item['cell_key'],shape=shape,dx=dx,dy=dy,estimate=estimate,snr=float(snr),signal_std=amplitude,oob=outside))
                        if dx==dy==0:central[shape]=z
                row.update(estimate=float(np.median(estimates)),snr_min=min(snrs),snr_max=max(snrs),
                    snr_threshold_crossing=min(snrs)<5<=max(snrs),max_oob=max(oob),signal_std_min=min(amplitudes))
                row['status']='geometry_out_of_frame' if max(oob)>0 else 'flat_signal' if min(amplitudes)<1e-5 else adjudicate(estimates,snrs)
                if row['status']!='candidate':all_rows.append(row);continue
                w={s:windows(z,fps,k,1.5) for s,z in central.items()}
                directions=[]
                for l,r in zip(w['circle'],w['ellipse']):
                    direction=l['direction'] if l['direction']==r['direction'] else 'uncertain'
                    directions.append(direction)
                    temporal.append(dict(arm=arm,cell_key=item['cell_key'],start=l['start_seconds'],end=l['end_seconds'],direction=direction))
                counts=Counter(directions);coverage=1-counts['uncertain']/len(directions)
                row.update(time_coverage=coverage,cw_windows=counts['cw'],ccw_windows=counts['ccw'])
                if min(counts['cw'],counts['ccw'])>=2:row['status']='temporal_bidirectional_candidate'
                elif coverage<.5:row['status']='temporal_unresolved'
                elif counts['cw'] and counts['ccw']:row['status']='temporal_mixed_evidence'
                else:
                    direction=1 if counts['ccw'] else -1
                    if direction!=np.sign(row['estimate']):row['status']='temporal_global_disagreement'
                    else:row.update(status='accepted_apparent_direction',direction=direction,sign_match=direction*row['gt']>0)
                all_rows.append(row)
            except Exception as e:
                row.update(status='execution_failed',error=f'{type(e).__name__}: {e}');all_rows.append(row)
            if (index+1)%30==0:print(arm,index+1,flush=True)
        write_csv(out/f'{arm}_rows.csv',[r for r in all_rows if r['arm']==arm])
        print(arm,dict(Counter(r['status'] for r in all_rows if r['arm']==arm)),flush=True)
    write_csv(out/'rows.csv',all_rows)
    if details:write_csv(out/'geometry_variants.csv',details)
    if temporal:write_csv(out/'temporal_windows.csv',temporal)
    summaries=[];pairs=[]
    for arm in arms:
        rows=[r for r in all_rows if r['arm']==arm];ok=[r for r in rows if r['status']=='accepted_apparent_direction']
        errors=sum(not r['sign_match'] for r in ok);n=len(rows)
        summaries.append(dict(arm=arm,input_n=n,accepted_n=len(ok),coverage=len(ok)/n,
            error_n=errors,conditional_error_rate=errors/len(ok) if ok else None,
            unresolved_n=n-len(ok),completion_lower_bound=errors/n,completion_upper_bound=(errors+n-len(ok))/n,
            statuses=dict(Counter(r['status'] for r in rows))))
        baseline='cw_A0' if arm.startswith('cw_') else 'A0'
        if arm==baseline:continue
        a={r['cell_key']:r for r in rows};b={r['cell_key']:r for r in all_rows if r['arm']==baseline}
        for band in ('all','2+'):
            common=sorted(set(a)&set(b));eligible=[key for key in common if band=='all' or a[key]['f_ratio']>=2]
            usable=[key for key in eligible if a[key]['status']==b[key]['status']=='accepted_apparent_direction']
            diff=np.array([int(not a[key]['sign_match'])-int(not b[key]['sign_match']) for key in usable])
            bounds=None
            if len(diff)>1 and len(set(diff))>1:
                rng=np.random.default_rng(1931);boot=rng.choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
                bounds=np.quantile(boot,[.025,.975]).tolist()
            pairs.append(dict(arm=arm,baseline=baseline,band=band,matched_input_n=len(eligible),both_accepted_n=len(usable),
                delta=float(diff.mean()) if len(diff) else None,bootstrap95=bounds))
    expected=sum(len(keys) for keys in identities.values()) if args.manifest else 1182
    assert len(all_rows)==expected and len({(r['arm'],r['cell_key']) for r in all_rows})==expected
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))
    (out/'paired.json').write_text(json.dumps(pairs,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
