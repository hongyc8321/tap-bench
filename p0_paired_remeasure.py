"""Unified remeasurement for original and Phase 3 CogVideoX arms."""
import csv, json
from pathlib import Path
import cv2
from p0_verify import ma, write_csv

ROOT=Path(__file__).resolve().parent
SNAP=ROOT/'p0_audit_results/cloud_snapshot_v1'
OUT=ROOT/'p0_audit_results/paired_v2'
ARMS={'orig':'phase2_bundle/phase2/gen_out/cogvideox2b_manifest.csv',
      'A0':'phase3/gen_A0/cogvideox2b_t2v_manifest.csv',
      'A1':'phase3/gen_A1/cogvideox2b_t2v_manifest.csv',
      'A2':'phase3/gen_A2/cogvideox2b_t2v_manifest.csv',
      'A3':'phase3/gen_A3/cogvideox2b_t2v_manifest.csv',
      'A4':'phase3/gen_A4/cogvideox2b_t2v_manifest.csv',
      'A4blur':'phase3/gen_A4blur/cogvideox2b_t2v_manifest.csv',
      'A5':'phase3/gen_A5/cogvideox2b_t2v_manifest.csv'}

def main():
    OUT.mkdir(exist_ok=False)
    with (SNAP/'phase2_bundle/phase2/gt_curves.csv').open() as f:
        gt={'_'.join(r['clip_id'].split('/')[1].split('_')[:-1]):r for r in csv.DictReader(f) if r['variant']=='point'}
    rows=[]
    for arm, rel in ARMS.items():
        with (SNAP/rel).open() as f: items=[r for r in csv.DictReader(f) if r.get('category') == 'wheel']
        for item in items:
            raw=Path(item['path']); base=SNAP/('phase3' if rel.startswith('phase3/') else 'phase2_bundle/phase2')
            path=next(p for p in [base/raw, (SNAP/rel).parent/raw, SNAP/raw] if p.is_file())
            cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS); cap.release()
            g=gt[item['stem']]
            meta={'fps':fps,'k':int(g['k']),'center':[float(x) for x in g['center'].split(';')],
                  'band_rmin':float(g['band_rmin']),'band_rmax':float(g['band_rmax'])}
            m=ma.measure_wheel(str(path), meta=meta, fps=fps)
            est=float(m['omega_est']); target=float(item['omega_gt'])
            rows.append(dict(arm=arm,cell_key=item['cell_key'],omega_gt=target,estimate=est,
                snr=float(m['snr']),low_conf=float(m['snr'])<ma.SNR_MIN,
                sign_match=est*target>0,motion=abs(est)>=.15,
                speed_ratio=abs(est)/max(abs(target),1e-6),f_ratio=float(item['f_tex_hz'])/float(item['fps_model'])))
        print(arm,len(items),flush=True)
    write_csv(OUT/'rows.csv',rows)
    by={}
    for arm in ARMS:
        x=[r for r in rows if r['arm']==arm]; c=[r for r in x if not r['low_conf']]
        by[arm]=dict(n=len(x),n_conf=len(c),low_conf_rate=1-len(c)/len(x),
            rr_all=sum(not r['sign_match'] for r in x)/len(x),
            rr_conf=sum(not r['sign_match'] for r in c)/len(c) if c else None,
            stationary_all=sum(not r['motion'] for r in x)/len(x),
            speed_ratio_all=sum(r['speed_ratio'] for r in x)/len(x))
    pairs={}
    cells=sorted({r['cell_key'] for r in rows})
    for arm in list(ARMS)[1:]:
        diffs=[]
        for cell in cells:
            o=next((r for r in rows if r['arm']=='orig' and r['cell_key']==cell),None)
            a=next((r for r in rows if r['arm']==arm and r['cell_key']==cell),None)
            if o and a: diffs.append(dict(rr_delta=int(not a['sign_match'])-int(not o['sign_match']),
                speed_delta=a['speed_ratio']-o['speed_ratio'], conf_both=not o['low_conf'] and not a['low_conf']))
        pairs[arm]=dict(n=len(diffs),rr_delta_mean=sum(x['rr_delta'] for x in diffs)/len(diffs),
            speed_delta_mean=sum(x['speed_delta'] for x in diffs)/len(diffs),
            paired_conf_n=sum(x['conf_both'] for x in diffs))
    (OUT/'summary.json').write_text(json.dumps({'arms':by,'paired_to_orig':pairs},indent=2))
    print(json.dumps({'arms':by,'paired_to_orig':pairs},indent=2))
if __name__=='__main__': main()
