"""Paired comparison of new A3/A0 with media and protocol checks."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent/'p0_audit_results'


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    out=ROOT/'formal_A3_scored_v1'
    a={r['cell_key']:r for r in read(out/'rows.csv')}
    b={r['cell_key']:r for r in read(ROOT/'formal_A0_scored_v1/rows.csv')}
    assert set(a)==set(b) and len(a)==141
    p=json.loads((out/'protocol.json').read_text())
    q=json.loads((ROOT/'formal_A0_scored_v1/protocol.json').read_text())
    assert p['config']==q['config']
    for name,digest in p['code_hashes'].items():
        if name!='p0_evaluate_v1.py':assert q['code_hashes'][name]==digest
    manifests={arm:read(ROOT/f'formal_{arm}_eval_download_v1/local_manifest.csv') for arm in ('A0','A3')}
    old={r['cell_key']:r for r in manifests['A0']}
    assert len(manifests['A3'])==154
    for r in manifests['A3']:
        for key in ('stem','category','k','omega_gt','fps_model','f_tex_hz'):
            assert r[key]==old[r['cell_key']][key]
        cap=cv2.VideoCapture(r['path']);fps=cap.get(cv2.CAP_PROP_FPS);n=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                assert frame.shape[:2]==(480,720);n+=1
        finally:cap.release()
        assert n==49 and fps==8
    pairs=[]
    for band in ('all','2+'):
        keys=sorted(k for k in a if band=='all' or float(a[k]['f_ratio'])>=2)
        common=[k for k in keys if a[k]['status']==b[k]['status']=='accepted_apparent_direction']
        diff=np.array([int(a[k]['sign_match']=='False')-int(b[k]['sign_match']=='False') for k in common])
        ci=None
        if len(diff)>1 and len(set(diff))>1:
            rng=np.random.default_rng(1931)
            ci=np.quantile(rng.choice(diff,(10000,len(diff))).mean(axis=1),[.025,.975]).tolist()
        pairs.append(dict(band=band,input_pairs=len(keys),both_accepted=len(common),
            a0_errors=sum(b[k]['sign_match']=='False' for k in common),
            a3_errors=sum(a[k]['sign_match']=='False' for k in common),
            delta=float(diff.mean()) if len(diff) else None,bootstrap95=ci,
            a0_only_accepted=sum(b[k]['status']=='accepted_apparent_direction' and a[k]['status']!='accepted_apparent_direction' for k in keys),
            a3_only_accepted=sum(a[k]['status']=='accepted_apparent_direction' and b[k]['status']!='accepted_apparent_direction' for k in keys)))
    result=dict(comparison='new_A3_minus_new_A0',pairs=pairs,media_verified=154,
        identical_measurement_config=True,identical_helper_hashes=True,
        entrypoint_change='Added group-name argument only; same decision logic',training_effect_proven=False)
    (out/'new_A0_comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
