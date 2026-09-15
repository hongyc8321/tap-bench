"""Combine frozen candidate outputs; do not relabel unresolved samples as correct."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from p0_verify import write_csv

ROOT=Path(__file__).resolve().parent/'p0_audit_results'
SOURCES={'tier':'tiered_results_v1/rows.csv','multiring':'multiring_v1/video_results.csv',
    'adaptive':'adaptive_harmonic_v1/video_results.csv','dual':'dual_timescale_v2/video_results.csv'}

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    out=ROOT/'evaluation_v2_candidate';out.mkdir(exist_ok=False)
    maps={name:{(r['arm'],r['cell_key']):r for r in read(ROOT/path)} for name,path in SOURCES.items()}
    keys=set(maps['tier']);assert len(keys)==282 and all(set(m)==keys for m in maps.values())
    rows=[]
    for arm,key in sorted(keys):
        t,m,a,d=(maps[x][(arm,key)] for x in ('tier','multiring','adaptive','dual'))
        target=m.get('direction','') if m.get('status')=='consensus' else ''
        visible=d.get('dominant_direction','') if d.get('status')=='dominant_resolved' else ''
        structure=a.get('status')=='selected' and a.get('selected_matches_target_k')=='True'
        strong=bool(target) and structure and bool(visible) and target==visible
        if strong:level='cross_method_target_consensus'
        elif target:level='target_harmonic_candidate_only'
        elif visible:level='visible_harmonic_candidate_only'
        else:level='unresolved'
        rows.append(dict(arm=arm,cell_key=key,f_ratio=t['f_ratio'],evidence_level=level,
            target_harmonic_direction=target,target_direction_error=(target!='ccw') if target else '',
            adaptive_status=a['status'],adaptive_k=a.get('selected_k',''),target_k=a['target_k'],
            adaptive_matches_target=structure,visible_dominant_direction=visible,
            local_behavior=d.get('local_behavior','not_run'),frozen_v1_tier=t['evidence_tier'],
            physical_direction_validity='not_established'))
    write_csv(out/'rows.csv',rows)
    groups=[]
    for arm in ('A0','A3'):
        rs=[r for r in rows if r['arm']==arm];target=[r for r in rs if r['target_harmonic_direction']];strong=[r for r in rs if r['evidence_level']=='cross_method_target_consensus']
        groups.append(dict(arm=arm,n=len(rs),target_candidate_n=len(target),target_candidate_coverage=len(target)/len(rs),
            target_candidate_error=sum(r['target_direction_error'] is True for r in target)/len(target) if target else None,
            strong_consensus_n=len(strong),strong_consensus_coverage=len(strong)/len(rs),
            strong_consensus_error=sum(r['target_direction_error'] is True for r in strong)/len(strong) if strong else None,
            evidence_levels=dict(__import__('collections').Counter(r['evidence_level'] for r in rs))))
    pairs=[];by={arm:{r['cell_key']:r for r in rows if r['arm']==arm} for arm in ('A0','A3')}
    for policy in ('target','strong'):
        for band in ('all','2+'):
            def use(r):return bool(r['target_harmonic_direction']) if policy=='target' else r['evidence_level']=='cross_method_target_consensus'
            keys2=[k for k in by['A0'] if (band=='all' or float(by['A0'][k]['f_ratio'])>=2) and use(by['A0'][k]) and use(by['A3'][k])]
            diff=np.array([int(by['A3'][k]['target_direction_error'])-int(by['A0'][k]['target_direction_error']) for k in keys2])
            ci=None
            if len(diff)>1 and len(set(diff))>1:
                rng=np.random.default_rng(1931);ci=np.quantile(rng.choice(diff,(10000,len(diff))).mean(axis=1),[.025,.975]).tolist()
            pairs.append(dict(policy=policy,band=band,both_available=len(keys2),delta=float(diff.mean()) if len(diff) else None,ci95=ci))
    report=dict(status='candidate_not_physical_benchmark',groups=groups,pairs=pairs,
        local_jitter_metric_released=False,source_hashes={k:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for k,v in SOURCES.items()},
        calibration_hash=hashlib.sha256((ROOT/'candidate_calibration_holdout_v2/summary.json').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
