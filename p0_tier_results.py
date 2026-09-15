"""Evidence tiers over frozen v1; no threshold or decision-rule changes."""
import csv,json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'p0_audit_results'


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def tier(row):
    status=row['status']
    if status=='accepted_apparent_direction':return 'tier1_global_and_local_agree'
    if status=='temporal_unresolved':return 'tier2_global_only_temporal_unresolved'
    if status in {'temporal_bidirectional_candidate','temporal_mixed_evidence','temporal_global_disagreement'}:return 'tier3_temporal_complex_or_conflicting'
    if status=='low_signal':return 'tier4_low_signal'
    if status in {'localization_rejected','geometry_temporal_unstable','geometry_direction_unstable','geometry_out_of_frame','flat_signal'}:return 'tier5_geometry_or_structure_unresolved'
    return 'tier6_execution_or_media_failure'


def main():
    out=ROOT/'tiered_results_v1';out.mkdir(exist_ok=False)
    rows=[]
    for arm in ('A0','A3'):
        for r in read(ROOT/f'formal_{arm}_scored_v1/rows.csv'):
            t=tier(r);direction=''
            if t=='tier1_global_and_local_agree':direction='ccw' if int(float(r['direction']))>0 else 'cw'
            elif t=='tier2_global_only_temporal_unresolved':direction='ccw' if float(r['estimate'])>0 else 'cw' if float(r['estimate'])<0 else ''
            rows.append(dict(arm=arm,cell_key=r['cell_key'],f_ratio=r['f_ratio'],frozen_status=r['status'],evidence_tier=t,
                candidate_direction=direction,direction_error=(direction!='ccw') if direction else '',
                direction_scope='apparent_harmonic_not_physical',time_coverage=r.get('time_coverage',''),
                snr_min=r.get('snr_min',''),snr_max=r.get('snr_max','')))
    with (out/'rows.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for arm in ('A0','A3'):
        subset=[r for r in rows if r['arm']==arm];counts=Counter(r['evidence_tier'] for r in subset)
        cumulative=[]
        for allowed in [('tier1_global_and_local_agree',),('tier1_global_and_local_agree','tier2_global_only_temporal_unresolved')]:
            selected=[r for r in subset if r['evidence_tier'] in allowed]
            cumulative.append(dict(tiers=list(allowed),n=len(selected),coverage=len(selected)/len(subset),
                conditional_candidate_error=sum(r['direction_error'] is True for r in selected)/len(selected) if selected else None))
        summary.append(dict(arm=arm,n=len(subset),tiers=dict(counts),cumulative=cumulative))
    pairs=[]
    by={arm:{r['cell_key']:r for r in rows if r['arm']==arm} for arm in ('A0','A3')}
    for policy,allowed in [('tier1',['tier1_global_and_local_agree']),
        ('tier1_plus_tier2',['tier1_global_and_local_agree','tier2_global_only_temporal_unresolved'])]:
        keys=[k for k in by['A0'] if by['A0'][k]['evidence_tier'] in allowed and by['A3'][k]['evidence_tier'] in allowed]
        diff=np.array([int(by['A3'][k]['direction_error'])-int(by['A0'][k]['direction_error']) for k in keys])
        ci=None
        if len(diff)>1 and len(set(diff))>1:
            rng=np.random.default_rng(1931);ci=np.quantile(rng.choice(diff,(10000,len(diff))).mean(axis=1),[.025,.975]).tolist()
        pairs.append(dict(policy=policy,both_available=len(keys),delta=float(diff.mean()) if len(diff) else None,ci95=ci))
    (out/'summary.json').write_text(json.dumps(dict(groups=summary,pairs=pairs,
        caveat='Tier2 is a candidate global sign with unresolved local time evidence; not calibrated medium confidence.'),indent=2))
    print(json.dumps(dict(groups=summary,pairs=pairs),indent=2))


if __name__=='__main__':main()
