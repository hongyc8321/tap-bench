"""Evidence ledger, not a new benchmark score. Missing checks stay unknown."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent/'p0_audit_results'
METHODS = {'scaled_circle','hough_or_fallback'}


def load(relative):
    with (ROOT/relative).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if any(None in r for r in rows): raise ValueError('Malformed CSV')
    return rows


def classify(pair):
    if set(pair)!=METHODS:
        return 'incomplete_methods'
    if any(r['status']!='measured' for r in pair.values()):
        return 'measurement_failed'
    if any(not math.isfinite(float(r[k])) for r in pair.values() for k in ('estimate','snr','out_of_frame_fraction')):
        return 'nonfinite_measurement'
    if any(float(r['out_of_frame_fraction'])>0 for r in pair.values()):
        return 'geometry_out_of_frame'
    signs={int(float(r['estimate'])>0)-int(float(r['estimate'])<0) for r in pair.values()}
    if len(signs)>1:return 'direction_disagreement'
    if signs=={0}:return 'zero_direction'
    if any(float(r['snr'])<5 for r in pair.values()):return 'low_signal'
    return 'basic_checks_passed_only'


def main():
    # Gate unit checks include boundary SNR and missing/invalid method outputs.
    base=dict(status='measured',estimate='1',snr='5',out_of_frame_fraction='0')
    assert classify({m:dict(base) for m in METHODS})=='basic_checks_passed_only'
    assert classify({})=='incomplete_methods'
    for patch,expected in [({'status':'failed'},'measurement_failed'),
        ({'estimate':'nan'},'nonfinite_measurement'),({'out_of_frame_fraction':'.1'},'geometry_out_of_frame'),
        ({'estimate':'-1'},'direction_disagreement'),({'snr':'4.99'},'low_signal')]:
        pair={m:dict(base) for m in METHODS}
        pair['scaled_circle'].update(patch)
        assert classify(pair)==expected
    groups=defaultdict(dict)
    for r in load('geometry_groups_v1/rows.csv'):
        key=(r['arm'],r['cell_key'])
        if r['method'] in groups[key]:raise ValueError('Duplicate method record')
        groups[key][r['method']]=r
    diagnostic={(r['arm'],r['cell_key']):r for r in load('disagreement_diagnostic_v1/cases.csv')}
    localization=defaultdict(list)
    for r in load('localization_v1/problem_frames.csv'):
        localization[(r['arm'],r['cell_key'])].append(r)
    mapping={r['sample_id']:r for r in load('localization_validation_v1/mapping.csv')}
    for r in load('localization_validation_v1/predictions.csv'):
        m=mapping[r['sample_id']]
        localization[(m['arm'],m['cell_key'])].append(r)
    # The two extra sensitivity cases use different rings; label that scope.
    extra=defaultdict(list)
    for r in load('localization_exceptions_v2/motion_sensitivity.csv'):
        m=mapping[r['group']+'_1']
        extra[(m['arm'],m['cell_key'])].append(r)
    rows=[]
    for key,pair in sorted(groups.items()):
        d=diagnostic.get(key)
        loc=localization.get(key,[])
        ex=extra.get(key,[])
        both_measured=set(pair)==METHODS and all(r['status']=='measured' for r in pair.values())
        row=dict(arm=key[0],cell_key=key[1],primary_status=classify(pair),
            geometry_failure=any(float(r.get('out_of_frame_fraction') or 0)>0 for r in pair.values()),
            any_low_snr=any(float(r.get('snr') or 0)<5 for r in pair.values()) if both_measured else 'unknown',
            direction_disagreement=(len({float(r['estimate'])>0 for r in pair.values()})>1) if both_measured else 'unknown',
            perturbation_status=('sensitive' if d['five_px_perturbation_changes_sign']=='True' else 'no_flip_in_tested_offsets') if d else 'not_checked',
            perturbation_scope='four_cardinal_5px_offsets_on_scaled_circle' if d else 'not_checked',
            localization_checked_frames=len(loc),localization_rejected_frames=sum(r['accepted']=='False' for r in loc),
            localization_status=('some_frames_rejected' if any(r['accepted']=='False' for r in loc) else 'sampled_frames_accepted_only') if loc else 'not_checked',
            additional_ring_sensitivity='not_checked',snr_threshold_sensitivity='not_checked',
            time_analysis='not_checked_in_this_ledger',structure_validation='not_checked_in_this_ledger',
            scientific_validity='not_established')
        if ex:
            row['additional_ring_sensitivity']='no_sign_flip_circle_ellipse_and_offsets' if len({float(r['estimate'])>0 for r in ex})==1 else 'sign_flip'
            row['snr_threshold_sensitivity']='crosses_5' if len({float(r['snr'])>=5 for r in ex})>1 else 'no_crossing_in_tested_variants'
        for method,r in pair.items():
            for field in ('estimate','snr','status'):
                row[method+'_'+field]=r.get(field,'')
        rows.append(row)
    assert len(rows)==1182
    out=ROOT/'quality_ledger_v1'
    out.mkdir(exist_ok=False)
    with (out/'samples.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    summary=dict(total=len(rows),primary_status=dict(Counter(r['primary_status'] for r in rows)),
        localization_checked=sum(r['localization_status']!='not_checked' for r in rows),
        perturbation_checked=sum(r['perturbation_status']!='not_checked' for r in rows),
        extra_ring_checks=sum(r['additional_ring_sensitivity']!='not_checked' for r in rows),
        gate_tests_passed=8,
        per_arm={arm:dict(Counter(r['primary_status'] for r in rows if r['arm']==arm)) for arm in sorted({r['arm'] for r in rows})})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    sources=['geometry_groups_v1/rows.csv','disagreement_diagnostic_v1/cases.csv',
        'localization_v1/problem_frames.csv','localization_validation_v1/mapping.csv',
        'localization_validation_v1/predictions.csv','localization_exceptions_v2/motion_sensitivity.csv']
    (out/'provenance.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
