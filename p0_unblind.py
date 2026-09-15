"""Validate completed labels and join estimates; retain raw human observations."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT/'p0_audit_results/anonymous_review_v1'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if any(None in row for row in rows):
        raise ValueError(f'Malformed CSV: {path}')
    return rows


def unique(rows, key):
    result = {key(r): r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate identity')
    return result


def main():
    labels = read_csv(REVIEW/'labels.csv')
    mapping = unique(read_csv(REVIEW/'mapping_DO_NOT_OPEN_BEFORE_REVIEW.csv'), lambda r:r['sample_id'])
    labelled = unique(labels, lambda r:r['sample_id'])
    if set(labelled) != set(mapping) or len(labels) != 18:
        raise ValueError('Expected 18 matching review identities')
    standard = unique(read_csv(ROOT/'p0_audit_results/paired_v2/rows.csv'), lambda r:(r['arm'],r['cell_key']))
    cw = unique(read_csv(ROOT/'p0_audit_results/snapshot_audit_v2/cw_remeasured.csv'), lambda r:(str(Path(r['manifest']).parent),r['cell_key']))
    groups = {'phase2_bundle/phase2/gen_out':'orig', 'phase3/gen_A0':'A0', 'phase3/gen_A3':'A3'}
    joined = []
    for raw in labels:
        m = mapping[raw['sample_id']]
        video = REVIEW/raw['video_path']
        if hashlib.sha256(video.read_bytes()).hexdigest() != m['sha256']:
            raise ValueError(f"Video identity changed: {raw['sample_id']}")
        a = cw[(m['group'],m['cell_key'])] if m['protocol']=='cw' else standard[(groups[m['group']],m['cell_key'])]
        direction = raw['dominant_direction'].strip().replace('counter_clockwise','counterclockwise')
        if direction not in {'clockwise','counterclockwise','mixed','none','unclear'}:
            raise ValueError(f'Unknown direction {direction}')
        warnings = []
        if direction != raw['dominant_direction']:
            warnings.append('direction_spelling_normalized')
        if raw['structure_stable'] not in {'yes','no','unclear'}:
            warnings.append('structure_free_text_unresolved')
        if not raw['uncertainty']:
            warnings.append('uncertainty_not_provided')
        if raw['sample_id']=='R12':
            warnings.append('jitter_yes_but_notes_uncertain')
        if raw['sample_id'] in {'R02','R04'}:
            warnings.append('spoke_count_numeric_but_notes_uncertain')
        estimate = float(a.get('est', a.get('estimate')))
        auto = 'counterclockwise' if estimate > 0 else 'clockwise' if estimate < 0 else 'none'
        comparable = direction in {'clockwise','counterclockwise'} and raw['sustained_change']=='no'
        # Conservative subset for direction checking, not an estimate of accuracy.
        eligible = comparable and raw['structure_stable']=='yes' and raw['reverse_jitter']=='no' and raw['camera_motion']=='no' and raw['wheel_flip']=='no'
        target_k = int(m['cell_key'].split('_')[0][1:])
        observed = raw['observed_spokes']
        numeric = observed.isdigit() and raw['sample_id'] not in {'R02','R04'}
        joined.append(dict(raw, protocol=m['protocol'], group=m['group'], cell_key=m['cell_key'],
            normalized_direction=direction, auto_direction=auto, estimate=estimate,
            auto_snr=a['snr'], auto_low_conf=a['low_conf'], warnings=';'.join(warnings),
            sustained_direction_comparable=comparable,
            direction_agrees=(direction==auto) if comparable else '',
            conservative_direction_subset=eligible, target_k=target_k,
            observed_spokes_match=(int(observed)==target_k) if numeric else '',
            human_target_agrees=(direction==('clockwise' if m['protocol']=='cw' else 'counterclockwise')) if comparable else ''))
    out = REVIEW/'unblinded_v1'
    out.mkdir(exist_ok=False)
    with (out/'comparison.csv').open('x', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(joined[0]))
        w.writeheader()
        w.writerows(joined)
    comparable = [r for r in joined if r['sustained_direction_comparable']]
    conservative = [r for r in joined if r['conservative_direction_subset']]
    summary = dict(n=len(joined), videos_hash_verified=len(joined),
        raw_labels_sha256=hashlib.sha256((REVIEW/'labels.csv').read_bytes()).hexdigest(),
        comparable_n=len(comparable), comparable_disagreements=[r['sample_id'] for r in comparable if not r['direction_agrees']],
        conservative_n=len(conservative), conservative_disagreements=[r['sample_id'] for r in conservative if not r['direction_agrees']],
        sustained_change=[r['sample_id'] for r in joined if r['sustained_change']=='yes'],
        structure_unstable=[r['sample_id'] for r in joined if r['structure_stable']=='no'],
        warning_samples=[dict(sample_id=r['sample_id'],warnings=r['warnings']) for r in joined if r['warnings']])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    for r in joined:
        print(r['sample_id'],r['protocol'],r['group'],r['cell_key'],
              'human='+r['normalized_direction'],'auto='+r['auto_direction'],
              'low_conf='+r['auto_low_conf'],'struct='+r['structure_stable'])


if __name__ == '__main__':
    main()
