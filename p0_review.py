"""Join human observations to historical estimates without overwriting labels."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT / 'p0_audit_results'
    with (out / 'blind_review_queue.csv').open(encoding='utf-8-sig', newline='') as f:
        reviews = list(csv.DictReader(f))
    with (ROOT / 'phase3_results_seedance/nyquist_rows.csv').open(encoding='utf-8', newline='') as f:
        estimates = list(csv.DictReader(f))
    lookup = {(r['model'], r['cell_key']): r for r in estimates}
    if len(lookup) != len(estimates):
        raise ValueError('Duplicate historical estimate keys')
    joined = []
    for r in reviews:
        a = lookup[(r['model'], r['cell_key'])]
        wheel = a['category'] == 'wheel'
        allowed = {'clockwise', 'counterclockwise', 'none', 'unclear'} if wheel else {
            'left', 'right', 'up', 'down', 'none', 'unclear'}
        if r['direction'] not in allowed:
            raise ValueError(f"Invalid direction: {r['cell_key']}")
        row = dict(r)
        row['normalization_note'] = ''
        if row['camera_stable'] == 'yea':
            row['camera_stable'] = 'yes'
            row['normalization_note'] = 'camera_stable: yea -> yes; original retained'
        for key in ('category', 'est', 'sign_match', 'motion', 'snr', 'low_conf'):
            row['historical_' + key] = a[key]
        row['estimate_source'] = 'phase3_results_seedance/nyquist_rows.csv; not rerun'
        row['assessment'] = (
            'structure_mismatch' if wheel and r['spokes_correct'] == 'no' else
            'direction_unresolved' if r['direction'] == 'unclear' else
            'outside_horizontal_estimator' if not wheel and r['direction'] in {'up', 'down'} else
            'requires_direction_calibration' if wheel else
            'horizontal_motion_observed')
        joined.append(row)
    with (out / 'review_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    print(f'Joined {len(joined)} human labels to historical estimates; original labels unchanged.')


if __name__ == '__main__':
    main()
