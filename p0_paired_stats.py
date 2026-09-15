import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
IN = ROOT / 'p0_audit_results/paired_v2/rows.csv'
OUT = ROOT / 'p0_audit_results/paired_v2'

def ci(values, seed=17):
    x = np.asarray(values, float)
    if len(x) < 2 or np.all(x == x[0]):
        return [None, None]
    rng = np.random.default_rng(seed)
    samples = [rng.choice(x, len(x), replace=True).mean() for _ in range(10000)]
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]

with IN.open(encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
arms = sorted({r['arm'] for r in rows})
original = {r['cell_key']: r for r in rows if r['arm'] == 'orig'}
result = {}
for arm in arms:
    if arm == 'orig':
        continue
    current = {r['cell_key']: r for r in rows if r['arm'] == arm}
    result[arm] = {}
    for label, lo, hi in [('0-0.5', 0, .5), ('0.5-1', .5, 1),
                          ('1-2', 1, 2), ('2+', 2, 999)]:
        all_diffs, paired_conf_diffs = [], []
        for key, old in original.items():
            new = current.get(key)
            if not new or not (lo <= float(old['f_ratio']) < hi):
                continue
            diff = int(new['sign_match'] == 'False') - int(old['sign_match'] == 'False')
            all_diffs.append(diff)
            if new['low_conf'] == 'False' and old['low_conf'] == 'False':
                paired_conf_diffs.append(diff)
        result[arm][label] = {
            'n': len(all_diffs),
            'delta_mean': float(np.mean(all_diffs)) if all_diffs else None,
            'delta_ci': ci(all_diffs),
            'both_conf_n': len(paired_conf_diffs),
            'both_conf_delta_mean': float(np.mean(paired_conf_diffs)) if paired_conf_diffs else None,
            'both_conf_ci': ci(paired_conf_diffs),
        }
(OUT / 'stratified_paired_stats.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))

# The planned intervention baseline is A0, not the pretrained original.
baseline = {r['cell_key']: r for r in rows if r['arm'] == 'A0'}
contrasts = []
for arm in arms:
    if arm in {'orig', 'A0'}:
        continue
    current = {r['cell_key']: r for r in rows if r['arm'] == arm}
    for label, minimum in [('all', 0), ('2+', 2)]:
        for scope in ('all_measured', 'both_conf'):
            differences = []
            for key, base in baseline.items():
                new = current[key]
                if float(base['f_ratio']) < minimum:
                    continue
                if scope == 'both_conf' and (base['low_conf'] == 'True' or new['low_conf'] == 'True'):
                    continue
                differences.append(int(new['sign_match'] == 'False') - int(base['sign_match'] == 'False'))
            contrasts.append(dict(arm=arm, baseline='A0', band=label, scope=scope,
                n=len(differences), delta=float(np.mean(differences)) if differences else None,
                ci95=ci(differences)))
(OUT / 'paired_vs_A0.json').write_text(json.dumps(contrasts, indent=2), encoding='utf-8')
print('A0 comparisons:', json.dumps([r for r in contrasts if r['scope']=='all_measured'], indent=2))
