"""Prepare a prospective paired dataset plan, without rendering or training."""
import argparse
import csv
import hashlib
import json
import random
from decimal import Decimal
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=root/'p0_audit_results/paired_training_plan_v1')
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=73191)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError('count must be positive')
    gt_path = root/'p0_audit_results/cloud_snapshot_v1/phase2_bundle/phase2/gt_curves.csv'
    with gt_path.open(encoding='utf-8') as f:
        gt = [r for r in csv.DictReader(f) if r['category']=='wheel']
    probe_speeds = sorted({Decimal(r['omega_gt']).copy_abs() for r in gt})
    ks = sorted({int(r['k']) for r in gt})
    # Integer hundredths avoid binary floating-point errors at exclusion edges.
    speeds = [Decimal(i)/100 for i in range(50,851)
              if all(abs(Decimal(i)/100-p)>Decimal('.10') for p in probe_speeds)]
    rng = random.Random(args.seed)
    base = []
    for index in range(args.count):
        base.append(dict(base_id=f'B{index:05}', k=rng.choice(ks),
            omega_rev_s=str(rng.choice(speeds)), appearance_seed=rng.randrange(2**31),
            fps=8, frames=49, width=720, height=480,
            direction='counterclockwise'))
    arms = {}
    for arm in ('A0','A1','A2','A3'):
        arms[arm] = [dict(row, arm=arm,
            temporal_variant='shutter180' if arm in {'A2','A3'} else 'point',
            sampling_weight=4 if arm in {'A1','A3'} and Decimal(row['omega_rev_s'])>=4 else 1)
            for row in base]
    for index,b in enumerate(base):
        assert all({key:arms[arm][index][key] for key in b}==b for arm in arms)
        assert all(abs(Decimal(b['omega_rev_s'])-p)>Decimal('.10') for p in probe_speeds)
        assert arms['A0'][index]['sampling_weight']==arms['A2'][index]['sampling_weight']
        assert arms['A1'][index]['sampling_weight']==arms['A3'][index]['sampling_weight']
    args.out.mkdir(exist_ok=False)
    for name,rows in [('base_cells',base), *arms.items()]:
        with (args.out/f'{name}.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    report=dict(status='plan_only_not_rendered',count=args.count,seed=args.seed,
        gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
        probe_speeds=[str(p) for p in probe_speeds],strict_exclusion_radius='0.10',
        paired_base_fields_verified=True,
        total_sampling_weight={arm:sum(r['sampling_weight'] for r in rows) for arm,rows in arms.items()},
        caveats=['Training pipeline does not yet consume sampling_weight.',
                 'Directions are counterclockwise only, matching original main ablation; no bidirectional claim.',
                 'Parameter separation does not establish pixel-level train/test independence.',
                 'No new videos or model training performed.'])
    report['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob('*.csv')}
    (args.out/'preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
