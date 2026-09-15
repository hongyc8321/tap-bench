"""Read-only remote inventory. No video decoding, model loading or remote writes."""
import os
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import csv, hashlib, json
from pathlib import Path
home = Path('/home/ubuntu')
roots = [home/'phase2_bundle/phase2', home/'phase3']
result = {'manifests': [], 'artifacts': [], 'cache_revisions': []}
for root in roots:
    for p in sorted(root.rglob('*manifest*.csv')):
        with p.open() as f:
            rows = list(csv.DictReader(f))
        found = missing = 0
        size = 0
        missing_keys = []
        for row in rows:
            raw = Path(row.get('path', ''))
            candidates = [raw] if raw.is_absolute() else [root/raw, p.parent/raw, home/raw]
            path = next((x for x in candidates if x.is_file()), None)
            if path:
                found += 1
                size += path.stat().st_size
            else:
                missing += 1
                missing_keys.append(row.get('cell_key'))
        result['manifests'].append(dict(path=str(p), rows=len(rows), found=found,
            unresolved=missing, missing_keys=missing_keys, video_bytes=size,
            columns=list(rows[0]) if rows else [],
            unique_cells=len({r.get('cell_key') for r in rows}),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in str(p):
            continue
        if p.suffix in {'.log', '.json', '.yaml', '.yml', '.sh', '.py', '.safetensors'} or p.name in {'requirements.txt','gt_curves.csv','prompts.txt','video_paths.txt'}:
            r = dict(path=str(p), bytes=p.stat().st_size)
            if p.suffix in {'.py','.sh'} or p.name == 'gt_curves.csv':
                r['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
            if p.name == 'gt_curves.csv':
                with p.open() as f:
                    rows = list(csv.DictReader(f))
                prompts = [r.get('prompt','').lower() for r in rows]
                r.update(rows=len(rows), counterclockwise=sum('counterclockwise' in s for s in prompts),
                    clockwise_only=sum('clockwise' in s and 'counterclockwise' not in s for s in prompts))
            result['artifacts'].append(r)
for cache in [home/'hf_cache', home/'.cache/huggingface']:
    if cache.exists():
        for p in cache.rglob('snapshots'):
            result['cache_revisions'].append(dict(model=str(p.parent), revisions=[x.name for x in p.iterdir() if x.is_dir()]))
print(json.dumps(result, indent=2))
'''


if __name__ == '__main__':
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'StrictHostKeyChecking=yes', '-i', '/home/rhap/.ssh/id_ed25519',
        '-p', '22', os.environ.get('NKST_REMOTE_HOST_LEGACY', ''), 'python3', '-'],
        input=REMOTE, text=True, capture_output=True, timeout=120, check=True)
    data = json.loads(p.stdout)
    out = Path(__file__).resolve().parent/'p0_audit_results/cloud_inventory.json'
    with out.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    for r in data['manifests']:
        print(r['path'], 'rows=', r['rows'], 'found=', r['found'],
              'unresolved=', r['unresolved'], 'MiB=', round(r['video_bytes']/2**20, 2))
    print('artifact_count=', len(data['artifacts']))
    print('cache_revisions=', data['cache_revisions'])
