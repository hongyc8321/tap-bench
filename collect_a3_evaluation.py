"""Collect completed A3 generation and verify every remote artifact hash."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from run_cloud_smoke import HOST, OPTIONS

ROOT=Path(__file__).resolve().parent
REMOTE='/home/ubuntu/nkst_formal_A3_eval_v1'


def main():
    out=ROOT/'p0_audit_results/formal_A3_eval_download_v1';out.mkdir(exist_ok=False)
    probe='''import hashlib,json,csv
from pathlib import Path
r=Path('/home/ubuntu/nkst_formal_A3_eval_v1')
s=json.loads((r/'watcher_status.json').read_text());assert s['status']=='completed' and s['generated']==154
with (r/'gen_out/cogvideox2b_t2v_manifest.csv').open() as f: rows=list(csv.DictReader(f))
assert len(rows)==154 and len({x['cell_key'] for x in rows})==154
files=[p for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
'''
    result=subprocess.run(['ssh',*OPTIONS,HOST,'/home/ubuntu/miniforge3/bin/python3','-'],input=probe,text=True,capture_output=True,check=True,timeout=120)
    index=json.loads(result.stdout)
    for name in index:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(target)],check=True,timeout=1800 if name.startswith('gen_out/') else 120)
    for name,digest in index.items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
    (out/'snapshot_index.json').write_text(json.dumps(index,indent=2))
    manifest=out/'gen_out/cogvideox2b_t2v_manifest.csv'
    with manifest.open() as f:rows=list(csv.DictReader(f))
    for row in rows:row['path']=str(out/Path(row['path']).relative_to(REMOTE))
    with (out/'local_manifest.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('154 A3 videos and',len(index),'artifacts collected; hashes verified')


if __name__=='__main__':main()
