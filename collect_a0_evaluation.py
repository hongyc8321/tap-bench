"""Collect completed generation and verify every artifact against remote hashes."""
import hashlib,json,subprocess
from pathlib import Path
from run_cloud_smoke import HOST,OPTIONS

ROOT=Path(__file__).resolve().parent
REMOTE='/home/ubuntu/nkst_formal_A0_eval_v1'


def main():
    code=f'''import hashlib,json,csv
from pathlib import Path
r=Path({REMOTE!r})
s=json.loads((r/'process_status.json').read_text());assert s['exit_code']==0
with (r/'gen_out/cogvideox2b_t2v_manifest.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==154 and len({{x['cell_key'] for x in rows}})==154
files=[p for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
print(json.dumps({{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}))
'''
    result=subprocess.run(['ssh',*OPTIONS,HOST,'/home/ubuntu/miniforge3/bin/python3','-'],input=code,text=True,capture_output=True,check=True,timeout=120)
    index=json.loads(result.stdout)
    out=ROOT/'p0_audit_results/formal_A0_eval_download_v1';out.mkdir(exist_ok=False)
    subprocess.run(['scp',*OPTIONS,'-r',HOST+':'+REMOTE+'/gen_out',str(out/'gen_out')],check=True,timeout=600)
    for name in index:
        if name.startswith('gen_out/'):continue
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(dest)],check=True,timeout=120)
    for name,digest in index.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
    (out/'snapshot_index.json').write_text(json.dumps(index,indent=2))
    manifest=out/'gen_out/cogvideox2b_t2v_manifest.csv'
    import csv
    with manifest.open() as f:rows=list(csv.DictReader(f))
    for row in rows:row['path']=str(out/Path(row['path']).relative_to(REMOTE))
    with (out/'local_manifest.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('154 records collected; all',len(index),'artifact hashes verified')


if __name__=='__main__':main()
