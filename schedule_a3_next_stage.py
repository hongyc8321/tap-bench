"""Schedule verified A3 probe generation after a three-hour delay."""
import hashlib
import json
import subprocess
from pathlib import Path
from run_cloud_smoke import HOST,OPTIONS

ROOT=Path(__file__).resolve().parent
REMOTE='/home/ubuntu/nkst_formal_A3_eval_v1'
WATCHER=r'''
import hashlib,json,subprocess,time,traceback
from datetime import datetime,timezone
from pathlib import Path
root=Path('/home/ubuntu/nkst_formal_A3_eval_v1')
train=Path('/home/ubuntu/nkst_formal_A3_v1')
status=root/'watcher_status.json'
def save(**values):
    current=json.loads(status.read_text()) if status.exists() else {}
    current.update(values,updated_at=datetime.now(timezone.utc).isoformat())
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(current,indent=2));tmp.replace(status)
try:
    save(status='sleeping',delay_seconds=10800,training=False)
    time.sleep(10800)
    save(status='waiting_for_A3')
    while True:
        state=json.loads((train/'process_status.json').read_text())
        if state['status']=='exited':break
        if state['status'] not in {'starting','running'}:raise RuntimeError('Unexpected A3 state: '+state['status'])
        time.sleep(300)
    if state.get('exit_code')!=0:raise RuntimeError('A3 exited unsuccessfully')
    runs=list((train/'weights').glob('run_*.json'))
    if len(runs)!=1:raise RuntimeError('Expected one A3 run metadata file')
    metadata=json.loads(runs[0].read_text())
    if metadata.get('status')!='completed' or metadata.get('global_step')!=2000:raise RuntimeError('A3 metadata incomplete')
    if metadata.get('dataset_rows')!=1319 or metadata.get('encoded_distributions')!=500:raise RuntimeError('A3 dataset counters unexpected')
    weights=train/'weights/pytorch_lora_weights.safetensors'
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    if metadata['final_weight_hashes'].get(weights.name)!=digest:raise RuntimeError('A3 weight hash mismatch')
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(300)
    save(status='generating',weight_sha256=digest)
    env={'HF_HOME':'/home/ubuntu/hf_cache','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    command=['/home/ubuntu/miniforge3/bin/python3','-c',
        "import generate_probe as g; g.MODEL_SPECS['cogvideox2b']['repo']='/home/ubuntu/hf_cache/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01'; g.main()",
        '--model','cogvideox2b','--subset','curve','--gt',str(root/'gt_curves.csv'),
        '--out',str(root/'gen_out'),'--lora',str(train/'weights'),'--samples','1']
    with (root/'generation.log').open('x') as log:
        process=subprocess.run(command,cwd=root,env={**__import__('os').environ,**env},stdout=log,stderr=subprocess.STDOUT)
    (root/'generation_exit_code.txt').write_text(str(process.returncode)+'\n')
    if process.returncode:raise RuntimeError('A3 generation failed')
    import csv
    with (root/'gen_out/cogvideox2b_t2v_manifest.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=154 or len({r['cell_key'] for r in rows})!=154:raise RuntimeError('A3 generation manifest incomplete')
    save(status='completed',generated=154,generation_exit_code=0)
except Exception as error:
    save(status='failed',error=repr(error),traceback=traceback.format_exc())
    raise
'''


def main():
    out=ROOT/'p0_audit_results/A3_scheduled_next_stage_v1';out.mkdir(exist_ok=False)
    check=subprocess.run(['ssh',*OPTIONS,HOST,'/home/ubuntu/miniforge3/bin/python3','-'],input='''import json
from pathlib import Path
s=json.loads(Path('/home/ubuntu/nkst_formal_A3_v1/process_status.json').read_text())
assert s['status']=='running'
assert not Path('/home/ubuntu/nkst_formal_A3_eval_v1').exists()
Path('/home/ubuntu/nkst_formal_A3_eval_v1').mkdir(mode=0o700)
print(json.dumps(s))
''',text=True,capture_output=True,check=True,timeout=120)
    uploads=[(ROOT/'phase2/generate_probe.py','generate_probe.py'),
        (ROOT/'p0_audit_results/cloud_snapshot_v1/phase2_bundle/phase2/gt_curves.csv','gt_curves.csv')]
    hashes={}
    for source,name in uploads:
        subprocess.run(['scp',*OPTIONS,str(source),HOST+':'+REMOTE+'/'+name],check=True,timeout=120)
        hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    local=out/'watcher.py';local.write_text(WATCHER)
    subprocess.run(['scp',*OPTIONS,str(local),HOST+':'+REMOTE+'/watcher.py'],check=True,timeout=120)
    code=f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({REMOTE!r});expected={hashes!r}
for name,digest in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
with (root/'watcher.log').open('x') as log:
    p=subprocess.Popen(['/home/ubuntu/miniforge3/bin/python3',str(root/'watcher.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,remote=str(root),delay_seconds=10800)))
'''
    result=subprocess.run(['ssh',*OPTIONS,HOST,'/home/ubuntu/miniforge3/bin/python3','-'],input=code,text=True,capture_output=True,check=True,timeout=120)
    (out/'launch.json').write_text(result.stdout)
    (out/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(result.stdout)


if __name__=='__main__':main()
