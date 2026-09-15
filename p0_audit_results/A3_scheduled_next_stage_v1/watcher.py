
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
