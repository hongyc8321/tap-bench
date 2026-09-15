"""Launch only probe generation for verified formal A0; no training."""
import hashlib
import json
import subprocess
from pathlib import Path
from run_cloud_smoke import HOST,OPTIONS,ssh

ROOT=Path(__file__).resolve().parent
REMOTE='/home/ubuntu/nkst_formal_A0_eval_v1'


def main():
    if ssh('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader').strip():raise RuntimeError('GPU busy')
    out=ROOT/'p0_audit_results/formal_A0_eval_v1';out.mkdir(exist_ok=False)
    run=json.loads((ROOT/'p0_audit_results/formal_A0_completed_v1/run.json').read_text())
    digest=run['final_weight_hashes']['pytorch_lora_weights.safetensors']
    assert ssh('sha256sum /home/ubuntu/nkst_formal_A0_v1/weights/pytorch_lora_weights.safetensors').split()[0]==digest
    ssh(f'test -d /home/ubuntu && test ! -e {REMOTE} && mkdir -m 700 {REMOTE}')
    uploads=[(ROOT/'phase2/generate_probe.py','generate_probe.py'),
        (ROOT/'p0_audit_results/cloud_snapshot_v1/phase2_bundle/phase2/gt_curves.csv','gt_curves.csv'),
        (ROOT/'formal_a0_supervisor.py','supervisor.py')]
    hashes={}
    for path,name in uploads:
        subprocess.run(['scp',*OPTIONS,str(path),HOST+':'+REMOTE+'/'+name],check=True,timeout=120)
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert ssh(f'sha256sum {REMOTE}/{name}').split()[0]==hashes[name]
    config=dict(command=['/home/ubuntu/miniforge3/bin/python3','-c',
        "import generate_probe as g; g.MODEL_SPECS['cogvideox2b']['repo']='/home/ubuntu/hf_cache/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01'; g.main()",
        '--model','cogvideox2b','--subset','curve','--gt',REMOTE+'/gt_curves.csv',
        '--out',REMOTE+'/gen_out','--lora','/home/ubuntu/nkst_formal_A0_v1/weights','--samples','1'],
        environment=dict(HF_HOME='/home/ubuntu/hf_cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),
        task='probe_generation_only',weight_sha256=digest,uploaded_hashes=hashes,
        expected_videos=154,expected_wheel_videos=141,auto_retry=False,training=False)
    p=out/'authorized.json';p.write_text(json.dumps(config,indent=2))
    subprocess.run(['scp',*OPTIONS,str(p),HOST+':'+REMOTE+'/authorized.json'],check=True,timeout=120)
    code=f'''import subprocess,json
from pathlib import Path
root=Path({REMOTE!r})
with (root/'supervisor.log').open('x') as log:
    p=subprocess.Popen(['/home/ubuntu/miniforge3/bin/python3',str(root/'supervisor.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=root,start_new_session=True)
print(json.dumps(dict(supervisor_pid=p.pid,remote=str(root))))
'''
    r=subprocess.run(['ssh',*OPTIONS,HOST,'/home/ubuntu/miniforge3/bin/python3','-'],input=code,text=True,capture_output=True,check=True,timeout=120)
    (out/'launch.json').write_text(r.stdout);print(r.stdout)


if __name__=='__main__':main()
