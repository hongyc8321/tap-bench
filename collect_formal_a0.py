"""Collect completed A0 artifacts without touching remote training data."""
import hashlib
import argparse
import json
import subprocess
from pathlib import Path
from run_cloud_smoke import HOST, OPTIONS, ssh

ROOT=Path(__file__).resolve().parent
REMOTE='/home/ubuntu/nkst_formal_A0_v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['A0','A3'],default='A0')
    arm=parser.parse_args().arm
    global REMOTE
    REMOTE=f'/home/ubuntu/nkst_formal_{arm}_v1'
    out=ROOT/f'p0_audit_results/formal_{arm}_completed_v1';out.mkdir(exist_ok=False)
    for name in ('train.log','authorized.json','environment.json','process_status.json','trainer.py'):
        subprocess.run(['scp',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(out/name)],check=True,timeout=180)
    names=ssh(f'find {REMOTE}/weights -maxdepth 1 -type f -name "run_*.json"').splitlines()
    if len(names)!=1:raise ValueError('Expected one run metadata file')
    subprocess.run(['scp',*OPTIONS,HOST+':'+names[0],str(out/'run.json')],check=True,timeout=120)
    metadata=json.loads((out/'run.json').read_text())
    assert metadata['status']=='completed' and metadata['global_step']==2000
    assert metadata['dataset_rows']==(1319 if arm=='A3' else 500) and metadata['encoded_distributions']==500
    assert hashlib.sha256((out/'trainer.py').read_bytes()).hexdigest()==metadata['trainer_sha256']
    for name,digest in metadata['final_weight_hashes'].items():
        assert Path(name).name==name
        subprocess.run(['scp',*OPTIONS,HOST+':'+REMOTE+'/weights/'+name,str(out/name)],check=True,timeout=300)
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest
        assert ssh(f'sha256sum {REMOTE}/weights/{name}').split()[0]==digest
    assert metadata['final_weight_hashes']
    (out/'verified.json').write_text(json.dumps(dict(completed=True,steps=2000,final_weights_verified=True,
        optimizer_checkpoints_downloaded=False),indent=2))
    print(arm,'final weights, code, logs and environment collected and verified.')


if __name__=='__main__':main()
