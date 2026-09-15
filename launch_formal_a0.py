"""Launch only explicitly authorized A0 after immutable input checks."""
import hashlib
import argparse
import csv
from collections import Counter
import json
import subprocess
from pathlib import Path
from run_cloud_smoke import HOST,OPTIONS

ROOT=Path(__file__).resolve().parent


def remote(code):
    result=subprocess.run(['ssh',*OPTIONS,HOST,'/home/ubuntu/miniforge3/bin/python3','-'],input=code,text=True,capture_output=True,check=True,timeout=180)
    return result.stdout


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['A0','A3'],default='A0')
    args=parser.parse_args()
    arm=args.arm
    config=json.loads((ROOT/'p0_audit_results/formal_A0_proposal_v1/proposal.json').read_text())
    data=ROOT/'p0_audit_results/paired_video_full_lossless_v1'
    plan=ROOT/'p0_audit_results/paired_training_plan_v1'
    if arm=='A3':
        def read(name):
            with (plan/(name+'.csv')).open() as f:return list(csv.DictReader(f))
        a0,a3=read('A0'),read('A3')
        assert len(a0)==len(a3)==500
        for left,right in zip(a0,a3):
            assert all(left[k]==right[k] for k in left if k not in {'arm','sampling_weight','temporal_variant'})
            assert right['temporal_variant']=='shutter180'
        paths=(data/arm/'videos.txt').read_text().splitlines()
        expected=Counter({r['base_id']:int(r['sampling_weight']) for r in a3})
        assert Counter(Path(p).stem.split('_')[0] for p in paths)==expected
        config['command']=[s.replace('nkst_formal_A0_v1','nkst_formal_A3_v1').replace('nkst_paired_data_v1/A0','nkst_paired_data_v1/A3') for s in config['command']]
        config.update(arm='A3',dataset_rows=1319,remote_directory='/home/ubuntu/nkst_formal_A3_v1',
            plan_sha256=hashlib.sha256((plan/'A3.csv').read_bytes()).hexdigest(),
            data_list_hashes={n:hashlib.sha256((data/arm/n).read_bytes()).hexdigest() for n in ('videos.txt','prompts.txt')},
            acceptance=['exit code 0','global_step exactly 2000','status completed',
                        'dataset_rows 1319 and encoded_distributions 500','final weight hashes verified'])
    assert hashlib.sha256((ROOT/'phase3/train_cogvideox_lora.py').read_bytes()).hexdigest()==config['trainer_sha256']
    config.update(status='authorized',timeout_seconds=None,termination_grace_seconds=None,
        authorization=f'User requested continuing with {arm}; only this arm, 2000 steps.',
        billing_note='No process time limit; instance billing continues until user stops it.',estimated_cost_formula='No hard cost cap')
    check='''import csv,hashlib,json,subprocess,shutil
from pathlib import Path
c=CONFIG
root=Path(c['remote_directory'])
assert not root.exists(), 'Output directory already exists'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip(), 'GPU busy'
assert shutil.disk_usage('/home/ubuntu').free>8*1024**3, 'Insufficient disk'
available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
assert available>16*1024**3, 'Insufficient RAM'
data=Path('/home/ubuntu/nkst_paired_data_v1')
arm=c['arm']
for name,digest in c['data_list_hashes'].items():assert hashlib.sha256((data/arm/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256((data/'encoding_audit.csv').read_bytes()).hexdigest()==c['audit_sha256']
with (data/'encoding_audit.csv').open(encoding='utf-8-sig') as f:audit={str((data/r['video']).resolve()):r for r in csv.DictReader(f)}
paths=(data/arm/'videos.txt').read_text().splitlines()
assert len(paths)==c['dataset_rows'] and len(set(paths))==500
for path in set(paths):
    p=(data/arm/path).resolve()
    assert hashlib.sha256(p.read_bytes()).hexdigest()==audit[str(p)]['video_sha256']
model=Path(c['command'][c['command'].index('--pretrained_model_name_or_path')+1]);assert model.is_dir()
root.mkdir(mode=0o700)
(root/'authorized.json').write_text(json.dumps(c,indent=2))
packages=json.loads(subprocess.check_output(['/home/ubuntu/miniforge3/bin/python3','-m','pip','list','--format=json'],text=True))
(root/'environment.json').write_text(json.dumps(packages,indent=2))
print('All 500 unique video hashes verified; GPU idle; directory reserved')
'''.replace('CONFIG',repr(config),1)
    print(remote(check),flush=True)
    out=ROOT/'p0_audit_results'/f'formal_{arm}_launch_v1';out.mkdir(exist_ok=False)
    (out/'authorized.json').write_text(json.dumps(config,indent=2))
    files=[(ROOT/'phase3/train_cogvideox_lora.py','trainer.py'),(ROOT/'formal_a0_supervisor.py','supervisor.py')]
    hashes={}
    for src,name in files:
        subprocess.run(['scp',*OPTIONS,str(src),HOST+':'+config['remote_directory']+'/'+name],check=True,timeout=120)
        hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    code='''import hashlib,json,subprocess
from pathlib import Path
root=Path(REMOTE)
for name,digest in HASHES.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
assert not (root/'weights').exists()
with (root/'supervisor.log').open('x') as log:
    p=subprocess.Popen(['/home/ubuntu/miniforge3/bin/python3',str(root/'supervisor.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
print(json.dumps(dict(supervisor_pid=p.pid,remote=str(root))))
'''.replace('REMOTE',repr(config['remote_directory'])).replace('HASHES',repr(hashes))
    output=remote(code);(out/'launch.json').write_text(output);print(output)


if __name__=='__main__':main()
