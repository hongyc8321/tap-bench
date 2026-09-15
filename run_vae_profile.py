"""Isolated, time-bounded remote VAE preflight, collecting success or failure logs."""
import hashlib
import json
import subprocess
from pathlib import Path
from run_cloud_smoke import HOST,OPTIONS,ssh

ROOT=Path(__file__).resolve().parent
REMOTE='/home/ubuntu/nkst_vae_profile_v2'


def main():
    if ssh('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader').strip():raise RuntimeError('GPU busy')
    out=ROOT/'p0_audit_results/cloud_vae_profile_v2';out.mkdir(exist_ok=False)
    ssh(f'test -d /home/ubuntu && test ! -e {REMOTE} && mkdir -m 700 {REMOTE}')
    hashes={}
    for source,name in [(ROOT/'profile_vae_cache.py','vae_resource_check.py'),(ROOT/'phase3/train_cogvideox_lora.py','trainer.py')]:
        subprocess.run(['scp',*OPTIONS,str(source),HOST+':'+REMOTE+'/'+name],check=True,timeout=120)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assert ssh(f'sha256sum {REMOTE}/{name}').split()[0]==digest
        hashes[name]=digest
    command=f'export HF_HOME=/home/ubuntu/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; timeout --signal=TERM --kill-after=30s 3600s /home/ubuntu/miniforge3/bin/python3 {REMOTE}/vae_resource_check.py --root {REMOTE} > {REMOTE}/run.log 2>&1; code=$?; printf "%s\\n" "$code" > {REMOTE}/exit_code.txt; exit "$code"'
    (out/'request.json').write_text(json.dumps(dict(command=command,hashes=hashes,training=False),indent=2))
    result=subprocess.run(['ssh',*OPTIONS,HOST,command],timeout=3700)
    for name in ('run.log','exit_code.txt'):
        subprocess.run(['scp',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(out/name)],check=True,timeout=120)
    exists=subprocess.run(['ssh',*OPTIONS,HOST,f'test -d {REMOTE}/profile']).returncode==0
    if exists:subprocess.run(['scp',*OPTIONS,'-r',HOST+':'+REMOTE+'/profile',str(out/'profile')],check=True,timeout=120)
    print('remote_exit',result.returncode)
    if result.returncode:raise RuntimeError('VAE preflight did not complete; logs preserved')
    print((out/'profile/summary.json').read_text())


if __name__=='__main__':main()
