"""Read-only backup of six final LoRA files; no training or remote writes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
HOST = os.environ.get('NKST_REMOTE_HOST', '')
OPTIONS=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-i','/home/rhap/.ssh/id_ed25519']


def main():
    os.umask(0o077)
    out=ROOT/'p0_audit_results/final_lora_backup_v1';out.mkdir(exist_ok=False)
    records=[]
    for arm in ('A0','A1','A2','A3','A4','A5'):
        remote=f'/home/ubuntu/phase3/lora_{arm}/pytorch_lora_weights.safetensors'
        before=subprocess.check_output(['ssh',*OPTIONS,HOST,'sha256sum',remote],text=True,timeout=120).split()[0]
        partial=out/f'{arm}.partial'
        subprocess.run(['scp',*OPTIONS,f'{HOST}:{remote}',str(partial)],check=True,timeout=300)
        digest=hashlib.sha256(partial.read_bytes()).hexdigest()
        after=subprocess.check_output(['ssh',*OPTIONS,HOST,'sha256sum',remote],text=True,timeout=120).split()[0]
        if digest!=before or digest!=after:raise RuntimeError(f'Hash mismatch: {arm}')
        final=out/f'{arm}.safetensors';partial.rename(final)
        records.append(dict(arm=arm,source=remote,file=final.name,bytes=final.stat().st_size,sha256=digest))
        print(arm,final.stat().st_size,'hash verified',flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(host=HOST,files=records,
        scope='final root weights only; optimizer states and training videos not backed up'),indent=2))


if __name__=='__main__':main()
