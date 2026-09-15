"""Copy full paired data into a fresh remote directory and audit it on CPU."""
import hashlib
import json
import subprocess
from pathlib import Path
from run_cloud_smoke import HOST, OPTIONS, ssh

ROOT=Path(__file__).resolve().parent
REMOTE='/home/ubuntu/nkst_paired_data_v1'
AUDIT=r'''
import csv,hashlib,json,time
from pathlib import Path
from collections import Counter
import decord
root=Path('/home/ubuntu/nkst_paired_data_v1')
index=json.loads((root/'transfer_index.json').read_text())
start=time.monotonic()
for name,digest in index.items():
    p=root/name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
with (root/'encoding_audit.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(rows)==1000
for i,r in enumerate(rows,1):
    reader=decord.VideoReader(str(root/r['video']),num_threads=1)
    assert len(reader)==49 and abs(reader.get_avg_fps()-8)<1e-6
    for j in range(49):
        frame=reader[j].asnumpy()
        assert frame.shape==(480,720,3)
    if i%100==0:print('decoded',i,flush=True)
orders={};counts={}
for arm in ('A0','A1','A2','A3'):
    paths=(root/arm/'videos.txt').read_text().splitlines()
    prompts=(root/arm/'prompts.txt').read_text().splitlines()
    assert len(paths)==len(prompts)
    assert all((root/arm/p).is_file() for p in paths)
    ids=[Path(p).stem.split('_')[0] for p in paths]
    assert len(set(ids))==500
    orders[arm]=ids;counts[arm]=dict(rows=len(ids),unique_cells=len(set(ids)))
assert orders['A0']==orders['A2'] and orders['A1']==orders['A3']
assert len(orders['A0'])==500 and len(orders['A1'])==1319
result=dict(files_hash_verified=len(index),videos_decoded=len(rows),frames_decoded=49000,
    arm_counts=counts,seconds=time.monotonic()-start,decord=decord.__version__,
    vae_encoded=False,optimizer_started=False,paired_order_verified=True)
(root/'remote_preprocessing.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
'''


def main():
    data=ROOT/'p0_audit_results/paired_video_full_lossless_v1'
    out=ROOT/'p0_audit_results/cloud_full_data_v1';out.mkdir(exist_ok=False)
    print(ssh('df -h /home/ubuntu'),flush=True)
    ssh(f'test -d /home/ubuntu && test ! -e {REMOTE} && mkdir -m 700 {REMOTE}')
    files=[p for p in data.rglob('*') if p.is_file()]
    index={str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'transfer_index.json').write_text(json.dumps(index,indent=2))
    for name in ('videos','A0','A1','A2','A3','encoding_audit.csv','preflight.json'):
        subprocess.run(['scp',*OPTIONS,'-r',str(data/name),HOST+':'+REMOTE+'/'],check=True,timeout=1800)
    subprocess.run(['scp',*OPTIONS,str(out/'transfer_index.json'),HOST+':'+REMOTE+'/transfer_index.json'],check=True,timeout=120)
    with (out/'audit.log').open('x') as log:
        process=subprocess.run(['ssh',*OPTIONS,HOST,'timeout','1800s','/home/ubuntu/miniforge3/bin/python3','-'],
            input=AUDIT,text=True,stdout=log,stderr=subprocess.STDOUT,timeout=1860)
    if process.returncode:raise RuntimeError('Remote preprocessing failed; inspect audit.log')
    subprocess.run(['scp',*OPTIONS,HOST+':'+REMOTE+'/remote_preprocessing.json',str(out/'remote_preprocessing.json')],check=True,timeout=120)
    print((out/'remote_preprocessing.json').read_text())


if __name__=='__main__':main()
