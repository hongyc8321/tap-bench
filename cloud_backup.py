"""Stream a read-only experiment snapshot, excluding training videos and weights."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
import hashlib, io, json, sys, tarfile
from pathlib import Path
home = Path('/home/ubuntu')
files = []
for root in [home/'phase2_bundle/phase2', home/'phase3']:
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_symlink(): continue
        rel = p.relative_to(home)
        if any(x.startswith(('train_', 'lora_', '__pycache__')) for x in rel.parts):
            if p.name not in {'prompts.txt','videos.txt','video_paths.txt'}: continue
        if p.suffix in {'.mp4','.csv','.json','.log','.py','.sh','.yaml','.yml'} or p.name in {'requirements.txt','prompts.txt','videos.txt','video_paths.txt'}:
            files.append(p)
index = []
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as tar:
    for p in files:
        data = p.read_bytes()
        rel = str(p.relative_to(home))
        index.append(dict(path=rel, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        info = tarfile.TarInfo(rel)
        info.size = len(data)
        info.mode = 0o600
        tar.addfile(info, io.BytesIO(data))
    data = json.dumps(index, indent=2).encode()
    info = tarfile.TarInfo('snapshot_index.json')
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
'''


if __name__ == '__main__':
    os.umask(0o077)
    out = ROOT/'p0_audit_results/cloud_snapshot_v1'
    out.mkdir(exist_ok=False)
    ssh = subprocess.Popen(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
        '-o','ConnectTimeout=15','-i','/home/rhap/.ssh/id_ed25519',
        os.environ.get('NKST_REMOTE_HOST_LEGACY', ''),'python3','-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ssh.stdin.write(REMOTE.encode())
    ssh.stdin.close()
    try:
        with tarfile.open(fileobj=ssh.stdout, mode='r|') as tar:
            for member in tar:
                target = out/member.name
                if not member.isfile() or not target.resolve().is_relative_to(out.resolve()):
                    raise ValueError('Unsafe archive member')
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    source = tar.extractfile(member)
                    for block in iter(lambda: source.read(1024*1024), b''):
                        f.write(block)
        if ssh.wait(timeout=30) != 0:
            raise RuntimeError('Remote snapshot failed')
    finally:
        if ssh.poll() is None:
            ssh.kill()
            ssh.wait()
    index = json.loads((out/'snapshot_index.json').read_text())
    for item in index:
        p = out/item['path']
        if p.stat().st_size != item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f"Hash mismatch: {item['path']}")
    result = dict(files=len(index), bytes=sum(x['bytes'] for x in index), all_hashes_match=True)
    (out/'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
