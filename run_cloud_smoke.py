"""Upload an isolated A0 fixture and execute exactly two update steps remotely."""
import os
import hashlib
import argparse
import csv
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
HOST = os.environ.get('NKST_REMOTE_HOST', '')
OPTIONS=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-i','/home/rhap/.ssh/id_ed25519']
REMOTE='/home/ubuntu/nkst_smoke_v2'


def ssh(command,timeout=120):
    return subprocess.run(['ssh',*OPTIONS,HOST,command],text=True,capture_output=True,timeout=timeout,check=True).stdout


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lossless',action='store_true')
    parser.add_argument('--weighted',action='store_true',help='A1 lossless two-cell fixture with repeated references')
    args=parser.parse_args()
    if args.weighted:args.lossless=True
    global REMOTE
    if args.lossless:REMOTE='/home/ubuntu/nkst_lossless_smoke_v1'
    if args.weighted:REMOTE='/home/ubuntu/nkst_weighted_smoke_v1'
    activity=ssh('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader')
    if activity.strip():raise RuntimeError('GPU busy; not starting smoke test')
    out=ROOT/'p0_audit_results'/('cloud_weighted_smoke_v1' if args.weighted else 'cloud_lossless_smoke_v1' if args.lossless else 'cloud_smoke_v2');out.mkdir(exist_ok=False)
    ssh(f'test -d /home/ubuntu && test ! -e {REMOTE} && mkdir -m 700 {REMOTE} && mkdir {REMOTE}/data {REMOTE}/data/A0 {REMOTE}/data/videos')
    source=ROOT/'p0_audit_results/paired_video_smoke_v1'
    uploads=[(ROOT/'phase3/train_cogvideox_lora.py',REMOTE+'/trainer.py')]
    uploads += [(source/'A0'/name,REMOTE+'/data/A0/'+name) for name in ('videos.txt','prompts.txt')]
    uploads += [(source/'videos'/name,REMOTE+'/data/videos/'+name) for name in ('B00000_point.mp4','B00001_point.mp4')]
    if args.lossless:
        source=ROOT/'p0_audit_results/paired_video_full_lossless_v1'
        paths=[];prompts=[];expected={}
        uploads=[(ROOT/'phase3/train_cogvideox_lora.py',REMOTE+'/trainer.py')]
        with (ROOT/'p0_audit_results/paired_training_plan_v1/A1.csv').open() as f:
            planned={r['base_id']:r for r in csv.DictReader(f)}
        for base in ('B00000','B00001'):
            for variant in (('point',) if args.weighted else ('point','shutter180')):
                name=f'{base}_{variant}.avi'
                meta=json.loads((ROOT/'p0_audit_results/paired_render_full_v1'/base/variant/'meta.json').read_text())
                b=meta['base'];expected[name]=meta['frame_pixels_sha256']
                weight=int(planned[base]['sampling_weight']) if args.weighted else 1
                paths.extend(['../videos/'+name]*weight)
                prompts.extend([f"a grayscale wheel with {b['k']} spokes rotating counterclockwise at exactly {b['omega_rev_s']} revolutions per second, static camera, white background"]*weight)
                uploads.append((source/'videos'/name,REMOTE+'/data/videos/'+name))
        for name,lines in [('videos.txt',paths),('prompts.txt',prompts)]:
            local=out/name;local.write_text('\n'.join(lines)+'\n')
            uploads.append((local,REMOTE+'/data/A0/'+name))
        expected_file=out/'expected_pixels.json';expected_file.write_text(json.dumps(expected))
        uploads.append((expected_file,REMOTE+'/expected_pixels.json'))
    hashes={}
    for local,remote in uploads:
        subprocess.run(['scp',*OPTIONS,str(local),HOST+':'+remote],check=True,timeout=120)
        digest=hashlib.sha256(local.read_bytes()).hexdigest()
        if ssh('sha256sum '+remote).split()[0]!=digest:raise ValueError('Upload hash mismatch')
        hashes[remote]=digest
    if args.lossless:
        code=f'''import json,hashlib
from pathlib import Path
import decord
import numpy as np
root=Path({REMOTE!r})
expected=json.loads((root/'expected_pixels.json').read_text())
for name,digest in expected.items():
    reader=decord.VideoReader(str(root/'data/videos'/name),num_threads=1)
    assert len(reader)==49 and abs(reader.get_avg_fps()-8)<1e-6
    h=hashlib.sha256()
    for i in range(49):
        rgb=reader[i].asnumpy()
        assert rgb.shape==(480,720,3)
        assert np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,0],rgb[:,:,2])
        h.update(np.ascontiguousarray(rgb[:,:,0]).tobytes())
    assert h.hexdigest()==digest
print(json.dumps(dict(videos=len(expected),decoded_pixel_hashes_match=True,decord=decord.__version__)))
'''
        checked=subprocess.run(['ssh',*OPTIONS,HOST,'/home/ubuntu/miniforge3/bin/python3','-'],input=code,text=True,capture_output=True,check=True,timeout=120)
        (out/'cloud_decode.json').write_text(checked.stdout)
        print(checked.stdout,flush=True)
    cmd=f'/home/ubuntu/miniforge3/bin/python3 {REMOTE}/trainer.py --pretrained_model_name_or_path THUDM/CogVideoX-2b --revision 1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01 --instance_data_root {REMOTE}/data/A0 --caption_column prompts.txt --video_column videos.txt --output_dir {REMOTE}/weights --height 480 --width 720 --max_num_frames 49 --fps 8 --rank 64 --lora_alpha 128 --learning_rate 1e-4 --max_train_steps 2 --train_batch_size 1 --gradient_accumulation_steps 4 --mixed_precision fp16 --gradient_checkpointing --checkpointing_steps 500 --checkpoints_total_limit 2 --seed 42 --num_validation_videos 0 --enable_slicing --enable_tiling'
    cmd=cmd.replace('THUDM/CogVideoX-2b','/home/ubuntu/hf_cache/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01')
    (out/'request.json').write_text(json.dumps(dict(command=cmd,uploaded_hashes=hashes,smoke_only=True,max_train_steps=2),indent=2))
    # GNU timeout bounds GPU use even if this SSH client disconnects.
    wrapped=f'export HF_HOME=/home/ubuntu/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled; timeout --signal=TERM --kill-after=30s 1200s {cmd} > {REMOTE}/train.log 2>&1; code=$?; printf "%s\\n" "$code" > {REMOTE}/exit_code.txt; exit "$code"'
    result=subprocess.run(['ssh',*OPTIONS,HOST,wrapped],text=True,capture_output=True,timeout=1300)
    for name in ('train.log','exit_code.txt'):
        subprocess.run(['scp',*OPTIONS,HOST+':'+REMOTE+'/'+name,str(out/name)],check=True,timeout=120)
    print('exit',result.returncode,flush=True)
    if result.returncode!=0:
        raise RuntimeError('See local train.log for failure; no automatic retry.')
    subprocess.run(['scp',*OPTIONS,'-r',HOST+':'+REMOTE+'/weights',str(out/'weights')],check=True,timeout=300)
    runs=list((out/'weights').glob('run_*.json'))
    if len(runs)!=1:raise ValueError('Unexpected run metadata')
    metadata=json.loads(runs[0].read_text())
    if metadata['status']!='completed' or metadata['global_step']!=2:raise ValueError('Incomplete run')
    if args.weighted and (metadata['dataset_rows']!=5 or metadata['encoded_distributions']!=2):
        raise ValueError('Weighted cache count mismatch')
    if not metadata['final_weight_hashes']:raise ValueError('Missing final weights')
    for name,digest in metadata['final_weight_hashes'].items():
        if hashlib.sha256((out/'weights'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Weight hash mismatch')
    print('Completed two steps; weights and run metadata verified.')


if __name__=='__main__':main()
