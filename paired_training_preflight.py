"""Check complete paired plans and exported data; never execute training."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def inspect(plan,data,arm,output):
    issues=[]
    rows=read_csv(plan/f'{arm}.csv')
    expected={r['base_id']:r for r in rows}
    if len(expected)!=len(rows):issues.append('duplicate_plan_base_id')
    source=ROOT/'phase3/train_cogvideox_lora.py'
    if output.exists():issues.append('training_output_already_exists')
    if not output.parent.is_dir():issues.append('training_output_parent_missing')
    paths_file=data/arm/'videos.txt';prompts_file=data/arm/'prompts.txt'
    actual=Counter();unique_files={}
    if not paths_file.is_file() or not prompts_file.is_file():
        issues.append('training_lists_missing')
    else:
        paths=paths_file.read_text().splitlines();prompts=prompts_file.read_text().splitlines()
        if len(paths)!=len(prompts):issues.append('list_length_mismatch')
        for rel,prompt in zip(paths,prompts):
            p=(paths_file.parent/rel).resolve()
            stem=p.stem;identity=stem.split('_')[0];actual[identity]+=1
            row=expected.get(identity)
            if not row:issues.append('unexpected_base_id:'+identity);continue
            if stem!=identity+'_'+row['temporal_variant']:issues.append('variant_mismatch:'+identity)
            wanted=f"a grayscale wheel with {row['k']} spokes rotating counterclockwise at exactly {row['omega_rev_s']} revolutions per second, static camera, white background"
            if prompt!=wanted:issues.append('prompt_mismatch:'+identity)
            if not p.is_file():issues.append('missing_video:'+rel)
            else:unique_files[str(p)]=p
        desired=Counter({k:int(r['sampling_weight']) for k,r in expected.items()})
        if actual!=desired:issues.append('incomplete_or_wrong_weighted_dataset')
    audit_file=data/'encoding_audit.csv'
    if not audit_file.is_file():issues.append('encoding_audit_missing')
    else:
        audited={str((data/r['video']).resolve()):r for r in read_csv(audit_file)}
        for name,p in unique_files.items():
            r=audited.get(name)
            if not r:issues.append('video_not_audited:'+p.name);continue
            if hashlib.sha256(p.read_bytes()).hexdigest()!=r['video_sha256']:issues.append('video_hash_mismatch:'+p.name)
            if [int(r[k]) for k in ('frames','fps','width','height')]!=[49,8,720,480]:issues.append('media_spec_mismatch:'+p.name)
            if r.get('apparent_direction_preserved') not in (None,'True'):
                issues.append('encoding_direction_mismatch:'+p.name)
    command=[sys.executable,str(source),'--pretrained_model_name_or_path','THUDM/CogVideoX-2b',
        '--revision','1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01','--instance_data_root',str((data/arm).resolve()),
        '--caption_column','prompts.txt','--video_column','videos.txt','--output_dir',str(output.resolve()),
        '--height','480','--width','720','--max_num_frames','49','--fps','8','--rank','64','--lora_alpha','128',
        '--learning_rate','1e-4','--max_train_steps','2000','--train_batch_size','1',
        '--gradient_accumulation_steps','4','--mixed_precision','fp16','--gradient_checkpointing',
        '--checkpointing_steps','500','--checkpoints_total_limit','2','--seed','42','--num_validation_videos','0']
    return dict(arm=arm,expected_base_cells=len(expected),listed_base_cells=len(actual),
        expected_rows=sum(int(r['sampling_weight']) for r in rows),actual_rows=sum(actual.values()),
        data_preflight_passed=not issues,issues=sorted(set(issues)),training_started=False,
        runtime_ready=False,command_proposal=command,
        trainer_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256((plan/f'{arm}.csv').read_bytes()).hexdigest(),
        blockers_not_checked=['GPU/model environment and full VAE dry-run',
            'final weight-save and shutdown integration test',
            'Accelerate sharding and resumed sampler determinism','training/runtime dependency lock'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,default=ROOT/'p0_audit_results/paired_training_plan_v1')
    parser.add_argument('--data',type=Path,default=ROOT/'p0_audit_results/paired_video_smoke_v1')
    parser.add_argument('--arm',choices=['A0','A1','A2','A3'],default='A0')
    parser.add_argument('--training-output',type=Path,default=ROOT/'p0_audit_results/future_lora_A0')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=inspect(args.plan,args.data,args.arm,args.training_output)
    with args.report.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    if not result['data_preflight_passed']:raise SystemExit(2)


if __name__=='__main__':main()
