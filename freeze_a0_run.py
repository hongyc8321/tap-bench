"""Freeze a single-arm proposal and hashes without connecting or launching."""
import hashlib
import json
from pathlib import Path
from paired_training_preflight import inspect

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'p0_audit_results/formal_A0_proposal_v1';out.mkdir(exist_ok=False)
    data=ROOT/'p0_audit_results/paired_video_full_lossless_v1'
    plan=ROOT/'p0_audit_results/paired_training_plan_v1'
    preflight=inspect(plan,data,'A0',ROOT/'p0_audit_results/formal_A0_future_weights')
    if not preflight['data_preflight_passed']:raise ValueError(preflight['issues'])
    remote='/home/ubuntu/nkst_formal_A0_v1'
    model='/home/ubuntu/hf_cache/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01'
    cmd=preflight['command_proposal'][:]
    cmd[:2]=['/home/ubuntu/miniforge3/bin/python3',remote+'/trainer.py']
    for flag,value in [('--instance_data_root','/home/ubuntu/nkst_paired_data_v1/A0'),
                       ('--output_dir',remote+'/weights'),('--pretrained_model_name_or_path',model)]:
        cmd[cmd.index(flag)+1]=value
    cmd+=['--enable_slicing','--enable_tiling']
    proposal=dict(status='awaiting_budget_authorization',arm='A0',dataset_rows=500,
        optimizer_steps=2000,effective_batch=4,seed=42,command=cmd,
        environment={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':'/home/ubuntu/hf_cache','WANDB_MODE':'disabled'},
        estimated_hours=10,timeout_seconds=43200,termination_grace_seconds=30,
        billing_note='Process timeout does not stop instance billing; no automatic machine shutdown.',
        estimated_cost_formula='10 * actual_hourly_rate; 12-hour process budget plus setup/idle and disk charges',
        remote_directory=remote,overwrite_allowed=False,auto_retry=False,auto_resume=False,
        run_other_arms=False,generate_evaluation_videos=False,
        trainer_sha256=preflight['trainer_sha256'],plan_sha256=preflight['plan_sha256'],
        data_list_hashes={n:hashlib.sha256((data/'A0'/n).read_bytes()).hexdigest() for n in ('videos.txt','prompts.txt')},
        audit_sha256=hashlib.sha256((data/'encoding_audit.csv').read_bytes()).hexdigest(),
        launch_checks=['GPU idle','remote trainer and data hashes match frozen proposal',
            'remote output directory absent','sufficient host RAM and disk',
            'dependencies captured; exact offline snapshot available'],
        acceptance=['exit code 0','global_step exactly 2000','status completed',
            'dataset_rows 500 and encoded_distributions 500','final weight hashes verified after retrieval'])
    (out/'proposal.json').write_text(json.dumps(proposal,indent=2))
    (out/'local_preflight.json').write_text(json.dumps(preflight,indent=2))
    print(json.dumps(proposal,indent=2))


if __name__=='__main__':main()
