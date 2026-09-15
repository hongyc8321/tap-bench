"""Audit saved training lists; no inference about unavailable video bytes."""
import csv
import hashlib
import json
import re
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parent
SNAP=ROOT/'p0_audit_results/cloud_snapshot_v1/phase3'
PATTERN=re.compile(r'with (\d+) spokes rotating (counterclockwise|clockwise) at exactly ([\d.]+) revolutions')


def main():
    out=ROOT/'p0_audit_results/training_provenance_v1';out.mkdir(exist_ok=False)
    reports=[];sets={}
    for arm in ('A0','A1','A2','A3','A4','A5'):
        folder=SNAP/f'train_{arm}'
        prompts=(folder/'prompts.txt').read_text().splitlines()
        videos=(folder/'videos.txt').read_text().splitlines()
        if len(prompts)!=len(videos):raise ValueError(f'Unaligned lists {arm}')
        parsed=[];unparsed=[]
        for i,p in enumerate(prompts):
            m=PATTERN.search(p)
            if not m:unparsed.append(i+1);continue
            k,direction,speed=m.groups()
            parsed.append((int(k),direction,float(speed)))
        sets[arm]=set(parsed)
        log=(SNAP/f'phase3_{arm}.log').read_text(errors='replace')
        base_names={re.sub(r'_dup\d+(?=\.mp4$)','',v) for v in videos}
        reports.append(dict(arm=arm,list_rows=len(prompts),unique_video_paths=len(set(videos)),
            unique_names_without_dup_suffix=len(base_names),unique_motion_parameter_tuples=len(sets[arm]),
            unparsed_prompt_lines=unparsed,direction_counts=dict(Counter(p[1] for p in parsed)),
            speed_min=min(p[2] for p in parsed),speed_max=max(p[2] for p in parsed),
            exact_probe_speed_rows=sum(p[2] in {.5,1,2,4,6,8} for p in parsed),
            within_0p1_probe_speed_rows=sum(min(abs(p[2]-v) for v in (.5,1,2,4,6,8))<=.100000001 for p in parsed),
            completed_2000_logged='2000/2000' in log,
            final_weight_save_logged=bool(re.search(r'Model weights saved in lora_'+arm+r'/pytorch_lora_weights.safetensors',log)),
            cuda_oom_logged='CUDA out of memory' in log,
            prompts_sha256=hashlib.sha256((folder/'prompts.txt').read_bytes()).hexdigest(),
            videos_list_sha256=hashlib.sha256((folder/'videos.txt').read_bytes()).hexdigest()))
    overlaps=[dict(left=a,right=b,shared_motion_tuples=len(sets[a]&sets[b]),
        left_unique=len(sets[a]),right_unique=len(sets[b])) for a in sets for b in sets if a<b]
    (out/'training_lists.json').write_text(json.dumps(reports,indent=2))
    (out/'motion_tuple_overlap.json').write_text(json.dumps(overlaps,indent=2))
    print(json.dumps(reports,indent=2))
    print(json.dumps(overlaps,indent=2))


if __name__=='__main__':main()
