"""Outcome-blind paired calibration sample, stratified by coverage evidence."""
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'p0_audit_results'


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def band(value):
    x=float(value)
    return '0-0.5' if x<.5 else '0.5-1' if x<1 else '1-2' if x<2 else '2+'


def main():
    out=ROOT/'manual_calibration_v1';out.mkdir(exist_ok=False);(out/'videos').mkdir()
    candidates={a:{r['cell_key']:r for r in read(ROOT/'evaluation_v2_candidate/rows.csv') if r['arm']==a} for a in ('A0','A3')}
    manifests={a:{r['cell_key']:r for r in read(ROOT/f'formal_{a}_eval_download_v1/local_manifest.csv') if r['category']=='wheel'} for a in ('A0','A3')}
    assert set(candidates['A0'])==set(candidates['A3'])==set(manifests['A0'])==set(manifests['A3'])
    strata=defaultdict(list)
    for key,a0 in candidates['A0'].items():
        a3=candidates['A3'][key]
        present=lambda r:bool(r['target_harmonic_direction'])
        transition='both' if present(a0) and present(a3) else 'A0_only' if present(a0) else 'A3_only' if present(a3) else 'neither'
        strata[(band(a0['f_ratio']),transition)].append(key)
    rng=random.Random(94021)
    # Low-frequency bins are small; quotas preserve all regions while assigning
    # half of the sample to the large 2+ population.
    quotas={'0-0.5':4,'0.5-1':4,'1-2':4,'2+':12}
    selected=[];selection=[]
    for frequency,quota in quotas.items():
        buckets={t:list(strata[(frequency,t)]) for t in ('A0_only','A3_only','neither','both')}
        for values in buckets.values():rng.shuffle(values)
        while len([x for x in selected if x[0]==frequency])<quota:
            moved=False
            for transition in ('A0_only','A3_only','neither','both'):
                if buckets[transition]:
                    key=buckets[transition].pop();selected.append((frequency,key));selection.append(dict(frequency_band=frequency,transition=transition,cell_key=key));moved=True
                    if len([x for x in selected if x[0]==frequency])==quota:break
            if not moved:raise ValueError(f'Not enough cells in {frequency}')
    assert len(selected)==24 and len({x[1] for x in selected})==24
    jobs=[]
    for frequency,key in selected:
        transition=next(r['transition'] for r in selection if r['cell_key']==key)
        for arm in ('A0','A3'):
            jobs.append(dict(arm=arm,cell_key=key,frequency_band=frequency,transition=transition,
                source=Path(manifests[arm][key]['path'])))
    rng.shuffle(jobs)
    labels=[];mapping=[]
    for index,job in enumerate(jobs,1):
        sample=f'M{index:03}';dest=out/'videos'/(sample+'.mp4');shutil.copyfile(job['source'],dest)
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        assert digest==hashlib.sha256(job['source'].read_bytes()).hexdigest()
        labels.append(dict(sample_id=sample,video_path=f'videos/{sample}.mp4',object_present='',dominant_direction='',
            observed_spokes='',structure_stable='',blur_level='',sustained_change='',change_description='',
            reverse_jitter='',camera_motion='',wheel_flip='',measurement_usable='',uncertainty='',notes=''))
        mapping.append(dict(sample_id=sample,arm=job['arm'],cell_key=job['cell_key'],frequency_band=job['frequency_band'],
            coverage_transition=job['transition'],source=str(job['source']),sha256=digest))
    for name,rows in [('labels.csv',labels),('mapping_DO_NOT_OPEN_BEFORE_REVIEW.csv',mapping),('selection_DO_NOT_OPEN_BEFORE_REVIEW.csv',selection)]:
        with (out/name).open('x',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(seed=94021,paired_cells=24,videos=48,frequency_quotas=quotas,
        selected_frequency_counts=dict(Counter(r['frequency_band'] for r in selection)),
        selected_transition_counts=dict(Counter(r['transition'] for r in selection)),
        all_copy_hashes_match=True,
        selection_did_not_use_candidate_direction_or_error=True,
        caveat='Stratified diagnostic calibration, not prevalence-weighted random model sample.')
    (out/'preflight.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
