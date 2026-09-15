"""Outcome-independent paired sampling, with anonymized media for human review."""
import csv
import hashlib
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAP = ROOT/'p0_audit_results/cloud_snapshot_v1'


def main():
    out = ROOT/'p0_audit_results/anonymous_review_v1'
    out.mkdir(exist_ok=False)
    (out/'videos').mkdir()
    rng = random.Random(73109)
    jobs = []
    # Separate CW and standard protocols. Select cells before inspecting scores.
    protocols = [
        ('cw', ['phase3/gen_cw_ORIG', 'phase3/gen_cw_A0', 'phase3/gen_cw_A5']),
        ('standard', ['phase2_bundle/phase2/gen_out', 'phase3/gen_A0', 'phase3/gen_A3'])]
    for protocol, groups in protocols:
        manifests = []
        for group in groups:
            folder = SNAP/group
            name = 'cogvideox2b_manifest.csv' if group.endswith('gen_out') else 'cogvideox2b_t2v_manifest.csv'
            with (folder/name).open() as f:
                records = [r for r in csv.DictReader(f) if r['category']=='wheel']
            manifests.append({r['cell_key']: r for r in records})
        common = sorted(set.intersection(*(set(m) for m in manifests)))
        cells = rng.sample(common, 3)
        for key in cells:
            for group, manifest in zip(groups, manifests):
                row = manifest[key]
                base = SNAP/('phase3' if group.startswith('phase3') else 'phase2_bundle/phase2')
                source = base/row['path']
                if not source.is_file():
                    raise FileNotFoundError(source)
                jobs.append(dict(protocol=protocol, group=group, cell_key=key, source=source))
    rng.shuffle(jobs)
    labels, mapping = [], []
    for i, job in enumerate(jobs, 1):
        sample = f'R{i:02}'
        dest = out/'videos'/f'{sample}.mp4'
        shutil.copyfile(job['source'], dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != hashlib.sha256(job['source'].read_bytes()).hexdigest():
            raise ValueError('Copy checksum mismatch')
        mapping.append(dict(sample_id=sample, protocol=job['protocol'], group=job['group'],
            cell_key=job['cell_key'], source=str(job['source']), sha256=digest))
        labels.append(dict(sample_id=sample, video_path=f'videos/{sample}.mp4',
            object_present='', dominant_direction='', sustained_change='', change_seconds='',
            reverse_jitter='', camera_motion='', wheel_flip='', structure_stable='',
            observed_spokes='', uncertainty='', notes=''))
    for name, records in [('labels.csv',labels), ('mapping_DO_NOT_OPEN_BEFORE_REVIEW.csv',mapping)]:
        with (out/name).open('x', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    print('Prepared', len(labels), 'anonymized videos; all copy hashes match.')


if __name__ == '__main__':
    main()
