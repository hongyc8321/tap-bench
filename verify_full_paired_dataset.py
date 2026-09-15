"""Finalize full dataset audit, including the training decoder's lossless pixels."""
import csv
import hashlib
import json
from pathlib import Path
import decord
import numpy as np
from paired_training_preflight import inspect

ROOT=Path(__file__).resolve().parent


def main():
    plan=ROOT/'p0_audit_results/paired_training_plan_v1'
    frames=ROOT/'p0_audit_results/paired_render_full_v1'
    data=ROOT/'p0_audit_results/paired_video_full_lossless_v1'
    out=ROOT/'p0_audit_results/full_dataset_release_v1';out.mkdir(exist_ok=False)
    with (data/'encoding_audit.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==1000
    decord.bridge.set_bridge('native')
    for i,row in enumerate(rows,1):
        reader=decord.VideoReader(str(data/row['video']),num_threads=1)
        assert len(reader)==49 and abs(reader.get_avg_fps()-8)<1e-6
        digest=hashlib.sha256()
        for j in range(49):
            rgb=reader[j].asnumpy()
            assert rgb.shape==(480,720,3)
            assert np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,0],rgb[:,:,2])
            digest.update(np.ascontiguousarray(rgb[:,:,0]).tobytes())
        meta=json.loads((frames/row['frames_path']/'meta.json').read_text())
        assert digest.hexdigest()==meta['frame_pixels_sha256']
        if i%100==0:print(i,'Decord pixel hashes verified',flush=True)
    reports=[]
    for arm in ('A0','A1','A2','A3'):
        r=inspect(plan,data,arm,ROOT/'p0_audit_results'/f'full_training_{arm}')
        assert r['data_preflight_passed'],r['issues']
        (out/f'{arm}_preflight.json').write_text(json.dumps(r,indent=2));reports.append(r)
    checks=json.loads((frames/'checks.json').read_text())
    assert len(checks['cases'])==500
    assert all(r['static_regions_identical'] and r['paired_appearance_identical'] for r in checks['cases'])
    result=dict(base_cells=500,unique_videos=1000,png_frames=sum(1 for _ in frames.rglob('frame_*.png')),
        opencv_pixel_exact=all(float(r['pixel_mae_255'])==0 for r in rows),decord_pixel_exact=True,
        codec='FFV1',container='AVI',four_arm_preflight_passed=True,
        list_rows={r['arm']:r['actual_rows'] for r in reports},
        video_bytes=sum((data/r['video']).stat().st_size for r in rows),
        png_bytes=sum(p.stat().st_size for p in frames.rglob('frame_*.png')),
        training_started=False,cloud_codec_validation_pending=True)
    assert result['png_frames']==49000
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
