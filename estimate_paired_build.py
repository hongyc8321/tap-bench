"""Measure the existing two-cell render/export path before a full CPU build."""
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'p0_audit_results/build_budget_v1';out.mkdir(exist_ok=False)
    free=shutil.disk_usage(ROOT).free
    def run(script,args):
        start=time.monotonic()
        with (out/(script+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(ROOT/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
        return time.monotonic()-start
    frames=out/'frames';videos=out/'videos'
    render_s=run('render_paired_smoke.py',['--out',str(frames)])
    export_s=run('export_paired_smoke.py',['--source',str(frames),'--out',str(videos)])
    old=json.loads((ROOT/'p0_audit_results/paired_render_smoke_v1/checks.json').read_text())
    fresh=json.loads((frames/'checks.json').read_text())
    assert old['base_plan_sha256']==fresh['base_plan_sha256']
    def manifest(path):
        with path.open() as f:return {(r['arm'],r['base_id']):(r['appearance_sha256'],r['pixels_sha256']) for r in csv.DictReader(f)}
    assert manifest(frames/'manifest.csv')==manifest(ROOT/'p0_audit_results/paired_render_smoke_v1/manifest.csv')
    png_bytes=sum(p.stat().st_size for p in frames.rglob('*.png'))
    mp4_bytes=sum(p.stat().st_size for p in videos.rglob('*.mp4'))
    projected=(png_bytes+mp4_bytes)*250
    result=dict(sample_base_cells=2,planned_base_cells=500,render_seconds=render_s,export_audit_seconds=export_s,
        projected_serial_hours=(render_s+export_s)*250/3600,
        sample_png_bytes=png_bytes,sample_mp4_bytes=mp4_bytes,
        projected_data_bytes=projected,recommended_free_bytes=projected*2,
        free_bytes_before=free,free_bytes_after=shutil.disk_usage(ROOT).free,
        repeat_render_pixel_hashes_match=True,
        estimated_2x_space_margin_available=shutil.disk_usage(ROOT).free>projected*2,
        limitations=['Only two sample cells; time/space vary with texture and machine load.',
            'No full dataset or cloud training started.',
            'Serial projection, not a guaranteed duration.'])
    (out/'budget.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
