"""Read-only probe generation status."""
import subprocess
from pathlib import Path
from run_cloud_smoke import HOST,OPTIONS

CODE='''import csv,json
from pathlib import Path
root=Path('/home/ubuntu/nkst_formal_A0_eval_v1')
manifest=root/'gen_out/cogvideox2b_t2v_manifest.csv'
rows=list(csv.DictReader(manifest.open())) if manifest.exists() else []
log=(root/'train.log').read_text(errors='replace').splitlines()
print(json.dumps(dict(process=json.loads((root/'process_status.json').read_text()),generated=len(rows),log_tail=log[-8:]),indent=2))
'''
if __name__=='__main__':
    r=subprocess.run(['ssh',*OPTIONS,HOST,'/home/ubuntu/miniforge3/bin/python3','-'],input=CODE,text=True,capture_output=True,check=True,timeout=120)
    (Path(__file__).resolve().parent/'p0_audit_results/formal_A0_eval_v1/latest_status.json').write_text(r.stdout)
    print(r.stdout)
