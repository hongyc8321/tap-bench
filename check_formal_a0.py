"""Read-only status check; never starts or resumes training."""
from launch_formal_a0 import remote,ROOT
import argparse

CODE='''import json,subprocess
from pathlib import Path
root=Path('/home/ubuntu/nkst_formal_A0_v1')
status=json.loads((root/'process_status.json').read_text())
log=(root/'train.log').read_text(errors='replace').splitlines()
print(json.dumps(dict(process=status,log_tail=log[-12:],gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)),indent=2))
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['A0','A3'],default='A0')
    args=parser.parse_args()
    result=remote(CODE.replace('nkst_formal_A0_v1',f'nkst_formal_{args.arm}_v1'))
    (ROOT/f'p0_audit_results/formal_{args.arm}_launch_v1/latest_status.json').write_text(result)
    print(result)
