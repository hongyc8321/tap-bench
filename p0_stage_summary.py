"""Publish audit coverage, not a new model ranking or physical reversal score."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'p0_audit_results'


def main():
    inputs=['quality_ledger_v1/summary.json','cloud_snapshot_v1/verification.json',
            'snapshot_audit_v2/summary.json','temporal_benchmark_v1/protocol.json']
    data={p:json.loads((ROOT/p).read_text()) for p in inputs}
    ledger=data[inputs[0]]
    assert sum(ledger['primary_status'].values())==ledger['total']
    rows=[]
    for arm,counts in ledger['per_arm'].items():
        n=sum(counts.values())
        rows.append(dict(arm=arm,input_n=n,basic_checks_only=counts.get('basic_checks_passed_only',0),
            low_signal_primary=counts.get('low_signal',0),direction_disagreement_primary=counts.get('direction_disagreement',0),
            out_of_frame_primary=counts.get('geometry_out_of_frame',0),
            basic_coverage=counts.get('basic_checks_passed_only',0)/n,
            physical_validity='not_established'))
    assert sum(r['input_n'] for r in rows)==ledger['total']
    out=ROOT/'stage_summary_v1';out.mkdir(exist_ok=False)
    with (out/'group_coverage.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(backup=data[inputs[1]],media_record_n=data[inputs[2]]['total'],
        decoded_record_n=data[inputs[2]]['decoded'],quality_ledger_video_n=ledger['total'],
        quality_counts=ledger['primary_status'],
        localization_checked_n=ledger['localization_checked'],
        perturbation_checked_n=ledger['perturbation_checked'],
        physical_reversal_conclusion='not_established',training_success='not_established',
        p0_complete=False,
        input_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},
        protocol_sha256=hashlib.sha256((ROOT/'EVALUATION_PROTOCOL_V1.md').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
