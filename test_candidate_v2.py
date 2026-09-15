import csv,json
from collections import Counter
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent/'p0_audit_results/evaluation_v2_candidate'

class CandidateV2(unittest.TestCase):
    def test_published_counts(self):
        with (ROOT/'rows.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        report=json.loads((ROOT/'summary.json').read_text())
        self.assertEqual(len(rows),282)
        self.assertEqual(len({(r['arm'],r['cell_key']) for r in rows}),282)
        for g in report['groups']:
            subset=[r for r in rows if r['arm']==g['arm']]
            self.assertEqual(len(subset),g['n'])
            self.assertEqual(sum(bool(r['target_harmonic_direction']) for r in subset),g['target_candidate_n'])
            self.assertEqual(sum(r['evidence_level']=='cross_method_target_consensus' for r in subset),g['strong_consensus_n'])
            self.assertEqual(dict(Counter(r['evidence_level'] for r in subset)),g['evidence_levels'])

if __name__=='__main__':unittest.main()
