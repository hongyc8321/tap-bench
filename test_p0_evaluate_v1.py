"""Contract tests for the frozen evaluator and its completed publication."""
import csv
import json
import unittest
from collections import Counter
from pathlib import Path
from p0_evaluate_v1 import adjudicate


class EvaluationContract(unittest.TestCase):
    def test_sign_and_snr_gates(self):
        self.assertEqual(adjudicate([1,-1],[10,10]),'geometry_direction_unstable')
        self.assertEqual(adjudicate([0,0],[10,10]),'geometry_direction_unstable')
        self.assertEqual(adjudicate([-1,-2],[5,5]),'candidate')
        self.assertEqual(adjudicate([-1,-2],[4.999,10]),'low_signal')

    def test_published_denominators(self):
        root=Path(__file__).resolve().parent/'p0_audit_results/evaluation_v1'
        with (root/'rows.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        summary=json.loads((root/'summary.json').read_text())
        self.assertEqual(len(rows),1182)
        self.assertEqual(len({(r['arm'],r['cell_key']) for r in rows}),1182)
        for s in summary:
            subset=[r for r in rows if r['arm']==s['arm']]
            accepted=[r for r in subset if r['status']=='accepted_apparent_direction']
            self.assertEqual(len(subset),s['input_n'])
            self.assertEqual(len(accepted),s['accepted_n'])
            self.assertEqual(sum(r['sign_match']=='False' for r in accepted),s['error_n'])
            self.assertEqual(dict(Counter(r['status'] for r in subset)),s['statuses'])
            self.assertEqual(s['accepted_n']+s['unresolved_n'],s['input_n'])
        for p in json.loads((root/'paired.json').read_text()):
            a={r['cell_key']:r for r in rows if r['arm']==p['arm']}
            b={r['cell_key']:r for r in rows if r['arm']==p['baseline']}
            common=[k for k in a.keys()&b.keys() if p['band']=='all' or float(a[k]['f_ratio'])>=2]
            accepted=[k for k in common if a[k]['status']==b[k]['status']=='accepted_apparent_direction']
            self.assertEqual(len(common),p['matched_input_n'])
            self.assertEqual(len(accepted),p['both_accepted_n'])


if __name__=='__main__':unittest.main()
