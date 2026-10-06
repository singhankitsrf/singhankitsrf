import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy import decide, normalized_entropy
from audit_evidence import audit, ROOT


class PolicyTests(unittest.TestCase):
    def test_agreement_retains(self):
        result = decide([.9,.04,.03,.03], [.85,.05,.05,.05])
        self.assertEqual(result['action'], 'retain_for_research')
        self.assertEqual(result['reasons'], [])

    def test_disagreement_refers_even_when_confident(self):
        result = decide([.9,.04,.03,.03], [.05,.85,.05,.05])
        self.assertTrue(result['roi_only_retained'])
        self.assertEqual(result['reasons'], ['branch_disagreement'])

    def test_uniform_probabilities_refer(self):
        result = decide([.25]*4, [.25]*4)
        self.assertEqual(result['action'], 'refer_for_human_review')
        self.assertIn('roi_high_entropy', result['reasons'])
        self.assertIn('latent_low_confidence', result['reasons'])
        self.assertAlmostEqual(normalized_entropy([.25]*4), 1)

    def test_confidence_boundary_is_inclusive(self):
        self.assertEqual(decide([.7,.3,0,0], [.8,.2,0,0])['action'], 'retain_for_research')
        self.assertIn('roi_low_confidence', decide([.6999,.3001,0,0], [.8,.2,0,0])['reasons'])

    def test_latent_confidence_boundary(self):
        self.assertEqual(decide([1,0,0,0], [.5,.5,0,0])['action'], 'retain_for_research')

    def test_invalid_inputs_fail_for_both_branches(self):
        bad = [[1,0], [float('nan'),0,0,1], [float('inf'),0,0,0],
               [-.1,.5,.3,.3], [.2]*4, [True,0,0,0], ['1',0,0,0], None]
        for values in bad:
            with self.subTest(values=values):
                with self.assertRaises(ValueError): decide(values,[1,0,0,0])
                with self.assertRaises(ValueError): decide([1,0,0,0],values)

    def test_class_order(self):
        r = decide([0,0,1,0], [0,0,1,0])
        self.assertEqual(r['roi_prediction'], 'CSOM')
        self.assertEqual(normalized_entropy([0,0,1,0]), 0)

    def test_supplied_examples(self):
        rows = json.loads((ROOT.parent/'examples/synthetic_probabilities.json').read_text())
        self.assertEqual([decide(r['roi'],r['latent'])['action'] for r in rows],
                         ['retain_for_research','refer_for_human_review','refer_for_human_review'])


class EvidenceTests(unittest.TestCase):
    def test_original_evidence(self):
        self.assertEqual(audit()['status'], 'passed')

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d)/'evidence'
            shutil.copytree(ROOT,dest)
            with (dest/'expert_roi_referral_ablation.json').open('a') as f: f.write(' ')
            with self.assertRaisesRegex(ValueError,'Hash mismatch'): audit(dest)


if __name__ == '__main__':
    unittest.main()
