"""Check original aggregate evidence consistency; does not rerun model inference."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'evidence'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root=ROOT):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                f'Hash mismatch: {name}')
    results = json.loads((root / 'expert_roi_referral_ablation.json').read_text())
    with (root / 'expert_roi_referral_classwise.csv').open() as f:
        rows = list(csv.DictReader(f))
    summaries = []
    for rule in results['rules']:
        group = [r for r in rows if r['rule'] == rule['rule']]
        require(len(group) == 4, 'Expected four classwise rows')
        total = sum(int(r['support']) for r in group)
        retained = sum(int(r['retained']) for r in group)
        require(total == 104, 'Unexpected primary test size')
        correct = 0
        for row in group:
            n, ref, support = (int(row[k]) for k in ('retained', 'referred', 'support'))
            require(n + ref == support, 'Class counts do not reconcile')
            count = n * float(row['retained_case_accuracy'])
            require(math.isclose(count, round(count), abs_tol=1e-8), 'Noninteger correct count')
            correct += round(count)
        error_capture = (rule['errors_total'] - (retained - correct)) / rule['errors_total']
        checks = {'coverage': retained / total, 'retained_accuracy': correct / retained,
                  'error_capture': error_capture}
        for name, actual in checks.items():
            require(math.isclose(actual, rule[name], abs_tol=1e-9), f'{name} mismatch')
        require(retained == rule['retained_n'], 'Retained count mismatch')
        require(rule['errors_referred'] == rule['errors_total'] - (retained - correct),
                'Referred error count mismatch')
        summaries.append({'rule': rule['rule'], 'retained': retained, 'correct': correct, **checks})
    csom = next(r for r in rows if r['Class'] == 'CSOM' and r['rule'] == 'Expert-ROI + latent consensus')
    require(int(csom['retained']) == 2 and float(csom['retained_case_accuracy']) == 0,
            'CSOM evidence changed; update the documented limitation')
    return {'status': 'passed', 'hashed_files': len(manifest['files']),
            'scope': 'source integrity and aggregate arithmetic only', 'rules': summaries,
            'critical_limitation': '0 of 2 retained CSOM cases correctly classified'}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
