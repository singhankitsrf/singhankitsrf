"""Portable adaptation of the v3.0 minor_primary_ablation.py referral gate.

Research-policy demonstration only; not a model or a clinical service.
"""
import argparse
import json
import math
from pathlib import Path

CLASSES = ('AOM', 'ASOM', 'CSOM', 'Normal')
POLICY_VERSION = 'pg-gecr-study-gate-1.0'


def validate_probabilities(values):
    if not isinstance(values, (list, tuple)) or len(values) != len(CLASSES):
        raise ValueError('Expected four probabilities in AOM, ASOM, CSOM, Normal order')
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in values):
        raise ValueError('Probabilities must be numeric and non-boolean')
    p = tuple(float(v) for v in values)
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p):
        raise ValueError('Probabilities must be finite and in [0, 1]')
    if not math.isclose(sum(p), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError('Probabilities must sum to one')
    return p


def normalized_entropy(p):
    return -sum(v * math.log(max(v, 1e-9)) for v in p) / math.log(len(CLASSES))


def decide(roi, latent):
    roi, latent = validate_probabilities(roi), validate_probabilities(latent)
    ri = max(range(len(CLASSES)), key=roi.__getitem__)
    li = max(range(len(CLASSES)), key=latent.__getitem__)
    re, le = normalized_entropy(roi), normalized_entropy(latent)
    reasons = []
    if max(roi) < .70:
        reasons.append('roi_low_confidence')
    if re > .65:
        reasons.append('roi_high_entropy')
    roi_only_retained = not reasons
    if ri != li:
        reasons.append('branch_disagreement')
    if max(latent) < .50:
        reasons.append('latent_low_confidence')
    if le > .75:
        reasons.append('latent_high_entropy')
    return {
        'policy_version': POLICY_VERSION,
        'scope': 'research_policy_demonstration',
        'class_order': list(CLASSES),
        'roi_prediction': CLASSES[ri], 'latent_prediction': CLASSES[li],
        'roi_confidence': max(roi), 'latent_confidence': max(latent),
        'roi_entropy': re, 'latent_entropy': le,
        'roi_only_retained': roi_only_retained,
        'action': 'refer_for_human_review' if reasons else 'retain_for_research',
        'reasons': reasons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    try:
        rows = json.loads(args.input.read_text())
        if not isinstance(rows, list):
            raise ValueError('Input must be a JSON list of examples')
        output = [{'example_id': r['example_id'], **decide(r['roi'], r['latent'])} for r in rows]
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
