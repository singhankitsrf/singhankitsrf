# PG-GECR | Auditable AI & Human-Supervised Decision Systems

**Array research publication → inspectable evidence → portable decision-policy companion.**

[Read the Array article](https://www.sciencedirect.com/science/article/pii/S2590005626006193) · [Architecture](ARCHITECTURE.md) · [Evidence & model card](MODEL_CARD.md) · [Project charter](PROJECT.md) · [Reproduce the checks](#run-locally)

**Ankit Kumar Singh**, with Ajay Singh Raghuvanshi and Rupa Mehta. *PG-GECR: A Provenance-Gated Hybrid Expert System for Auditable Otoscopic Classification and Human-Supervised Referral.* Array, 2026.

## The engineering problem

A model can look convincing while learning acquisition shortcuts, leaking duplicate images across splits, or failing on a small minority class. This project makes the evidence behind a decision inspectable: which data may enter evaluation, which augmentation is admitted, how uncertainty triggers review, and which claims the evidence supports.

For **Senior ML Engineer, Applied AI Lead, AI Platform Architect and Responsible AI Engineer** interviews, the central signal is ownership of evaluation and governance decisions across an ML system. The publication supplies the research foundation; this public companion makes aggregate evidence and the referral policy easy to inspect.

## What is available here

- Original cVAE/ROI training source, reviewer analyses, split metadata and historical prediction artifacts.
- Portable expert-only ROI feature extraction, training, calibration and model export.
- Five original aggregate evidence files from the v3.0 study supplement, with SHA-256 provenance.
- A dependency-free Python implementation of the study's ROI confidence/entropy and latent-consensus gate, with explicit input validation and reason codes.
- Synthetic probability examples for demonstrating the policy without clinical images.
- An aggregate audit that recomputes coverage, retained accuracy and error capture from classwise counts, verifies hashes, and checks minority-class failures.
- Tests and a scoped GitHub Actions workflow; architecture, model card, and a staged deployment plan.

**Training source is now bundled:** [74 recovered research files](research_archive/) plus a [portable expert-only ROI training/export path](BUNDLE.md). Original trained checkpoints and primary clinical images were absent from the recovered packages; historical prediction artifacts are included. The public bundle supports source inspection, aggregate replay and new ROI training on authorized data. It does not yet supply the original trained model for image inference.

## Study evidence at a glance

| Evidence | Result | Scope |
|---|---:|---|
| Expert-only ROI accuracy | 78.85% | All 104 expert-labelled test images |
| Macro F1 / macro AUC / MCC | 0.677 / 0.862 / 0.598 | Primary ROI classifier |
| ROI + latent retained accuracy | 94.44% (51/54) | Exploratory referral ablation; **51.92% coverage** |
| Errors routed to review | 19/22 (86.36%) | Exploratory ROI + latent policy |
| ROI-only at matched coverage | 88.89% (48/54) | Post hoc comparator; not independent validation |
| CSOM retained cases correctly classified | **0/2** | Full gate; aggregate accuracy does not establish clinical safety |

Source: [`evidence/expert_roi_referral_ablation.json`](evidence/expert_roi_referral_ablation.json) and [classwise counts](evidence/expert_roi_referral_classwise.csv). These are archived study outputs, not newly measured deployment outcomes. The primary split was image-level, not verified patient-independent, and ASOM was confounded with acquisition format. Synthetic augmentation was rejected by the validation gate. [Read the complete evidence boundary](MODEL_CARD.md).

## Why this matters to an engineering team

| Capability | Concrete evidence | Engineering relevance |
|---|---|---|
| Data provenance and leakage governance | Expert-only evaluation policy; duplicate/source audits in the study | Prevent misleading evaluation and make data admission auditable |
| Validation-led model decisions | Expert-only ROI retained as primary; unsupported synthetic augmentation rejected | Avoid complexity without demonstrated value |
| Reliability evaluation | Matched-coverage comparisons and classwise failure disclosure | Evaluate automation coverage alongside error and review workload |
| Explainable orchestration | Portable policy emits reason codes and explicit review actions | Make decisions inspectable at an API/workflow boundary |
| Reproducibility and release discipline | Hash manifest, aggregate audit, policy tests, CI | Detect evidence drift and policy regressions |

## Run locally

Python 3.10+; standard library only. From the repository root:

```bash
python projects/pg-gecr/audit_evidence.py
python projects/pg-gecr/policy.py --input projects/pg-gecr/examples/synthetic_probabilities.json
python -m unittest discover -s projects/pg-gecr/tests -v
```

The probability example is synthetic and tests control flow only. `retain_for_research` is a research-policy outcome, not permission to diagnose or bypass a clinician. A model-serving system must supply calibrated probability vectors in the documented class order.

## Interview walkthrough

1. Explain why expert, pseudo-labelled and generated evidence have different permissions.
2. Show why the ROI classifier stayed primary and augmentation could be rejected.
3. Run the policy examples and inspect the reasons for referral.
4. Run the evidence audit; compare 94.44% retained accuracy with 51.92% coverage and the CSOM failures.
5. Discuss the [deployment stages and release gates](ARCHITECTURE.md) required before any clinical use.

**Research-to-engineering summary:** First author of an Array study on provenance-gated otoscopic AI, combining expert-only evaluation, conditional latent evidence, uncertainty-aware referral, and auditable data/model admission decisions. Public companion includes portable policy code, aggregate evidence checks and CI.

[Back to profile](../../README.md) · [Recruiter guide](../../RECRUITER_START_HERE.md)
