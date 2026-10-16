---
title: OtoSeva AI
emoji: 👂
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: true
license: apache-2.0
short_description: MONAI-enabled human-supervised otoscopic screening and referral workflow
---

# OtoSeva AI

**MONAI-enabled, human-supervised otoscopic screening and referral workflow**

OtoSeva AI is a healthcare-AI product concept designed for clinics, schools, screening camps and tele-ENT pathways. The workflow combines guided image capture, quality gating, reproducible MONAI preprocessing, calibrated confidence, uncertainty-aware accept/defer logic, clinician review and auditable routing.

## Current evidence boundary

Internal retrospective technical validation supports feasibility of the governed screening and referral workflow. OtoSeva AI is **not** presented as autonomous diagnosis. External device-diverse prospective validation and applicable regulatory assessment are planned before clinical deployment.

### Internal technical evidence
- 687 QA-clean otoscopic images after data-integrity audit
- 115-image locked internal test set
- Macro ROC-AUC: 0.873
- Expected calibration error after calibration: 0.082
- Retained-set accuracy at 80% coverage: 0.815
- Median CPU baseline: 20.8 ms/image

## Public product links

- Product website: https://rawcdn.githack.com/singhankitsrf/singhankitsrf/db07c704b5e6d71668154c5b2f98dee72b8ba075/otoseva/index.html
- Interactive workflow demo: https://rawcdn.githack.com/singhankitsrf/singhankitsrf/db07c704b5e6d71668154c5b2f98dee72b8ba075/otoseva/demo.html
- GitHub portfolio: https://github.com/singhankitsrf/singhankitsrf/tree/main/otoseva

## Team

**Ankit Kumar Singh — Team Lead & Sole Innovator**  
PhD Research Scholar, National Institute of Technology Raipur

This Space is a portfolio/demo surface for workflow demonstration and does not provide medical diagnosis.
