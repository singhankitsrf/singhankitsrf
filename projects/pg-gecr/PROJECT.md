# Project charter

**Name:** PG-GECR — Auditable AI & Human-Supervised Decision Systems  
**Author:** Ankit Kumar Singh; research co-authors Ajay Singh Raghuvanshi and Rupa Mehta  
**Publication:** Array, 2026; [publisher record](https://www.sciencedirect.com/science/article/pii/S2590005626006193)

## Objective
Turn published research into an inspectable engineering case study for senior ML, AI architecture and responsible-AI roles. Demonstrate sound technical decisions, reproducible evidence handling and clear operational boundaries.

## Delivered scope
Research case study, aggregate results with provenance, a portable referral-policy implementation adapted from the study script, synthetic examples, automated tests, and a CI workflow. Original recovered research source is now included under `research_archive/`, alongside portable expert-only ROI training/export. Original trained weights and private clinical images remain unavailable.

## Ownership and attribution
Ankit Kumar Singh is the first author of the study. The paper credits Ajay Singh Raghuvanshi and Rupa Mehta as co-authors. The portable companion and documentation were prepared with AI-assisted engineering support in October 2026; they should not be described as the original experimental runtime or as work previously deployed in a hospital.

## Milestones

| Stage | Status | Exit evidence |
|---|---|---|
| Publication-linked portfolio and aggregate audit | Implemented | Publisher link, immutable-source file hashes, passing local audit |
| Portable policy and automated regression checks | Implemented | Synthetic examples, unit tests, scoped CI configuration |
| Research source bundle and portable ROI training/export | Implemented; source and synthetic smoke checks | 74 archived files, configuration and model-export commands |
| Complete original-model reproduction | Pending original checkpoints/data | Released environment, feature artifacts, checkpoints and split metadata under appropriate permissions |
| Service integration | Planned | Typed inference contract, model/config versions, access control, trace storage and integration tests |
| External validation | Required before clinical use | Patient-grouped, device-diverse evaluation and prespecified classwise acceptance criteria |

No salary, revenue, latency, throughput, hospital adoption or cost-saving outcomes are attributed to this companion.
