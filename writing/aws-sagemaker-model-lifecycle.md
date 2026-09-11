# Architecting an AWS SageMaker Model Lifecycle with Registry, Async Inference and Terraform

A cloud ML architecture becomes meaningful when it explains the lifecycle of a model, not merely the list of managed services around it.

For many applied-AI systems, the real engineering questions are: How does data enter the training workflow? Where is data quality enforced? What blocks a weak model from promotion? Who approves deployment? How is inference decoupled from a client request when processing may take time? How is infrastructure reproduced? And which claims are actually supported by execution evidence?

These questions shape the architecture of my **OtoSage-AWS** portfolio project, an AWS-native reference system for an otoscopic image-analysis workload.

> The project is a research and engineering portfolio implementation. It is not a medical device and does not claim autonomous clinical use.

## 1. Model lifecycle is a state machine

It is tempting to describe MLOps as a sequence of services: S3, SageMaker, Lambda, API Gateway, DynamoDB, SNS, CloudWatch and Terraform.

The more useful view is a sequence of controlled state transitions:

```text
Raw data
  ↓
Data QA / split creation
  ↓
Training candidate
  ↓
Evaluation
  ↓
Registry candidate
  ↓
Human approval
  ↓
Deployable model
  ↓
Inference job
  ↓
Result + operational evidence
```

Each transition should have an explicit contract. Data should not silently pass through a quality failure. A model should not be promoted merely because training completed. A deployment should not be described as production-ready without operational evidence.

## 2. Data integrity belongs before training

The OtoSage workflow starts with data integrity rather than model selection.

For the associated otoscopic archive, preprocessing computes SHA-256 hashes, detects exact duplicates, hard-fails on cross-label exact duplicates, retains a canonical copy, and creates deterministic stratified splits.

Why put this inside the lifecycle?

Because a model registry cannot rescue an invalid evaluation. If duplicated or conflicting images cross train/test boundaries, later metrics may look excellent while the underlying experiment is weak.

In medical imaging especially, evaluation design is part of system architecture.

## 3. Managed training is only one stage

The project uses a SageMaker Pipeline pattern:

```text
S3 → SageMaker Processing → Training → Evaluation → Model Registry
```

The important part is not that SageMaker launches training. It is that preprocessing, training and evaluation become explicit workflow stages whose inputs and outputs can be versioned and audited.

The evaluation contract covers metrics such as accuracy, balanced accuracy, macro-F1, ROC-AUC, expected calibration error and Brier score where appropriate.

A quality gate can then decide whether a trained candidate is eligible to move forward.

The thresholds in a portfolio or reference implementation should be described as engineering configuration—not confused with clinical acceptance criteria.

## 4. Registry promotion should be governed

A model registry is valuable because it separates **candidate creation** from **deployment authorization**.

The OtoSage architecture uses manual approval before deployment. That is deliberate.

In high-stakes or regulated domains, automation does not always mean removing humans. A mature system often automates evidence collection and blocks invalid transitions while preserving explicit approval at the point where organizational accountability matters.

A good promotion record should answer:

- Which model version is being promoted?
- Which dataset/split produced the evaluation?
- Which metrics were measured?
- Which code/environment generated the artifact?
- Which quality gate passed?
- Who or what approved the state transition?

That is much stronger than “deploy the latest model.”

## 5. Why asynchronous inference?

Not every inference request should behave like a low-latency web API call.

The OtoSage design uses asynchronous inference because image-processing workloads can benefit from decoupling submission from completion.

The client flow is:

```text
Client
  ↓
POST /upload
  ↓
Pre-signed S3 URL
  ↓
Direct S3 upload
  ↓
S3 event / Lambda submission
  ↓
SageMaker asynchronous inference
  ↓
S3 result + SNS notification
  ↓
Completion Lambda
  ↓
DynamoDB job state
  ↓
GET /jobs/{job_id}
```

This pattern provides several useful properties.

First, large payloads do not need to travel through the application server. Second, the client does not hold an HTTP connection while inference runs. Third, job state becomes explicit. Fourth, completion/failure can be event-driven.

The architecture therefore models inference as a **job lifecycle**, not just a function call.

## 6. Job state is part of the product contract

Once inference is asynchronous, the API needs to expose useful state.

A request can move through states such as:

```text
created → uploaded → submitted → processing → completed
                                      ↘ failed
```

DynamoDB is used as the job-state store in the reference architecture. SNS carries completion or failure notifications, and a Lambda updates the state so the client can later retrieve the result through an expiring URL.

This design makes failure observable instead of hiding it behind a generic timeout.

## 7. Infrastructure as code is an evidence artifact

Terraform is not just a deployment convenience. It is a description of intended infrastructure that can be linted, validated and reviewed.

The repository includes infrastructure-as-code assets plus CI checks. That supports a defensible claim that the AWS environment has been specified reproducibly.

But there is an important evidence boundary:

**Terraform validation does not prove that the infrastructure has been deployed, benchmarked or operated at production scale.**

Those stronger claims require cloud-execution evidence.

This distinction is one of the most important habits in public AI architecture work.

## 8. CI/CD should protect the lifecycle

Cloud AI repositories benefit from CI that validates more than Python syntax.

Useful gates include:

- unit tests;
- static/lint checks;
- Terraform formatting and validation;
- data-contract checks;
- reproducible evaluation steps where feasible;
- security scanning;
- release provenance;
- identity patterns such as OIDC instead of long-lived cloud credentials.

The purpose is to make invalid changes expensive to merge and valid releases easier to reproduce.

## 9. Observability should follow the job, model and infrastructure

CloudWatch or another observability layer becomes useful when it can answer questions at multiple levels:

**Infrastructure:** Are components healthy?  
**Application:** Are upload/submission/completion paths functioning?  
**Inference:** How long do jobs take? Which fail?  
**Model:** Is input or prediction behavior changing?  
**Cost:** What does a completed inference workload actually cost?

A reference architecture can define these observability points, but measured latency, cost and SLO claims should only be published after execution in an owned environment.

## 10. Evidence boundary: architecture vs cloud execution

This is where many portfolio systems become misleading.

For OtoSage, I separate what the repository can demonstrate now from what requires an executed AWS environment.

### Supported by repository evidence

- source code and project structure;
- SageMaker Pipeline definition;
- data-integrity logic;
- evaluation contract;
- Model Registry/manual-approval design;
- async-inference deployment configuration;
- event-driven Lambda/DynamoDB/SNS architecture;
- Terraform definition and validation;
- CI/CD design.

### Requires owned-cloud execution before claiming

- actual SageMaker training duration;
- deployed endpoint latency/throughput;
- measured AWS cost;
- real availability/SLOs;
- autoscaling behavior under load;
- production monitoring history.

Making that boundary explicit increases credibility rather than weakening the project.

## 11. The broader lesson

A good ML cloud architecture should make the model lifecycle inspectable.

The model is not only a file produced by training. It moves through data provenance, evaluation, approval, deployment, inference, monitoring and retirement states.

When those state transitions are explicit, several things improve at once:

- reproducibility;
- governance;
- operational clarity;
- security review;
- rollback capability;
- evidence quality.

That is the real value of treating MLOps as a system rather than a collection of cloud services.

## Inspect the project

- GitHub: https://github.com/singhankitsrf/OtoSage_AWS_GitHub
- Hugging Face architecture demo: https://huggingface.co/spaces/singhankit491/otosage-aws
- Portfolio index: https://github.com/singhankitsrf/AgentForge-Enterprise-Agentic-RAG/blob/main/PORTFOLIO_PROJECTS.md

**Author:** Ankit Kumar Singh