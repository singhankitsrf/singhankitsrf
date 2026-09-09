<div align="center">

# Ankit Kumar Singh

### AI Lead · Senior/Staff AI Engineering · GenAI & Agentic Systems · AI Platform Architecture

Building production-oriented AI systems across **LLMs, LangChain/LangGraph, MCP, RAG, computer vision, NLP, AWS, Azure, PySpark, MLOps, CI/CD, Docker, Kubernetes and responsible AI**.

[![GitHub](https://img.shields.io/badge/GitHub-singhankitsrf-181717?logo=github)](https://github.com/singhankitsrf)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-singhankit491-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/singhankit491)
[![Portfolio](https://img.shields.io/badge/Portfolio-5%20Flagship%20Projects-2563EB)](https://github.com/singhankitsrf/AgentForge-Enterprise-Agentic-RAG/blob/main/PORTFOLIO_PROJECTS.md)

</div>

---

## Executive profile

I build AI systems as **engineering products**, not isolated notebooks. My portfolio focuses on the full path from data and model design to evaluation, APIs, cloud/platform architecture, CI/CD, observability, governance and deployment evidence.

My strongest current portfolio signals are:

- **Agentic AI / LLM engineering:** LangChain, LangGraph, MCP, governed RAG, human approval, evaluation and observability
- **Computer vision / medical AI:** PyTorch, transfer learning, uncertainty-aware routing, explainability and MLOps
- **NLP / ReleaseOps:** reproducible model baselines, regression gates, API contracts, security and browser deployment
- **AWS AI platform engineering:** SageMaker Pipelines, Model Registry, async inference, Terraform and event-driven architecture
- **Azure data platform engineering:** PySpark, Databricks, Delta Lake, medallion architecture, streaming and IaC
- **Production discipline:** Docker, Kubernetes, FastAPI, GitHub Actions, testing, security gates and responsible-AI boundaries

> **Portfolio principle:** executed evidence is labeled separately from architecture or future deployment work. I do not present unexecuted cloud/model claims as measured results.

---

# Flagship AI Engineering Portfolio

## 1. AgentForge Enterprise — Agentic AI / LLM Platform

**Governed multi-agent RAG platform with LangGraph, LangChain, MCP, evaluation, observability and human-in-the-loop control.**

**Architecture signal**

```text
User / API
   ↓
Policy pre-check
   ↓
LangGraph Supervisor
   ├── Retrieval / Evidence
   ├── MCP Tool Registry
   └── Human Approval Boundary
   ↓
Answer Synthesis
   ↓
Trace + Evaluation Artifacts
```

**Evidence:** deterministic agent orchestration, MCP v2 typed tools, FastAPI, offline evaluation contract, Docker, CI and dependency-security audit. Provider-backed LLM benchmarking is intentionally not claimed yet.

[GitHub](https://github.com/singhankitsrf/AgentForge-Enterprise-Agentic-RAG) · [Live Hugging Face Space](https://huggingface.co/spaces/singhankit491/agentforge-enterprise) · [Project Charter](https://github.com/singhankitsrf/AgentForge-Enterprise-Agentic-RAG/blob/main/PROJECT.md) · [Roadmap](https://github.com/singhankitsrf/AgentForge-Enterprise-Agentic-RAG/issues/1)

---

## 2. OtoVision MLOps — Computer Vision / Medical AI

**Production-oriented otoscopic image classification architecture with leakage-aware data QA, uncertainty-aware referral, explainability, FastAPI, Docker and Kubernetes.**

**Architecture signal**

```text
Images → Integrity & Duplicate Audit → Leakage-Aware Split
       → PyTorch Training → Evaluation / Calibration
       → Risk-Aware Review + Grad-CAM
       → FastAPI → Docker → Kubernetes
```

**Evidence:** data-quality/leakage controls, reusable PyTorch pipeline, evaluation/calibration design, Grad-CAM, API serving, container/Kubernetes assets and green CI. No disease-performance metric is published without a genuine five-class checkpoint and reproducible evaluation evidence.

[GitHub](https://github.com/singhankitsrf/Otovision-MLOps) · [Live Hugging Face Space](https://huggingface.co/spaces/singhankit491/otovision-mlops) · [Project Charter](https://github.com/singhankitsrf/Otovision-MLOps/blob/main/PROJECT.md) · [Roadmap](https://github.com/singhankitsrf/Otovision-MLOps/issues/3)

---

## 3. OtoSage AWS — AWS AI Platform / SageMaker

**AWS-native event-driven healthcare AI reference platform using SageMaker Pipelines, Model Registry, async inference, S3, Lambda, DynamoDB, SNS, CloudWatch and Terraform.**

**Architecture signal**

```text
S3 Data → SageMaker Processing → Training → Evaluation
        → Model Registry → Manual Approval → Async Endpoint
Client → API Gateway / Lambda → S3 → Inference → SNS / DynamoDB
                         ↘ CloudWatch / Terraform / OIDC CI-CD
```

**Evidence:** source code, Terraform validation, CI, model-lifecycle controls and Hugging Face architecture demo. Actual SageMaker training, endpoint latency, cloud cost and production SLOs are not claimed until executed in an owned AWS environment.

[GitHub](https://github.com/singhankitsrf/OtoSage_AWS_GitHub) · [Live Hugging Face Space](https://huggingface.co/spaces/singhankit491/otosage-aws) · [Project Charter](https://github.com/singhankitsrf/OtoSage_AWS_GitHub/blob/main/PROJECT.md) · [Roadmap](https://github.com/singhankitsrf/OtoSage_AWS_GitHub/issues/3)

---

## 4. ClinRoute NLP ReleaseOps — NLP / Model-Aware CI-CD

**Production-oriented NLP pipeline with reproducible TF-IDF/logistic baselines, optional Transformer path, redaction, confidence routing, FastAPI, Docker, CodeQL and model regression gates.**

**Architecture signal**

```text
Referral Text → Redaction → Baseline / Transformer Candidate
              → Route + Urgency + Entity Extraction
              → Confidence Policy → API / Browser Runtime
              → Regression Gate → Validated Release
```

**Evidence:** reproducible synthetic benchmark, data-contract checks, browser/scikit-learn parity validation, Python 3.11/3.12 quality gates, unit tests, security scan, CodeQL and live static inference. Synthetic performance is explicitly separated from clinical generalization.

[GitHub](https://github.com/singhankitsrf/ClinRoute-NLP-ReleaseOps) · [Live Hugging Face Space](https://huggingface.co/spaces/singhankit491/clinroute-nlp) · [Project Charter](https://github.com/singhankitsrf/ClinRoute-NLP-ReleaseOps/blob/main/PROJECT.md) · [Roadmap](https://github.com/singhankitsrf/ClinRoute-NLP-ReleaseOps/issues/4)

---

## 5. MedLake Azure PySpark — Data / Lakehouse Engineering

**Healthcare lakehouse reference architecture using PySpark, Azure Databricks, Delta Lake, ADLS Gen2, Event Hubs, Data Factory, Terraform and GitHub Actions.**

**Architecture signal**

```text
Batch / Event Input → Bronze Delta
                   → Quality + Quarantine
                   → Silver Deduplicated Data
                   → Gold KPI Marts
                   → Analytics / AI / BI
                   ↘ Databricks + Event Hubs + Terraform + CI/CD
```

**Evidence:** executable local PySpark pipeline, synthetic-event quality checks, quarantine/deduplication/reconciliation evidence, Spark 3.5.3 CI evaluation, mypy, pytest and Terraform validation. Azure cloud performance/cost is not claimed until measured in Azure.

[GitHub](https://github.com/singhankitsrf/MedLake-Azure-PySpark) · [Live Hugging Face Space](https://huggingface.co/spaces/singhankit491/medlake-pyspark) · [Project Charter](https://github.com/singhankitsrf/MedLake-Azure-PySpark/blob/main/PROJECT.md) · [Roadmap](https://github.com/singhankitsrf/MedLake-Azure-PySpark/issues/7)

---

# Capability Matrix

| Capability | AgentForge | OtoVision | OtoSage | ClinRoute | MedLake |
|---|:---:|:---:|:---:|:---:|:---:|
| LLM / Agentic AI | ✅ |  |  |  |  |
| LangChain / LangGraph | ✅ |  |  |  |  |
| MCP / Tool Use | ✅ |  |  |  |  |
| RAG / Retrieval | ✅ |  |  |  |  |
| Computer Vision / PyTorch |  | ✅ | ✅ |  |  |
| NLP / Transformers |  |  |  | ✅ |  |
| FastAPI | ✅ | ✅ |  | ✅ |  |
| AWS / SageMaker |  |  | ✅ |  |  |
| Azure / Databricks |  |  |  |  | ✅ |
| PySpark / Delta Lake |  |  |  |  | ✅ |
| Docker | ✅ | ✅ | ✅ | ✅ |  |
| Kubernetes |  | ✅ |  |  |  |
| Terraform / IaC |  |  | ✅ |  | ✅ |
| CI/CD / GitHub Actions | ✅ | ✅ | ✅ | ✅ | ✅ |
| Model / Data Evaluation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Security / Governance | ✅ | ✅ | ✅ | ✅ | ✅ |
| Responsible AI Boundaries | ✅ | ✅ | ✅ | ✅ | ✅ |

---

# Engineering Approach

```text
Problem / Product Requirement
        ↓
Data & Contract Design
        ↓
Model / Retrieval / Pipeline Architecture
        ↓
Evaluation & Quality Gates
        ↓
API / Tool / Data-Platform Boundary
        ↓
Container / Cloud / Infrastructure Design
        ↓
CI-CD + Security + Observability
        ↓
Measured Evidence + Release Provenance
```

I prioritize:

- reproducibility before headline metrics
- data leakage and provenance controls
- human review for uncertain or side-effecting AI actions
- automated regression and security gates
- clear separation of research demos from production/clinical claims
- measurable latency, cost and reliability only after execution

---

# Recruiter / Hiring Manager View

This portfolio is designed for roles where the requirement is broader than model training alone:

**AI Lead · Staff/Senior AI Engineer · GenAI/LLM Engineer · AI Platform Architect · Senior ML Engineer · Senior Data Scientist · MLOps / Applied AI Lead**

### What the portfolio demonstrates

- ability to design **end-to-end AI systems**, not only train models
- breadth across **GenAI, CV, NLP, cloud AI and data engineering**
- production thinking around **evaluation, governance, security, CI/CD and deployment**
- architecture spanning **AWS + Azure + container/Kubernetes ecosystems**
- public, clickable deployments with explicit evidence boundaries

---

## Live portfolio

**GitHub:** https://github.com/singhankitsrf  
**Hugging Face:** https://huggingface.co/singhankit491  
**Portfolio index:** https://github.com/singhankitsrf/AgentForge-Enterprise-Agentic-RAG/blob/main/PORTFOLIO_PROJECTS.md

> Medical/healthcare repositories are research and engineering portfolio systems unless explicitly supported by separate validation evidence. They are not presented here as autonomous clinical products or medical devices.
