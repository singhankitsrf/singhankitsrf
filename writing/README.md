# Research & Technical Writing

This collection translates my AI research and engineering work into practitioner-focused articles for research, architecture and production audiences. It also serves as the source library for **Engineering Reliable AI**, my Substack publication system.

The writing is organized around four themes:

1. **Reliable Medical AI** — evaluation, calibration, uncertainty, explainability, human review and research-to-deployment boundaries.
2. **Agentic AI & RAG Platforms** — governed tool use, retrieval, MCP, evaluation, observability, human approval and enterprise architecture.
3. **AI Platform Engineering** — MLOps, AWS, Azure, CI/CD, data platforms, infrastructure and release provenance.
4. **PhD → Production** — reproducibility, evidence quality and converting research ideas into inspectable engineering systems.

## Launch library

### 1. Designing Reliability-Aware Medical AI: From Research Model to Production Architecture
Why accuracy alone is insufficient for high-stakes AI, and how calibration, uncertainty routing, leakage controls, explainability, provenance and human review fit into a trustworthy system.

- [Read the article](./reliability-aware-medical-ai.md)
- Related project: [OtoVision MLOps](https://github.com/singhankitsrf/Otovision-MLOps)
- Live portfolio demo: [OtoVision on Hugging Face](https://huggingface.co/spaces/singhankit491/otovision-mlops)

### 2. Building Governed Agentic RAG Systems: Architecture Beyond the Chatbot
A systems-level view of evidence-bearing retrieval, typed tools, MCP, graph orchestration, human approval, evaluation, observability, security and explicit evidence boundaries.

- [Read the article](./governed-agentic-rag-systems.md)
- Related project: [AgentForge Enterprise](https://github.com/singhankitsrf/AgentForge-Enterprise-Agentic-RAG)
- Live portfolio demo: [AgentForge on Hugging Face](https://huggingface.co/spaces/singhankit491/agentforge-enterprise)

### 3. What Should Count as Evidence in an AI Engineering Portfolio?
A practical evidence hierarchy distinguishing architecture, implementation, reproducible evaluation, deployment, production operation and external validation.

- [Read the article](./evidence-in-ai-engineering-portfolios.md)
- Related index: [Five flagship AI engineering projects](https://github.com/singhankitsrf/AgentForge-Enterprise-Agentic-RAG/blob/main/PORTFOLIO_PROJECTS.md)

### 4. Architecting an AWS SageMaker Model Lifecycle with Registry, Async Inference and Terraform
A lifecycle-oriented view of data QA, managed training, evaluation gates, manual registry approval, event-driven asynchronous inference, infrastructure-as-code and evidence boundaries.

- [Read the article](./aws-sagemaker-model-lifecycle.md)
- Related project: [OtoSage AWS](https://github.com/singhankitsrf/OtoSage_AWS_GitHub)
- Portfolio demo: [OtoSage on Hugging Face](https://huggingface.co/spaces/singhankit491/otosage-aws)

### 5. Model-Aware CI/CD for NLP Systems: Regression Gates, Security and Portable Inference
Why an ML release pipeline must validate model behavior in addition to software behavior, including reproducible benchmarks, regression gates, runtime parity, security and exact-artifact promotion.

- [Read the article](./model-aware-cicd-nlp.md)
- Related project: [ClinRoute NLP ReleaseOps](https://github.com/singhankitsrf/ClinRoute-NLP-ReleaseOps)
- Live demo: [ClinRoute on Hugging Face](https://huggingface.co/spaces/singhankit491/clinroute-nlp)

### 6. Designing Healthcare Lakehouse Pipelines with PySpark, Delta Lake and Data Quality Controls
A systems approach to Bronze/Silver/Gold responsibilities, quality quarantine, deduplication, replayability, schema drift, batch/stream convergence and data provenance for AI.

- [Read the article](./healthcare-lakehouse-pyspark-delta.md)
- Related project: [MedLake Azure PySpark](https://github.com/singhankitsrf/MedLake-Azure-PySpark)
- Portfolio demo: [MedLake on Hugging Face](https://huggingface.co/spaces/singhankit491/medlake-pyspark)

## Publication operating system

The portable Substack launch system is in [`../substack/`](../substack/README.md). It includes:

- finished publication identity and positioning;
- finished About page;
- finished welcome email;
- 12-week editorial calendar;
- Substack Notes playbook;
- exact dashboard/setup checklist;
- cross-platform linking and measurement strategy.

## Recommended cadence

Publish **one substantial article every two weeks**, with 2–3 concise research/engineering Notes per week. Each long-form article should link to inspectable code, a relevant demonstration where available, and the supporting evaluation or architecture evidence.

## Evidence policy

Every public claim should be labeled according to what supports it. Architecture, implementation, reproducible offline evaluation, cloud execution, production operation and clinical validation are not interchangeable evidence tiers.

## Author

**Ankit Kumar Singh**  
AI Lead · Senior/Staff AI Engineering · GenAI & Agentic Systems · AI Platform Architecture

[GitHub](https://github.com/singhankitsrf) · [Hugging Face](https://huggingface.co/singhankit491) · [LinkedIn](https://www.linkedin.com/in/ankit-kumar-singh-data-scientist-434404203)

> These articles discuss research and engineering systems. Medical/healthcare examples are not presented as autonomous clinical products or medical devices unless supported by separate validation and regulatory evidence.