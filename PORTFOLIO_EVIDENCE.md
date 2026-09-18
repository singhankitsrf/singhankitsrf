# Portfolio evidence guide

This guide helps hiring managers assess Ankit Kumar Singh's portfolio for senior AI engineering, AI-platform architecture, MLOps and healthcare-AI leadership roles.

## Ownership and implementation context

Ankit Kumar Singh states that he personally implemented the portfolio projects. The real-world implementation context includes:

- AI-assisted medical-support work associated with the ENT Department at AIIMS Raipur;
- medical-support activity across hospital-facing environments;
- multidisciplinary institutional engineering through IReSOpM, an Indo–Norway consortium supported by DST (India) and RCN (Norway).

These statements describe implementation context. They do not disclose confidential data or imply regulatory clearance, autonomous diagnosis, or unrestricted publication rights.

## Recommended review order

| Priority | Repository | Senior-level signal | Inspect first |
|---:|---|---|---|
| 1 | [AgentForge Enterprise](https://github.com/singhankitsrf/AgentForge-Enterprise-Agentic-RAG) | Governed agentic RAG, MCP, approval controls and evaluation | README, EVIDENCE.md, architecture, tests, baseline evaluation |
| 2 | [OtoVision MLOps](https://github.com/singhankitsrf/Otovision-MLOps) | Medical CV, leakage control, uncertainty routing and MLOps | EVIDENCE.md, dataset audit, model card, CI |
| 3 | [ClinRoute NLP ReleaseOps](https://github.com/singhankitsrf/ClinRoute-NLP-ReleaseOps) | Model-aware CI/CD, safety boundaries and release engineering | EVIDENCE.md, evaluation outputs, model gate, CodeQL |
| 4 | [OtoSage AWS](https://github.com/singhankitsrf/OtoSage_AWS_GitHub) | AWS SageMaker lifecycle and event-driven architecture | EVIDENCE.md, Terraform, pipeline code, CI |
| 5 | [MedLake Azure PySpark](https://github.com/singhankitsrf/MedLake-Azure-PySpark) | Streaming lakehouse, quality controls and Azure/Databricks design | EVIDENCE.md, local Spark output, jobs, Terraform |

## Claim and verification model

- **Repository-verifiable:** directly inspectable source, tests, workflows or committed outputs.
- **Publicly runnable:** available through documented GitHub setup or a linked Hugging Face/live surface.
- **Author-confirmed:** institutional execution described by the author but not fully reproducible from public records because of confidentiality or access constraints.
- **Not established:** performance, deployment or validation that the repository explicitly does not claim.

## Senior engineering assessment path

A reviewer can evaluate architecture judgment, implementation ownership, reproducibility, safety boundaries, cloud/platform breadth, operational thinking and communication without relying on unsupported headline claims. Each flagship repository includes an `EVIDENCE.md` file using the same framework.
