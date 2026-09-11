# Designing Healthcare Lakehouse Pipelines with PySpark, Delta Lake and Data Quality Controls

Data platforms fail in less dramatic ways than models. A record arrives late. A schema changes. Two copies of the same event enter through different paths. A malformed value is coerced instead of rejected. A streaming job restarts and reprocesses data. A downstream KPI looks plausible even though the upstream data contract has drifted.

For AI and analytics systems, these failures matter because model quality cannot exceed the quality and provenance of the data pipeline feeding it.

This is the systems problem behind my **MedLake Azure PySpark** portfolio project: a healthcare lakehouse reference architecture using synthetic events, PySpark, Delta Lake and Azure-oriented infrastructure patterns.

> The project uses synthetic healthcare data only. Cloud performance, cost and scale should be claimed only after execution in an owned Azure environment.

## 1. A medallion architecture is useful when each layer has a contract

The familiar pattern is:

```text
Sources → Bronze → Silver → Gold
```

The names alone do not improve reliability. The value comes from giving each layer a clear responsibility.

### Bronze: preserve and replay

Bronze should retain the raw or minimally transformed event representation together with ingestion metadata. The objective is recoverability and provenance.

If a transformation rule later changes, Bronze allows the downstream layers to be rebuilt without asking the source system to resend history.

### Silver: validate and reconcile

Silver is where records become trustworthy enough for broader use. Typical responsibilities include:

- schema enforcement;
- type normalization;
- deduplication;
- domain validation;
- null handling;
- quality flags;
- late-arrival logic;
- controlled schema evolution.

### Gold: expose stable analytical products

Gold should represent business or analytical contracts—KPI marts, reporting tables, feature-ready datasets or other consumption-oriented views.

Gold is not simply “more transformed data.” It is the layer whose semantics consumers should be able to rely on.

## 2. Bad data needs an explicit destination

One of the most dangerous data-pipeline behaviors is silent coercion.

If an invalid record can be converted into something that looks valid, the pipeline may continue while the dataset becomes less trustworthy.

The MedLake architecture uses a **quality branch**:

```text
Bronze
  ↓
Quality rules
  ├── valid   → Silver
  └── invalid → Quarantine
```

Quarantine is valuable because it preserves evidence of failure.

Instead of deleting a record or forcing it through, the system can retain:

- the original payload;
- the failed rule;
- ingestion time;
- source identity;
- pipeline/run version;
- any remediation status.

That turns data quality into an operational workflow rather than an invisible preprocessing step.

## 3. Deduplication is not just `dropDuplicates()`

Healthcare and event-driven systems can receive repeated records for legitimate operational reasons: retries, replay, multiple ingestion paths, or source-system duplication.

A robust deduplication strategy needs a stable definition of identity.

Depending on the source, identity may involve:

- event ID;
- source-system ID;
- entity ID + timestamp + event type;
- content hash;
- a composite business key.

The system also needs to decide which copy wins when records differ.

A useful rule might prefer the latest valid version while retaining enough metadata to audit what happened.

The important point is that deduplication is a data contract, not a generic Spark function call.

## 4. Batch and streaming should converge on the same semantics

The MedLake reference architecture supports both batch files and streaming events:

```text
Batch files → Auto Loader ┐
                          ├→ Bronze Delta
Event Hubs → Streaming ───┘
```

If batch and streaming paths apply different validation rules, the same logical record can behave differently depending on how it arrived.

A stronger architecture centralizes reusable PySpark quality and transformation modules so both modes converge on the same Silver semantics.

This improves testability and reduces “two pipelines for the same data” drift.

## 5. Replayability is a reliability feature

Streaming systems are often presented as continuously moving forward. Real systems need to go backward too.

You may need to replay data because:

- a transformation bug was fixed;
- a new quality rule was introduced;
- a dimension/reference table changed;
- a downstream model needs historical features regenerated;
- a failed period needs reconciliation.

Replayable Bronze storage plus deterministic transformations makes that possible.

Without replayability, a pipeline failure can become a permanent data-quality defect.

## 6. Delta Lake helps, but semantics still belong to the application

Delta Lake provides useful primitives such as transactional writes, schema controls, versioned tables and merge operations.

Those capabilities can make a lakehouse more reliable, but they do not define the business rules.

A transactionally correct table can still contain semantically invalid data.

For that reason, engineering should separate:

**storage guarantees** — consistency/versioning;  
**data-contract guarantees** — schema/domain rules;  
**business guarantees** — what a Gold metric actually means.

All three layers matter.

## 7. Schema drift should be classified, not merely accepted

Automatic schema evolution is convenient, but not every new column or type change is harmless.

A useful drift policy distinguishes changes such as:

- additive optional field;
- additive required field;
- type widening;
- incompatible type change;
- field removal;
- semantic redefinition.

Some changes can be accepted automatically. Others should block the pipeline or route the affected data to quarantine.

The goal is not to make schema rigid forever. It is to make schema change **observable and governed**.

## 8. Gold tables should carry data-quality context

A dashboard can display a clean KPI while hiding upstream failures.

For operational analytics, it is often useful to expose data-quality indicators alongside business metrics, for example:

- records ingested;
- records quarantined;
- duplicate rate;
- late-arrival rate;
- schema-drift count;
- reconciliation difference;
- processing freshness.

Now a consumer can distinguish “KPI changed” from “data pipeline changed.”

This becomes particularly important when Gold data later feeds ML training or monitoring.

## 9. CI/CD for data platforms should test transformations

Infrastructure validation is useful, but a data platform also needs executable tests around the transformation logic.

The MedLake repository includes local PySpark execution, tests, lint/type checks and Terraform validation.

A strong data-platform CI system can include:

- unit tests for transformation functions;
- contract tests for expected schema;
- deterministic sample datasets;
- duplicate/quarantine test cases;
- reconciliation assertions;
- static type/lint checks;
- infrastructure validation;
- deployment-bundle validation.

These checks make data engineering behave more like software engineering.

## 10. Cloud architecture and cloud evidence are different

The Azure-oriented design includes patterns involving Databricks, ADLS Gen2, Event Hubs, Data Factory, Key Vault, Azure Monitor, Terraform and OIDC-style CI/CD.

Those artifacts demonstrate architecture and implementation intent.

They do not by themselves prove:

- Databricks runtime performance;
- Event Hubs throughput;
- end-to-end streaming latency;
- Azure cost;
- operational availability;
- behavior under production load.

Those metrics should be published only after they are measured in an owned Azure environment.

The distinction matters because a cloud reference architecture is valuable even before production operation—as long as it is labeled accurately.

## 11. Data provenance becomes AI provenance

When a lakehouse feeds model development, each training artifact should be traceable back to data state.

A mature flow might record:

```text
model version
   ↓
feature/training dataset version
   ↓
Gold/Silver table version
   ↓
transformation code version
   ↓
Bronze source provenance
```

This makes it possible to investigate a model regression as a data problem rather than assuming every behavior change comes from the model code.

For regulated or high-stakes domains, this lineage is especially valuable.

## 12. The broader lesson

A healthcare lakehouse should not be evaluated only by whether it can ingest large volumes of data.

The stronger questions are:

- Can the pipeline explain where every record came from?
- Can bad data be isolated without disappearing?
- Can transformations be replayed?
- Can batch and streaming produce consistent semantics?
- Can schema changes be governed?
- Can downstream consumers see data-quality state?
- Can a model later be traced to the exact data version used?

Those capabilities create a data platform that can support trustworthy analytics and AI rather than merely storing data at scale.

## Inspect the project

- GitHub: https://github.com/singhankitsrf/MedLake-Azure-PySpark
- Hugging Face portfolio demo: https://huggingface.co/spaces/singhankit491/medlake-pyspark
- Portfolio index: https://github.com/singhankitsrf/AgentForge-Enterprise-Agentic-RAG/blob/main/PORTFOLIO_PROJECTS.md

**Author:** Ankit Kumar Singh