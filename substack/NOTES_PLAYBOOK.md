# Substack Notes Playbook — Engineering Reliable AI

Substack Notes should extend the long-form publication, not become a stream of generic AI headlines.

## Content mix

Use five repeatable Note formats:

### 1. Engineering principle

One strong claim plus a short explanation.

Example:

> A production AI claim should name its evidence tier. Architecture, implementation, offline evaluation, deployment and production operation are different kinds of proof.

Then link to the relevant article only when the link adds value.

### 2. Architecture fragment

Share one small system flow from a flagship project and explain one design decision.

Good topics:

- policy pre-check → agent supervisor → tools → human approval;
- image QA → classifier → calibration → uncertainty routing;
- registry → approval → async endpoint;
- model regression gate → release;
- Bronze → quality/quarantine → Silver → Gold.

### 3. Failure mode

Explain one failure that a polished demo can hide.

Examples:

- patient or near-duplicate leakage;
- RAG answer without evidence-bearing retrieval;
- agent tool call without authorization boundary;
- benchmark metric without reproducible configuration;
- data pipeline that silently drops invalid events.

### 4. Research-to-production note

Take one research concept and state what changes when it becomes an engineering system.

Examples:

- accuracy → calibration + selective prediction;
- model → model + data contract + release provenance;
- prompt → graph + tool contract + trace + eval;
- notebook → tested package + API + CI + container + monitoring boundary.

### 5. Evidence update

Share a real repository improvement, CI result, benchmark, test, demo update or newly measured result. State exactly what changed and what it does **not** prove.

## Frequency

Target **2–3 Notes per week**. Quality and professional relevance are more important than daily volume.

Suggested rhythm:

- early week: engineering principle or architecture fragment;
- mid/late week: failure mode or research-to-production note;
- after a meaningful repository change: evidence update.

## Voice

- technical but readable;
- specific rather than motivational;
- evidence-aware rather than promotional;
- open about trade-offs;
- avoid engagement bait and exaggerated claims.

## Comment strategy

Use comments to contribute technical substance to relevant AI/ML/research conversations. Do not promote the publication in unrelated threads. A useful comment should stand alone even if nobody visits the profile.

## Restack strategy

Restack only when adding a short technical reason the item matters. Prefer papers, engineering postmortems, evaluation work, responsible-AI analysis, systems research and high-quality architecture discussions.

## Conversion path

A useful Note should naturally lead a reader through:

`Note → long-form article → relevant GitHub project → live Hugging Face demo / evidence artifact`

Not every Note needs a link. Repeatedly linking without adding value weakens credibility.

## Ten launch Notes

1. Accuracy is not confidence reliability: why calibration belongs in medical AI evaluation.
2. Architecture is evidence of design, not evidence of execution.
3. The safest agent architecture assumes tool calls are side effects, not just text generation.
4. RAG should make evidence inspectable, not merely make answers sound grounded.
5. A model registry is useful because promotion is a governed state transition, not because it is another cloud service.
6. CI for ML should test data contracts and model behavior, not only syntax and unit tests.
7. In healthcare data engineering, quarantine is often better than silently coercing bad records.
8. A Hugging Face demo and a production deployment answer different questions; label them accordingly.
9. A strong AI portfolio tells the reader exactly which metrics were measured and where.
10. The research-to-production gap is often a missing evidence system, not a missing model.