# Liver Wala — Project Charter

**Project type:** Live telemedicine product engineering portfolio project

## Product objective

Provide one digital workflow for account access, clinician onboarding, appointment booking, payment-state progression, consultation access, document exchange, and operational administration.

## Current product scope

- separate patient and clinician access flows
- email/mobile OTP-based access requirement
- administrator approval boundary for clinician onboarding
- fourteen one-hour appointment windows between 09:00 and 23:00
- payment-state-driven consultation activation
- browser audio/video consultation capability
- clinician-to-patient consultation document exchange
- operational owner/admin panel
- live public website

## Delivery roadmap

### Phase 1 — End-to-end consultation workflow — implemented
- account access and role separation
- appointment selection
- payment state
- consultation activation
- browser communication
- consultation document delivery

### Phase 2 — Production hardening
- centralized audit events
- stronger session and role authorization tests
- delivery/retry telemetry
- appointment conflict protection
- encrypted document-storage policy validation
- backup and recovery evidence

### Phase 3 — Scale and assurance
- measured availability and latency
- load and concurrency testing
- independent security review
- formal privacy/compliance review for target operating jurisdictions
- operational runbooks and incident response

## Evidence principle

The live website and implemented workflow are portfolio evidence. Regulatory approval, independent security certification, and large-scale service-level performance are not claimed without separate evidence.
