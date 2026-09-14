# Liver Wala — Product Architecture

```text
Patient / Clinician Web Access
        ↓
Role-aware authentication
        ↓
Scheduling and availability
        ↓
Payment-state transition
        ↓
Consultation-session activation
        ↓
Browser audio/video communication
        ↓
Consultation document exchange
        ↓
Operational records and follow-up state
```

## Principal product domains

| Domain | Responsibility |
|---|---|
| Identity | Patient/clinician access and role separation |
| Clinician onboarding | Approval state before clinician activation |
| Scheduling | One-hour booking windows and conflict control |
| Payment state | Successful payment advances the consultation workflow automatically |
| Consultation | Session/link availability and browser communication |
| Document exchange | Clinician-to-patient consultation document delivery |
| Operations | Admin visibility into users, appointments and consultation state |

## Scheduling contract

The current product requirement exposes fourteen one-hour consultation windows each day from **09:00 through 23:00**.

## State-oriented design

A consultation should progress through explicit states rather than disconnected pages:

```text
registered → eligible → slot_selected → payment_confirmed
→ consultation_active → consultation_completed → document_available
```

This state-machine view is the main engineering differentiator of the project: authentication, scheduling, payment, communication and document delivery are treated as one transactional product journey.
