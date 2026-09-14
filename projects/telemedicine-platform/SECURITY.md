# Liver Wala — Security, Privacy and Safety Boundaries

Liver Wala handles identity, scheduling, payment state, consultation access, and sensitive health-related communication. The public portfolio therefore documents controls and boundaries without exposing operational secrets.

## Security principles

- do not publish OTP secrets, payment credentials, private QR payloads, session tokens, or patient records
- separate patient, clinician, and administrator permissions
- require explicit clinician activation before clinician access is enabled
- treat payment confirmation as a server-side state transition, not a client-side trust decision
- use expiring consultation-session access rather than permanent public links
- record security-relevant state changes in an audit trail
- minimize retained personal/consultation data
- protect consultation documents at rest and in transit
- revoke or expire access when consultation state changes

## Healthcare safety boundary

The platform supports remote consultation workflow. It should not be positioned as an emergency service, autonomous diagnosis system, or replacement for appropriately licensed clinical judgment.

## Production-assurance roadmap

- independent security review
- privacy and retention policy review
- threat modeling for account takeover and broken access control
- payment replay/idempotency testing
- appointment race-condition testing
- consultation-session access testing
- backup/restore validation
- incident-response procedures
- jurisdiction-specific telemedicine and privacy compliance review before material scale-up
