# Incident Response Runbook

This runbook describes how Northwind Analytics detects, classifies, and responds
to security and availability incidents.

## Severity levels

- **SEV1 — Critical**: customer-facing outage or confirmed data breach. Initial
  response within 15 minutes; target resolution within 4 hours.
- **SEV2 — High**: major feature degraded for many customers. Initial response
  within 30 minutes; target resolution within 8 hours.
- **SEV3 — Moderate**: limited impact or workaround available. Response within 4
  business hours.
- **SEV4 — Low**: minor issue, no customer impact. Handled in the normal backlog.

## Roles

Every incident has an Incident Commander who coordinates the response, a
Communications Lead who handles customer and internal updates, and one or more
Responders. The on-call rotation is managed in PagerDuty and covers 24 hours a
day, 7 days a week.

## Process

1. Detect and declare the incident, assigning a severity.
2. Assemble the response team in the incident channel.
3. Contain, then eradicate the root cause.
4. Recover services and confirm normal operation.
5. Within 5 business days, publish a blameless post-incident review.

## Breach notification

If personal data is confirmed to be affected, the Data Protection Officer is
engaged immediately and affected supervisory authorities are notified within 72
hours, consistent with the privacy policy. Customers are notified without undue
delay. Report suspected incidents to security@northwind.example.
