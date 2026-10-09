# Data Retention Policy

This policy defines how long Northwind Analytics retains different categories of
data and when that data is deleted. It applies to all customer data, personal
data, and internal records processed by the InsightHub platform and corporate
systems.

## Retention periods

- **Customer account data**: retained for the life of the subscription and for
  24 months after account closure, after which it is permanently deleted.
- **Personal data of end users**: retained for 24 months after the associated
  account is closed, unless a longer period is required by law.
- **Biometric data**: Northwind does not use biometric data for its core
  product. Where a customer explicitly enables an optional biometric
  authentication feature, biometric templates are retained for a maximum of 12
  months and are deleted within 30 days of feature deactivation.
- **Application and audit logs**: retained for 13 months to support security
  investigations, then rotated out of storage.
- **Financial and tax records**: retained for 7 years to meet regulatory
  obligations.
- **Backups**: encrypted backups are retained for 35 days on a rolling basis.

## Data subject rights

Under the EU General Data Protection Regulation (GDPR), Northwind fulfills data
subject access, correction, and erasure requests within 30 days. Under the
California Consumer Privacy Act (CCPA), verifiable consumer requests are
fulfilled within 45 days, extendable once by an additional 45 days when
reasonably necessary.

Erasure requests are honored across primary storage within 30 days and are
purged from backups on the next backup rotation cycle. To submit a request,
contact the data protection officer at dpo@northwind.example.
