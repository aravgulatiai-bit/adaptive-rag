# Information Security Policy

Northwind Analytics maintains an information security program aligned with ISO
27001 and is audited annually against the SOC 2 Type II framework. The program
is owned by the Chief Information Security Officer, Priya Nair.

## Encryption

- All customer data is encrypted at rest using AES-256.
- All data in transit is encrypted using TLS version 1.2 or higher.
- Encryption keys are managed in AWS Key Management Service and rotated
  automatically every 12 months.

## Authentication

- Multi-factor authentication (MFA) is mandatory for all employee accounts and
  for all administrative access to production systems.
- Employee passwords must be at least 14 characters and are rotated every 90
  days.
- Single sign-on (SSO) via SAML is available to all Enterprise customers.

## Endpoint and network security

- Company laptops use full-disk encryption and centrally managed anti-malware.
- Production environments are isolated in a separate AWS account with no direct
  internet ingress except through a hardened load balancer.
- Vulnerability scans run weekly, and critical vulnerabilities are remediated
  within 7 days.

## Certifications

Northwind holds SOC 2 Type II and ISO 27001 certifications. Copies of the
current SOC 2 report are available to customers under NDA by request to
security@northwind.example. Penetration tests are conducted by an independent
third party at least once per year.
