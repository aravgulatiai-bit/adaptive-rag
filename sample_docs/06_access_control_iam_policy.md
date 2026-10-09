# Access Control and Identity Management Policy

Northwind Analytics enforces the principle of least privilege: every person and
system is granted the minimum access required to perform their role.

## Provisioning and roles

Access is granted based on role. New employees receive a baseline set of
application access during onboarding. Any access to production customer data
requires manager approval and a documented business justification, and is
granted through time-limited, role-based entitlements.

## Reviews

- User access rights are reviewed every quarter by system owners.
- Privileged (administrator) accounts are reviewed monthly.
- Service accounts must have a named owner and are reviewed twice a year.

## Offboarding

When an employee or contractor leaves, all access is revoked within 24 hours of
their departure. For involuntary terminations, access is revoked immediately,
before or at the time of notification.

## Privileged access

Administrative access to production systems requires multi-factor authentication
and is performed through a bastion host that records session logs. Standing
administrative access is discouraged; just-in-time elevation is preferred and
expires automatically after 4 hours. All privileged sessions are logged and
retained for 13 months.

Questions about access requests should go to the IT service desk at
ithelp@northwind.example.
