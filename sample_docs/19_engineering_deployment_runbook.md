# Engineering Deployment Runbook

This runbook describes how Northwind Analytics ships changes to the InsightHub
production environment.

## Continuous delivery

Changes are merged to the main branch only after passing automated tests and at
least one peer code review. Every merge triggers a continuous integration
pipeline that builds artifacts and runs the full test suite.

## Deployment strategy

Northwind uses blue-green deployments. A new version is deployed to an idle
(green) environment, smoke-tested, and then traffic is switched over from the
current (blue) environment. If a problem is detected, traffic is switched back
instantly, providing a fast rollback.

## Deployment windows

Routine deployments happen during business hours, Monday through Thursday.
Deployments are frozen on Fridays and during any active SEV1 incident. Emergency
fixes may be deployed at any time with approval from the on-call Incident
Commander.

## Database changes

Schema migrations must be backward compatible so that the old and new versions
can run simultaneously during a blue-green switch. Destructive migrations, such
as dropping a column, are split across at least two releases.

## Observability

Every service emits metrics, logs, and traces. Deployments are monitored for 30
minutes after release, and an automated rollback triggers if error rates exceed
the defined threshold. Questions go to platform@northwind.example.
