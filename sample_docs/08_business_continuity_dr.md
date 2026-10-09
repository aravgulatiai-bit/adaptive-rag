# Business Continuity and Disaster Recovery

Northwind Analytics maintains a business continuity plan (BCP) and disaster
recovery (DR) plan to ensure InsightHub remains available and customer data is
protected against loss.

## Recovery objectives

- **Recovery Time Objective (RTO): 4 hours.** In a disaster, core InsightHub
  services are restored within 4 hours.
- **Recovery Point Objective (RPO): 1 hour.** No more than 1 hour of data is at
  risk of loss.

## Backups

Customer data is backed up daily. Backups are encrypted with AES-256 and stored
in a separate AWS region from the primary environment. Backups are retained on a
rolling 35-day basis. Restore procedures are tested at least once per quarter,
and the results are documented.

## Redundancy

Production runs across multiple AWS Availability Zones. The database uses
synchronous replication within the primary region and asynchronous replication
to a secondary region for regional failover.

## Testing and review

The full DR plan is tested at least once per year through a simulated regional
failover. The BCP is reviewed annually and after any SEV1 incident. Plan
ownership sits with the VP of Engineering, and questions can be sent to
ops@northwind.example.
