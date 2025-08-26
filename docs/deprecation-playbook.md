# Deprecation Playbook

This playbook defines the **safe path for removing or replacing features** without harming user experience or breaking invariants.

---

## Principles

- **No surprises**: communicate early, clearly, and often.  
- **User-first**: migrations must prioritize minimal disruption.  
- **Telemetry-driven**: removal only when usage = 0%.  
- **Guardrails**: every deprecation must have a rollback plan.

---

## Deprecation Stages

1. **Proposal (ADR Required)**  
   - Create an ADR documenting rationale, alternatives, and migration plan.  
   - Confirm invariants (security, SLOs, data integrity) remain intact.  

2. **Announcement**  
   - Mark feature as “Deprecated” in code/docs.  
   - Notify internal/external users, provide timelines and guides.  

3. **Dual Mode**  
   - Add adapters/shims.  
   - Support both old + new paths (dual-read/dual-write if data).  
   - Shadow traffic to new system for validation.  

4. **Migration**  
   - Provide tooling or scripts to ease migration.  
   - Track adoption via telemetry.  
   - Block new usage of deprecated feature.  

5. **Validation**  
   - Monitor metrics and error rates.  
   - Validate new system meets or exceeds prior SLOs.  

6. **Removal**  
   - Remove code behind a feature flag.  
   - Confirm telemetry = 0 active usage.  
   - Communicate removal milestone.  

---

## Required Checklist

- [ ] ADR written and accepted.  
- [ ] Deprecation timeline approved.  
- [ ] Migration tooling/docs created.  
- [ ] Feature flagged off by default.  
- [ ] Telemetry proves zero active usage.  
- [ ] Rollback plan documented.  

---

## Rollback Protocol

- Maintain previous code path until final removal.  
- Rollback must be possible within **1 deploy cycle**.  
- For data migrations, dual-write until validation passes.  

---

## Example Communication
Feature XYZ will be deprecated on 2025-12-31.
Please migrate to ABC following the guide at /docs/migrations/xyz-to-abc.md.
Telemetry shows 80% adoption; final removal occurs when adoption = 100% or after 2025-12-31.


---

## References

- Related ADRs  
- Migration Guides  
- Feature Flag registry
