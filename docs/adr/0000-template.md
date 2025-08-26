# [ADR-0000] Title of Decision

**Date:** YYYY-MM-DD  
**Status:** Proposed | Accepted | Superseded by [ADR-XXXX]  
**Authors:** [Name(s)]  
**Version:** 1.0

---

## Context

- What problem are we solving?  
- Why does this matter to the *user experience*?  
- What constraints (technical, business, legal) shape this decision?  
- What alternatives were considered and why rejected?

---

## Decision

- **Chosen Approach:** [clear, concise statement of the path forward]  
- **Confidence Level:** ≥95% before implementation  
- **Tradeoffs:** Call out impacts on complexity, performance, maintainability, cost.  
- **Invariants Preserved:** Security, data integrity, SLOs, observability.

---

## Consequences

- **Positive Outcomes:** [expected benefits]  
- **Risks/Mitigations:** [what could go wrong + safety nets]  
- **User Impact:** [how this improves UX, metrics affected]  
- **Lifecycle:** How this decision evolves (migrations, rollbacks, deprecations).  

---

## Rollout Plan

- Feature flag(s): [yes/no, name]  
- Deployment strategy: canary → % rollout → GA  
- Monitoring & alerts tied to user-facing metrics.  
- Rollback plan if metrics regress.

---

## Alternatives Considered

- Alt A: [pros/cons]  
- Alt B: [pros/cons]  

---

## References

- Related issues / tickets / PRs  
- Supersedes / related ADRs  