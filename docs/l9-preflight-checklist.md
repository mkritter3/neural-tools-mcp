# L9 Preflight Checklist

**Purpose:**  
Before making *any* significant change (especially code modifications, migrations, or architecture updates), confirm ≥95% confidence by walking through this checklist.

---

## 1. Clarity
- [ ] Is the **goal of the change clear** in one sentence?
- [ ] Does it directly improve **end-to-end user experience**?
- [ ] Is the scope defined as **Patch / Minor / Major**?

---

## 2. User Impact
- [ ] Have I written the **expected user-facing outcome**?
- [ ] Are there **metrics** (latency, reliability, adoption, satisfaction) tied to this change?

---

## 3. Invariants
- [ ] Security/privacy protections remain intact.  
- [ ] Data model integrity rules are preserved.  
- [ ] SLOs/SLIs (latency, error rate, availability) remain within budget.  
- [ ] Observability contracts (logs, metrics, traces, IDs) remain valid.

---

## 4. Evidence & Testing
- [ ] Confidence is ≥95% and supported by **tests**.  
- [ ] Unit / integration / regression tests updated or added.  
- [ ] Rollout experiments / chaos / perf tests considered for critical paths.

---

## 5. Safety & Rollback
- [ ] Is the change **small and reversible**?  
- [ ] If Major: is there an **ADR documenting rationale**?  
- [ ] Is there a **rollback plan** documented and tested?  
- [ ] Is the change **feature-flagged** or staged (canary → % rollout → GA)?

---

## 6. Lifecycle & Communication
- [ ] Migration or deprecation paths documented (if applicable).  
- [ ] Stakeholders (PM, design, ops) informed or aligned.  
- [ ] Operational runbooks updated if behavior changes.

---

## 7. Tradeoffs & Alternatives
- [ ] Have I written down the **tradeoffs** (complexity vs maintainability, performance vs cost)?  
- [ ] Have alternatives been considered and rejected with reasons?

---

## Final Gate
- [ ] ✅ Confidence ≥95%  
- [ ] ✅ Purpose clear  
- [ ] ✅ User impact positive  
- [ ] ✅ Rollback possible within one deploy cycle  
- [ ] ✅ Documentation updated (ADR, migration guide, changelog)  

**If any box above is unchecked → STOP. Clarify, test, or document before proceeding.**

---