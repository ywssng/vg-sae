# Verdict

**The audited algebra is correct, and I found no counterexample.**

- **(T1)–(T3) hold.** The kernel equals the n = 3 k-statistic, 3/[(n−1)(n−2)]·Σ = (3/2)Σ. The expansion (T2) matches as cubic forms, which is enough because both sides are symmetric. Unbiasedness follows from independence and zero means.
- **Coefficients check out.** Writing xr − x̄ = Σs(δrs − 1/3)xs gives coefficients 1/3, −1/6 and 1/3. Their squared sum is 3/9 + 18/36 + 6/9 = 3/2.
- **(V2) holds.** Var(h_abc) = (3/2)v³ for distinct teacher-basis indices. There are 16·15·14 = 3360 such ordered entries, so the off-diagonal part is at least 5040v³.
- **(B1) holds.** At p = 1/16, ν/(1−2p)² = 15/196, and this ratio increases on (0, 1/2). The bound 5040·15·20/(196·2730) = 1800/637 is exact. So ε/κ ≥ √(1800/637) ≈ 1.68.
- **(B2) holds.** The constant 5040·(15/196)·2000 = 5,400,000/7 is exact.

**Sample-size feasibility is a separate question.** Exact validity concerns only the displayed confidence formula. To reach ρ = 0.1 it needs n ≥ 2,314,287 rows, about 283 times the 8192 available. This does not bound other estimators or the actual realized error.

**The .02 paired C2 contrast is a third, separate question.** It is a selector-difference criterion. Failure of this geometry certificate neither refutes nor establishes a .02 paired improvement. It only blocks certifying that improvement through this tensor route.

# Major defects with exact fixes

1. **V̄ is undefined, and (C1) silently depends on it.** Markov's inequality is valid only if V̄ ≥ V_H. Substituting (V2), which is a lower bound, makes (C1) invalid. *Fix:* state "V̄ is any certified upper bound on V_H; (V2) may be used only in necessary-condition (floor) arguments."

2. **The rotation step behind (V2) is implicit.** *Fix:* add one sentence. h is equivariant, h(Ax) = A^{⊗3}h(x) for orthogonal A, and the Frobenius norm is invariant under orthogonal maps applied to all modes. Therefore E‖h − T‖² may be computed for Y = D_*ᵀ(X − μ), whose coordinates are independent with variance v.

3. **Undefined symbols.** *Fix:* define ν = p(1−p). Attach the label (V2) to the displayed inequality. State that ε_T denotes ε_mean, or the MOM radius, on the corresponding event.

4. **Edge case.** *Fix:* replace "n ≤ 8192" with "3 ≤ n ≤ 8192", since N = 0 leaves the estimator undefined.

5. **"Radius > 1" is the wrong threshold language.** The trivial Frobenius scale here is O(2√K) = 8, so radius > 1 is not by itself vacuous. *Fix:* state the exact floor ε/κ > 1.68 and compare it with the target ρ = 0.1.

6. **Unverified dependencies.** The claim "(C2) ≥ (C1) at α = 1/20", the bridge from .02 to .1, and (V1) are not checkable from this note. *Fix:* label them provisional. B1 and B2 do not depend on them, since any valid V̄ ≥ V_H ≥ (V2).

7. **Limiting-case caveat to record.** (V2) omits the diagonal and repeated-index entries.
   - As p → 0, the (V2)-based ratio scales like p and vanishes. The true V_H/κ² instead grows like 1/p, through Bernoulli moments E(S−p)⁶ ≈ p.
   - As p → 1/2, κ → 0 and the ratio diverges, as expected.
   - So p = 1/16 is the weakest point of (V2), not of the problem. This makes the obstruction conservative, which is consistent with the note's claim.

# Next falsifiable calculation

(V2) also implies a mean-square floor that does not depend on α, which makes it a sharp target for simulation. Set K = 16, p = 1/16, σ = 0 and N = 2730. Then:

- E‖T̂ − T‖²_F / κ² ≥ 2700/(7·2730) ≈ 0.1413.
- Equivalently, the RMS relative error is at least ≈ 0.376.

Simulate T̂ with D_* = I and fixed triples, using one predeclared seed, and estimate this ratio. Also check the per-entry second moment (3/2)v³ for a single off-diagonal entry. A value reliably below 0.1413, beyond Monte Carlo error, would falsify (V2) or the coefficient algebra.