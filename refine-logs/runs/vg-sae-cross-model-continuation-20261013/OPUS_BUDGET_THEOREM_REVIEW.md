# Verdict

**Conditionally correct.** The algebra checks out. The conclusion is exact, not statistical: if the imported definitions hold, no audit array with n ≤ 8192 can pass (G2). The note does not show that the certificate is impossible in general, and it says nothing about the C2 contrast.

**Arithmetic checks**
- dσ² = 16/400 = 1/25.
- V_ψ ≥ 3·256 = 768.
- 8·768/8192 = 3/4, so e_C ≥ √3/2 ≈ 0.866 > 4/5.
- C_U ≥ −1/25 + e_C > 19/25. The constant can be sharpened to √3/2 − 1/25 ≈ 0.826.
- Kp ≤ 4 on p ∈ [1/16, 1/4], so the (G2) ceiling is 67/(16·10¹²) ≈ 4.19×10⁻¹².
- The contradiction misses by about 11 orders of magnitude. It therefore survives any plausible slack in g(p).

# Major defects and exact fixes

1. **Imported, unverified premises.** The contradiction rests on four inequalities that are not derived in the note:
   - ψ ≥ −dσ². This needs ψ to have the SURE form: nonnegative terms + 2σ²·div f − dσ², with no hidden negative cross terms.
   - Q₄ ≥ K². If Q₄ were, for example, (Σm_i²)² with m_i ∈ [0,1], it would satisfy Q₄ ≤ K² instead, and the bound would fail.
   - g(p) ≤ 3/16 on [1/16, 1/4].
   - Nonnegativity of the J_U/k term and of the other C_U coefficient in (G2).
   
   *Fix:* quote each definition from ASTRA_INITIAL_REVIEW.md and give a one-line proof of each inequality for the witness.

2. **Notation collision.** The symbol b is used both for the witness offset (b = 0) and for the block count in l = ⌊n/b⌋. *Fix:* rename the block count to B ≥ 1. Then l = ⌊n/B⌋ ≤ n.

3. **The stated boundary is not the true limit.** n ≤ 8192 is sufficient but far from the edge. With V_ψ ≥ 768, the contradiction holds whenever √(6144/l) > 1/25 + 4.19×10⁻¹², which means l ≤ 3,839,999.
   - At l = 3.84×10⁶, e_C = 1/25 exactly. Then C_U = max(0, med + e_C) can equal 0, and this argument yields nothing.
   - *Fix:* state "for all n ≤ 3,839,999 (any B)". Do not imply that larger budgets are also ruled out by this argument.

4. **"Only worsen" is asserted, not shown.** The claim that simultaneous-bank correction and p-uncertainty can only enlarge e_C, or tighten (G2), needs a one-line justification. For example, Bonferroni over H increases B, which decreases l.

# Exact validity vs. feasibility vs. C2

- **Exact:** The deterministic failure for n ≤ 8192 holds, conditional on item 1.
- **Practical feasibility:** Even when n > 3.84×10⁶, passing would require the median of ψ to fall at or below roughly 4×10⁻¹² − e_C. That means ψ must sit essentially at its floor. Clearing the necessary sample size is not sufficient to pass.
- **The .02 paired C2 contrast:** This is a separate estimand. The obstruction does not bear on whether a 4096- or 8192-row audit can resolve a 0.02 paired difference. The advice "do not allocate an audit" should be scoped strictly to this certificate. Any C2 audit needs its own paired-variance power calculation.

# Next falsifiable calculation

Using the report's exact ψ, compute or lower-bound the population median m* of ψ for the witness, with p ∈ [1/16, 1/4].

As n grows, C_U ≥ med + e_C and med → m*. So if m* > 67/(16·10¹²), this generic-envelope certificate fails for every n with high probability, not just at small budgets. If instead m* ≤ 0, report the smallest l at which e_C + m* < 4.19×10⁻¹². Any sharper envelope must first show that its radius at n = 8192 can fall below this margin.