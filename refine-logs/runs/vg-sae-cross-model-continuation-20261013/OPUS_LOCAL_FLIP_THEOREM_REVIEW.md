# Review: local-flip geometry lemma and hard-coefficient implication (§3–§4)

## Verdict

**The lemma is mathematically valid as stated, but it is a near-vacuous and practically infeasible certificate for C2.**

I re-derived each step and found no counterexample. The steps checked were:

- **Lemma (L):** `(1−z)(1−δ/z)−δ = 1−z−δ/z`.
- **Row-norm step:** `σ_min(A)=1/s`, so each row has an entry `≥1/(s√K) > t`. This holds because G1 is stronger.
- **(L2):** the bound with `a_K(p)`.
- **(L3):** `m > u > r`, and KL is monotone above r. The union bound is `1{h>η} ≤ v/V + KL/k`. `EΣv ≤ C` holds by unit columns and conditionally independent gates.
- **Permutation step:** by Weyl's inequality.
- **Column bounds:** `|α_j−1| ≤ ‖r_j‖ ≤ (K−1)t`.
- **(R):** via `T=PS`, `κ√E‖Zcode−T‖² ≤ √C + B + ε_D√(Kp)`.
- **§3.1 numerics:**
  - `(7/8)^16 ≈ .118 < .125`
  - `.75^16 ≈ .010`
  - `.8125^16 ≈ .036`

## Major defects and exact fixes

1. **(R) is undefined at the normalization level.**
   - (R) silently uses `NMSE_hard = E‖H∘a − PS‖² / E‖S‖²`, with `E‖S‖² = Kp`, and P taken from (G3).
   - **Fix:** state this definition explicitly.
   - If NMSE is instead centered (per-coordinate variance), the denominator becomes `Kp(1−p)`, so the prefactor is `2/(Kp(1−p)κ²)`.
   - If NMSE uses a best permutation, (R) remains an upper bound.

2. **The bound ε_D is correct but near-vacuous at the G1 edge.**
   - G1 gives `t < 1/(s√(K(K−1)))`, so `ε_D < 2√(K−1)/s ≈ 7.75` for K=16. The trivial bound is `2√K = 8`.
   - **Fix:** state that informative recovery needs t far below the G1 limit.
   - Optional sharpening, using the same proof: `‖D_* − DN‖_F ≤ s‖A−N‖_F`, which gives `ε_D' = 2s√(K(K−1))·t`. This is better whenever `s < √(K−1)`.

3. **A load-bearing premise is implicit.**
   - The step "`F0 < −e` ⇒ bad" requires `a_i ≥ 0`, so that `h ≥ 0`.
   - **Fix:** state the softplus or nonnegative-magnitude premise explicitly.
   - Neither σ, Gaussianity, nor matched β enters §3 or (R). The matched-noise contract matters only for estimating C, which is a clean-target risk. Say so, rather than presenting it as a premise of (R).

4. **Exact validity is not practical feasibility.** At p=.125, K=16:
   - `g = .109` and `a_K = 184`.
   - With ideal gates, `J/k > Kp`, so the slack must come from `u→1` (`V ≪ η²`) and tiny C.
   - **For NMSE_hard < .02 via (R)**, taking B=0:
     - (R) requires `ε_D < .1κ`.
     - This gives `t ≲ 8×10⁻⁴κ` and `e ≲ 4×10⁻⁴κ`.
     - G2 then needs `C ≲ 10⁻¹⁰κ⁴`.
   - **Order-of-magnitude feasibility (outside packet):** an unbiased SURE-type estimate of C has per-sample SD of roughly `σ²√(2d) ≈ .014` at `σ²=1/400`. Certifying `C_U ≈ 10⁻¹⁰` would then need about 10¹⁶ samples. The certificate is valid but practically unreachable. The sharpened ε_D' improves this only to roughly `10⁻⁹κ⁴/s²`.

5. **The .02 paired C2 contrast is unaddressed.**
   - (R) upper-bounds one arm only. A paired ≥.02 advantage needs either a lower bound on the comparator's NMSE or a paired-difference confidence interval, with decoder guard and world-level inference.
   - The note already says this correctly. Keep that statement, and do not cite (R) as C2 evidence.

## Next falsifiable calculation

For K=16 and each p in `I_p ⊃ {.125, .1875, .25}`:

1. Set J at its ideal-gate value, `Kp·log(1/r) + K(1−p)·log(1/(1−r))`.
2. Compute `C_max(p) = sup` over `(η, e, V, r)` of the C allowed by G1–G2.
3. Compute the minimal value of (R) over the same feasible set, with B=0, under both ε_D and ε_D'.

**Falsification test:**
- If `min (R) > .02` for all feasible parameters, the hard-NMSE route cannot certify C2 at all.
- If `C_max` is below the planned confidence half-width for C, the certificate is infeasible at the planned sample size, even though it is exactly valid.