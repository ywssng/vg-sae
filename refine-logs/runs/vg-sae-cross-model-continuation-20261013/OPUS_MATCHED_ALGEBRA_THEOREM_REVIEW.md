# Verdict

**(A)–(C) are mathematically valid for the two frozen cases.** I found no counterexample and no incorrect bound. The corrections below concern presentation and interpretation, not validity.

Re-derived facts:
- The stochastic gap is `m·g_t`. Here `g_t = y²r(2−r)`, with `y = x−1` and `r ∈ (0,t)`.
- The posterior identity holds because `βσ² = 1`.
- `μ2 = erf(1/√2) − √(2/π)e^{−1/2} ≈ 0.19875`.
- The tail moment is `E[Z²1{Z>α}] = αφ(α) + Q(α)`.
- `a_t ≤ max(1,x)` holds on `x > h > 0`.
- `dG_t/dt|₀ = −βσ²Kq(1−e^{−1})`.

Expected closed-form values:

| Bound | q = 1/10 | q = 3/10 |
|---|---|---|
| (A) | ≈ 0.010498 | ≈ 0.031495 |
| (B) | ≈ +4.872×10⁻⁵ | ≈ +4.872×10⁻⁵ |
| (C) | ≈ 6.25×10⁻⁴ | ≈ 6.25×10⁻⁴ |

- The common inputs are `c ≈ 0.313143` and `c(2−c) ≈ 0.528229`.
- The tail terms in (B) are about 2×10⁻²³ and 1×10⁻²³. They are negligible.
- In (C), `t²σ² = 6.25×10⁻⁴` dominates; the other terms are about 10⁻²².

# Major defects and exact fixes

1. **Hidden specialization `βσ² = 1`.** The posterior gate, `h`, (A), and the derivative all use `β = 1/σ²` silently.
   - Fix: state (A) as `G₀−G_t ≥ (βσ²Kq/2)·c(2−c)·μ2`.
   - Write `h = 1/2 + γ/β`.
   - Note that the gates are the exact posterior only when `β = σ⁻²`.
   - The frozen numbers are unaffected.

2. **Effect size is not reported.** The margin in (B) is positive but tiny, about 4.9×10⁻⁵. The true gap is roughly `E[r²]σ² ≈ 2.5×10⁻⁴`.
   - Fix: state the magnitude, not only the sign.
   - Add the sandwich `4.87×10⁻⁵ ≲ NMSE_t − NMSE₀ ≤ t²σ² + O(10⁻²²)`.

3. **The C2 scale needs a stronger statement.**
   - `NMSE₀ ≈ 10⁻²²` and `NMSE_t ≤ (C) ≈ 6.25×10⁻⁴`.
   - So `|NMSE_t − NMSE₀| < 6.3×10⁻⁴`.
   - This pair therefore **cannot exhibit a 0.02 paired contrast in either direction**. The certified degradation is under about 3% of 0.02.
   - Fix: add this sentence. Keep "(C) < 1/50" as a scale remark, not a success criterion; the note already does so correctly.

4. **Exact validity versus practical feasibility.**
   - The certificate is exact and needs no sampling.
   - Empirically, the paired per-active-coordinate difference `≈ r²σ²Z²` has a signal-to-noise ratio of roughly 0.7. Its sign would be detectable with modest paired samples (rough heuristic: tens to hundreds of active coordinates).
   - Its magnitude is irrelevant to C2b's 0.02 threshold.
   - Do not phrase the result as "practically significant degradation."

5. **Minor wording.** "Derivative lies in (0,1)" should read "`∂ₓa_t = t·σ(·) ∈ (0,t) ⊂ (0,1)`". The strict betweenness claim relies on this.

The mechanism is unsurprising: with gates fixed, G pointwise prefers `a → x`. The note correctly claims nothing beyond this pair.

# Next falsifiable calculation

Use Arb on the closed forms only, with no quadrature. For each q, enclose:

- `c`;
- `μ2`;
- `α` and `δ`, and check `0 < h < 1−σ`;
- the right-hand sides of (A), (B) and (C).

Pass criteria:
- (B)'s lower endpoint exceeds 0. Its enclosure should contain about 4.8723×10⁻⁵.
- (C)'s upper endpoint is below 1/50. Its enclosure should contain about 6.25×10⁻⁴.
- (A)'s enclosure is near 0.0105 (q = 1/10) and 0.0315 (q = 3/10).

Any enclosure outside these values, or any (B) enclosure that straddles 0, indicates an implementation error rather than a failure of the algebra. Also check that `dG/dt|₀` encloses `−Kq(1−e⁻¹)`: about −0.1264 for q = 1/10 and −0.3793 for q = 3/10.