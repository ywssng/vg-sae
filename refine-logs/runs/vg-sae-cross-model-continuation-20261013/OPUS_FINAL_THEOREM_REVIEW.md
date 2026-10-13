# Verdict

**Note 1 (budget obstruction): correct as scoped.** I checked each step:
- dσ² = 1/25.
- Vψ ≥ 3·256 = 768.
- e_C ≥ √(6144/8192) = √0.75 > 4/5.
- Since every ψ ≥ −1/25, C_U > 19/25.
- G2 forces C_U < (67/16)·10⁻¹².

The contradiction holds for every array, every N_blocks and every p ∈ [1/16, 1/4]. Rejecting the "median of individual ψ" substitution is correct, because block means target Eψ = C.

**Note 2 (cumulant geometry): algebra correct. Acceptable as a conditional sufficient certificate.** I checked:
- κ₃ = p(1−p)(1−2p).
- Gaussian noise and offset do not change T_*.
- E²/κ₃² = 2K − 2ΣA³.
- W doubly stochastic, so the Birkhoff step holds.
- The triangle step through the polar factor.
- κ_L ≤ κ₃ and √K·|κ̂ − κ₃| ≤ ε_T, which gives the +2ε_T term.

I found no counterexample, no incorrect bound and no blocking defect.

# Non-blocking corrections

1. **Note 1, minimum block size.** State l ≥ 1, i.e. n ≥ N_blocks. Otherwise e_C is undefined and no certificate is issued.
2. **Note 1, budget boundary (optional).** The deterministic contradiction persists while e_C ≥ 1/25 + (67/16)·10⁻¹², i.e. up to l ≈ 6144·625 ≈ 3.84×10⁶.
   - Beyond that, passing requires the median to fall below C ≥ 0 by about e_C, which is a probability-≤δ event.
   - High-probability passing therefore needs l ≳ 6144/(4.19×10⁻¹²)² ≈ 3.5×10²⁶.
   - The note rightly does not claim 8192 is maximal.
3. **Note 2, missing link.** Make the permutation identity explicit. With P_{i,π(i)} = 1:
   ‖QP − D_*‖²_F = 2K − 2Σ_i A_{i,π(i)}.
   This is the step connecting Birkhoff to the distance.
4. **Note 2, hypotheses.** Q need only be orthogonal; the polar factor is one choice. Full rank of D is never used.
5. **Note 2, certificate floor.** Even at Q = D_*, the bound is at least 2ε_T/κ_L = 2√K·ε_T/(‖T̂‖ − ε_T).
   - A target radius r therefore requires ε_T ≤ r‖T̂‖/(2√K + r).
   - For K = 16 this is ≈ κ₃r/2.
6. **Note 2, invariance.** The guarantee holds only up to permutation, so downstream risk and contrast metrics must be permutation-invariant.

# Exact validity vs. feasibility vs. the 0.02 paired C2 contrast

- **Note 1** is an exact infeasibility result for one certificate formula at n ≤ 8192. It does not imply VG failure, and it says nothing about whether a 0.02 paired C2/C2b contrast exists.
- **Note 2** is an exact implication from a given ε_T to geometry error. It implies no affordable budget, coefficient risk, density identification, or 0.02 C2 contrast.
- κ₃ peaks at p = (3 − √3)/6 ≈ 0.211, which lies inside [1/16, 1/4]. So the non-injectivity warning applies on the witness range itself.

# Next falsifiable calculation

Bound the certificate floor 2ε_T/κ_L at n = 8192 with an explicit radius. That radius must cover unknown-mean centering, Gaussian tails and failure probability δ.

Back-of-envelope estimate (not a radius; ignores centering bias and δ):
- E‖T̂ − T_*‖² ≈ (E‖X − EX‖⁶ − ‖T_*‖²)/n.
- At p = 1/4 and K = 16: E(Σ(S_j − p)²)³ ≈ 27 + 6.75 + 0.19 ≈ 34.
- So RMS ε_T ≈ 0.065, against ‖T_*‖ = 0.375.
- This gives a floor ≈ 1.7 in Frobenius norm. The random-rotation scale is about √32 ≈ 5.7.
- For r = 0.5 one needs ε_T ≤ 0.022, which suggests roughly 9× more rows from the RMS term alone.

**Falsification criterion:** if the rigorous floor exceeds the geometry tolerance required by the downstream clean-risk bridge, the cumulant route is unaffordable at 8192 rows. That should be settled before any Colab allocation.