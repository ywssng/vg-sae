# Independent finite-noise derivation review

Date: 2026-10-03. Reviewer: independent code-audit agent, same-family/provisional. Scope: analytic review of `FINITE_NOISE_DERIVATION.md` and the finite-noise portions of `EXECUTION_PROTOCOL.md`. No local numerical computation, model evaluation, test, or experiment was performed. Implementation review is pending at this revision.

**Analytic verdict: coherent under the stated assumptions.** I found no mathematical blocker in the explicit variance compactness, one-sided finite-noise remainder, clipped-profile penalties, infinite-tail exclusion, or fixed-grid sufficient-threshold statement. This verdict does not certify an implementation, numerical result, novelty, or native VG-SAE density estimator.

## Reviewed bytes and provenance

| Input | SHA-256 |
|---|---|
| `FINITE_NOISE_DERIVATION.md` | `a9148f5c376d3282a6c4287bffb32de9be9880f647b55241d654910b8afaa77b` |
| `EXECUTION_PROTOCOL.md` | `baa2a0bfc06537d3c9565597d18ef0c0f189ed8c0e89b33941b1360378fadecf` |
| Imported `joint_global_sign_proof_20261002/PROOF_PACKAGE.md` | `4607582522616354d8e04278e53c1ff19ec2636e7039f4ca32540e6679c3193a` |
| Imported `researcher_constructive/PRELIMIT_VARIANCE_PROOF.md` | `3e5c4209865b921df0f43e688da6991cb6701d16af21908272d73de539a77f92` |

The imported population freeze references original proof hash `34f24ca48e7f59d117bda35403179fd17c6cbeaeca5e9f646ebb4bc4f5b78c75`. `research-import-20261002T131813Z/metadata/CURRENT_SOURCE_RECONCILIATION.json` explicitly maps that hash to delivered hash `460758...` with `editorial_copy=true`. The current review reads the delivered document; it does not claim byte identity with the original frozen proof. No evidence of transport corruption follows from this documented difference.

## Independent analytic checks

### 1. Explicit variance domain

For `d=2m`, the volume of a radius-`sqrt(t)` ball multiplied by the standard Gaussian density supremum is `t^m/(2^m m!)`. The four-center union bound therefore gives the stated condition `4 t^m/(2^m m!) <= 1/2` and distortion lower bound `eta=t/2`, after conditioning on the true state. The centers are fixed relative to each observation's Gaussian noise, which is why this step applies to the supplied codebook and does not transfer to arbitrary input-dependent amplitudes.

The expected log prior at true `q0` is `-2 H_2(q0)`, so the feasible generating-state comparison at variance ratio 1 yields `B_ref=-d/2-2H_2(q0)`. Although that state is unavailable to an estimator, the pointwise inference supremum exceeds every candidate point mass, so using it as a lower comparison is valid.

The upper envelope derivative is `U'(v)=(eta-dv)/(2v^2)`. Thus the strict lower-end exclusion at a rational `v_minus<eta/d` propagates to every smaller positive variance. A rational `v_plus >= exp(1+4H_2(q0)/d)` satisfies `-(d/2)log(v_plus)<=B_ref`; the additional strictly negative distortion term excludes the upper tail, including equality of the exponential comparison. These exclusions do not require a previously estimated limiting variance.

### 2. Entropy and wrong-group remainder

The product reconstruction variance is exactly `sum_j m_j(1-m_j)` because the two columns have unit norm. Comparing the optimizing product law with the generating point mass bounds its expected reconstruction energy by `sigma^2(d+2v_plus C)`. With `t_j=min(m_j,1-m_j) <= 2m_j(1-m_j)`, two applications of entropy concavity give `E H(Q*)<=2H_2(A)` when `A<=1/2`.

Under `h=(lambda sigma/2)^2<=3/4`, distances between different declared groups are at least 1. On `||Z||<=1/(4sigma)`, any cross-group squared-error difference is at least `delta^2-delta/2>=1/2`. The log-prior advantage is at most `C`, hence `1/(4v_plus sigma^2)>C` is sufficient to exclude a wrong group. On the complement, `C+||Z||^2/(2v_minus)` bounds any improvement over the generating state and therefore over the correct-group hard optimum.

For `W=||Z||^2~chi-square_(2m)` and `x=1/(32sigma^2)`, the threshold is `W>2x`. The exact formulas are

```
P(W>2x) = exp(-x) sum_(j=0)^(m-1) x^j/j!
E[W 1_(W>2x)] = d exp(-x) sum_(j=0)^m x^j/j!
```

They reproduce the documented coefficient `d/(2v_minus)`. Exact coupling makes the correct-group hard objective equal to `J_lambda`, so `E_sigma>=H_sigma>=J_lambda` and the remainder is one-sided. A factor of two on `epsilon_sigma` is unnecessary in the final gap test. An approximate lambda path would require the omitted coupling remainder and is outside scope.

### 3. Restricted profiles and clipping

For a fixed threshold, optimizing its completed-label objective over `q` in any closed subinterval and over `v>0` produces the clipped cross entropy and `v=M/d`. Interchanging suprema is valid, and every unrestricted hard assignment for a sparse `q<1/2` is a threshold above `lambda/2`.

For upper exclusion, when `u<q0`, the unrestricted fitted activity is exactly `u`, so the profile loss is `2 KL(u||q0)`. Its derivative in `u` is negative for `u<q0`, making an upper enclosure endpoint `u_U` a valid penalty lower bound. Returning zero when the enclosure crosses `q0` is conservative.

For lower exclusion, the constrained fit is `r` because `u>=r`. The difference from the full profile is

```
f(u) = 2[u log(clip_I(u)/r)
         +(1-u) log((1-clip_I(u))/(1-r))].
```

For `r<=u<=beta` it equals `2KL(u||r)`; for `u>=beta` it is affine with positive slope `2log[beta(1-r)/(r(1-beta))]`. These pieces meet continuously with the documented derivative. Consequently `f(max(r,u_L))` is a valid lower bound, including upper clipping. Replacing an interval-rounding negative lower endpoint by zero is safe because the exact penalty is nonnegative, not because floating output is trusted.

### 4. Infinite threshold tail and complete domain

Every restricted fitted `q` is at least `r`. Also `v(c)` is increasing on `c>=lambda/2` and bounded above by `1+b lambda^2/d`. The derivative formula therefore gives `P_J'(c)<=0` beyond the stated common cutoff. Rounding that cutoff upward preserves this conclusion, and the restricted profile value at the rounded cutoff bounds its whole infinite tail.

The implementation must use the **complete terminal partition**, including previously rejected leaves. A leaf that was dominated for the unrestricted optimum can still contain the best constrained parameter, so original active near-optimum leaves alone cannot bound the excluded set. Source-level verification of this requirement remains necessary.

### 5. Witness comparison, zero crossover, and threshold quantifiers

A directly verified feasible limiting value `L` lower-bounds the prelimit global optimum because `E_sigma>=J_lambda` holds at every positive `v`, including a witness outside the explicit compact variance domain. Only the upper comparison at prelimit global optimizers needs that compact domain. This resolves a possible ambiguity without requiring a new variance restriction on the witness.

At `lambda=0`, `J_0` separates exactly, its global maximum is `(r,1)`, and upper exclusion has gap `2KL(r||q0)`. The lower excluded interval contains that limiting optimum, so no strict lower-bracket certificate follows. The document correctly restricts this anchor to the upper side.

As sigma decreases on the exact fixed-lambda path, `h` and `A` decrease, the prior-separation inequality strengthens, and the positive Gaussian tail expectation decreases. Binary entropy is increasing on the allowed `A<=1/2` range. Thus any successful bound at `sigma0` remains sufficient for every `0<sigma<=sigma0`; the result is a continuous one-sided threshold guarantee, not just a list of tested sigma values. A dyadic replacement for extreme tails must remain a valid positive upper bound. The largest certified *grid* value must not be presented as the largest mathematically admissible noise.

## Implementation and reporting obligations

- Bind evaluated specifications, terminal partitions, source hashes, and completed audit receipts. Skip finite-noise evaluation for incompletely verified cells.
- Check all compactness, geometry, entropy, and strict prior-separation prerequisites by outward interval inequalities.
- Verify a feasible objective lower witness directly; do not infer it from a reported optimum or parameter enclosure.
- Bound both constrained profiles across the full terminal partition and their common infinite tail. Preserve clipping distinctions between count `u` and fitted `q`.
- Report nonpositive and inconclusive margins, including sigma `1/20`, and keep upper/lower certifications distinct. At lambda zero, the lower side is inapplicable.
- Preserve old primary resolution, old full-tree audit coverage, new terminal audit coverage, and finite-noise exclusion as separate evidence fields.
- The protocol's separate native finite-beta witness job is not reviewed here and still needs its own mathematical/source review before execution.

The stated Colab-only resource protocol, no-restart rule for imported optimizers, fixed panel, and separate result locations are consistent with this analytic task. No approval for additional data, training, confirmation access, or LLM activations is implied.

## Initial source review

Reviewed `scripts/certify_finite_noise.py` at SHA-256 `deeb2099ea45d66f26fd524168b48dbd1888a36cb51a37a4b5f17f02123ae48b` and `tests/test_finite_noise_certificate.py` at `ff5a0b3a37284135dace883fe9bb60a4d011fd9d4e430f4aa9815ff830d936a4`. The terminal verifier was read only to verify the output schema and constrained-profile connection; its separate independent source review is not duplicated here.

The source correctly implements the small-ball coefficient with exact rational bisection; the improved feasible-reference variance upper bound; the strict lower-cutoff test; `h<=3/4`; the entropy-domain and prior-margin prerequisites; and the two even-dimensional chi-square sums. Scientific exclusions use strict exact-rational comparisons. It skips cells without verified status and prevents a lambda-zero lower-bracket claim. The finite sigma grid is fixed in descending order, so the first passing value is the largest tested sufficient value.

The extreme-tail implementation checks the Arb magnitude against a positive dyadic floor **before** converting its endpoint to a Python rational. This avoids materializing denominators proportional to an enormous negative exponential exponent. Its replacement is an upper bound, and both entropy and tail rounding are conservative. Static API use matches the Arb methods already present in the independent/archived source; actual runtime compatibility is still a Colab-test obligation.

**Input-contract findings requiring correction before this source version is used:**

1. The main input roster check compared only the set of `(q0,d,lambda)` triples. It accepted changed candidate interval `I`, changed/duplicate cell IDs, or a different order despite the fixed 20-specification contract. Require the full original ordered specification list, including `I=[1/100,2/5]` and each exact ID.
2. The input consumer accepted any JSON with cell-level `status="verified"`; computing its hash after reading only records which unchecked input was used. Bind the consumed terminal audit to its output seal and expected verifier version/source and historical run-seal provenance. Permit a valid verified prefix when the overall audit is partial; do not require all 20 cells to pass merely to use an individually complete cell. Seal consistency is provenance checking, not a substitute for the independent mathematical validation performed by the producer.

No further mathematical source error was found. The d=2 radial-tail fixture, strict-margin fixture, lambda-zero separation, unsupported-input rejection, and tiny-tail fixture are relevant tests. None was executed by this reviewer. This initial implementation assessment remains pending the input-contract corrections and remote test results.

## Final static source verdict after input-contract corrections

**Accepted for the specified bounded Colab evaluation, subject to focused remote tests.** The two findings above are corrected in `scripts/certify_finite_noise.py` SHA-256 `27dbbd74a0e9bdd76c583c1ed6c570847017b478fa5da4f2c93b965835174fcb`, with tests at `38abef31ec0c047ce8ed07828ce987a3e6686bb5bd85ce71fb625e2621e30e48`.

`fixed_panel()` now preserves the entire original ordered specification, including interval and IDs. `read_verified_terminal()` verifies the sibling audit/prefix seal, exact verifier version and caller-pinned source hash, original run/source-manifest identities, completed-row count, and journal content. Individually verified rows from a partial audit remain usable. The result records the consumed audit, seal, and verifier hashes. The mathematical calculation is unchanged from the reviewed version. This is static same-family/provisional review; no tests or numerical outcomes were run or certified by this reviewer.
