# Explicit finite-noise exclusion for the coupled population limit

2026-10-03. ARIS formula-derivation continuation. This document concerns the
normalized supplied two-atom model in the imported joint-global-sign proof,
not the native input-dependent-amplitude VG-SAE. No empirical result is assumed.

## Target and status

**Target:** turn the existing eventual small-noise optimizer statement into a
checkable sufficient positive-noise threshold, separately for q < q0 and
q > r = q0(1-q0). A failed sufficient test is inconclusive.

**Status:** COHERENT AS STATED, pending independent exact-source review and
Colab numerical evaluation. Neither novelty nor practical usefulness is claimed.

## Invariant object and assumptions

The invariant object is the globally optimized population product-Bernoulli
ELBO E_sigma(q,v), with its common Gaussian normalization removed. Use the
precise assumptions in the imported
`vg-sae-generative-reassessment-20261002/joint_global_sign_proof_20261002/PROOF_PACKAGE.md`:
K=2, supplied unit columns, iid Bernoulli(q0) latent bits, unit exogenous
amplitudes, known zero offset, Gaussian noise, and jointly fitted scalar
variance v sigma^2. Candidate q belongs to I=[alpha,beta] contained in (0,1/2)
with [r,q0] contained in I. Both inference and parameter optimization are
global; all expectations are population expectations.

For this finite-noise result set h=(lambda sigma/2)^2 **exactly**. The dictionary
columns are (sqrt(1-h),sqrt(h),0,...) and (-sqrt(1-h),sqrt(h),0,...).
The prescribed lambda is fixed and finite. We require h<=3/4, so every pair of
codewords in different groups {00,11}, {10}, {01} is separated by at least one.
The computation covers even d=2 or 10, the existing fixed table dimensions.
All lambda, sigma, variance ratios and thresholds are dimensionless relative
to unit active amplitude. This is no physical-temperature interpretation.

## Derivation map

1. A four-center Gaussian distortion floor confines every prelimit variance
   optimum without assuming a limiting optimum or a fitted variance cap.
2. The imported entropy and wrong-group bounds give a one-sided finite-noise
   envelope on that whole compact domain.
3. The threshold profile and exact Bernoulli cross-entropy penalties bound the
   entire excluded q intervals, not just stationary roots.
4. A feasible limiting value exceeding the excluded upper bound plus the
   one-sided remainder excludes every prelimit global optimum there.

## 1. Explicit common variance domain

Write d=2m. Choose a positive rational t satisfying

    4 t^m / (2^m m!) <= 1/2,
    eta = t/2.

Here t is the squared radius of a ball. The standard Gaussian density upper
bound and union bound over at most four centers give
E min_t ||Z-c_t||^2 >= eta. This is uniform over all codeword locations.
Since prior plus entropy is minus a KL divergence,

    E_sigma(q,v) <= U(v) = -(d/2) log v - eta/(2v).

The generating-state point mass at (q,v)=(q0,1) is feasible as an objective
comparison and gives the lower reference

    B_ref = -d/2 - 2 H_2(q0).

It is not an oracle estimator. A rational v_plus no smaller than
exp(1+4 H_2(q0)/d) excludes every v>v_plus because U(v)<B_ref there.
Choose rational 0<v_minus<eta/d with U(v_minus)<B_ref. Since U is increasing
on (0,eta/d), this excludes every v<=v_minus. Thus **all** prelimit global
maximizers lie in V=[v_minus,v_plus]. The code uses outward interval bounds
for both strict comparisons. The improved reference is a permitted feasible
comparison, not a variance range inferred from previously observed optima.

## 2. One-sided finite-noise remainder

Set C=-2 log alpha and A=sigma^2(d+2 v_plus C). The supplied-column product
variance identity and comparison with the generating state yield

    0 <= E_sigma-H_sigma <= 2 H_2(A),   provided A<=1/2,

uniformly on I x V. H_sigma is the all-support hard objective. This is the
same entropy bound as the imported prelimit proof, with the explicit compact
constants above. No softplus/native amplitudes are substituted into it.

If ||Z||<=1/(4 sigma), a wrong-group codeword has a squared-error disadvantage
at least 1/2. It cannot beat the generating support if

    1/(4 v_plus sigma^2) > C.

Outside that event the excess hard objective is at most
C+||Z||^2/(2 v_minus). Write x=1/(32 sigma^2). The exact even-dimensional
chi-square tail identities therefore give the remainder bound

    T_sigma = exp(-x) [ C sum_{j=0}^{m-1} x^j/j!
                       + d/(2 v_minus) sum_{j=0}^{m} x^j/j! ],
    epsilon_sigma = 2 H_2(A) + T_sigma.

The correct-group objective is **exactly** J_lambda when
h=(lambda sigma/2)^2, so

    0 <= E_sigma(q,v)-J_lambda(q,v) <= epsilon_sigma.

An approximate lambda path would require an additional two-sided coupling
term and is outside this implementation. Extreme Gaussian tails may be
replaced by a rigorously larger positive dyadic bound to avoid serializing
astronomical rational denominators; they are never rounded down to zero.

## 3. Certifying the whole excluded parameter set

Use the imported threshold notation u(c), M(c), v(c)=M(c)/d and

    K_I(u)=2 u log clip_I(u)+2(1-u)log(1-clip_I(u)),
    P_I(c)=K_I(u(c))-(d/2)log v(c)-d/2.

For any closed subinterval J of I, interchanging suprema in the same
completed-label objective gives

    sup_{q in J, v>0} J_lambda(q,v)
      = sup_{c>=lambda/2} P_J(c).

In particular this upper-bounds the excluded set with v in V. A verified
finite partition of the threshold domain, including the original rejected
leaves, is needed. Keeping only the final near-optimum leaves would omit
potential constrained optima.

On a leaf let U bound P_I and let [u_L,u_U] enclose the count update. We always
have u>=r>=alpha. For the upper excluded interval J_up=[q0,beta], a safe
penalty lower bound is

    p_up = 2 KL(Bern(u_U)||Bern(q0)) if u_U<q0, else 0.

For J_low=[alpha,r], define z=max(r,u_L), s=min(z,beta). A safe lower bound is

    p_low = 2 [ z log(s/r)+(1-z)log((1-s)/(1-r)) ].

The first penalty decreases to zero as u increases to q0. The second is
nondecreasing for u>=r: its derivative is
2 log[clip_I(u)(1-r)/(r(1-clip_I(u)))] >= 0.
Thus P_J<=U-p_J throughout each leaf. All transcendental penalties use
interval lower bounds; negative rounding spill is conservatively replaced by
zero only because the exact penalty is analytically nonnegative.

The original finite tail cutoff C_tail is also valid for both restricted
profiles. Their clipped q is >=r and v(c)<=v_max, so beyond
lambda/2 + (2 v_max/lambda) log[(1-r)/r] their derivatives are nonpositive.
Their value at the outward-rounded cutoff bounds the complete infinite tail.
The maximum over every terminal-partition leaf and that tail is U_up or U_low.

A feasible point of the original profile supplies a lower value L. Independent
interval evaluation verifies it; it is not inferred from a reported optimum.
The terminal verifier imports no original numerical engine and binds complete
final certificates to their creation/precision-update events.

At lambda=0 use the exact analytic profile: the unrestricted maximum is at
(r,1), and the upper excluded optimum is (q0,1), with gap 2 KL(r||q0).
There is no lower strict-bracket certificate at this anchor.

## 4. Finite-noise exclusion and quantifiers

If L > U_up+epsilon_sigma, every global prelimit optimizer has q<q0.
If lambda>0 and L > U_low+epsilon_sigma, every such optimizer has q>r.
Both inequalities certify the strict bracket. Indeed E_sigma at the feasible
limiting witness is >=J_lambda>=L, while its supremum on either excluded
set is <=U+epsilon_sigma. The entire excluded set is separated strictly.

For a successful sigma0, all 0<sigma<=sigma0 with the same exact lambda path
also succeed: the entropy bound decreases, the Gaussian tail event shrinks,
and every geometry/small-noise prerequisite remains true. This is a sufficient
threshold, not the largest true admissible noise.

## Evaluation contract and boundaries

Preserve the original 20 specifications: q0=1/10 or 3/10, d=2 or 10, and
lambda=0,1/2,2,8,16. New results retain the old statuses rather than overwrite
them. For each fully verified cell evaluate the fixed grid sigma=1/20 and
10^(-k), k=2,...,20. Report all evaluated bounds and the largest certified grid
value separately for the two sides. In particular report the sufficient test
at the original noise .05, including failure or inapplicability. Do not replace
failed cells or select new lambda values after seeing margins.

These are deterministic bounds, not samples, fitted models, new development
worlds, confidence coverage, or LLM experiments. Global-inference, known-D,
unit-amplitude and population assumptions remain. Numerical inconclusiveness
does not refute the existing analytic theorem. An extremely small threshold
does not establish practical usefulness or native VG density recovery.

## Open risks

- The existing finite partition can be too coarse to give a positive excluded
  objective margin even when the analytic theorem guarantees one.
- The generic compactness/entropy constants can be very conservative.
- Independent numerical review is same-family and uses the same official Arb
  elementary library; it is not proof-assistant or cross-library verification.
- The native finite-beta recovery gap and hard-between selection law remain
  separate open obligations. This package cannot clear BR5, W2 or C2.
