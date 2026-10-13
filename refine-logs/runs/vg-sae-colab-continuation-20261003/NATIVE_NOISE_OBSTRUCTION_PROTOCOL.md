# Native finite-precision objective versus clean-risk obstruction

2026-10-03. Frozen candidate protocol before Colab execution. This is a new
bounded original-native follow-up, distinct from the noiseless restricted bias
profile, the normalized BSC cancellation limit, and closed training/benchmark
branches. No local numerical calculation or test has been performed for this
protocol. Independent theory review precedes execution.

The independent derivation is recorded in
[NATIVE_FINITE_BETA_NOISE_OBSTRUCTION_REVIEW.md](NATIVE_FINITE_BETA_NOISE_OBSTRUCTION_REVIEW.md).

## Research question and claim

At specified finite overprecision, can an explicit finite native VG-SAE model
have strictly lower population free energy than **every** native model with
small clean stochastic reconstruction risk?

The proposed certificate compares a native noise-fitting witness with a bound
for the entire clean-risk class. It addresses the actual objective-to-clean-risk
link, without assuming that an optimizer attains a global minimum. It is not a
dictionary-recovery gap, an unknown-dictionary consistency result, or a
generating-density estimator. No novelty claim follows from applying elementary
Hilbert-space and exponential-moment bounds.

## 1. Invariant model, objective and frozen cases

The data population is X=S+sigma Z in d=K=2, with independent source bits
S_j~Bernoulli(q), Z~N(0,I), and sigma=1/20. The generating dictionary is I and
generating bias is zero. The target clean risk is the full squared quantity

```text
C(theta) = E ||S - [b + D(B * a(X))]||^2,
B_j | X ~ Bernoulli(m_j(X)), conditionally independent.
epsilon = d*sigma^2/100 = 1/20000.
```

The objective, after omitting only its common Gaussian normalization at fixed
beta, is

```text
G_beta,gamma(theta)
 = beta/2 * E ||X - [b + D(B * a(X))]||^2
   + E sum_j KL(Bernoulli(m_j(X)) || Bernoulli(sigmoid(-gamma))).
```

This is the implemented native conditional-support objective with full variance
correction, entropy coefficient one, nonnegative softplus amplitudes, affine
sigmoid gates, centering bias and unit decoder columns. Use the learned-beta
branch with beta held fixed. Profiling beta, removing variance or entropy, or
changing entropy weight changes the theorem. Objective values are compared only
within a fixed beta/gamma case.

The full twelve-case panel is fixed before execution:

- q in {1/10, 3/10};
- beta in {400, 1600, 25600};
- gamma in {1, 16}.

Every case uses sigma=1/20, epsilon=1/20000, the same witness construction and
512-bit arithmetic. Matched beta=400 remains in the denominator even when the
sufficient bound is inconclusive. There are no seeds, data samples, quadrature,
optimizer steps, training updates or adaptive choices.

## 2. Explicit finite native witness

Fix the exact rational orthogonal decoder

```text
D = [[3/5, -4/5],
     [4/5,  3/5]],
c=10, L=32,
b=-c*D*1 = (2,-14),
t=D^T(X-b)=D^T X+c*1,
a(X)=softplus(t),
m_j(X)=sigmoid(L) for both coordinates.
```

The native encoder consumes X-b. Its finite tensors are therefore

```text
decoder.weight = D,
pre_bias = (2,-14),
amplitude_encoder.weight = D^T,
amplitude_encoder.bias = (0,0),
gate_encoder.weight = 0,
gate_encoder.bias = (32,32),
log_beta = log(beta).
```

The parameters describe the real-valued native architecture, with rational
matrix entries and a stated analytic log-beta value. Formal arithmetic claims
do not transfer automatically to float64-rounded tensors. The optional source
parity test is separate from the exact-rational certificate.

Let m=sigmoid(32) and delta=sigmoid(-32)=1-m. Orthogonality and the Bernoulli
second moment give the exact identity

```text
E ||X-b-D(B*a)||^2
 = delta E||t||^2 + m E||softplus(t)-t||^2.
```

No large-t approximation is used in this identity. In particular, replacing
delta by zero would discard a positive risk contribution.

## 3. Analytic upper bound on the witness objective

The first moment term is exactly

```text
T2 = E||t||^2
   = d*q + 2*c*q*(1^T D 1) + d*c^2 + d*sigma^2
   = 200 + 26*q + 1/200,
```

since 1^T D 1=6/5. For every real t,
`softplus(t)-t=softplus(-t)<=exp(-t)`. Gaussian moment generation and independent
source coordinates therefore give

```text
L2_bound = exp(-2*c+2*sigma^2)
           * sum_j product_i [(1-q)+q*exp(-2*D_ij)],
E||softplus(t)-t||^2 <= L2_bound.
```

The product uses decoder **columns**, because t=D^T X+c*1. With the fixed matrix
this is

```text
exp(-20+1/200) * [
 ((1-q)+q exp(-6/5))*((1-q)+q exp(-8/5))
 + ((1-q)+q exp(8/5))*((1-q)+q exp(-6/5))
].
```

For gamma>0, the Bernoulli KL as a function of m is convex on [0,1], so its
maximum is an endpoint value. Those values are softplus(-gamma) and
softplus(gamma); the latter is larger. Thus the finite witness satisfies

```text
G_dense <= UB
 = beta/2 * [delta*T2 + m*L2_bound] + d*softplus(gamma).
```

This may be conservative. An inconclusive result does not authorize replacing
the witness or optimizing c/L after seeing values.

## 4. Uniform lower bound for the entire clean-risk class

Let H=b+D(B*a(X)) be any native stochastic reconstruction with finite second
moment. Apply the reverse triangle inequality in the L2 space of the same joint
(S,Z,B) variables:

```text
sqrt(E||X-H||^2)
 >= sqrt(E||sigma Z||^2) - sqrt(E||S-H||^2).
```

If C(theta)<=epsilon<d*sigma^2, the right side is positive. Since normalized
Bernoulli KL is nonnegative,

```text
G(theta) >= LB
 = beta/2 * (sigma*sqrt(d)-sqrt(epsilon))^2
 = 81*beta/40000.
```

The lower bound does not require a correct dictionary, bounded bias,
orthogonality of the candidate model, independence between B and noise, or a
particular inference algorithm. It uses the declared common joint probability
space, fixed finite beta, full stochastic risk and normalized KL. This is why it
applies to the entire native clean-risk class rather than a selected competitor.

If an outward upper endpoint of UB is strictly below the exact rational LB, the
entire clean-risk class has higher objective than the explicit finite witness.
Consequently `inf_clean G > inf_all G`. No attainment assumption is necessary.
Every eta-optimal model with eta smaller than the certified positive gap must
have C>epsilon. Actual SGD convergence or its optimization gap is not inferred.

## 5. The good class is nonempty in the actual finite encoder family

Use D=I, b=0, a_j(X)=1 and

```text
A=1/(2*sigma^2)=200,
m_j(X)=sigmoid(A*(X_j-1/2)).
```

The constant amplitude uses zero amplitude weights and finite softplus bias
`log(exp(1)-1)`. Gate weights are 200I and biases (-100,-100). Gaussian symmetry
makes the active and inactive stochastic support-error terms equal, regardless
of q. The inequality sigmoid(u)<=exp(u) gives

```text
C <= d E exp(A*(sigma Z-1/2))
   = d exp(-A/2+A^2*sigma^2/2)
   = 2 exp(-50) < epsilon.
```

The same 512-bit outward computation verifies the final strict inequality. This
finite model consumes only X, not the unobserved S. Nonemptiness therefore does
not depend on an unattainable oracle support, infinite encoder limit or a
postulated clean-risk class with no feasible member.

## 6. Numerical certificate and artifacts

All scientific inputs enter as exact Python Fraction/fmpq values, never decimal
Python floats. Official python-flint Arb runs at 512 bits and one thread. Finite
outward endpoints are saved as exact rational strings. The sole success test is
strict `UB_upper < LB_exact`, together with the nonempty-class check. Equality,
interval overlap or failure to separate yields `bound_inconclusive`; it does not
prove absence of an objective obstruction. There is no tolerance or successful
row replacement. Python floats are used only for runtime accounting and the
optional noncertifying parity test.

The runner outputs under `COLAB_TASK_OUTPUT`:

- `native_noise_obstruction_results.json`: all twelve cases, all component
  enclosures, exact witness parameters, good-class proof data and source hashes;
- `native_noise_obstruction_summary.csv`: every case, separation status and exact
  upper/lower/gap endpoints;
- `native_noise_obstruction_report.md`: counts and interpretation limits;
- `native_noise_obstruction_runtime_receipt.json`: total process CPU/wall usage
  including imports and principal output serialization.

`VG_SAE_COLAB_CPU_ONLY=1` is required. A visible Colab GPU flag/device causes
rejection; no GPU library is imported by the runner. The root must use the
verified Colab CPU allocation. Existing results and all outputs use exclusive
creation, so failed attempts are preserved. There is no local output fallback.

The process has a 60 CPU-second and 120 wall-second limit, preserving any tighter
inherited CPU bound. Scientific work stops five seconds before those limits and
records interrupted/unstarted rows as `budget_unresolved`; all twelve IDs remain.
The resource limit and wall alarm stay active during serialization. Linux
process-age accounting includes startup with a conservative 0.05-second guard;
the fallback is disclosed. No memory cap is imposed on virtual address space.

Tests compare the moment formulas against four-state enumeration, verify the
rational orthogonal risk identity by independent mask enumeration, check the
factor of one half and strict separation, retain the matched-precision case,
and test runtime/output/timeout safeguards. An optional Torch test verifies
native free-energy parity on artificial inputs at float64 tolerance only.
The main analytic certificate uses no Torch or sampled inputs.

Execute only on the authorized Colab CPU runtime, from the repository root:

```text
python -m pytest tests/test_native_noise_obstruction.py -q
python -m scripts.certify_native_noise_obstruction
```

## 7. Scope and relation to the research goal

The prior dense-offset construction showed an unbounded learned-precision
direction. This protocol instead asks a fixed, finite-beta question and
compares against all models meeting a meaningful, nonempty clean-risk bound.
The distinction matters: a valid finite-beta counterexample need not rely on
an optimizer escaping to infinite precision.

Nevertheless the certificate supplies neither generating-q estimation nor a
correction. It cannot explain the current beta=400 states when its bound there
is inconclusive. It does not show that all low-objective dictionaries are wrong:
the teacher dictionary itself can be combined with noise-fitting dense offsets.
It is also not a finite-sample guarantee or a claim about the behavior of a
particular training run. The actual learned-beta process and the hard-between
repeat-selection law remain separate original-method obligations.

The population model, error threshold and overprecision cases are explicit
assumptions. Successful arithmetic certification is not a publication-novelty
claim, formal proof-assistant verification, or completion of the pre-LLM goal.

Arithmetic source contracts were checked against the official
[python-flint general concepts](https://python-flint.readthedocs.io/en/latest/general.html)
and [Arb documentation](https://python-flint.readthedocs.io/en/latest/arb.html).
Scientific assumptions and proof are project derivations, not claims attributed
to the numerical library.
