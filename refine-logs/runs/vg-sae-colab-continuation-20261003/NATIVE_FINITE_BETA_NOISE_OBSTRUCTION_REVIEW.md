# Finite-precision noise-copying obstruction for the native VG objective

Date: 2026-10-03. Independent analytic review using `physics-guide-solver`
and `physicist-analyst`. No local computation, experiment, numerical test,
training, sampling, or model forward was run. The concurrently executing
population verifier and its frozen sources were not modified.

**Verdict:** the proposed finite-parameter witness and uniform clean-risk
lower bound are valid under the explicit native-model contract below. They
give a substantive finite-beta, positive-noise obstruction to a universal
objective-to-small-clean-risk implication. The comparison covers every model
in the declared clean-risk class, which is demonstrably nonempty. It does
not prove a dictionary-recovery gap failure, an error of the native density
estimator, or a failure at correctly calibrated noise precision.

This is a same-family semantic review, not a numerical execution receipt or
a novelty determination. The proposed fixed 12-row Colab calculation remains
separate from this analytic review.

## 1. Contract and invariant objective

The generator is the original square binary-source model

\[
Y=S\in\{0,1\}^d,\qquad S_i\overset{\mathrm{iid}}\sim
\mathrm{Bernoulli}(q),\qquad X=Y+\sigma Z,\qquad Z\sim N(0,I_d),
\]

with `d=K=2`, independent `S,Z`, supplied positive `sigma`, and the same noisy
input `X` available to every encoder. No candidate observes `Y`, `S`, or the
realized noise at inference time.

Use the native affine sigmoid-gate/softplus-amplitude architecture, enabled
centering bias, full Bernoulli reconstruction-variance term, entropy coefficient
one, and the normalized Bernoulli prior
\(\pi=\operatorname{sigmoid}(-\gamma)\), with fixed \(\gamma>0\).
The precision \(\beta\in(0,\infty)\) is supplied and held fixed for every
competing model. In the implementation this corresponds to the normalized
`learned` branch with `log_beta` frozen, not the profiled-beta branch.

For a native model \(\theta\), let
\(A_\theta(X,B)=b+D[B\odot a(X)]\), where the mask bits are generated with
the model's conditional probabilities given `X`. Write

\[
R_X(\theta)=\mathbb E\|X-A_\theta(X,B)\|^2,\qquad
C(\theta)=\mathbb E\|Y-A_\theta(X,B)\|^2,
\]
\[
J_\gamma(\theta)=\mathbb E\sum_j
\mathrm{KL}(\mathrm{Bern}(m_j(X))\|\mathrm{Bern}(\pi))\ge0.
\]

All risks are full sums over coordinates and include the Bernoulli draw, not
mean-only reconstruction errors. After subtracting the common Gaussian
normalizer \(N_\beta=-(d/2)\log[\beta/(2\pi)]\), the population objective is

\[
\bar F_\beta(\theta):=F_\beta(\theta)-N_\beta
=\frac\beta2R_X(\theta)+J_\gamma(\theta).\tag{1}
\]

The signs, half factor, prior normalizer and centering convention in (1) were
checked against `src/sae_model.py`: the encoder receives `X-pre_bias`, the
decoder adds `pre_bias`, and the standard prior-minus-entropy term is exactly
the stated nonnegative KL. Removing the variance term, increasing entropy
weight above one, fitting beta separately for each competitor, or changing
the noise model would require another argument.

Units are consistent: `C`, `R_X`, and `epsilon` have squared observation units;
beta has inverse squared observation units; the objective and gate logits are
dimensionless in nats. The dimensionless precision calibration variable is
\(\kappa=\beta\sigma^2\).

## 2. Exact finite native dense construction

Take any square orthogonal \(D\), a finite positive offset `c`, and a finite
gate logit `L`. Set

\[
b=-cD\mathbf1,\qquad
t=D^T(X-b)=D^TX+c\mathbf1,\qquad
a_j(X)=\operatorname{softplus}(t_j),
\]
\[
m=\operatorname{sigmoid}(L),\qquad \delta=1-m.
\]

This is represented by finite native affine parameters:

```text
decoder.weight           = D
pre_bias                 = -c D 1
amplitude_encoder.weight = D^T
amplitude_encoder.bias   = 0
gate_encoder.weight      = 0
gate_encoder.bias        = L 1
log_beta                 = log(beta), held fixed
```

The decoder has unit columns. There is no amplitude-closure value, infinite
logit, arbitrary per-input code lookup, or teacher-label-dependent encoding.
The generating `q` enters the population expectation, not the encoder.

Let \(\ell(t)=\operatorname{softplus}(t)-t=\log(1+e^{-t})\).
Orthogonality gives, exactly for each input,

\[
\begin{aligned}
\mathbb E_B\|X-b-D(B\odot a)\|^2
&=\sum_j\mathbb E_B(t_j-B_ja_j)^2\\
&=\sum_j\{(1-m)t_j^2+m(t_j-a_j)^2\}\\
&=\delta\|t\|^2+m\|\ell(t)\|^2.\tag{2}
\end{aligned}
\]

No cross term is missing. The formula is an expectation over the full
stochastic decoder, including its variance correction. The mathematical
sigmoid has `0<m<1`, so the construction uses strictly finite parameters.

## 3. Closed-form population upper bound

The exact second moment is

\[
T_2=\mathbb E\|t\|^2
=dq+2cq\,\mathbf1^TD\mathbf1+dc^2+d\sigma^2.\tag{3}
\]

The equivalent cross term initially appears as
\(\mathbf1^TD^T\mathbf1\); it equals the displayed scalar by transposition.
There is no missing covariance term because \(\|D^TX\|=\|X\|\).

For every real `t`, \(0\le\ell(t)\le e^{-t}\), so

\[
L_2:=\mathbb E\|\ell(t)\|^2
\le L_2^{\mathrm{up}}
=e^{-2c+2\sigma^2}
\sum_{j=1}^d\prod_{i=1}^d
\{(1-q)+q e^{-2D_{ij}}\}.\tag{4}
\]

Indeed, \(t_j=c+\sum_iD_{ij}S_i+\sigma(D^TZ)_j\), each transformed noise
coordinate is standard normal, and its exponential moment contributes
\(e^{2\sigma^2}\). Independence of the source bits gives the product over
`i`. The matrix entries must be read down each **column**, because the
encoder uses `D^T`. Independence between transformed noise coordinates is
not needed to sum the expectations in (4).

For positive gamma, convexity of Bernoulli KL, or its direct expression, gives

\[
\mathrm{KL}(m\|\pi)
=m\gamma+\operatorname{softplus}(-\gamma)-h_2(m)
\le\operatorname{softplus}(\gamma).
\]

Consequently the finite constructed model satisfies

\[
\boxed{\bar F_\beta(\theta_{\mathrm{dense}})
\le U_{\mathrm{dense}}
:=\frac\beta2\{\delta T_2+mL_2^{\mathrm{up}}\}
+d\operatorname{softplus}(\gamma).}\tag{5}
\]

This bound retains the normalized prior cost. Dropping that cost would
spuriously make the correctly calibrated small-beta cases look adverse.

## 4. A lower bound for every sufficiently clean native model

Take the same joint random variables `(S,Z,B)` in both risks, for any native
model. Correlation between the model output and the observation noise is
allowed. The reverse triangle inequality in the Hilbert space of
square-integrable random vectors gives

\[
\sqrt{R_X(\theta)}
=\|(Y-A_\theta)+\sigma Z\|_{L^2}
\ge\sigma\sqrt d-\sqrt{C(\theta)}.
\]

If \(C(\theta)\le\varepsilon<d\sigma^2\), the right side is nonnegative,
and squaring preserves the inequality. Equation (1) and `KL >= 0` yield

\[
\boxed{\bar F_\beta(\theta)
\ge L_{\mathrm{clean}}
:=\frac\beta2(\sigma\sqrt d-\sqrt\varepsilon)^2
\quad\text{whenever }C(\theta)\le\varepsilon.}\tag{6}
\]

This comparison does not assume a particular recovering dictionary, gate,
amplitude, bias, or optimization path. It covers the entire declared native
class with the given clean-risk property. It does not require a divergence
calculation, independence of the residual and noise, or access to clean data
by the encoder. An invalid replacement would be `R_X=C+d sigma^2`, which
generally fails for input-dependent reconstructions.

## 5. The clean comparator class is nonempty with finite native weights

For the same noisy information, set `D=I`, `b=0`, constant amplitude `a_j=1`,
and

\[
m_j(X)=\operatorname{sigmoid}\{A(X_j-1/2)\},\qquad
A=\frac1{2\sigma^2}.
\]

This is a native finite affine model: amplitude weights are zero and their
bias is \(\log(e-1)\), while gate weights are `A I` and biases are `-A/2`.
The true source is binary, and the decoded Bernoulli output is binary too.
Thus a squared clean-coordinate error is exactly a gate mistake. Gaussian
symmetry and \(1-\operatorname{sigmoid}(x)=\operatorname{sigmoid}(-x)\) give

\[
\begin{aligned}
\Pr(B_j=1\mid S_j=0)
&=\mathbb E\operatorname{sigmoid}\{A(\sigma Z-1/2)\},\\
\Pr(B_j=0\mid S_j=1)
&=\mathbb E\operatorname{sigmoid}\{A(\sigma Z-1/2)\}.
\end{aligned}
\]

Their mixture therefore does not depend on `q`. Since
\(\operatorname{sigmoid}(x)\le e^x\),

\[
C(\theta_{\mathrm{clean}})
\le d\exp\{-A/2+A^2\sigma^2/2\}
=d\exp\{-1/(8\sigma^2)\}.\tag{7}
\]

At \(\sigma=1/20\), `A=200` and the bound is \(2e^{-50}\).
The proposed tolerance is \(\varepsilon=1/20000\). The strict inequality
\(2e^{-50}<1/20000\) follows without numerical evaluation, for example from
the fourth-order term of the positive exponential series:
\(e^{50}>50^4/4!>40000\). Hence the recovering comparator class is genuinely
nonempty at the prescribed positive noise level.

The bound is a population risk statement. It does not claim exactly zero
support error in overlapping Gaussian distributions.

## 6. Precise obstruction and approximate-optimization consequence

Let \(\Theta_\varepsilon=\{\theta:C(\theta)\le\varepsilon\}\).
If the directly checkable difference

\[
\Delta=L_{\mathrm{clean}}-U_{\mathrm{dense}}>0,\tag{8}
\]

then

\[
\inf_{\theta\in\Theta}\bar F_\beta(\theta)
\le U_{\mathrm{dense}}<L_{\mathrm{clean}}
\le\inf_{\theta\in\Theta_\varepsilon}\bar F_\beta(\theta).
\]

This avoids assuming that a global minimizer exists in the noncompact neural
parameter space. If one exists, its clean stochastic risk exceeds epsilon.
More generally, an `eta`-optimal model with `eta<Delta` cannot belong to
\(\Theta_\varepsilon\): its objective is at most
\(\inf_\Theta\bar F_\beta+\eta\le U_{\mathrm{dense}}+\eta<L_{\mathrm{clean}}\).
The statement concerns objective values, not the ability of training to reach
such values.

This is stronger than comparing a dense solution against one arbitrarily
chosen recovery checkpoint. It remains narrower than ruling out every
meaning of feature recovery.

## 7. Exact prespecified substitutions and interpretation of unresolved rows

For

\[
D=\begin{pmatrix}3/5&-4/5\\4/5&3/5\end{pmatrix},\quad
c=10,\quad L=32,\quad\sigma=1/20,\quad d=2,
\]

equations (3), (4), and (6) simplify to

\[
T_2=200+26q+1/200,
\]
\[
L_2^{\mathrm{up}}=e^{-20+1/200}
\big[(1-q)+qe^{-6/5}\big]
\big[2(1-q)+q(e^{-8/5}+e^{8/5})\big],
\]
\[
\delta=(1+e^{32})^{-1},\qquad
\varepsilon=1/20000,\qquad
L_{\mathrm{clean}}=81\beta/40000.
\]

The fixed panel is the full product `q in {1/10,3/10}`, `beta in
{400,1600,25600}`, and `gamma in {1,16}`. It has 12 parameter cells, no sampled
observations, no optimizer runs, and no independent-world interpretation.

For all four `beta=400` rows, this declared certificate necessarily fails:
\(L_{\mathrm{clean}}=81/100\), whereas
\(U_{\mathrm{dense}}\ge2\operatorname{softplus}(\gamma)>2\) for the declared
gamma values. The two `beta=1600,gamma=16` rows also fail this sufficient
comparison since `L_clean=81/25` and `U_dense>32`.

These failures do not prove that a recovering model is globally optimal,
that the dense family is irrelevant, or that no other obstruction exists.
They say only that the fixed upper/lower bounds do not separate the two classes
in those rows. They must remain in the table without choosing new offsets,
logits, tolerances or prior fields after the result.

The remaining six rows have favorable analytical margins already under coarse
bounds. One can bound the dense observed risk uniformly over the two q values by

\[
\delta T_2+mL_2^{\mathrm{up}}
<208e^{-32}+2e^{-18}<1/5000000.
\]

The last inequality follows from `e>2` and `e^3>16`, both immediate from the
positive exponential series. Also \(2\operatorname{softplus}(1)<3\), while
\(2\operatorname{softplus}(16)<32+2^{-15}\). These bounds establish positive
separation for `beta=1600,gamma=1` and both prior fields at `beta=25600`.
This analytic classification is not an assertion that the prescribed Colab
interval-arithmetic run has already executed or passed. That run should record
the exact outward margins, source hashes and complete fixed denominator.

## 8. What this adds to the prior native theory

The imported
`refine-logs/runs/vg-sae-cloud-20260930/AVERAGE_RISK_MARGIN_BRIDGE_20261001.md`
explicitly distinguished the old unbounded-precision dense-offset sequence
from failure at fixed finite beta. Its Section 5 correctly left the latter
unproved. The present argument changes two important quantifiers:

- beta is finite and shared by every comparator, and the dense witness itself
  has finite `c`, `L` and encoder parameters;
- the lower comparison is uniform over every model satisfying the stated
  clean-risk target, whose class is nonempty under the same noisy observations.

Thus this closes part of the objective-to-clean-risk issue identified there
for the separated parameter regimes. It does not rely on a free-energy limit
of minus infinity, the profiled-beta floor, or the earlier noiseless
zero-centering-bias descent witness.

The mechanism is a calibration tradeoff. Put
\(\rho=\varepsilon/(d\sigma^2)\). Dense noise copying can make observed risk
arbitrarily small with gate KL approaching
\(d\operatorname{softplus}(\gamma)\), whereas a sufficiently clean model
pays at least \(d\kappa(1-\sqrt\rho)^2/2\) in observed-risk cost. A sufficient
finite-parameter existence condition is therefore

\[
\kappa>\frac{2\operatorname{softplus}(\gamma)}{(1-\sqrt\rho)^2}.
\]

The current panel has `rho=1/100` and `kappa in {1,4,64}`. Its separation
appears only when the precision exceeds the reciprocal generating noise
variance. The matched value `beta=1/sigma^2=400` is not refuted by this witness.
Using the term "fixed precision" without also stating `beta sigma^2` would
hide the relevant scientific condition.

## 9. Remaining scope limits and next gate

This theorem does not establish a positive or negative gap for dictionary
identity alone. Although the supplied dense witness is rotated away from the
positive-permutation teacher dictionary, the same construction works with
`D=I`. A global optimizer could retain a teacher dictionary while encoding
noise densely. The dense witness also uses a large unconstrained bias; a
separately imposed small-bias contract may exclude it.

Excluding `C <= epsilon` does not by itself prove wrong hard supports or wrong
generating-activity estimates. The clean-risk target here is stringent: one
percent of raw noise energy. A model can miss that target and still recover
useful features or activity. The all-on hard density of this particular
witness is not a statement about every global optimizer. BR5 measurement,
W2, C2 and the hard-between repeat law remain separate gates.

The next justified gate is the bounded Colab CPU check of the frozen 12 cells
with outward-rounded exponentials/logarithms, complete source binding, fixed
parameter values and explicit unresolved classifications. Its test suite
should challenge column orientation, the half factor, the normalized prior,
the good-class Gaussian symmetry, and the matched-precision negative cases.
The directed-arithmetic certificate concerns the mathematical finite native
model; any claim about rounded Torch parameter tensors or sampled performance
would need an additional error or execution contract.

After that check, update the native research plan to require an explicit
noise/precision contract before claiming that objective improvement supplies
the small clean-risk premise of the average-risk bridge. A positive recovery
theorem in the remaining calibrated regime, or an appropriate counterexample,
is still needed. The original density estimator additionally needs its own
identified observable and repeat-law evidence. No LLM activation experiment is
authorized or needed for this bounded theory step.

## 10. Independent final implementation review

The final candidate script, focused tests and protocol were read in full after
the author reported them stable. No mathematical or implementation blocker was
found in this static review. Provisional clearance applies to these exact
versions, subject to the focused tests on the authorized Colab CPU runtime:

| File | SHA-256 |
|---|---|
| `scripts/certify_native_noise_obstruction.py` | `30d75112ee17dfc8e912939331ccbb44ccc70572c07f3fc7f2d205f71ec28768` |
| `tests/test_native_noise_obstruction.py` | `dde4cd24e832a3972015445376963bfbdb71929fff65b002eee7665f61f3af93` |
| `refine-logs/runs/vg-sae-colab-continuation-20261003/NATIVE_NOISE_OBSTRUCTION_PROTOCOL.md` | `7abe387aa021103fce26b645da16d213a66503f6fe0ab2f3352bc9897ea975c7` |

The implementation preserves exact Fraction inputs and converts them to Arb
through `fmpq`, with 512-bit precision and one arithmetic thread. Its column
products implement equation (4) with `D[i][j]`; the moment term implements
equation (3), including the supplied observation-noise variance. The risk term
retains both `delta*T2` and `m*L2_bound` before multiplication by `beta/2`.
The normalized KL ceiling is stably evaluated as
`d*(gamma + log1p(exp(-gamma)))`, equal to `d*softplus(gamma)`.

The lower bound is the exact rational `81*beta/40000`. The code separately
encloses the square-root reverse-triangle formula and checks that this rational
value belongs to the resulting outward interval. It independently records the
finite good-class gate slope `200`, gate bias `-100`, constant amplitude one,
and upper bound `2*exp(-50) < epsilon`. The strict success comparison uses the
outward upper endpoint of the dense bound and the exact clean lower bound.
Equality or lack of separation receives `bound_inconclusive`; no tolerance or
successful-case substitution is used.

All twelve prescribed tuples are fixed in the executable. The outputs retain
every tuple, exact endpoints, model parameters, precision calibration ratio,
and source hashes. The construction is explicitly a real-valued native model.
The optional Torch parity test checks finite artificial inputs at float64
tolerance and is not used as evidence that the rational-weight population
certificate transfers to rounded tensors. The actual runner uses no Torch,
samples, quadrature, optimizer or training.

The entrypoint requires `COLAB_TASK_OUTPUT` and the authorized CPU marker,
rejects a visible Colab GPU, and refuses an existing result. Output files use
exclusive creation. Before importing the arithmetic backend, it installs a
60-second total-process CPU limit and a 120-second wall limit, retaining a
tighter inherited CPU allowance. Scientific computation stops five seconds
before the limits; interrupted and unstarted prescribed cases retain explicit
budget-unresolved statuses. Finalization keeps CPU and wall limits active.
Process-age accounting and its fallback are disclosed. No memory cap is
claimed. A hard resource kill can leave a partial artifact and must be reported
as incomplete rather than passed.

The focused tests cover the fixed panel, rational geometry, source-state
enumeration of the second moment, column-wise exponential moments, independent
Bernoulli-mask enumeration of the stochastic risk, the half factor, nonempty
finite recovery class, strict separation and matched-precision nonseparation.
They also challenge runtime authorization, existing-output preservation and
complete case retention after an artificial budget stop. These tests were
reviewed but not executed locally by this reviewer. No new numerical result is
asserted in this implementation review.
