# Matched-precision native amplitude relaxation: independent analytic review

2026-10-13. Independent review prepared for the root research agent. No local
numerical calculation, model execution, tests, training, quadrature, or compute
allocation was performed. This document is a derivation and a proposed fixed
Colab arithmetic check, not its execution receipt. Existing results were read
and were not modified. The review itself does not establish cross-family
agreement or the availability of any other requested reviewer.

## Target and status

**COHERENT AS STATED for the finite native amplitude-relaxation proposition
below.** The unrestricted matched-precision objective-to-recovery gap and the
native generating-q measurement law remain unresolved.

The smallest useful next result is the following falsifiable statement:

> At actual noise sigma=1/20 and supplied beta=400, even the finite native
> unit-amplitude solution with the exact, correctly specified support posterior
> has a strict objective-decreasing finite native amplitude perturbation.
> Decoder identity and native hard masks remain exactly fixed. At the fixed
> perturbation t=1/2, population hard coefficient NMSE strictly increases,
> while retaining an explicit small upper bound.

This tests whether calibrating beta, and even supplying the correct prior,
makes the generative unit-amplitude solution stationary in the full native
model. It does not. It does **not** compare against the best model in an entire
recovering class; therefore it is not the missing all-class recovery obstruction.
An unrestricted positive recovery gap may still hold for a neighborhood
containing both models.

## Assessment of the four completed results

The scope stated in the 2026-10-03 continuation report is appropriately narrow.

1. The saved population terminal audit certifies its saved partition and the
   images of each surviving connected component. Multiple components prevent
   interpreting this as optimizer uniqueness or a bound on their union diameter.
2. The explicit finite-noise bounds make an asymptotic population argument
   quantitative but do not certify sigma=.05 and concern a different normalized
   population model. Their successful smaller-noise cells are not native q
   estimation evidence.
3. The noiseless finite-beta bias witness keeps the teacher decoder. The small
   or unresolved beta400 differences cannot explain a substantive decoder or
   hard-support failure.
4. The positive-noise witness supplies a valid all-clean-class comparison in
   its separated overprecision cases. Its beta400 rows are inconclusive. Its
   epsilon=1/20000 threshold is not the C2 recovery-success definition.

There is no reason to rerun those jobs for the new question. This review relied
on their mathematical contracts and scoped reports, rather than reperforming
their numerical or archive audits.

## Original endpoints that the new result must not replace

The imported `refine-logs/EXPERIMENT_PLAN.md`, P2-B, fixes:

- C2a: relative error in the expected generating count, report rate, and
  all-world within-20%-and-reported rate. Broad support requires report rate
  >=80%, reported median relative error <=20%, all-world joint rate >=70%, and
  each cell reporting >=4/6.
- C2b: hard coefficient NMSE averaged across repeats, same-bank oracle regret,
  and a paired selector contrast against reconstruction-only. The practical
  superiority margin is .02, with its stated confidence-interval condition and
  c_dec non-inferiority guard. It is **not** an absolute NMSE <=.02 criterion.
  The <=.05 regret criterion is secondary.

The new two-case bound below is a population mechanism check. It is not a
selector comparison, an independent world, a report/abstention assessment, or
confirmation evidence. No threshold or coverage gate changes as a result.

## Invariant object, assumptions, and notation

Let K=d=2, S_j independently follow Bernoulli(q), and

    X=S+sigma Z,   Z~N(0,I_K),   sigma=1/20,
    beta=sigma^(-2)=400,   q in {1/10,3/10}.

Source amplitudes are exactly one, the generating dictionary is I, and the
generating bias is zero. This is a structural population assumption. Every
encoder observes X only. Set

    gamma_q=log((1-q)/q),   pi=sigmoid(-gamma_q)=q,
    m_j(X)=sigmoid(beta*(X_j-1/2)-gamma_q).

These are the exact support marginals P(S_j=1|X), because independent bits,
orthogonal dictionary, and known Gaussian noise make the posterior factorize.
Using gamma_q supplies the true prior to this control; it is not a truth-free
q estimator. The forthcoming pointwise inequality holds for any fixed finite
gates, so the result does not depend on finding a favorable prior mismatch.

The mask B is drawn conditionally independently of S and Z given X, with
conditionally independent B_j~Bernoulli(m_j(X)). Let a_t be the amplitudes
specified below, D=I, b=0, and define

    R_X(t)=E ||X-B*a_t(X)||^2,
    C(t)=E ||S-B*a_t(X)||^2,
    J=E sum_j KL(Bern(m_j(X)) || Bern(q)),
    G(t)=beta R_X(t)/2+J.

G omits only the common Gaussian normalization at fixed beta. It retains the
full Bernoulli reconstruction variance, normalized prior, and entropy
coefficient one. Risks are full sums, not per-coordinate MSEs. Beta has inverse
squared observation units and G is dimensionless; all amplitudes and inputs are
expressed in the structural unit source-amplitude scale.

## Derivation map

1. Represent both models with finite native sigmoid/softplus weights.
2. Hold gates fixed, so gate KL cancels exactly.
3. Use monotonicity and the global derivative bound of softplus to contract
   each active reconstruction residual toward the observed input.
4. Integrate a strictly positive part of this exact gain on one fixed Gaussian
   event to obtain a finite population lower bound.
5. Independently bound the hard coefficient NMSE and its increase using the
   unchanged decision threshold, amplitude Lipschitz bound, and Gaussian tails.
6. Keep stochastic clean risk, hard support, dictionary identity, and q
   measurement separate.

## Finite parameter mapping

Write u0=log(expm1(1)), so softplus(u0)=1. For 0<=t<=1 define

    a_t,j(X)=softplus(u0+t*(X_j-1)).

The proposed fixed comparison uses t=0 and t=1/2 only. Native parameters are

    decoder.weight          = I_K
    pre_bias                = 0
    gate_encoder.weight     = beta I_K
    gate_encoder.bias       = (-beta/2-gamma_q) 1_K
    amplitude_encoder.weight= t I_K
    amplitude_encoder.bias  = (u0-t) 1_K
    log_beta                = log(beta), frozen

Use nonnegative amplitudes, normalized gate prior with lambda_sparsity=gamma_q,
variance enabled, entropy enabled with coefficient one, and the learned-beta
branch with beta fixed. This is not the profiled branch. All parameters are
finite and decoder columns have unit norm. The encoder centering convention in
`src/sae_model.py` agrees because pre_bias=0.

The exact real-valued construction and its outward arithmetic certificates
must remain separate from float64-rounded tensors. A small source-parity check
can be reported as an implementation check, not as interval certification of
those rounded tensors.

## Main derivation

### 1. Strict pointwise objective decrease: exact proposition

Fix a scalar x and let Delta_t(x)=a_t(x)-1. Softplus is strictly increasing and
has derivative strictly between zero and one. For x!=1 and 0<t<=1,

    sign(Delta_t(x))=sign(x-1),
    0 < |Delta_t(x)| < t |x-1| <= |x-1|.

Thus a_t(x) lies strictly between 1 and x, even when x<0. Consequently

    (x-1)^2-(x-a_t(x))^2
      = Delta_t(x)*(2*(x-1)-Delta_t(x)) > 0.

Conditional on X, the full stochastic risk per coordinate is

    E_B (X_j-B_j a_t,j)^2
      = (1-m_j) X_j^2 + m_j (X_j-a_t,j)^2.

The first term is unchanged. Every finite sigmoid gate has m_j>0. Therefore
the conditional stochastic risk decreases whenever any X_j!=1. A continuous
Gaussian mixture assigns probability one to this condition. J is unchanged,
and all second moments are finite, so

    G(t)<G(0),  for every 0<t<=1.

No oracle amplitude, optimization, approximation, or change of beta appears
in this argument. It also holds for fixed non-Bayes gates, including the old
gamma=1 or16 gates, but those extra cases are not required numerical jobs.

### 2. Exact derivative in the correctly specified control

Let s0=sigmoid(u0)=1-exp(-1). Differentiation at t=0 gives

    G'(0)=-beta*s0*sum_j E[m_j(X)*(X_j-1)^2].

Dominated differentiation is justified by Gaussian-mixture polynomial moments
and the bounded softplus derivative. Because m_j=P(S_j=1|X),

    E[m_j(X)*(X_j-1)^2]
      =E[S_j*(X_j-1)^2]=q sigma^2.

Hence, exactly at beta sigma^2=1,

    G'(0)=-K q (1-exp(-1)) < 0.

This is an O(q) dimensionless objective direction at matched precision. It is
not the exponentially small noiseless bias effect. It still says nothing about
the sign of the change in every recovery metric.

### 3. Explicit finite improvement lower bound

Freeze t=1/2. On the event S_j=1 and |Z_j|<=1, write
u=X_j-1=sigma Z_j and define

    s_min=sigmoid(u0-t sigma),
    c=t s_min,

The mean-value theorem gives Delta_t(1+u)/u in [c,t] for u!=0. Since
r(2-r) is increasing for 0<=r<=1,

    (X_j-1)^2-(X_j-a_t,j)^2 >= c(2-c) u^2.

The exact Bayes-posterior tower identity strengthens the bound: for any
integrable scalar function g,

    E[m_j(X) g(X_j)]=E[S_j g(X_j)]=q E[g(1+sigma Z)].

Thus no lower approximation to m_j is required. All omitted pointwise
improvements are nonnegative. For a standard normal scalar Z define

    mu2=E[Z^2 1{|Z|<=1}]
       =erf(1/sqrt(2))-sqrt(2/pi)*exp(-1/2)>0.

The population objective improvement has the rigorous lower bound

    G(0)-G(t)
      >= beta K q c(2-c) sigma^2 mu2 /2
       = K q c(2-c) mu2 /2
       =: L_gain >0.                                      (A)

Using S_j to analyze a population expectation supplies no latent labels to the
encoder. The event is not selected from samples or tuned after a result. This
bound uses only scalar exp/log/erf and exact rational inputs; it needs neither
sampling nor quadrature.

### 4. Unchanged hard support and explicit hard coefficient error bound

Use the native strict threshold H_j=1{m_j>1/2}. Both models share

    H_j=1{X_j>h},   h=1/2+sigma^2 gamma_q.

Decoder identity, every input's hard mask, hard count, FP/FN probabilities, and
support F1 are exactly unchanged between t=0 and t=1/2. Hard coefficient values
H_j a_t,j can change. Set

    alpha=h/sigma,   delta=(1-h)/sigma.

The prescribed cases have 0<h<1, so alpha,delta>0. Write phi for the standard
normal density and Q for its upper tail. For S_j=1,

    (H_j a_t,j-1)^2
      <=t^2 (X_j-1)^2 + 1{X_j<=h},

and the expectation is at most t^2 sigma^2+Q(delta). For S_j=0 and X_j>h>0,
a_t(X_j) lies between 1 and X_j, giving a_t(X_j)^2<=1+X_j^2. Thus

    E[H_j a_t,j^2 | S_j=0]
      <=(1+sigma^2)Q(alpha)+sigma^2 alpha phi(alpha).

The population hard coefficient NMSE, with the actual denominator E||S||^2=Kq,
satisfies

    NMSE_H(t)
      <=t^2 sigma^2+Q(delta)
        +(1-q)/q * [(1+sigma^2)Q(alpha)+sigma^2 alpha phi(alpha)].

Using Q(v)<=phi(v)/v for v>0 gives the convenient arithmetic bound

    NMSE_H(t) <= U_H
      :=t^2 sigma^2+phi(delta)/delta
        +(1-q)/q * phi(alpha)*[(1+sigma^2)/alpha+sigma^2 alpha]. (B)

The root can certify U_H<.02 as an explicitly labeled scale comparison. This
does not turn .02 into an absolute C2 success criterion. It also cannot be
reported as the attained NMSE or as a selector advantage. The dominant term
in this upper bound is t^2 sigma^2=1/1600; no decimal evaluation is needed to
identify that scale.

### 5. Strict increase in hard coefficient NMSE

The root agent proposed the following strengthening, which was independently
checked in this review. Because gates are shared, the difference in active
hard coefficient error is exactly H_j(a_t,j-1)^2, everywhere nonnegative.
When S_j=1 and |Z_j|<=1, the condition h<1-sigma guarantees H_j=1, and the
mean-value bound supplies (a_t,j-1)^2>=c^2 sigma^2 Z_j^2. The inactive difference
is H_j(a_t,j^2-1), bounded below by -H_j. Dividing by Kq yields

    NMSE_H(t)-NMSE_H(0)
      >=c^2 sigma^2 mu2-(1-q)/q * Q(alpha)
      >=c^2 sigma^2 mu2-(1-q)/q * phi(alpha)/alpha
       =: L_H_increase.                                  (E)

For the fixed cases this lower bound is strictly positive even without a
numerical calculation. Here gamma_q is positive and <=log(9)<3, so alpha>10
and h<1-sigma. Also u0>1/40, hence c>1/4; integrating only
1/2<=|Z|<=1 gives mu2>=phi(1)/4. Since (1-q)/q<=9 and phi(v)/v decreases for
v>0,

    L_H_increase
      > phi(0)*[exp(-1/2)/25600-(9/10)exp(-50)] >0.

The last strict inequality follows from exp(99/2)>23040. For example,
exp(99/2)>2^15>23040 since e>2 and 99/2>15. These are deliberately loose
analytic constants; the proposed Arb check evaluates (E) itself.

Thus the fixed finite perturbation lowers the actual native objective and
increases population hard coefficient NMSE while preserving D and every hard
support. This is a controlled disagreement between two quantities. Its small
absolute error bound (B), unchanged support, and comparison with only one
reference prevent interpreting it as C2b failure or an all-recovery-class gap.

### 6. Stochastic clean risk remains a different quantity

For the construction,

    C(t)=sum_j E[S_j*((1-m_j)+m_j*(a_t,j-1)^2)
                +(1-S_j)*m_j*a_t,j^2].

The exact expression retains stochastic false activations and amplitude error.
It does not equal the hard coefficient error from (B), and its change has not
been assigned a sign by the pointwise observed-risk proof. In particular,
lower G is not itself proof of higher clean risk or worse support.

The same conditional independence gives the matched-precision identity

    G=beta C/2 + J - E div f + K/2,   f(X)=m(X)*a(X),    (C)

under the usual integrable derivatives, which these finite native weights
satisfy. Here

    div f_t=sum_j [beta*m_j*(1-m_j)*a_t,j
                  +m_j*t*sigmoid(u0+t*(X_j-1))].

Thus G alone does not isolate C; changing input response changes the divergence
term. Equation (C) is an identity, not a theorem that its sufficient bounds
will be useful for arbitrary trained models. A supplied upper bound nu on
E div f and a lower bound J_min on J would imply

    C<=2 sigma^2 [G-J_min+nu-K/2].

The divergence is observable from the model and noisy inputs, while supplied
noise calibration, population expectations versus finite-sample confidence,
and conditioning/geometry remain separate premises.

## A useful general hard-code bridge

Let T_i be a binary teacher coordinate after an independently justified
positive permutation, Z_i=B_i a_i, and z_i^H=a_i 1{m_i>tau}. For any a_i and
0<tau<1, conditioning on X,S yields

    E[(z_i^H-T_i)^2] <= E[(Z_i-T_i)^2]/min(tau,1-tau).

When m_i>tau, its numerator is (a_i-T_i)^2 and the stochastic error contains
at least m_i times this. When m_i<=tau, its numerator is T_i^2 and the
stochastic error contains at least (1-m_i)T_i^2. This also handles exact ties.

With native tau=1/2, M=E||S||^2, sigma_min(D)>=kappa_D, and a separately
established ||DP-D_*||_op<=epsilon_D, the imported coefficient-risk bridge
therefore gives

    NMSE_H <= 2/(M kappa_D^2)
               * [sqrt(C)+||b||+epsilon_D sqrt(M)]^2.     (D)

This exposes the missing geometry/bias premises rather than assuming them.
For exact geometry and zero bias, the old epsilon=1/20000 condition with K=2
implies the unusually stringent upper bounds 1/2000 at q=.1 and 1/6000 at
q=.3. Neither is the original C2 selector criterion. An overcomplete decoder
cannot satisfy a global positive kappa_D and needs a different theorem.

## Smallest fixed Colab check

Exactly two cases: K=2, sigma=1/20, beta=400, t=1/2,
q in {1/10,3/10}, gamma=log((1-q)/q). No seeds, training, sampled evaluations,
parameter search, gamma refinement, integration grid, or optimizer is needed.

Use outward arithmetic to retain for both cases the real-valued parameter
definitions, L_gain, U_H, alpha/delta positivity, and the exact derivative
expression. Also retain L_H_increase and the h<1-sigma event condition.
Scientific pass for the new finite direction and the metric disagreement is
strict `lower(L_gain)>0` and `lower(L_H_increase)>0`.
`upper(U_H)<1/50` is an auxiliary scale statement only.
Interval overlap or budget exhaustion stays unresolved; do not replace a row.
A single small native objective/parameter parity check may be run on Colab CPU
against this exact mapping. It does not replace the analytic proof or certify
rounded-tensor population integrals. Source hashes and actual resource limits
belong in the root's frozen protocol and execution receipt.

The analytic claim is already established by the displayed inequalities. The
arithmetic check's role is transparent finite constants and implementation
agreement, not discovering a new favorable parameter regime.

## Open risks and boundaries

- This result compares two members of the same finite native family. It does
  not lower-bound every recovering model, characterize global minimizers,
  establish SGD convergence, or settle the requested all-class gap.
- The exact decoder and unchanged support mean the witness is not a dictionary
  or support failure. C2b coefficient values are reported separately.
- Supplying gamma_q is an oracle-prior control. It cannot be fed into an
  unknown-q estimator and then counted as q recovery.
- The old average-risk anchor needs quantitative C/J/conditioning and source
  margins. A constant-unit-amplitude recovering model pays inactive gate KL;
  hence J approximately wM does not follow merely from accurate support. In
  the high-SNR limit its J is K[q gamma+softplus(-gamma)], exceeding
  wM=Kq[gamma+softplus(-gamma)] by K(1-q)softplus(-gamma). Treating this as zero
  would silently grant an extra amplitude/gate regime.
- No hard-between repeat-selection law has been identified. Holding one
  model's gates fixed does not explain the distribution over independent
  optimizer runs; a fitting template or unchanged hard mask cannot identify q.
- BR5 remains measurement_unresolved, W2 remains gated, and C2a/C2b remain
  unentered. Confirmation worlds remain untouched. The pre-LLM objective is
  not complete.

## Source and workflow provenance

The reviewed research sources are the 2026-10-03 continuation plan/report,
noise-obstruction protocol/review, the imported
`AVERAGE_RISK_MARGIN_BRIDGE_20261001.md`, and the imported P2-B endpoints.
The native mapping was read against `src/sae_model.py`.

Project-local ARIS formula-derivation informed the fixed invariant object and
the distinction between identities, propositions, and scope. The
scientific-critical-thinking workflow informed endpoint separation and
consistent evidentiary standards. physics-guide-solver and physicist-analyst
informed the dimensional and limiting-regime checks. These workflows are not
evidence for the mathematical claims, whose derivations are above.

Workflow reference, checked against the current arXiv metadata on 2026-10-13:
Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M.
Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge
for Research Agents*, [arXiv:2609.00065](https://arxiv.org/abs/2609.00065).
The record exposes version 2 and no journal reference was shown in the queried
metadata. The citation intentionally resolves to the current version.

## Static pre-execution implementation review

The new runner, focused test, protocol, both configs, the reused resource guard,
native model, and controller bundle behavior were read without execution.
No mathematical, native-mapping, or source-bundle blocker was found in these
versions:

| File | SHA-256 |
|---|---|
| `scripts/certify_matched_amplitude.py` | `db0ef559a72d767e1edc33f73fc0aacac6c490e72176d21410f2160e31b5f899` |
| `tests/test_matched_amplitude.py` | `74cd9c331166ee58e16a2151ba33a4299fd1d3a3dc137cf76d4abfcd77c535e9` |
| `configs/colab_matched_amplitude.json` | `e215ef63a4b1d7e0fb5c93b66a1b87a80e0cb4240dfe340d9a14992bf9c97a76` |
| `configs/colab_matched_amplitude_sources.json` | `2f3d5621fa8b508d8b9eb4aba3956ca7628cd046192aa7963ce62651fd2c1a0b` |
| `MATCHED_AMPLITUDE_PROTOCOL.md` | `76a31ba69c6bbfcf055c3f0c569375c7a568766632fa3ef524f0bdb034477635` |

The scalar interval formulas implement (A), (B), and (E) correctly, with exact
Fraction/fmpq inputs and 512-bit one-thread Arb. The native parity test retains
the full Bernoulli variance and common prior/entropy, uses finite weights, and
does not infer population accuracy from its artificial input points. The
controller automatically includes all `src/*.py`, so the source manifest's
additional-file list supplies the remaining dependencies. The reused resource
guard retains hard limits during result serialization.

One nonblocking labeling refinement was sent to the root before execution:
`pair_certified` in this version requires the auxiliary U_H<.02 condition as
well as the two strict scientific signs. Keep the auxiliary comparison separate
from scientific pair status, or name the combined status to indicate the full
protocol. This cannot create a false positive here but avoids defining the
mechanism result through an illustrative scale threshold. A subsequently changed
file has a different source version and must not be represented by this hash.

This is static review clearance only. Test passage, interval endpoints, actual
resource use, and requested external-family reviews remain separate records.
