# Limiting population proof and certificate audit

Date: 2026-10-03. Applied skills: `physics-guide-solver` and
`physicist-analyst`. This is an analytic and source audit. No experiment,
numerical evaluation, local test, Colab allocation, or training was performed
for this report. The imported preservation sources were not changed.

## Assessment

The supplied proof supports the strict activity bracket

\[
q_0(1-q_0)<q_*<q_0
\]

for every global optimizer of its specified limiting population problem and
each fixed finite positive crossover. I found no mathematical blocker in the
threshold reduction, Gaussian-hazard comparison, or prelimit compactness and
uniform-convergence argument. This is a same-family semantic review, not a
formal proof-assistant certificate or a novelty determination.

The frozen numerical panel is incomplete in two separate senses: 12 positive
cells did not resolve parameter images; 17 of the 20 rows did not finish the
independent arithmetic/tree replay. Stored-leaf filtering can improve the first
issue. It cannot, by itself, resolve the second issue or establish a native-VG
activity estimator.

## Sources inspected

All following paths are under
`research-import-20261002T131813Z/project/` unless stated otherwise.

- The transfer root's `START_HERE_KO.md`.
- `refine-logs/runs/vg-sae-generative-reassessment-20261002/joint_global_sign_proof_20261002/PROOF_PACKAGE.md`.
- The same reassessment run's `researcher_constructive/PRELIMIT_VARIANCE_PROOF.md`
  and the corresponding independent theory reviews.
- `refine-logs/runs/vg-sae-population-bias-table-20261002/RESULT_README.md`,
  `reports/COMPACT_TABLE.md`, `input_spec.json`, `engine.py`, final independent
  numerical review, unexecuted stored-leaf draft and its status document.
- The reassessment run's `population_illustration_proposal_20261002/PROTOCOL.md`
  and `implementation_ultra/replay_population_certificates.py`.

The older prelimit appendix says that the all-crossover sign remains open.
That historical sentence is superseded by the later all-finite-crossover proof;
the appendix's compactness and convergence argument remains its dependency.

## Contract and physical interpretation

The model has two supplied unit-norm dictionary atoms, exogenous active
amplitudes equal to one, zero supplied offset, independent Bernoulli activity
\(q_0\in(0,1/2)\), and isotropic Gaussian observation noise. Dimension \(d\ge2\)
is fixed. The fitted activity interval contains
\([r,q_0]\), where \(r=q_0(1-q_0)\), and remains strictly inside \((0,1/2)\).
Both inference over product-Bernoulli laws and population parameter
optimization are global. A common scalar variance is fitted without a cap.

The coupled limit is

\[
\sigma_n\to0,\qquad
\lambda_n=2\sqrt{h_n}/\sigma_n\to\lambda<\infty,\qquad
v=\tau_n^2/\sigma_n^2.
\]

The pair of cancellation means has separation \(2\sqrt{h_n}\). Thus \(\lambda\)
is separation in units of generating noise standard deviation, and \(v\) is a
variance ratio. Neither is an inverse temperature with physical units. The
threshold, normal variate, residual \(M\), objective in nats, and all displayed
ratios are dimensionless. The reconstruction precision in the unscaled model
has inverse-square observation units. Keeping \(-(d/2)\log v\) is essential:
removing it changes the variance optimization problem.

The mechanism is informative: a correlated mixture of states \(00\) and
\(11\) can preserve a cancelling output, while an interior product law also
places mass on costly singleton states. As the unit-atom penalty is divided by
\(\sigma_n^2\), the optimizing product law approaches a point mass. At exact
cancellation and \(q<1/2\), the prior selects \(00\). True singleton states
remain resolved, so half the inferred active count is \(r\). This gives
\(q_*=r\) and missing activity \(q_0^2\) at \(\lambda=0\).

This is a statistical variational mechanism. A finite two-atom model and a
crossover table do not prove a thermodynamic phase transition. Energy or entropy
language here does not invoke a new conservation law or the second law of
thermodynamics.

## Independent algebra and global argument

Write \(a=(1-q_0)^2\), \(b=q_0^2\), \(w=a+b=1-2r\),
\(k=\lambda/2\), and \(g(c)=a\phi(c)+b\phi(c-\lambda)\).
The cancellation channel chooses state \(11\) precisely when

\[
c=c(q,v)=k+\frac{2v}{\lambda}\log\frac{1-q}{q}.
\]

This follows by subtracting its two log-joint branches. It verifies the factor
\(2v/\lambda\): replacing variance by its square root or reciprocal is invalid.
The fixed-threshold expected count update and squared residual are

\[
u(c)=r+aQ(c)+bQ(c-\lambda),
\]
\[
M(c)=d+\lambda^2\{aQ(c)+bQ(\lambda-c)\}-2\lambda g(c).
\]

For a true \(00\) assigned \(11\), the excess residual is
\(\lambda^2-2\lambda Z\). For a true \(11\) assigned \(00\), it is
\(\lambda^2+2\lambda Z\). Integrating the corresponding normal tails gives
the displayed \(M\). The unaffected residual term in the limiting objective
is \(2rd+w(d-1)=d-w\), which includes all singleton noise coordinates.

Direct differentiation gives

\[
u'(c)=-g(c),\qquad M'(c)=2\lambda(c-k)g(c).
\]

The completed-label objective is

\[
F=2u\log q+2(1-u)\log(1-q)-\frac d2\log v-\frac{M}{2v}.
\]

Its unique conditional optima are \(q_I=\operatorname{clip}_I u\) and
\(v=M/d\). In particular the denominator is \(d\), not \(d-w\).
After profiling,

\[
P_I'(c)=g(c)\left[2\log\frac{1-q_I(c)}{q_I(c)}
-\frac{\lambda(c-k)}{v(c)}\right].
\]

The clipping transitions are continuously differentiable because the interior
activity score vanishes at each transition. Interchanging suprema is legitimate
and does not assert convergence of alternating optimization.

The decisive global comparison is also consistent. Let
\(H=\phi/Q\) and \(B_\lambda=\lambda/[2H(k)]\). The proof establishes
strict hazard convexity using the residual normal lifetime survival inequality
\(S_x(s+t)<S_x(s)S_x(t)\). Integrating this strict inequality gives
\(\mathbb E T_x^2<2(\mathbb E T_x)^2\), hence \(H''>0\).
For every threshold where \(u(c)\ge q_0\), the auxiliary fixed point
\(\bar p=\sqrt{Q(c)}/[\sqrt{Q(c)}+\sqrt{Q(\lambda-c)}]\), monotonicity of the
count map in its generating activity, and Jensen's inequality imply

\[
\frac{\lambda(c-k)}{2\log[(1-u(c))/u(c)]}\le B_\lambda<1.
\]

Meanwhile \(M\) increases on \([k,\infty)\), so

\[
v(c)-B_\lambda\ge(1-B_\lambda)
\left[1-\frac{2w\lambda\phi(k)}d\right]>0.
\]

The final inequality follows from
\(\sup_{\lambda\ge0}\lambda\phi(\lambda/2)=\sqrt{2/(\pi e)}<1\),
\(w<1\), and \(d\ge2\). Upper clipping only increases the positive logarithm
in the derivative. Thus the profile increases through the entire region
\(u\ge q_0\), including the truth boundary. Every finite global maximizing
threshold lies beyond it. There \(r<u<q_0\), clipping is inactive, and the strict
bracket follows for every global maximizer. Uniqueness is unnecessary.

At \(\lambda=0\), direct branch comparison gives the admitted unique
\((q_*,v_*)=(r,1)\). At any finite positive \(\lambda\), a rounded endpoint equal
to \(r\) or \(q_0\) is compatible with a strictly positive but unresolved gap.
No uniform gap follows when \(\lambda\to\infty\).

## Prelimit transfer and its remaining quantitative obligation

The finite-codebook distortion argument does not assume a known latent state
available to an estimator. It uses that state only for a feasible objective
comparison. With four deterministic centers, bounded Gaussian density gives

\[
\mathbb E\min_j\|Z-c_j\|^2\ge
\eta_d=\left[\frac{\Gamma(d/2+1)}8\right]^{2/d}>0.
\]

For \(d=2\), this elementary bound is \(\eta_2=1/8\). Together with
prior-plus-entropy \(=-\mathrm{KL}\le0\), it makes fitted \(v\) lie in a
common positive compact interval before taking the limit. This avoids assuming
prelimit compactness from the limiting objective.

The unit-column product reconstruction variance yields an expected entropy
bound \(2h_{\rm bin}(\sigma_n^2[d+2v_+C])\), where \(C=-2\log\alpha\), once
its argument is at most \(1/2\). Gaussian wrong-group tails and a coupled
Lipschitz bound in \(|\lambda_n-\lambda|\) then give the appendix's explicit
uniform error bound (11). Compact uniform convergence implies outer convergence
to the limiting global argmax set. Its positive distance from both bracket
boundaries gives the eventual strict prelimit bracket for fixed \(\lambda>0\).

The remaining optional quantitative task is precise: certify a limiting
objective gap \(\Delta\) between a feasible global witness and a declared
excluded parameter region, bound the complete prelimit uniform error by
\(\varepsilon\), and require \(2\varepsilon<\Delta\). Check every condition
used to obtain the error bound, including compact-variance constants, the
entropy argument, Gaussian group separation, and fixed crossover. This would
yield a conservative sufficient finite-noise certificate for this same
population model. It would not be a finite-sample guarantee, an attained
algorithmic error rate, or a native-VG result. No such numerical certificate was
evaluated during this audit.

## Numerical evidence and genuine unresolved items

| Item | Frozen evidence | Current interpretation |
|---|---|---|
| Four zero-crossover anchors | Exact analytic rows | \(q_*=r\), \(v_*=1\) |
| Sixteen positive-crossover rows | All recorded objective widths meet \(10^{-10}\) | Primary recorded certificate claim; incomplete independent replay remains |
| Parameter images | Four \(\lambda=16\) rows meet both \(10^{-6}\) widths; 12 do not | Magnitude/variance illustration is incomplete |
| Independent replay | Only \(q_0=1/10,d=2,\lambda=0,1/2,2\) complete | 17 audit-unresolved rows remain |
| Table/receipt check | 20 rows and 120 exact-rational checks passed | Reporting/identity checks do not replace arithmetic replay |
| Native VG | Import summary: BR5 measurement unresolved, W2 blocked, C2 unentered | No success claim follows from this toy theorem |

The engine removes a dominated box only when it is popped from its priority
queue. Repeated refinement of higher upper bounds can leave many already
dominated boxes in the active image union. This preserves all-maximizer coverage
but can obscure parameter localization. Many reported possible branches are
not evidence for multiple exact maximizers.

## Stored-leaf repair: justified method and missing binding

For a valid feasible lower bound \(L\) on the global objective and certified
leaf upper bound \(U_B\), removing a leaf when \(U_B<L\) is sound. Equality or
overlap must retain the leaf. Recomputing the surviving exact-rational image
union is appropriate as a new, separately versioned continuation. It does not
require new samples, fitting, a new parameter grid, or a restart of optimization.

There is a specific audit obligation in the preserved draft. The existing
independent replayer's `remaining` branch (lines 163–166) checks the final leaf
IDs and domains, but does not compare its duplicated `upper`, `q`, `v`,
`variance_error`, derivative and witness fields with the earlier validated
creation records. The unexecuted postprocessor consumes those terminal fields.
Therefore the earlier checked-prefix result alone is insufficient to certify
the new postprocessor's exact input values. This is a dependency gap in the
proposed derivative workflow; it does not establish corruption of the old run.

The new verifier must:

1. Bind frozen source/result seal hashes and every consumed artifact.
2. Reconstruct exact tree splits and verify the root is covered by historical
   rejected leaves plus terminal active leaves, with no missing interval.
3. Bind the entire terminal certificate to its latest creation/precision-refresh
   record, not just its ID and domain.
4. Independently evaluate a feasible witness and the bounds needed by the final
   partition using outward arithmetic. Ancestor function evaluations are not
   needed once complete terminal cover and leaf bounds are proved.
5. Require independently supported strict exclusion for every removed leaf;
   retain ties and report all surviving components.
6. Compute stable direct-tail activity gaps and direct residual variance error.
   Retain the theorem's strict positivity even when numerical lower endpoints
   are zero. Do not label an interval containing zero with a variance sign.
7. Keep all 20 cells and separate original image status, new derived image
   status, and new independent audit status. A timeout preserves only the
   verified prefix and leaves the remaining cells audit-unresolved.

The draft's old wall deadline and approval file refer to the closed historical
run. They must not be bypassed or silently rewritten. A new source, budget,
output directory and semantic review implement the user's present continuation
request while preserving that record. CPU-only Colab execution is sufficient.

## Boundaries before later activation work

Neither the proof nor a complete leaf replay supplies learned dictionaries,
input-dependent amplitudes, amortized encoder behavior, finite-sample accuracy,
SGD convergence, uniqueness, or novelty. Those are different obligations.
The finite-codebook lower bound in particular cannot simply be copied to an
observation-dependent codebook. Any native-VG density claim must identify its
own observable, invariances, estimand and information assumptions and address
the existing negative/measurement-unresolved evidence.

No LLM activation experiment is needed to finish this proof/certificate line.
The next justified computational work is the bounded independent terminal
partition audit and, if pursued, the explicitly specified quantitative
finite-noise population bound. Their outcomes should determine the exact
claims available for the later research plan.

## New terminal-partition verifier: pre-execution method review

The continuation implementation is `scripts/verify_population_terminal.py`,
with artificial regression challenges in `tests/test_population_terminal.py`.
These are new sources; no historical engine or replayer is imported. The
implementation uses official python-flint/Arb elementary functions at 512 bits.
At this writing the sources have been statically reviewed but the tests and
historical replay have not been run by this reviewer. Execution belongs to the
root agent's Colab CPU workflow.

The finite domain is `[k,C]`, with the outward tail ceiling independently
checked. Streaming trace reconstruction keeps each active node's exact domain
and complete certificate digest. Splits must cover their parent exactly,
historical rejected certificates must match their latest creation record,
precision refreshes must preserve domains, and every terminal field must match
its latest creation/refresh record. The union of historical rejected leaves
and terminal active leaves must exactly partition the root. This makes the
final partition sufficient; ancestor transcendental evaluations are unnecessary.

For every leaf, the verifier independently constructs the clipped count image,
monotone variance image, derivative enclosure, midpoint witness, monotonicity
upper bound and centered mean-value upper bound. It verifies that the stored
certificate contains these independent bounds. It independently evaluates the
best recorded midpoint candidate and both domain endpoints to establish an
attainable lower bound at least as large as the original incumbent `L`.
All historical exclusions must then satisfy the independently checked strict
inequality `U < L`. New filtering removes only terminal leaves already satisfying
the same strict inequality in both stored and independently recomputed bounds.
Ties remain. The surviving union therefore covers every global maximizer.

Original component bounds and source/output hashes are also checked. An old
nonempty outer component is allowed to become independently empty after a
tighter theorem intersection: the old outer enclosure did not assert existence
of an optimizer there. An originally excluded component must still be excluded
by the independently checked raw intervals and theorem. Regression tests target
this asymmetry, tampered terminal values, lost leaves, wrong exact cover,
nonstrict rejection and unverified host execution.

The resource contract is one Colab CPU core, at most 600 process CPU seconds,
900 wall seconds and 512 MiB address space. No GPU is used. The entrypoint
requires the Colab CPU marker before creating output. A timeout returns a durable
completed prefix, explicit unresolved rows for every unchecked cell, and no
claim of completed all-cell replay. New output consists of provenance receipts,
per-cell summaries, small component/leaf-ID lists and exact bounds; the large
historical traces are not copied into the result. Only the fixed manifest and
run-seal bytes and the consumed summary, cell results and traces are rehashed;
the receipt does not claim a new hash pass over all archival dependencies.

### Additional restricted-activity bounds for a finite-noise consumer

The same independently verified partition supplies global objective upper
bounds for activity in `q <= r` and `q >= q0`, without a new optimizer. For
fixed threshold, write `K_I(u)` for the unrestricted clipped count objective.
Because `u(c) > r >= alpha`, the optimum over `[alpha,r]` is `q=r`. The
nonnegative penalty

\[
\psi_-(u)=K_I(u)-2u\log r-2(1-u)\log(1-r)
\]

is increasing: its derivative is
\(2\log[q_I(u)(1-r)/(r(1-q_I(u)))]\ge0\). This remains true when the upper
activity constraint clips `q_I` at `beta`. Therefore an outward lower bound
for `psi_-(max(u_lo,r))` is a valid penalty throughout each leaf.

For `[q0,beta]`, the penalty is

\[
\psi_+(u)=
\begin{cases}
2\,\mathrm{KL}(\mathrm{Bern}(u)\|\mathrm{Bern}(q_0)),&u<q_0,\\
0,&u\ge q_0.
\end{cases}
\]

It decreases up to `q0` and then vanishes. Hence a lower bound evaluated at
`min(u_hi,q0)` is valid. Subtracting either lower penalty from a certified
unrestricted leaf upper `U` gives a restricted objective upper on that leaf.
All final-partition leaves are used, including historically rejected leaves;
the global unrestricted rejection does not justify dropping a leaf from a
different restricted supremum.

Both restricted profiled activities are at least `r`, so the same derivative
tail cutoff applies: each restricted profile decreases beyond `C`. Its tail
supremum is bounded at `C`, and included in the outside-region upper bound.
The resulting bounds cover the entire threshold half-line, hence the complete
restricted joint limiting population objective with scalar variance fitted.

At `lambda=0`, the consumer receives the direct outward bounds from

\[
L\le2r\log r+2(1-r)\log(1-r)-d/2,
\]
\[
\sup_{q\ge q_0,v>0}J_0(q,v)
=2r\log q_0+2(1-r)\log(1-q_0)-d/2\le U_+.
\]

No strictly positive lower-region margin is claimed there because `q=r` is
the actual global optimum. It is represented as `None`.

Consumer schema: `TERMINAL_AUDIT.json` has `cells`; a completed row has
`status="verified"`, the original `spec`, exact rational `feasible_lower`,
and `outside_upper={"q_le_r": ..., "q_ge_q0": ...}`. Audit-unresolved rows
have no usable numeric bounds. Image-resolution status is independent of
arithmetic audit status. A nonpositive exclusion margin remains unresolved;
it is not a counterexample to the analytic theorem.

For an exactly fixed-crossover path `lambda_n=lambda`, the prelimit objective
dominates the within-group hard objective `J_lambda` pointwise. An independently
bounded one-sided excess `epsilon` therefore permits the sharper sufficient
condition `L > U_out + epsilon`. For a general converging crossover sequence,
the Lipschitz coupling error must also be charged to the feasible reference.
The verifier itself makes no finite-noise claim; the consumer must establish
these additional error bounds and all their applicability conditions.
