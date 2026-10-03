# Bounded literature gap and next research obligation

2026-10-03. Separate scope reviewer. Primary-source reading and analytic critique
only; no experiment, training, numerical quadrature, old-result recomputation or
Colab allocation. This review supplements `RESEARCH_STATE_AUDIT.md` and preserves
its open pre-LLM requirements.

## Decision

An explicit positive-noise optimizer-exclusion certificate would close a real
remaining quantitative obligation in the supplied two-atom generative theory. It
would not establish a new sparse-coding model, a new posterior method, native
VG-SAE consistency, a practical estimator or publication novelty. The broader
original-method obligation remains an objective-to-recovery guarantee or a
precisely scoped counterexample, followed by a justified link to the chosen
repeat-selection observable.

The count-fiber criterion is more specific than “variational inference can be
biased”: it distinguishes same-count ambiguity from count-changing ambiguity.
The existing proof of an all-finite-positive-crossover sign, with jointly fitted
scalar variance, is likewise more specific than generic classification-bias
precedent. The present bounded source check found no theorem-level replacement
for those statements in the inspected text. Full-text access is incomplete for
two historical sources, so absence of such a theorem is not established.

## Primary-source comparison

| Source and access depth | Established adjacent result | Consequence for this project |
|---|---|---|
| Turner and Sahani, *Two problems with variational expectation maximisation for time-series models*; official author PDF and preserved full-text §§1.2, 1.4.4–1.4.7 inspected | Parameter-dependent approximation gaps bias learned noise and weights; tighter approximations need not reduce parameter bias. Joint fitting changes those biases. | Generic VI bias, geometry/noise coupling, or monotone lower-bound improvement are established context. This text does not provide the specified Bernoulli count-fiber criterion or the present all-crossover global exclusion result. |
| Bryant and Williamson (1978), *Asymptotic behaviour of classification maximum likelihood estimates*; publisher and IBM author-institution abstract | Hard all-or-nothing classification maximum likelihood can be asymptotically biased. | A hard-assignment bias example alone is not new. Full theorem text was not available; no claim that the whole paper lacks an equivalent special case. |
| Henniges, Puertas, Bornschein, Eggert and Lücke (2010), *Binary Sparse Coding*; preserved primary author PDF text, equations (1), (2), (6)–(8), discussion | Bernoulli activity, Gaussian noise and dictionary are learned together using truncated posterior sums; activity truncation needs explicit correction. | Learning q/noise/D in a normalized binary model predates this project. Count underestimation and a correction are already explicit, although truncation bias and globally optimized product-family bias are different mechanisms. |
| Drefs, Guiraud and Lücke (2022), *Evolutionary Variational Optimization of Generative Models*; official JMLR PDF model §3.1, Eq. (11), method and recovery sections | BSC has homogeneous Bernoulli activation and Gaussian observations; flexible retained-state variational distributions are optimized with established parameter updates. | Generic nonfactorized retained supports and activity learning cannot be presented as a new fix. Any count-aware approximation must demonstrate a distinct error/cost benefit against this family. |
| Haft, Hofman and Tresp (2004), *Generative binary codes*; publisher abstract, metadata and linked erratum record | Binary continuous-observation coding, mean-field inference and EM were already used for sparse representations. | A binary mean-field autoencoding precursor must be acknowledged. Abstract access does not support a theorem-level conclusion about count bias. Publisher lists “Hofman”; the author's institutional publication list uses “Hofmann.” Preserve this bibliographic discrepancy until full article/erratum text resolves it. |

Direct sources:

- [Turner–Sahani author PDF](https://www.gatsby.ucl.ac.uk/~maneesh/papers/turner-sahani-2010-ildn.pdf).
- [Bryant–Williamson publisher record](https://academic.oup.com/biomet/article/65/2/273/236816)
  and [IBM author-institution record](https://research.ibm.com/publications/asymptotic-behaviour-of-classification-maximum-likelihood-estimates).
- [Henniges et al. author-institution PDF](https://www.honda-ri.de/pubs/pdf/1917.pdf).
  Fresh PDF extraction timed out; the repository's preserved primary text was
  read. The accessible primary indexed abstract agrees with the model attribution.
- [Drefs et al. JMLR paper](https://jmlr.org/papers/volume23/20-233/20-233.pdf)
  and [article metadata](https://www.jmlr.org/papers/v23/20-233.html).
- [Generative binary codes publisher record](https://link.springer.com/article/10.1007/s10044-003-0194-x)
  and [publisher erratum record](https://link.springer.com/article/10.1007/s10044-004-0216-3).

No secondary source is used for a mathematical attribution. Search terms were
the exact listed titles plus a small terminology check for count-fiber and
minimum-count classification likelihood; irrelevant returned results were not
used. This is a bounded gap check, not a systematic novelty audit. No unpublished
project file was sent to an external model service.

## What is already proved in the imported line

Use `research-import-20261002T131813Z/project/refine-logs/runs/` as the prefix for
paths below.

- `vg-sae-generative-reassessment-20261002/researcher_constructive/REASSESSMENT.md`
  records the count-fiber criterion and rejects generic rate-separation and
  normalized-model redesign as new contributions.
- The final `joint_global_sign_proof_20261002/PROOF_PACKAGE.md` already proves
  every limiting global optimum has `r < q* < q0` for fixed finite positive
  crossover, with `r=q0(1-q0)`, supplied two-atom geometry, unit amplitudes and
  jointly fitted scalar variance. Uniqueness is not needed and is not proved.
- `researcher_constructive/PRELIMIT_VARIANCE_PROOF.md` establishes a uniform
  compact fitted-variance domain, explicit error bound (11), and eventual
  prelimit optimizer transfer. Its earlier small-crossover restriction is
  superseded by the final all-crossover sign proof for that implication.
- `theory_ultra/PHYSICIST_SCALING_ADDENDUM.md` already gives a conservative
  finite-noise bound for fixed D with matched supplied noise. The new task must
  therefore explicitly concern the coupled q/variance cancellation limit; merely
  restating a generic finite-noise entropy bound would duplicate prior work.

## The new finite-noise obligation

Let `E_sigma(q,v)` be the original globally optimized product population ELBO
and `J_lambda(q,v)` the limiting objective. Freeze q0, d, the candidate interval
I, a positive finite lambda, and a target interior interval B before computing
any numerical result. The desired result is a verifiable positive sigma0 for
which every global prelimit population maximizer lies in B.

The required logical chain is:

1. Give a common positive compact V containing **all prelimit global optima**,
   using the reviewed finite-codebook distortion envelope. A numerically narrow
   limiting variance range cannot silently replace this prelimit domain.
2. Provide a feasible limiting objective lower value L and a certified upper
   value U for `sup J_lambda(q,v)` over **all** `q in I\B, v in V`. For the exact
   one-threshold reduction, clipping must use each excluded q interval, not the
   original unrestricted optimum. Selected stationary roots and their images do
   not establish U.
3. Bound the entropy term, Gaussian-tail term and any lambda-path error in
   prelimit equation (11) at the fixed positive sigma0, while checking every
   prerequisite (group separation, small-noise condition, interval membership).
4. Under a two-sided bound `|E_sigma-J_lambda|<=epsilon`, require
   `L-U>2*epsilon`. If `lambda_sigma=lambda` exactly, the reviewed decomposition
   is one-sided: product >= all-state hard >= correct-group hard, so
   `0<=E_sigma-J_lambda<=epsilon`; then `L-U>epsilon` suffices. A conservative
   two-sided test remains valid but may be much looser.
5. Account for numerical interval and tail errors in L, U and epsilon. Preserve
   nonpositive margins and capped outcomes; no root-selecting or candidate
   replacement after seeing results. Record Colab CPU runtime, dependencies and
   exact source/protocol hashes.

This closes the explicit noise-scale and global excluded-set obligation left by
an eventual asymptotic statement. It does not rerun the closed conditional-score
benchmark or enlarge the old 20-cell parameter-location table. A finite number
of predetermined certified examples is sufficient for this obligation; a broad
parameter grid is not warranted.

## Applicability critique and stopping interpretation

The finite-codebook compactness constants may make V very wide, which enlarges
the entropy bound and can force sigma0 to be extremely small. The signal scale
is unit active amplitude; the relevant geometric ratio is
`lambda=||d1+d2||/sigma`. A small numerical sigma alone is not a general low-noise
statement. Finite K and supplied geometry are material.

For each fixed scenario report the exact noise threshold and all contributing
error terms. Separately show whether the same sufficient bound is informative at
the project's original noise .05. If it is not, report “bound inconclusive at
.05”; do not claim that the actual model is unbiased or that the theorem fails.
If only a very small sigma is certified, the result is a rigorous illustrative
quantification, not demonstrated relevance to the retained orthogonal native
states. Do not choose an easier geometry after a frozen case fails merely to
obtain a positive row.

Even a fully successful finite-noise certificate leaves population versus sample,
global optimum versus practical optimizer, supplied D versus learned D, fixed
amplitudes versus a(X), and Bernoulli-prior q versus native gamma/occupancy
unbridged. These are independent substantive assumptions, not engineering
details that numerical precision can remove.

## Original native route after this bounded closure

The exact open target is already stated in
`vg-sae-cloud-20260930/AVERAGE_RISK_MARGIN_BRIDGE_20261001.md` §5: a positive
population free-energy gap outside an epsilon-neighborhood of the positive
permutation teacher orbit, at fixed finite beta/gamma, supplied sigma, explicit
bias/conditioning bounds and the actual amortized encoder class; or a
representable competitor that disproves the gap in that contract.

The finite-panel and average-clean-risk inequalities are already available.
Their stringent hypotheses are not automatically supplied by the native training
loss. Another SURE/distance pass over already-failed states would quantify known
failure without creating the needed guarantee. A useful analytic next step must
connect the actual objective to those premises, or identify the restriction that
provably prevents it.

The root proposed a concrete restricted-family check: for a noiseless scalar
Bernoulli(q) coordinate, unit decoder, native prior pi=sigmoid(-gamma),
`0<=b<=1`, optimize the two nonnegative amplitudes and the two gate values. In
the softplus closure `a0=0`, `a1=1-b`, `m0=pi`, and
`m1=sigmoid(beta*(1-b)^2/2-gamma)`. The reduced objective, omitting only constants
independent of b, is

```text
F(b) = beta/2 * [(1-q)b^2 + q(1-b)^2]
       + q*softplus(-gamma)
       - q*softplus(beta*(1-b)^2/2-gamma).
F'(b) = beta * [b-q+q(1-b)m1].
```

The derivative at zero is strictly negative for finite beta/gamma and interior q.
It is positive for b>=q, so every global minimum on [0,1] lies in (0,q). If
beta/2<=gamma and the native decision is the strict threshold m>.5, every gate in
this restricted family is hard-inactive. The argument must state gamma>=0 in
that all-off claim, which is automatic when beta>0 and beta/2<=gamma.

On two noiseless input values, two finite target gate logits are exactly
representable by one affine sigmoid encoder. Positive finite a0 and a1 are
exactly representable by one affine softplus encoder; the zero-amplitude limit
must be approached with finite parameters, and a strict objective gap must
survive the approximation to give a genuine finite witness. Gaussian positive
noise changes this two-point interpolation argument and needs separate work.

No identical profiled formula was found in the inspected imported notes. Its
nearest antecedents are the hard oracle-subset envelope and G1/G2's negative
native amplitude-slope derivative. The profile is a modest original-model scope
lemma: exact zero-offset parameter recovery is not the correct finite-beta
claim. It does **not** refute a dictionary gap (D stays the teacher dictionary),
an epsilon recovery bound, or unrestricted native optimization. At beta=400 and
gamma=16 its zero-offset defect scales with exp(-184), so it does not explain the
large retained support errors. An unrestricted all-off-optimum statement cannot
be inferred from the restricted-family all-off result.

Recommended use: include this as a small independently checked lemma and use it
to sharpen the finite-beta recovery contract. Do not let it replace the larger
objective-gap or repeat-selection problem. Analytic construction and bounded
Colab CPU identity/parameter-witness checks are enough; no new fitting campaign
is justified by this lemma alone.

## Completion boundary

Keep BR5 `measurement_unresolved`, W2 blocked and C2a/C2b unentered until a
reviewed method contract genuinely clears their gates. The present analytic
candidate and its numerical certificate can be completed while these remain
open. It is not faithful to the user request to mark the full pre-LLM research
goal complete solely because this small theorem package is done.

Workflow provenance: project-local ARIS `analyze-results` and Scientific Agent
Skills `scientific-critical-thinking`; verified reference metadata is recorded
in `RESEARCH_STATE_AUDIT.md`. Their use structures this review and supplies no
independent scientific authority or cross-family validation.
