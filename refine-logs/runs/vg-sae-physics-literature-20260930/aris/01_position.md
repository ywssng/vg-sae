# ARIS scientist: independent position on the physics literature

Date: 2026-09-30. Author role: `/root/aris_physicist`. Status: independent position, before peer discussion. Review independence: same-family; acceptance status: provisional. No experiment, GPU job, or source-code change was made.

## Position

The three supplied papers strengthen the case for a VG-SAE paper built around an explicit statistical model, inference, and empirically validated density selection. They do not supply a theorem that the present across-initialization uncertainty curve identifies generating density. The most useful revision is to make the **probability space of each uncertainty** explicit, and to test the ensemble definition before committing to 7,240–10,376 fits.

The central concern is sharper than a generic “stability is not truth” warning. With a fixed training dataset, correct convergence to a unique solution, modulo the allowed permutation, can make the proposed between-fit uncertainty zero at every sparsity control. Thus a density signal may depend on optimization ambiguity even in an otherwise successful SAE. This is a testable limitation of the proposed estimator, not an argument against VG-SAE or against density inference as the paper's objective.

I recommend retaining Phase 1, retaining C2a generating-density and C2b recovery-selection as distinct endpoints, and adding a small bridge that distinguishes fixed-model posterior response, training-data variability, and initialization variability. The large confirmatory bank becomes conditional on that bridge. I do not recommend replacing the project with an RBM, an identifiability-only paper, or a generic diagnostic benchmark.

## What I actually read

Project context read: root `AGENTS.md`, `.agents/project-memory.md`, `PAPER_PLAN.md`, `refine-logs/FINAL_PROPOSAL.md`, `refine-logs/EXPERIMENT_PLAN.md`, `refine-logs/EXPERIMENT_RESULTS.md`, the 2026-09-29 `LITERATURE_GROUNDING.md`, relevant 2026-09-23 literature notes, Phase 1 proposal and relevant normalization/response passages in `REPRODUCTION_NOTES.md`. The exact conditional model and gamma sign were checked in `src/sae_inference.py`; this was a targeted equation check, not an implementation audit. No descendant AGENTS file was found in the output path.

Skill use: project-local `research-lit/SKILL.md`, `alphaxiv/SKILL.md`, and `research-review/SKILL.md` were read. I use their primary-source, source-depth, assumptions, actionable critique, and same-family qualification guidance. The PI's bounded two-scientist workflow controls delegation and write scope; no extra reviewer or unrelated pipeline was launched. The project's actual reference library is `refs/`, notwithstanding the skill's generic `papers/` convention. Existing reference-manager sources were not requested. Wiki ingestion and canonical edits belong to the PI's integration scope, not this agent's assigned `aris/` directory.

| Source | Version and depth | Important access qualification |
|---|---|---|
| Klindt, O'Neill, Reizinger, Maurer, Miolane, *A unifying framework from neural superposition to sparse interpretable codes*, Nature Machine Intelligence 8, 1025–1037 (2026) | [Publisher page](https://www.nature.com/articles/s42256-026-01259-z): metadata, abstract, subscription notice, references. Associated [author preprint 2503.01824v1](https://arxiv.org/html/2503.01824v1), *From superposition to sparse codes: interpretable representations in neural networks*: main argument and detailed §§4–5.3, Theorem 1, Eqs. (6)–(9), conclusion. | Nature's full published text was not accessible. First web request failed at an identity-provider redirect; direct download exposed the subscription preview. The preprint is the primary full-text basis; its numbering and wording must not be attributed to the final published version without checking. |
| Tubiana and Monasson, *Emergence of Compositional Representations in Restricted Boltzmann Machines*, PRL 118, 138301 (2017) | [APS metadata/abstract](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.138301); [arXiv 1611.06759v2 full main text](https://arxiv.org/html/1611.06759v2), Eqs. (1)–(2), Figs. 2–4. [Author supplement](https://www.phys.ens.fr/~monasson/Articles/a105-si.pdf), selectively read §III A–B, Eq. (7)–(8) participation-ratio argument, and inspected section structure. | APS full text is gated. Author-hosted supplement is accessible. Author publication list has a 138501 typo; the verified APS article number is 138301. Supplement's full training/sampling derivations were not audited. |
| Hou and Huang, *Statistical Physics of Unsupervised Learning with Prior Knowledge in Neural Networks*, PRL 124, 248302 (2020) | [APS metadata/abstract](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.248302); [arXiv 1911.02344v2 main text and supplement](https://arxiv.org/html/1911.02344v2): Eqs. (1)–(6), order-parameter and symmetry discussion; Appendix A's factor graph/Bethe setup and Appendix E Eqs. (S38)–(S50) read; Appendix D stability result inspected. | Main text and relevant supplements read in author version. I did not independently reproduce the full replica saddle-point algebra in Appendix B or certify the replica-symmetric approximation. |
| Soh et al., *Variational Garrote for Statistical Physics-based Sparse and Robust Variable Selection* (2025 preprint) | [2509.06383v1 §III.1.4, Eqs. (18)–(19), Appendix A Eqs. (20)–(23)](https://arxiv.org/html/2509.06383v1) were rechecked. | This is an empirical/model-based template in fixed variable coordinates. Its stated ensemble varies datasets. |

AlphaXiv overview succeeded for 2503.01824 and returned HTTP 404 for both PRL IDs; claims below rely on the primary full texts, not the generated overview. Downloaded source files are in local-only `aris/sources/`; the manifest records access and content hashes. New code or claims from papers cited inside these sources were not assumed verified merely because they appear in their reference lists.

## What each new source contributes, and what does not transfer

**Klindt et al.** organize the argument into linear identifiability, sparse recovery, and independent interpretability evaluation. The preprint's Theorem 1 uses a specified cluster-centric generating process, a continuous encoder, global cross-entropy optimization, and matched latent/representation dimensions. Its limitations explicitly identify the gap to overcomplete sparse representations. Equation (7) is a compressed-sensing scaling relation; Eqs. (8)–(9) separate code inference and dictionary learning. These are reasons to state assumptions and separate inference from learning, not a proof of VG-SAE identifiability. The quantitative interpretability step also does not make a recovered synthetic coordinate a semantic concept in an LM. [Preprint §§4–5.3](https://arxiv.org/html/2503.01824v1)

For this project, that organization supports Phase 1's exact→mean-field→amortized-encoder separation and C2's independent recovery evaluation. It also blocks a tempting overclaim: `d=8, K=16` plus sparsity is not sufficient to import a known-dictionary compressed-sensing guarantee into joint SAE training. The preprint describes Eq. (8)'s fixed-dictionary L1 objective as non-convex/NP-hard; that displayed objective is convex in the coefficients. We should not reproduce that sentence: hard L0 optimization and joint dictionary learning pose different difficulties. This local mathematical correction does not invalidate the perspective's overall motivation.

**Tubiana and Monasson.** Their RBM has binary visible units and hidden-unit potentials in Eq. (1). Compositional behavior depends jointly on weight sparsity, hidden nonlinearity, effective temperature, and visible activity fields. The random-weight ensemble yields a ground-state analysis, Eq. (2), and a sparse-limit count of strongly active hidden units scaling as `L=ell*/p`. Here `p` is weight connectivity sparsity, not a feature firing rate. Their supplementary §III A–B distinguishes an effective participation count from literal nonzero count and derives when its proxy converges. [Main text](https://arxiv.org/html/1611.06759v2), [supplement](https://www.phys.ens.fr/~monasson/Articles/a105-si.pdf)

The useful lesson for VG-SAE is that effective feature composition depends on geometry, signal amplitude/noise, and inference, not only a sparsity knob. The effective-temperature parameter of that RBM cannot be equated numerically with Gaussian observation precision in VG-SAE. Nor can `L~1/p` be used as a density-estimator formula here. A participation ratio could be an inexpensive **secondary amplitude-concentration measurement** under the exponential-amplitude stress condition; it must not replace hard L0 or be introduced as a new learned selector. Their actual phase claim has a thermodynamic ensemble and scaling argument. A bend in our finite K=16 response curve does not.

**Hou and Huang.** Their teacher–student RBM has two hidden units and binary synaptic weights. Equation (2) infers global weights from many observations. The parameter `q` is the correlation between the two receptive fields, not activation density. Equation (3) separates truth overlap, self-overlap, and permutation overlap. Equation (4) gives a critical sample-density expression in that model. Equations (5)–(6) and Appendix E derive joint noise/correlation inference using a normalized hyperparameter posterior; the authors explicitly discuss non-convexity with insufficient data. [Main text and Appendix E](https://arxiv.org/html/1911.02344v2)

For us, the transferable principle is **matched generative assumptions before interpreting uncertainty or empirical-Bayes parameters**. The global weight posterior here is distinct from a per-input Bernoulli support posterior, and both are distinct from the distribution over Adam outcomes. A decoder appearing consistently after initialization is analogous to a self-consistency observation, not direct evidence of truth overlap. The Nishimori result does not license calling learned beta Bayes-optimal, identifying gamma with true sparsity, or calling independently trained networks posterior replicas. Any such statement needs its own joint model and a valid inference argument.

## A compact VG theory bridge worth adding

The following equations are my direct derivations for the project's stated conditional model, not new results attributed to the RBM papers. They are familiar exponential-family/mean-field identities and should be presented as an explanatory bridge, not standalone novelty.

Fix an input, the decoder `D`, amplitudes `a`, bias `b`, and precision `beta`. Write `u_j=a_j d_j`, `r=x-b`, `s_j∈{0,1}`, and `N(s)=sum_j s_j`. The normalized support prior has `pi=sigmoid(-gamma)`.

The conditional posterior is

`p_gamma(s|x,D,a,b,beta) ∝ exp[-beta/2 ||r-sum_j u_j s_j||² - gamma N(s)]`.

Its log unnormalized density is a binary pairwise model:

`sum_j [beta u_jᵀr - beta ||u_j||²/2 - gamma] s_j - beta sum_{i<j} u_iᵀu_j s_i s_j`.

This exhibits exactly where explaining-away/competition can arise: the off-diagonal Gram matrix. For orthogonal nonzero `u_j`, the posterior factorizes. The relevant interaction includes amplitudes and precision, not just raw decoder cosine. Positive overlap with nonnegative amplitudes penalizes simultaneous activation in this conditional energy. Negative overlaps have different effects; geometry alone does not determine all posterior dependencies.

Let `Z_x(gamma)` be the finite-state partition function with gamma-independent likelihood. Then

`∂gamma log Z_x = -E[N|x]`,

`-∂gamma E[N|x] = Var(N|x) = sum_j Var(s_j|x) + 2 sum_{i<j} Cov(s_i,s_j|x)`.

With density normalization, divide the second line by K. The exact marginal sum `sum_j m_j(1-m_j)` is only the diagonal part. It is the full count variance in the orthogonal factorized case; overlapping features can invalidate that equality. Finite differences of a reoptimized decoder/amplitude/beta path contain additional total-derivative terms and are not this conditional susceptibility.

For the interior factorized mean-field free energy, the fixed-D,a,beta Hessian is

`H_ii = 1/[m_i(1-m_i)]`, `H_ij=beta u_iᵀu_j` for `i≠j`.

The likelihood's diagonal quadratic term cancels the corresponding diagonal from the Bernoulli variance correction. On a differentiable stable stationary branch with nonsingular Hessian,

`dm/dgamma = -H^{-1} 1`, so `-d(sum m)/dgamma = 1ᵀH^{-1}1`.

Thus even optimized mean-field response is not generally its factorized count variance `sum m(1-m)`. A branch switch or singular Hessian needs separate treatment. None of these identities equates response, within-gate uncertainty, or initialization disagreement.

The Phase 1 orthogonal agreement and high-overlap reverse-KL gap already fit this explanation. Existing frozen results further show that improving reverse KL need not improve marginal Brier/NLL/error. A covariance/response readout would make that evidence more informative without requiring a new model family. The no-variance result remains a total effect involving profiled beta; a fixed-beta side control is the minimal way to isolate the intended term more directly if the paper needs that causal claim.

An optional mathematical note, not a new learned-gamma proposal: with a fixed **proper generative likelihood**, the evidence score for the normalized Bernoulli prior is `∂gamma log p(X|gamma)=n K pi - sum_x E[N|x]`. An interior empirical-Bayes stationary point therefore matches average posterior occupancy to the prior mean. This is a self-consistency equation, not a finite-sample truth guarantee. The repository already records an analogous gamma gradient in the earlier pilot. Input-dependent point amplitudes prevent silently rebranding the joint SAE objective as that generative evidence. I would defer a new empirical-Bayes method; it is enough to explain why simply learning gamma does not settle Phase 2.

## The missing ensemble bridge

The current plan has important safeguards already: finite-ensemble decomposition, no-FP/no-FN and exchangeability conditions, native hard-density matching, truth-free signed matching, an abstention state, and all-world recovery evaluation. I endorse these. I would not present them as new discoveries of this review.

The remaining issue is the change in randomness. Soh et al. §III.1.4 explicitly average selection across realizations of synthetic datasets; their real-data analysis also builds an ensemble through data choices. The new plan freezes training examples and changes initialization and batch order. Both are legitimate experimental quantities, but the second does not inherit the first's curve merely by keeping the same formula. [Soh et al., Eq. (18)–(19) and Appendix A](https://arxiv.org/html/2509.06383v1)

There are at least three ensembles:

1. Support draws at fixed learned parameters and input: conditional posterior randomness.
2. Independent training datasets under the same generating dictionary: sampling variability in learned parameters.
3. Initialization/batch-order draws on a fixed training dataset: algorithm-induced variability.

Their sum can be decomposed only after specifying a hierarchy and correspondence. The present finite-R variance identity does not identify these measures with each other or make any one a Bayesian posterior over dictionaries.

**Converged-correct counterexample.** Suppose for each control all repeat optimizers reach the same selected dictionary and hard support function up to an exactly recovered permutation. Then `H[r,x,j]=H[x,j]`, so `U_hard=0` for every available density, including the true density and wrong densities. At a density that recovers truth this is a successful SAE. A flat curve therefore need not indicate incorrect features. The current abstention is an honest response, but such cases can block the planned broad C2a report-rate criterion. This is a logical counterexample; it does not assert that the existing optimizer actually reaches this regime.

A related opposite case is a reproducible mixed dictionary with `U=0` and bad recovery. Together they show why a peak/minimum is not intrinsically a generative-density estimator. The desired empirical finding is more specific: in a specified training ensemble, instability has a shape that predicts density and improves held-out recovery. That is a meaningful SAE-method result, even if it is algorithm- and regime-dependent.

## Minimal discriminating work before scaling

These are future experiments, not results from this review. They should replace or reorder early work, not be blindly added to every factorial cell.

| Priority | Minimal comparison | What it distinguishes | Decision consequence |
|---|---|---|---|
| 1 | Reuse the tiny exact-posterior fixture: orthogonal vs high-overlap atoms, fixed amplitudes and beta, a short gamma sweep. Report marginal variance sum, full count variance, exact finite-difference response and MF branch response. | The true fixed-model statistical-mechanical quantity versus its approximations. | Validate the derivation/measurement before naming a response “susceptibility.” No new joint training needed. |
| 2 | Same-data initialization repeats vs independently redrawn training samples from the same synthetic teacher. Keep teacher D, fixed align/select/test bank, training size, control grid, and initialization pairing identical between arms. | Optimization variability vs data-realization variability; direct transfer gap from Soh. | Choose and name the target ensemble during development, then lock it before confirmatory worlds. Neither arm is automatically a Bayesian posterior. |
| 3 | Current fixed horizon vs the already planned longer-horizon continuation, on the same bridge worlds. Track density, recovery, matching and U together. | Signal arising from unfinished optimization versus robust ensemble behavior. | If recovery improves while U and density identifiability disappear, revise the estimator interpretation or ensemble before spending the full budget. |
| 4 | Fixed known decoder/amplitudes with exact or converged inference, then frozen-decoder learned encoder, then joint VG on a few common worlds. | Conditional ambiguity, amortization/optimization, and dictionary-learning ambiguity. | Establish which component actually creates the candidate density signal. Exact/frozen controls are explanatory; no truth enters the deployed estimator. |
| 5 | Only after the above: primary VG vs L1 on the same small mechanism worlds; retain TopK as a recovery comparator. | VG-specific probabilistic mechanism versus a generic repeated sparse-training selector. | A generic component should be credited as such; it need not invalidate a VG-SAE contribution. |

A feasible **proposed development reallocation**, still requiring a revised protocol before execution, is 2 families × 2 development densities × 1 world = 4 worlds, each with 1 reference + 3 repeats, 17 controls, and 2 training-ensemble policies: 544 base fits / 800 fits at the existing 25-control maximum. This is the same fit scale as the current 8-world single-policy development stage. R=3 is suitable for a falsification/development screen, not a precise uncertainty band; do not apply the confirmatory R=5 bootstrap gate to it. Keep the existing 36 fresh worlds and R=5 as a conditional later design, rather than silently cutting their independent units.

The primary amplitude=1 homogeneous support case is a useful bridge because support count has a direct meaning. A small exponential-amplitude stress check then separates support existence from observable/effectively recoverable support. The heterogeneous firing test matters more than a broad grid of methods if the paper's claim depends on exchangeability. Support co-firing and decoder overlap must remain distinct manipulations.

## Why the current 7k–10k fit plan is premature

The arithmetic is consistent: 7,240 base / 10,376 maximum core fits, with 24 continuation segments counted separately; the stated GPU-hour range is an unmeasured planning assumption. The problem is decision efficiency, not that a small model can never justify this many fits.

The core campaign repeats a single unvalidated ensemble-transfer assumption thousands of times. Thirty-six worlds improve estimates of performance under a chosen protocol; they cannot establish that initialization variability is the theoretical ensemble or repair an uninformative curve. The expensive precision/entropy/variance factorial should follow a visible density signal and a viable estimator, except for the few controls needed to explain it. Repeated evaluation of existing checkpoints and exact/frozen controls is much cheaper than retraining every ablation bank.

The lowest true density `.0625` has only three target points strictly below it (`.02,.03,.045`) on the proposed native grid. Losing any of these to common coverage, drift, or matching eliminates a candidate near truth under the three-points-on-each-side rule. Requiring common success across six independently trained runs compounds this risk. This is a development issue to measure and lock, not a reason to tune the grid using test truth. The same check is relevant for deployment coverage: a fitted q can be scientifically meaningful even when the discrete checkpoint bank cannot deploy it.

The proper gate is not “the pilot must show positive results.” It is “the pilot identifies a measurable ensemble and tells us what null/failure behavior means.” If the same-data arm is consistently flat while the data-realization arm is informative, preserve density inference and change the ensemble with fresh confirmation. If neither supports the template, reconsider the density estimator within VG-SAE; do not declare Phase 2 complete using Phase 1 evidence, and do not relabel the whole research program as a diagnostic study without the user's agreement.

## Relationship to the existing literature and novelty

The existing repo grounding remains valuable. Soh supplies the selection model and piecewise template; Sparse but Wrong supplies the density/recovery motivation. VAEase, Learned Thresholding, Variational Sparse Coding, and Entropy-Based ELBOs show that variational sparse representations and entropy-based derivations are established. Toward Identifiable Sparse Autoencoders and Unstable Features, Reproducible Subspaces make it inappropriate to claim feature matching or seed stability alone as new. Their exact results were not re-audited in this agent pass; use the PI's source verification and the existing [literature-grounding artifact](../../vg-sae-unified-paper-20260929/LITERATURE_GROUNDING.md) for attributed details.

The new three-paper set does not appear to contain this project's specific uncertainty-curve density estimator in learned SAE coordinates. That is a bounded observation, not an exhaustive novelty certification. The plausible contribution is a coherent conditional VG-SAE model with measured inference limits, followed by a density-selection procedure whose necessary ensemble assumptions and empirical recovery utility are demonstrated. The physics contributes a precise distinction between the model, approximation, observables, and averaging measure. It cannot be reduced to applying physics names to familiar plots.

The existing Phase 1 results are sufficient to motivate this bridge: orthogonal exact/MF agreement, dependence at high overlap, finite encoder optimization error, nonmonotonic relation between reverse KL and marginal accuracy, and the gap between mean/stochastic/hard readouts. They are not evidence of semantic truth, a new theorem, or density-estimator success. Neither SOTA performance nor importing a new neural component is necessary to make the proposed explanation worthwhile.

## Adopt / defer / reject

| Decision | Recommendation |
|---|---|
| Adopt now | Add an explicit model/latent/ensemble table; retain normalized prior, conditional-model wording, C2a/C2b separation and truth-free evaluation. |
| Adopt now | Present the Gram interaction and fixed-model covariance/response identities as explanatory derivations; separate them from native-density retraining curves. |
| Adopt before large execution | Test initialization vs training-data ensembles and optimization horizon at development scale; include a converged-correct flat-U control. |
| Adopt conditionally | Use participation ratio only as a secondary effective-amplitude count if the amplitude stress experiment needs it. |
| Defer | Thermodynamic limits, replica calculations, full phase diagrams, new empirical-Bayes gamma learning, TAP/Bethe inference, structured posterior architectures, extra large-LM training. |
| Reject | RBM=VG-SAE equivalence; transferring RBM phase thresholds or `L~1/p`; self-overlap=truth; learned beta=Nishimori matching; sharp finite response=phase transition; universal density or semantic guarantees. |
| Reject | Treating the entire 7k–10k fit bank as the minimum prerequisite for deciding whether the ensemble template is plausible. |

## Questions for the scientific-skills peer

1. Is same-data seed instability a defensible primary estimand for generating density, or should data-realization variability be primary once the small paired bridge is available? I see a scientific justification for measuring both before choosing; I do not think the original VG template privileges the seed-only choice.
2. Can the converged-correct flat-U example be integrated as an estimator applicability condition without abandoning density inference? My answer is yes, but it weakens claims of broadly automatic q estimation.
3. Which smallest observation would persuade you that the Phase 1 quantity actually helps Phase 2 rather than merely sharing notation? I propose covariance/response distinction plus a within/between and ensemble comparison, not a new formal phase diagram.
4. Which currently mandatory ablations deserve to precede the main 36-world run? I favor only those that distinguish a real density signal from coverage or unfinished-optimization artifacts; reserve the full three-objective bank for explanation after that.

Residual uncertainty at this stage: no experiment establishes which ensemble works, the final Nature version's full text has not been read, and novelty has not undergone an exhaustive search. These uncertainties are explicit reasons for bounded next steps, not reasons to stop the VG-SAE research program.
