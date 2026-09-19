# VG-SAE code and artifact audit for idea discovery

Date: 2026-09-19. Scope: read-only inspection of the current working tree and
existing local artifacts. No training or checkpoint re-evaluation was performed
for this audit. The only file written is this report. Existing user changes were
preserved. Root `AGENTS.md` and `.agents/project-memory.md` were read; no nested
instructions were found in the inspected source, docs, config, or output paths.

## 1. What is actually implemented

The local model is an amortized, vector-output extension of Variational Garrote:

\[
\bar x=x-b,\quad m=\sigma(W_g\bar x+b_g),\quad
a=\operatorname{softplus}(W_a\bar x+b_a),\quad h=m\odot a,
\quad \hat x=b+Dh.
\]

Both encoders are linear, with independent trainable parameters. Copying the
decoder transpose into the amplitude encoder is initialization, not permanent
weight tying. Decoder columns are normalized by default. Source:
`src/sae_model.py:100-140,159-190`; `REPRODUCTION_NOTES.md:63-70`.

For one observation, the mean-field expected Gaussian energy is

\[
E(x)=\tfrac12\|\bar x-D(m\odot a)\|^2+
\tfrac12\sum_j m_j(1-m_j)a_j^2\|D_j\|^2.
\]

With learned precision, the implemented loss is the sample mean of

\[
\beta E(x)-\tfrac d2\log(\beta/(2\pi))+
\gamma\sum_jm_j+K\operatorname{softplus}(-\gamma)-\eta\sum_jH(m_j).
\]

The default posterior objective has `eta=1`; removing variance or changing
entropy weight is an ablation. In profiled mode, the whole minibatch uses
`beta*=B*d/(2*sum_b E_b)` and the Gaussian term becomes
`(d/2)*log(2*sum_b E_b/(B*d))` in the batch-averaged loss. This is a minibatch
profile objective, not a fixed known observation precision. Source:
`src/sae_model.py:218-284`; `REPRODUCTION_NOTES.md:26-45`.

Public inference uses hard support and unshrunk amplitude:
`s=1[m>threshold]`, `z_hard=s*a`, with default threshold `0.5`
(`src/sae_model.py:302-310`). The SAELens adapter also exposes hard codes while
the training loss reconstructs with expected codes
(`src/saelens_vg.py:195-225,373-407`). Stage-1 evaluation currently uses `>=`
for its explicit hard metric family (`src/sae_sweep_eval.py:146-168,258`), so an
exact-boundary toy experiment should choose and record one tie convention.

Interpretation limits:

- `m` is a variational support responsibility conditional on the learned,
  input-dependent deterministic amplitude. Its name alone establishes neither
  frequentist calibration against generating support nor epistemic uncertainty.
- Mean-field support independence omits posterior competition between coherent
  atoms. Linear amortization is another independent approximation.
- Profiling or learning beta also absorbs model misspecification; reported beta
  is not automatically the generating noise precision.
- Inference-only decoding changes cannot improve the saved dictionary's recovery
  or MCC. Such a claim needs retraining and separate evidence.

## 2. Prior floor: a critical interpretation check

Holding all other quantities fixed, if `a_j=0`, likelihood energy no longer
depends on `m_j`. Its remaining objective is

\[
f(m_j)=\gamma m_j-H(m_j)+\operatorname{softplus}(-\gamma),\qquad
f'(m_j)=\gamma+\log\frac{m_j}{1-m_j}.
\]

For unit entropy weight the optimum is therefore
`m_j=pi=sigmoid(-gamma)`, even when that latent makes no reconstruction
contribution. For entropy weight `eta>0`, this stationary point is instead
`sigmoid(-gamma/eta)`; that is an ablated objective.

Consequently `sum(m)` can grow with dictionary width and prior probability even
when most amplitudes vanish. A large expected/hard L0 ratio alone is insufficient
evidence of useful dense computation or failed posterior calibration. The
hard/expected reconstruction gap is a separate observable and does remain in
the long-run artifacts below. `sum(m)-K*pi` is only an algebraic diagnostic;
it is not a validated new L0 definition and may be negative.

Appropriate additional measurements are amplitude-weighted posterior variance,
per-latent KL to the prior, distribution of `m-pi` conditional on amplitude, and
the actual energy carried by the hard/expected reconstruction difference.
Current unweighted uncertainty is `mean_x m_j(1-m_j)`
(`src/sae_evaluate.py:57-60`), so it can be dominated by the same prior floor.

## 3. Existing data, training, and evaluation seams

| Component | Available capability and source |
| --- | --- |
| Generic synthetic data | Known dictionary, supports, amplitudes, clean/noisy observations; independent Bernoulli firing, frequency skew, constant/uniform/exponential amplitude laws, common-mode dictionary construction, Gaussian observation noise (`src/sae_data.py:25-104,127-212`). |
| Stage-1 standard data | No noise or added coherence, overcomplete dictionary, independent firing, skew default 0.5, matched active second moments across amplitude laws. `SweepConfig` explicitly rejects noise/coherence for this named data kind (`src/sae_sweep.py:245-263`). A new controlled noisy pilot should use the generic generator in its own output root rather than mutate this baseline. |
| Train/test generation | One dictionary is generated for a combined sample pool before splitting; do not independently call the generator with new seeds for test data and accidentally change the dictionary (`src/sae_sweep.py:406-446`). |
| Local pilot trainer | `fit_sae` handles VG and official SAELens models, deterministic seeds, loss histories, best/last state (`src/sae_train.py:111-224,227-290`). The official-baseline path uses upstream trainer/Adam; local VG uses AdamW with zero weight decay by default. |
| Baselines | Pinned official SAELens L1/ReLU, TopK, BatchTopK, JumpReLU, Gated, not project substitutes (`README.md:15-20`; `REPRODUCTION_NOTES.md:46-62`). |
| Stage-1 metrics | Hungarian atom matching with rectangular union: unmatched true latents are false negatives, unmatched learned latents are false positives. Separate hard/expected reconstruction, support PR/F1/AP/AUC, recovery, shrinkage, dead fraction (`src/sae_sweep_eval.py:35-92,199-265,324-365`). |
| Stage-2 | Fixed official SynthSAEBench model at revision `b2efd8b919ae46d6d487c73d46db5ee52813621d`; input 768, truth width 16,384, learned width 4,096; streaming MCC, uniqueness, per-latent classifier metrics and hard/expected VG diagnostics (`README.md:175-205`). |
| Stage-3 | Existing Gemma/Llama streaming and CE/KL intervention stack; real data has no support/dictionary ground truth (`docs/stage3_real_activations.md:7-26,109-126`). Full default is 390 train/eval jobs and is inappropriate for an initial idea pilot (`docs/stage3_real_activations.md:44-52`). |

The generic generator's `coherence` is a construction parameter, not achieved
pairwise cosine: a unit common vector is mixed with unnormalized Gaussian
columns (`src/sae_data.py:139-144`). A coherence experiment should report actual
Gram statistics or explicitly construct a pair with prescribed cosine.

Current diagnostic code has no proper scoring-rule calibration evaluation for
support probabilities. AP/AUC are ranking metrics; F1 at a threshold is a
classification metric. They do not establish probability calibration.

## 4. Raw artifact observations and exact conditions

These are observations from existing CSVs, not a new experiment or an integrity
certification. Existing local artifacts include many ignored checkpoints and
untracked summaries; only some plots/diagnostic CSVs are tracked by Git.

Primary VG source:

`outputs/runs/stage2_synthsaebench16k_l0calibrated_sae4096_train200m_test25m_beta_learned_seed0/summary/last/final_metrics.csv`

Filter: `method=vgsae`, `checkpoint_kind=last`, seed 0. The neighboring
`sweep_config.json` records 199,999,488 training samples, 24,999,936 evaluation
samples, batch size 1,024, learned beta initialized at 1, constant learning rate
0.0003, and the pinned 768/16,384/4,096 generator/SAE dimensions. The training
config enables BF16 autocast for data and SAE. The pinned generator has
`scale_children_by_parent=false`.

| gamma | hard L0 | expected L0 | `4096*sigmoid(-gamma)` | hard EV | expected EV | MCC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6.00 | 15.2734 | 18.3170 | 10.1279 | 0.77466 | 0.77983 | 0.62042 |
| 2.80 | 19.2196 | 253.8832 | 234.7998 | 0.77355 | 0.83368 | 0.71734 |
| 2.22 | 25.7196 | 440.8360 | 401.2802 | 0.77692 | 0.85262 | 0.71680 |
| 1.99 | 30.2536 | 543.1684 | 492.5721 | 0.77504 | 0.86096 | 0.70155 |
| 1.82 | 36.3314 | 633.2790 | 571.1211 | 0.78086 | 0.86894 | 0.69698 |
| 1.71 | 40.4441 | 697.1217 | 627.3586 | 0.77350 | 0.87291 | 0.67619 |
| 1.63 | 45.6633 | 744.6815 | 671.0492 | 0.78305 | 0.87783 | 0.68413 |

The prior-floor column is newly calculated from saved gamma and width; it is
not an existing evaluation field. At gamma 1.99 the expected minus prior-floor
count is 50.5963. The proximity suggests a testable explanation, not proof that
every low-amplitude feature sits at its prior.

The profiled counterpart has the same path with `beta_profiled_seed0`. At gamma
2.01 it reports hard L0 30.1156, expected L0 531.4262, hard EV 0.77685, expected
EV 0.86055, MCC 0.71111. Thus the observed gap is not confined to learned beta.

Preserved baseline source:

`outputs/runs/stage2_synthsaebench16k_l0calibrated_sae4096_train200m_test25m_seed0/summary/last/final_metrics.csv`

| method/control | hard L0 | hard EV | MCC |
| --- | ---: | ---: | ---: |
| L1 / 1.36 | 30.0724 | 0.78645 | 0.61801 |
| TopK / 30 | 29.9945 | 0.80059 | 0.71472 |
| BatchTopK / 30 | 30.0533 | 0.80148 | 0.75806 |
| JumpReLU / 0.61 | 29.8805 | 0.80413 | 0.74501 |
| Gated / 1.38 | 30.0783 | 0.80021 | 0.66299 |

These local observations do not support a general claim that the current VG
model dominates existing architectures. They do identify an inference-quality
problem worth isolating. Comparisons are close in achieved hard L0, not exactly
matched. The evaluation stream was reused for coefficient calibration, so it
is not an untouched test set (`README.md:215-237`;
`REPRODUCTION_NOTES.md:127-153`). One seed is insufficient for a robust ranking.
Stage-2's best-match per-latent classification also differs from Stage-1's
rectangular-union metric (`REPRODUCTION_NOTES.md:105-112`).

The older tracked Stage-1 root `outputs/runs/stage1_custom_baseline` has
`d_in=16`, truth/learned width 128, 512 training and 512 test observations,
support density 0.1, seed 1, 1,000 updates at learning rate 0.01. It should not
be presented as the current default 128/1,024/1,024 experiment. The mode-specific
Stage-1 roots record 8,196 training and 1,024 test observations, density 0.01,
seed 0, and 1,000 updates. Config provenance matters more than the directory
name alone.

## 5. Three bounded pilots

### P0. Measure whether the raw uncertainty signal is mostly a prior floor

Use one saved Stage-2 learned-beta checkpoint near hard L0 30 and a fresh,
bounded generator stream of 4,096–16,384 observations. Repeat for gamma 6 as
a sparse anchor if the initial result is informative. No retraining is needed.
Save raw per-latent aggregates and inspect:

- `m`, `m-pi`, `a^2`, `m(1-m)`, and `m(1-m)*a^2*||D_j||^2` by amplitude
  quantile and firing-frequency quantile.
- Bernoulli `KL(q_j||prior_j)` and the fraction of raw `sum(m)` contributed by
  latents with negligible amplitude/contribution.
- Expected/hard reconstruction MSE, EV, hard L0, and the norm of the difference
  between their reconstructions on exactly the same samples.

Distinguishing question: is large expected L0 principally an uninformative
prior count, while the actual reconstruction decision error is concentrated in
a smaller set of consequential ambiguous features? Stop this idea if those
signals are not distinguishable or only restate ordinary amplitude magnitude.
Amplitude magnitude, activation frequency, and decoder norm are required
non-Bayesian ranking controls. Information gain is a candidate diagnostic, not
a claimed novelty or proven feature-quality score.

### P1. Separate amortization error from mean-field error with exact support

Construct a tiny fixed-dictionary oracle model: `d=6`, `K=8`, constant known
amplitude, known independent support prior, and known Gaussian noise. Enumerate
all 256 support patterns to calculate exact `p(s|x)` and exact marginals. Use
three dictionary seeds, two noise settings, and an independent-atom versus
prescribed-correlated-pair condition. A few thousand observations suffice for
a screening experiment; keep train/calibration/test separate.

Compare (a) exact support posterior, (b) per-sample optimized mean-field
probabilities under the identical known model, and (c) the current linear
amortized gate trained on the same objective with dictionary, amplitude, and
beta frozen. Fixed beta can be represented by constructing learned-beta mode
and freezing `log_beta`; no change to the public model modes is needed. Constant
amplitude is achieved by freezing zero amplitude weights and its softplus
inverse bias. Record this as an oracle conditional-support model, not the full
learned VG-SAE.

Evaluate marginal Brier/log scores, reliability conditioned on feature
prevalence, support AP/F1, and exact/mean-field/amortized reconstruction risks.
Compare free energies using multiple initializations for per-sample optimization
to avoid confusing a poor optimizer with a variational-family limit. This can
identify whether the next method should target gate architecture, dependence,
or the decision rule; it does not establish calibration of full learned SAE
features on real activations.

### P2. Determine whether hard-decoding loss is recoverable at fixed sparsity

On a saved checkpoint and bounded fresh samples, compare default posterior
thresholding to amplitude-weighted ranking, residual-aware greedy support
selection, and fixed-support nonnegative amplitude refitting. Use a calibration
split to select controls, then freeze them on a test split. Match actual hard
L0, report per-sample L0 distribution and latency, and give the same refitting
opportunity to applicable deterministic baselines.

First apply the alternatives on P1's exact tiny model, where exhaustive
support enumeration supplies a meaningful oracle. Then test one Stage-1 or
Stage-2 checkpoint. The principal outcomes are hard reconstruction quality and
support quality, not dictionary recovery (the dictionary is frozen). Record
whether the gain comes from support choice or amplitude refitting. A gain that
requires much more compute than a baseline is not a free inference improvement.

Go/no-go should depend on consistent held-out gains beyond threshold tuning and
plain refitting at matched sparsity and measured inference cost. A small
evaluation-only screening can justify a later training change; it does not
justify launching the full real-activation sweep.

## 6. Implications for idea selection

The codebase already supports most conventional SAE comparisons, realistic
synthetic data, and real-model intervention evaluation. Simply adding VG to
another benchmark has limited methodological novelty. The most defensible
project-specific starting point is the relationship between support prior,
amplitude, uncertainty diagnostics, and hard reconstruction decisions. The
three pilots above can disprove attractive but vague Bayesian-uncertainty
claims before spending large training budgets.

Literature novelty was not audited in this code subtask; these are evidence-
grounded candidate questions for the main idea-discovery workflow.
