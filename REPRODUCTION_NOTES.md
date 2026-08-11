# Reproduction Notes

This implementation follows the paper where specified and records choices where
the paper leaves details open.

## Adopted Choices

- **Gamma sign:** the code uses `+ gamma * sum(m)` in the negative log posterior.
  The printed reduced equation has `- gamma * sum(m)`, but the stated prior
  `P(s_i) proportional to exp(-gamma s_i)` and the prose that larger gamma gives
  stronger sparsity imply the positive penalty.
- **Mask parameterization:** masks are represented as logits and mapped through
  `sigmoid`. This keeps `m_i` inside `(0, 1)` while preserving gradients near the
  paper's boundary initialization.
- **Mask initialization:** `mask_init=0.999`, a near-one value. Exact `1.0` is
  singular for the entropy term and can freeze gradients under direct clamping.
- **Weight decay:** AdamW defaults to `weight_decay=0.0`, consistent with the
  paper's flat prior over weights.
- **Synthetic active count:** finite-`N` spike-and-slab weights use
  `round(rho_data * N)` active variables, matching sparse paper examples such as
  `5/256`.
- **Synthetic noise:** targets include additive Gaussian noise calibrated to the
  requested SNR.
- **Appendix B inference:** SciPy NNLS is used when available; otherwise the code
  falls back to projected-gradient nonnegative fitting.
- **VG-SAE vector output:** the SAE objective treats each reconstructed vector as
  an isotropic Gaussian observation and replaces the scalar squared residual in
  the paper with a squared Euclidean norm. The Bernoulli variance correction
  becomes `sum_j m_j (1-m_j) a_j^2 ||d_j||_2^2 / 2`, which is the direct
  vector-output extension of Eq. (10).
- **VG-SAE sparsity field:** `lambda_sparsity` is the paper's `gamma` and may be
  any finite real number. The normalized expected negative log prior is
  `gamma * sum(m) + n_latents * softplus(-gamma)` per sample. Positive gamma
  favors sparse support; negative gamma is a valid dense-prior regime. The
  minus sign printed in Eqs. 9/13 conflicts with Eq. 6 and is treated as a typo.
- **VG-SAE precision:** `beta_mode="profiled"` preserves the beta-eliminated
  objective; `"fixed"` uses the full Gaussian NLL with constant beta; and
  `"learned"` optimizes `log_beta`. The legacy `trace_beta` flag remains an
  alias when `beta_mode` is omitted. `--beta` does not set the profiled optimum;
  it is an initial/reporting value in that mode.
- **Entropy ablations:** the variational objective has unit entropy weight.
  `entropy_weight != 1` and `use_entropy_term=False` are explicit ablations,
  not alternative derivations of the same posterior.
- **SAELens baselines:** SAELens is installed from commit
  `8be14080485952f729ed58d674bcddf9778e0aa4` (v6.47.0). The project names for
  Standard, TopK, BatchTopK, JumpReLU, and Gated are direct identity aliases of
  the upstream training classes and configs. The local factory only translates
  `input_dim`/`n_latents` into upstream `d_in`/`d_sae`; architecture behavior,
  losses, optimizer steps, and inference state conversion remain upstream.
  The former project `L1ReLUSAE`/`L1SAEConfig` names and the local `SAEConfig`,
  `UnitNormDecoderMixin`, and `CenteredLinearSAE` symbols were removed; direct
  construction now uses
  `StandardSAE(StandardSAEConfig(d_in=..., d_sae=...))`. The factory accepts
  `standard` variants while retaining `l1` variants as experiment-compatible
  selectors. Omitting L1-specific arguments uses the upstream defaults
  (`l1_coefficient=1.0`, `decoder_init_norm=0.1`), rather than the former local
  coefficient default of `1e-3`; experiment call sites that require `1e-3` set
  it explicitly. Old local-L1 checkpoints and the old
  `config`/`encoder`/`decoder`/dict-forward interface are not compatible with
  the official `cfg`/`W_enc`/`W_dec`/tensor-forward interface.
- **SAELens VG architecture:** `src.saelens_vg` follows the upstream custom-SAE
  registry/config/trainer/save boundaries. Public `encode` and dead-feature
  statistics use hard posterior support; `training_forward_pass.sae_out` and the
  free-energy loss use the expected code. Their log names are deliberately
  distinct. Training-time activation/reconstruction intervention hooks are not
  supported because one upstream pre/post hook cannot unambiguously represent
  VG's gate probability, expected code, and hard code; ordinary inference hooks
  remain available.
- **SAELens activation boundary:** `expected_average_only_in` scaling is the
  scalar `sqrt(d_in) / mean(||x||_2)` fitted on training activations only, with no
  centering. A single Arrow cache can be split by cache-row group as a leakage
  reduction fallback, but separately caching source-dataset train/validation
  splits is stronger because packed rows need not coincide with documents.
- **SynthSAEBench notebook:** notebook 09 uses fresh seeded official synthetic
  activation streams and `eval_sae_on_synthetic_data`. Common hard comparisons
  use `sae_l0 / d_sae`; VG posterior density and expected reconstruction are
  separate diagnostics. The benchmark model snapshot is fixed at
  `b2efd8b919ae46d6d487c73d46db5ee52813621d`. Full mode remains an exploratory
  sweep rather than the official 200M-training-sample leaderboard protocol.
- **Stage-1 custom baseline:** observations are noiseless linear mixtures of an
  overcomplete random unit dictionary. Feature supports are independent
  Bernoulli draws with marginal probabilities rank-skewed from
  `support_density` using `frequency_skew=0.5`. Stage-1 configs that would need
  the generic generator's `0.95` probability cap are rejected so the requested
  mean remains exact. Nonzero amplitudes are independent `Exponential(scale=1)`
  draws. The baseline adds no
  dictionary coherence, correlated firing, or hierarchy.
  `ground_truth_num_features` sets the generating dictionary width, while
  `sae_width` independently sets the learned latent width. Legacy `n_features`
  sweep configs remain readable and set both widths. The default matched
  baseline uses `input_dim=128`, `ground_truth_num_features=1024`,
  `sae_width=1024`, and `support_density=0.01` (expected true L0 `10.24`). Its
  calibrated TopK and BatchTopK grids currently span `k<=128`. Sweep output
  directories and W&B groups use a resolved-config ID such as
  `stage1_din128_gt1024_sae1024_sd001_seed0`; W&B also records it as `exp_id`
  in config and summary. The manifest fingerprint remains the guard for config
  axes omitted from this readable ID.
- **Width-aware sparsity plots:** `rho_model`, average L0, and expected L0 are
  defined over all `sae_width` learned latents. The data reference line is
  `sum(feature_probabilities) / sae_width`, rather than `support_density`, so it
  remains valid when the ground-truth and SAE widths differ. Dictionary previews
  report empirical pairwise cosine similarity instead of a configured coherence
  parameter. VG-SAE's `rho_model` is the posterior mean probability `mean(m)`,
  whereas baseline `rho_model` values are hard inference densities. Therefore
  VG peak locations on this soft-density axis are not directly hard-L0 matched;
  `average_l0 / sae_width` is the corresponding hard-density diagnostic.
- **Rectangular recovery matching:** decoder atoms use rectangular Hungarian
  matching. Unmatched ground-truth features remain zero predictions (false
  negatives), while unmatched learned latents are appended against zero targets
  (false positives). Support and latent-recovery metrics use this union; raw
  model density and L0 always use every SAE latent.
- **Multi-seed mask panels:** representative heatmaps use the lowest seed shared
  by every plotted method, so all rows visualize the same regenerated dataset.
- **Experiment 07 controls:** L1's GMM mask threshold is fitted on training
  activations and reused unchanged on held-out data. Dead features use the
  SAELens strict `steps_since_fired > window` boundary; notebook 07 sets the
  window to 100 for its 1000-step run (1 for the four-step smoke run), so AuxK
  is exercised. Full mode is a seed-0 calibration: controls within each method
  share initialization and batch order, and the recorded `init_seed` is stable
  when a grid changes. The grids target paired VG density `0.0038--0.962` where
  feasible. Gated's zero-coefficient run is an unregularized boundary anchor
  (`rho_model` about `0.984`), not an interior matched-density point. TopK is
  bounded below near `1 / d_sae`; Standard's train-fitted GMM support reaches
  only about `0.46`; and JumpReLU reaches about `0.011` but is nonmonotone in its
  L0 coefficient under this 1000-step protocol. JumpReLU is therefore compared
  only by measured density, not coefficient order. These seed-calibrated ranges
  do not establish cross-seed robustness. Revised six-method outputs use a new
  directory; the original four-method artifacts remain explicitly marked as
  legacy.
- **Experiment 07 artifact pipeline:** the parallel runner uses one subprocess
  per run so method-paired initialization and batch seeds cannot race. Stage-1
  command-line overrides expose `input_dim`, `ground_truth_num_features`,
  `sae_width`, `support_density`, seed, and repeatable per-method sparsity
  controls; a sweep directory contains one fixed data condition. Every run
  stores resolved data/model/training metadata and both `last` and `best` state
  dictionaries. `best` is the minimum full-training objective at a recorded
  history step. The evaluator processes both checkpoint kinds by default, while
  notebook 10 plots `last` by default because notebook 07 used the final model.
  Evaluation refits the L1 GMM only on the regenerated training split, persists
  aligned masks for heatmaps, and aggregates only after worker completion.
  Source/package fingerprints and the train/eval devices are recorded, and
  changed evaluator code invalidates old caches. The serialized data `kind` is
  the extension seam for later
  SynthSAEBench or real-activation adapters. Notebook 10 deliberately uses the
  shared method order for heatmap rows instead of notebook 07's accidental
  alphabetical groupby order.
- **uv GPU environment:** Linux `uv` resolution takes PyTorch 2.13 from the
  official CUDA 12.6 wheel index. This keeps the project compatible with the
  installed NVIDIA driver while preserving the locked PyTorch version. The SAE
  sweep launcher reads `WANDB_API_KEY` from the ignored project-root `.env`
  without overriding an explicitly exported value; credentials are never part
  of sweep configs or artifact fingerprints.
- **SAE optimizer prior:** `fit_sae` defaults to `weight_decay=0.0`, avoiding an
  extra Gaussian weight prior that is not part of the VG construction.

## Known Limits

- The exact gamma sweep used in the paper is not reported.
- Real-world UCI column handling is exposed through generic CSV utilities rather
  than hardcoded dataset-specific column lists.
- The paper reports large ensemble averages; this repository provides the core
  functions and small tests, not a full 20,000-ensemble reproduction run.
