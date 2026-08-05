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
  Standard/L1, TopK, BatchTopK, JumpReLU, and Gated are direct identity aliases
  of the upstream training classes and configs. The local factory only translates
  `input_dim`/`n_latents` into upstream `d_in`/`d_sae`; architecture behavior,
  losses, optimizer steps, and inference state conversion remain upstream.
  The former local `SAEConfig`, `UnitNormDecoderMixin`, and `CenteredLinearSAE`
  symbols were removed; direct L1 construction now uses
  `L1SAEConfig(d_in=..., d_sae=...)`. Omitting L1-specific arguments now uses
  the upstream defaults (`l1_coefficient=1.0`, `decoder_init_norm=0.1`), rather
  than the former local coefficient default of `1e-3`; experiment call sites
  that require `1e-3` set it explicitly. Old local-L1 checkpoints and the old
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
- **Experiment 07 controls:** L1's GMM mask threshold is fitted on training
  activations and reused unchanged on held-out data. Dead features use the
  SAELens strict `steps_since_fired > window` boundary; notebook 07 sets the
  window to 100 for its 1000-step run (1 for the four-step smoke run), so AuxK
  is exercised. Revised six-method outputs use a new directory; the original
  four-method artifacts remain explicitly marked as legacy.
- **SAE optimizer prior:** `fit_sae` defaults to `weight_decay=0.0`, avoiding an
  extra Gaussian weight prior that is not part of the VG construction.

## Known Limits

- The exact gamma sweep used in the paper is not reported.
- Real-world UCI column handling is exposed through generic CSV utilities rather
  than hardcoded dataset-specific column lists.
- The paper reports large ensemble averages; this repository provides the core
  functions and small tests, not a full 20,000-ensemble reproduction run.
