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
- **VG-SAE sparsity field:** `lambda_sparsity` is the paper's `gamma`, so it is
  constrained to be non-negative. Negative values would correspond to a dense
  prior and break the stated physical meaning that larger `gamma` enforces
  sparsity.
- **SAE optimizer prior:** `fit_sae` defaults to `weight_decay=0.0`, avoiding an
  extra Gaussian weight prior that is not part of the VG construction.

## Known Limits

- The exact gamma sweep used in the paper is not reported.
- Real-world UCI column handling is exposed through generic CSV utilities rather
  than hardcoded dataset-specific column lists.
- The paper reports large ensemble averages; this repository provides the core
  functions and small tests, not a full 20,000-ensemble reproduction run.
