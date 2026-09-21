# P3: exact joint-support mechanism pilot — frozen before execution

Protocol version 1, 2026-09-22. This is a five-latent mechanism test within the
original VG-SAE development project. It does not supply a scalable SAE or claim
a full generative ELBO: amplitudes remain input-dependent point estimates.

- Worlds 210/211/212; latent Gaussian hub correlation −.4/+.4; firing probability
  .4; d=20, J=5; QR-orthonormal truth; max(0, Normal(1,.15)) amplitudes; no noise.
- Independent train/cal/test counts 32768/8192/16384. Dictionary and model seed
  = world; split seeds = 10000/20000/30000 + world; minibatch seed = 40000 + world.
  Actual shared provider is shuffled cycling through finite data without
  replacement within each epoch (not the jury's initial replacement wording).
- Gamma 0/4/8; beta fixed at 10 for every arm, represented by an explicitly
  frozen log_beta. Arms: current amortized product VG, optimized product VG,
  exact 32-state VG. All share core architecture, initial decoder/amplitude
  heads/bias, data and minibatches. Unused gate heads in the last two arms are
  frozen. Decoder bias starts at zero, gate bias at −2. Trainable decoder bias
  also centers the head inputs, as in current VG.
- Adam lr .003, betas (.9,.999), weight decay 0, float32, gradient norm clip 1;
  remove radial decoder gradient and normalize columns after each step.
  2000 updates, batch 128, checkpoints 500/1000/2000: 54 primary fits.
- Exact energy: E(s)=beta/2 ||x−b−D(s*a)||² + gamma |s|. Exact F is
  −logsumexp(−E) + J softplus(−gamma) −d/2 log(beta/(2pi)). Enumerate binary
  states in ascending integer index; joint MAP ties use the first state.
- Product F includes complete analytic quadratic risk, Bernoulli entropy and
  normalized prior with the same Gaussian normalizer. Product optimizer uses
  3 constant starts (.1,.5,.9), cyclic coordinates 0..4, maximum 50 sweeps.
  Per-sample convergence requires max iterate change <1e−5 AND simultaneous
  fixed-point residual <5e−5. Choose lowest complete terminal F among all starts
  independently for each sample, breaking ties by start index. Differentiate
  with detached selected m for all samples to preserve paired data and updates.
  For unconverged samples this is the fixed-q partial gradient of F(q,theta),
  not an envelope gradient of optimized F. No sample is excluded. A batch passes only if
  every selected sample converged. Report every failed batch and per-start
  convergence. Less than 95% passing training batches, or evaluation residual
  failures, leaves covariance attribution unresolved. Three starts do not
  certify a global mean-field optimum.
- Native readout: exact joint MAP; others marginal>.5. Exact marginal>.5 is a
  prespecified common-readout control. Decoder matching is signed Hungarian;
  retain cosine/recovered fraction, signed hub leakage, squared mixing and
  negative mixing, outside-span energy, support micro/macro/per-feature F1 and
  recall, native/expected/true L0, coefficient error, hard/mean/expected risk,
  hard EV and c_dec. Geometry is frozen before support evaluation.
- Final checkpoint primary. Calibration-only selection at targets 1.8/2/2.2
  minimizes native hard-L0 distance, then hard MSE, then gamma index. Target
  tolerance .15 and pair-L0 tolerance .10 are mandatory for matched claims.
  Save config, hashes, every calibration result and chosen IDs before creating
  test tensors. Evaluate every predeclared final fit exactly once; intermediate
  checkpoints provide calibration trajectories only. Test never tunes gamma.
- Acceptance: exact beats optimized MF in ≥2/3 worlds for each sign by cosine
  ≥.05 OR recovered fraction ≥.20, AND support F1 ≥.03 AND mixing decrease ≥.01,
  with matched L0. Dictionary benefit must persist with common marginal readout.
  All same-gamma paired differences are descriptive when grid matching fails.
  EV loss >.02 is a fidelity tradeoff. Missing convergence/coverage is unresolved.
- The optional 18 oracle mixed-initialization fits are dropped before any test
  access to preserve the minimal 54-fit random-initialization mechanism budget.
  No post-test expansion is authorized by this protocol.
- Resource plan: CPU with two Torch threads, 100-update optimized-MF runtime
  probe before grid; no GPU allocation. Forecast must fit two hours wall time.
  Report CPU wall/core hours separately from GPUh=0. No external API or installs.
- Numerical gates: state-weighted risk equals direct enumeration; product state
  risk equals mean-risk plus variance; orthogonal exact factorization; exact
  log-partition and envelope gradient equality; monotone coordinate descent,
  stationarity and finite differences; converged MF envelope finite differences;
  known dictionary/code identity metrics; explicitly frozen beta.

This finite, repeated-data 256000-presentations-per-fit pilot deliberately differs
from the paper's online 15-million-sample training and optimized orthogonalization.
Fixed beta10 differs from P1's profiled precision and does not isolate an effect
relative to P1. Negative or unresolved P3 results do not reject the VG-SAE project.
