# P1 current-VG feature identity pilot — preregistration

Saved before scientific training or test generation on 2026-09-22. Scope is a
paper-inspired implementation and direction pilot, not a faithful reproduction
of the paper's 15M-example, online-data, original-library training budget.

## Hypothesis and fixed design

Determine whether current VG improves signed dictionary/support identity over
L1, calibration-rescaled L1, Gated, Anthropic-style tanh JumpReLU, and BatchTopK.
Narrow positive magnitudes max(0, Normal(1,.15)), d=20, truth/SAE width=5,
marginal firing p=.4, latent Gaussian hub correlations -.4, 0, +.4. Gaussian
correlation is not Bernoulli Pearson correlation; report realized marginals and
support correlations. QR orthogonal directions differ from original optimized
directions. No outcome-conditioned initialization, retries, or seed exclusion.

Paired worlds 210,211,212. Dictionary and per-model initialization seed=world;
train/cal/test seeds are 10000/20000/30000 + world; batch shuffle seed=40000+world.
All methods reuse the same fixed train32768/cal8192/test16384 independent streams
and dictionary. Train shared batches256 for 3000 updates at fixed lr=.003;
checkpoints after completed updates500,1500,3000. This is 768000 sample
presentations with repeated examples. No input rescaling or LR warmup. JumpReLU
alone uses the jury-prescribed shortened1000-update sparsity warmup.

Five controls per method (equal candidate count, not commensurable parameter units):

| Method | Controls |
|---|---|
| Native full profiled VG | gamma = -2,0,2,4,6 |
| Official SAELens L1 | coefficient = 0,.03,.1,.3,1 |
| Official SAELens Gated | coefficient = 0,.03,.1,.3,1 |
| Official SAELens Anthropic-style JumpReLU | tanh coefficient = .03,.1,.3,1,3 |
| Official SAELens BatchTopK | batch k = 1,1.5,2,2.5,3 |

VG uses all current defaults, full variance/entropy, profiled beta, gate bias -2,
zero decoder/pre-bias, native threshold m>.5 and positive softplus amplitude.
Baselines use pinned SAELens commit8be14080485952f729ed58d674bcddf9778e0aa4,
official SAETrainer, apply_b_dec_to_input=True, initial bias0 and decoder norm .1.
VG and baselines use Adam(weight_decay0). VG unit decoder normalization differs
from official baseline norm handling; record complete model configs. Both center
encoder inputs and add the learned decoder bias to reconstruction. JumpReLU uses
tanh scale4, pre-activation loss3e-6, initial threshold .1, bandwidth2 and warmup1000.
Compared with the paper helper, this uses lr .003 instead of .0003, batch256,
shorter sample budget, fixed QR/data, current pinned implementation, no activation
normalization and shortened1000-step warmup; these choices are fixed before outcomes.

## Selection and leakage prevention

Calibration-only target native achieved L0 =1.8,2,2.2 with absolute distance<=.15.
Primary uses final-step controls ranked by absolute calibration L0 distance,
then calibration MSE and ascending control index. Pairwise comparisons additionally
require achieved calibration L0 difference<=.10. A separate oracle ceiling uses
all checkpoints and controls: signed Hungarian mean cosine descending, mixing
ascending, support micro-F1 descending, then later checkpoint and ascending
control index. Empty feasible sets
are coverage failures, never nearest-L0 substitutions. The jury must be read
before scientific launch and any required changes registered before that launch.

These target/recovery selectors are oracle calibration analyses: true L0 and
true dictionary/support are unavailable at deployment. Separately record
deployment-feasible calibration min-c_dec and min-MSE selectors across final controls without targets
or any truth metric. These are diagnostic comparison selectors, not claims of
successful automatic true-L0 selection.

Store all675 calibration records/checkpoints, then freeze selected IDs and file
hashes before creating any test batch. Evaluate each selected checkpoint once
and the entire225-model final grid once, deduplicating overlap. Full final-grid
curves are preregistered descriptive outputs, not test-set tuning. Do not infer
a wider effective parameter band by comparing fractions of differently scaled
gamma, penalty or k grids. Claims about robustness require actual achieved-L0
curves and adequate coverage; no interpolation through coverage gaps.

L1 rescaling is secondary. Fit five strictly positive per-latent gains by
calibration-input least squares with learned dictionary and native support fixed.
No ground-truth dictionary/support/coefficients enter the fit. Freeze gains with
calibration records and apply once to test. Native L1 remains the primary readout;
rescaling preserves its support and dictionary geometry and is not an extra HPO
chance. No inference function receives hidden ground truth.

## Measurements and interpretation

Primary dictionary matching maximizes signed cosine, with no sign correction for
nonnegative coefficients. Report signed mean cosine, fraction>=.95, off-diagonal
mixing energy, negative mixing energy, signed hub leakage, off-manifold energy,
and c_dec. Absolute matching is secondary. Report support micro/macro F1,
per-feature recall, native L0, sample L0 error, coefficient NMSE and true-positive
coefficient bias/MAE, MSE/EV. Exclude zero-input examples only from norm ratios
and report their count. For VG additionally expected L0, mean-code MSE,
Bernoulli-sampled expected MSE, gate entropy and evaluated profiled beta.
Matching depends only on the fixed learned/true dictionaries, so the same mapping
is used on calibration/test; test data never changes it.

All seed outcomes and coverage failures remain visible. A positive isolated
oracle result is not automatic sparsity discovery. Reduced amplitude bias alone
does not establish VG-specific mixing improvement. This pilot establishes neither
convergence nor broad-regime superiority or formal significance with three worlds.

## Compute gate and reproducibility

Physical GPU1 exclusively (CUDA_VISIBLE_DEVICES=1, logical cuda:0), threads2,
float32, deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8.
Existing project .venv and verified local kernel witness are reused. No installs.
Before full grid, measure150 updates for each of five methods on an engineering
world999210 with no scientific test or scientific interpretation. Save timings.
Estimated total including25% overhead must be<=2 GPUh. If necessary reduce the
uniform budget and checkpoint schedule before scientific test or launch and
record amendment. Scientific process uses a7200s hard timeout and self-stops at
6900s during training; never consume more than2 GPUh for P1. Store per-model
timings and total wall time including setup, snapshots, I/O, calibration and test.

Portable JSON/CSV outputs live under outputs/sbw_20260922/baseline; binary
checkpoints remain there and are excluded from Git. Save protocol, jury, source,
dataset and checkpoint SHA256 values, all model configs, seed rules, dependency
versions, calibration freeze receipt, raw metrics, and completion status.

Prelaunch jury integration: scientific run uses the final-step/independent-oracle
selection rules and JumpReLU1000 warmup in pilot_jury.json. Root explicitly retained
official baseline norm training instead of imposing jury-wide unit decoder
normalization, which would alter the Anthropic norm-weighted recipe. The shared
stream cycles shuffled finite data without replacement within each epoch, unlike
the jury text's with-replacement wording. This is fixed and shared across methods.
Five-method150-update engineering probe projected3800 seconds training and4750
seconds with25% overhead (1.32 GPUh); the full3000-update grid fits the2 GPUh cap.

Invocation (from repository root):

```bash
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -B scripts/run_sbw_baseline_pilot.py --mode throughput
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 7200 .venv/bin/python -B scripts/run_sbw_baseline_pilot.py --mode run
```
