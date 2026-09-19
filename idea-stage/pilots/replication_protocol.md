# Decoder replication pilot: preregistered numerical audit

Protocol fixed at 2026-09-19 02:34 UTC, before execution. Repository source
baseline: `074ad5455adbe18744b502b6badddbce938f8f27`.

## Question and construction

Does the existing `VariationalGarroteSAE.free_energy` admit a replicated-direction
configuration with accurate posterior-mean reconstruction, zero posterior/prior
KL, and decreasing uncertainty energy even though thresholded reconstruction is
zero? This is a constructed feasibility test, not an optimization experiment.
Decoder-duplication reduction of dropout variance is an established mechanism:
[Cavazza et al., AISTATS 2018](https://proceedings.mlr.press/v84/cavazza18a.html),
Section 4, Equation 13 / Proposition 1, and width-dependent retention in Section 5.
This pilot maps that mechanism to the repository's Bernoulli KL and hard inference
implementation; it does not propose a new general theorem or correction method.

Use one input `x = A u`, `A = 1`, input dimension `d = 8`, and unit vector
`u = (1, 0, ..., 0)`. All `r` decoder columns equal `u`. Set all encoder weights
to zero, gate biases to `logit(pi)`, amplitude biases to
`softplus_inverse(A / (r pi))`, and disable decoder bias. The model still uses its
actual sigmoid gate, softplus amplitude, unit decoder, and free-energy code.
Use float64, CPU, seed 0, and `r in {4, 16, 64, 256}`. No training or checkpoint.

Two predetermined prior rules:

1. Fixed field: `gamma = 2`, `pi = sigmoid(-gamma) < 0.5`.
2. Fixed prior expected count: `kappa = 1`, `pi = kappa/r`,
   `gamma = log((r-kappa)/kappa)`; again `pi < 0.5`.

For each rule evaluate both configured beta modes. In `learned` mode keep the
trainable precision parameter at `beta = 2` without updating it; this is **not**
a fixed-mode API, and no claim is made about learned-beta training.

## Predictions and pass/falsification criteria

For all cases, `D (m a) = x`, `KL = prior - entropy = 0`, and
`E = variance = A²(1-pi)/(2 r pi)`. The existing inference rule `m > 0.5`
selects no atoms: hard reconstruction is zero and hard NMSE is 1.

Actual implementation predictions, including its normalization conventions:

- `learned`: `F = beta E - (d/2) log(beta/(2 pi_math)) + KL`,
  with effective precision 2.
- `profiled`: `F = (d/2) log(2 E/d) + KL`, effective precision `d/(2E)`.
  Its omitted additive constants prohibit direct absolute-loss comparisons
  against the learned mode. Energy and scaled-energy epsilon clamps must be
  inactive for all tested cases.

Under fixed gamma, variance ratios follow `E(r2)/E(r1) = r1/r2`, and profiled
loss increments equal `-(d/2) log(r2/r1)`. The zero-clamp analytic objective
decreases without bound along this family as width grows. The finite-epsilon
implementation eventually reaches its numerical floor, outside this grid.

Under fixed kappa, `E = A²(1-kappa/r)/(2 kappa)` increases toward `A²/(2 kappa)`;
this particular `1/r` energy decrease is absent. This does not establish general
replication invariance or a practical mitigation.

Pass only if **all** actual free-energy components agree with these independent
closed forms at `atol = rtol = 1e-10`; posterior/prior KL magnitude is at most
`1e-10`; unit-norm and posterior-mean residual errors are at most `1e-10`; hard
active count is exactly zero and hard NMSE equals 1. The tests also enumerate
all 16 Bernoulli masks for `r=4` to compare direct expected squared error with
the actual energy, independently of the variance formula. Any violated
prediction falsifies this numerical claim (or identifies a construction bug
requiring an explicitly documented correction); do not relabel it successful.

## Budget, evidence, and limits

Expected execution cost: seconds on one CPU thread; 0 GPU-hours. Use the existing
Python 3.14 environment; no packages, network, or external compute required.
The JSON artifact records all model configuration fields, seed, runtime,
platform/Python/PyTorch versions, source/script/protocol SHA-256 digests,
source baseline commit, commands, check outcomes, and every numeric row. CSV
contains the exact same unrounded numeric rows. No existing results are changed.

Commands (run from repository root):

```bash
.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_idea_discovery_replication.py -q
.venv/bin/python -B scripts/idea_discovery_replication_pilot.py --output-dir outputs/idea_discovery_20260919/replication
```

The construction establishes existence at selected widths, not how often
training finds it, whether it lowers loss against optimized alternatives, or
whether it degrades real feature recovery. The input has one fixed positive
amplitude and one direction, and does not test signed/multiple directions,
data heterogeneity, generalization, learned beta dynamics, or baseline SAEs.
Decoder bias is explicitly disabled. With a free decoder bias, this single
constant input admits a trivial bias-only fit. The audit has not been extended
to centered nonconstant data, so it does not establish that default-bias training
chooses replicated atoms over a bias-only solution.
The matched-count control changes the prior field with width. Passing it is
evidence about this single constructed family, not causal proof of a remedy.
