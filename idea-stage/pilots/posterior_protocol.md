# B05 exact-posterior diagnostic — preregistered bounded pilot

Status at registration: no B05 experiment results observed. This file is written
before the pilot is executed. Run date: 2026-09-19. This is one exploratory seed,
not a confirmatory result or a general SAE calibration evaluation.

## Question and controls

Separate an achieved finite-budget amortization/training gap from an achieved
mean-field approximation gap and irreducible support ambiguity. The generative
model is known: `x = D s + epsilon`, independent `s_j ~ Bernoulli(.15)`, constant
amplitude 1, `epsilon ~ N(0, .2^2 I)`, beta 25, d = K = 8. Conditions: `D = I`
and the same matrix with column 1 replaced by `.95 e_0 + sqrt(1-.95^2) e_1`.
The dictionary is full rank in both cases; coherence induces posterior ambiguity
at finite noise, not structural nonidentifiability. Paired conditions use the
same sampled supports/noise. The data seed is 20260919, with 8192 training and
2048 independent held-out samples. Support draws are ground truth; enumerated
posterior probabilities are a separate probabilistic reference.

Use the actual `VariationalGarroteSAE.free_energy`, learned-beta mode with beta
frozen at 25, decoder bias disabled, dictionary frozen, amplitude weights zero
and biases inverse-softplus(1) frozen. Only the existing linear gate is trained:
optimizer/init seed 0, AdamW, learning rate .003, weight decay 0, batch 256,
gradient clip 1, exactly 1500 updates, shuffled repeated epochs, final iterate.
Keep existing default gate initialization. No checkpoint, hyperparameter,
iteration-count, or seed selection after seeing results.

## Inference references

Enumerate all 256 supports with the normalized Gaussian likelihood and normalized
Bernoulli prior in float64. Compute exact joint posterior, marginals, evidence,
entropy, Bayes support error, and posterior signal variance. In the orthogonal
control, verify `m_exact = sigmoid(beta D^T x - beta/2 - gamma)`, where
`gamma = log((1-pi)/pi)`. This also proves the linear gate can represent the exact
orthogonal posterior; its fitted error cannot establish a capacity limitation.

Optimize per-example factorized mean-field with 100 fixed cyclic coordinate
sweeps, in order j = 0,...,7, from each of three starts: prior .15, final amortized
gate, and seeded uniform random probabilities (seed 20260920). Update
`m_j = sigmoid(beta D_j^T (x - sum_{k != j} D_k m_k) - beta ||D_j||^2/2 - gamma)`.
Choose the smallest final free energy separately for each sample. Report both
all-start final energies and simultaneous fixed-point residuals. Three starts do
not certify the global mean-field optimum. Sweep count is never extended based
on test results.

`F(q) + log p(x) = KL(q || p(s|x))` uses the same normalized joint density as
enumeration. The exact joint's F is `-log p(x)` and KL is zero. A product of exact
marginals is *not* the exact joint in the coherent condition; its separate
mean-field energy is reported only as a diagnostic.

## Registered outcomes and interpretation

Validity gates: orthogonal marginal max absolute error < 1e-10; independent
model-versus-enumerated free-energy test error < 1e-8; orthogonal optimized
mean-field mean absolute KL < 1e-8. A failed validity gate invalidates the pilot.

Primary coherent-condition refinement outcome:

- **Positive:** mean amortized-minus-refined F >= .1 nat/sample, at least 25%
  reduction in squared distance to exact marginals, refined Brier no worse than
  amortized Brier by > .002, and max fixed-point residual < 1e-6.
- **Negative:** convergence residual < 1e-6, F reduction < .01 nat/sample, and
  marginal-distance reduction < 5%.
- **Null/inconclusive:** remaining cases, including failed convergence.

Separately flag a practically visible achieved mean-field limitation only when
coherent refined KL >= .05 nat/sample and marginal MSE >= .001, with residual
< 1e-6 and valid orthogonal control. The exact posterior's Brier and Bayes support
error describe irreducible ambiguity under this known model. These thresholds
are pilot triage rules, not significance tests. No multi-seed claims.

Report overall/active/inactive Brier, sampled and predicted prevalence, true
support joint NLL, per-coordinate marginal NLL, distance to exact marginals,
mean-field free energy/KL and observed reconstruction energy, posterior mean and
hard-threshold reconstruction risk against sampled clean signals and posterior
expected clean-signal risk, plus convergence residual. All MSE and Brier values
are per coordinate; NLL and KL explicitly distinguish per sample from per atom.
Hard threshold is > .5. The exact joint, product-marginal score, and q-posterior
energy must not be conflated.

## Execution and evidence

Reuse the existing project `.venv`, with no installation or rebuild. One idle
local GPU (physical GPU 2), one CPU thread; planned < 5 minutes and < .1 GPU-hour.
The seeded tensor/kernel and formula tests precede training. Runtime, peak GPU
memory, environment versions, source/protocol hashes, training trace, weights,
per-sample/per-atom CSV, summaries and exact-posterior arrays are saved under
`outputs/idea_discovery_20260919/posterior/`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_idea_discovery_posterior.py -q
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 .venv/bin/python -B scripts/idea_discovery_posterior_pilot.py --device cuda
```

This diagnostic cannot establish calibration with learned dictionaries,
learned amplitudes or beta, real activations, or held-out distributions. The
amortized-minus-refined difference includes training optimization and finite
sample effects. A remaining refined-minus-exact gap includes possible local
optimization error as well as the restriction to independent Bernoulli factors.
