# B05 known-model posterior pilot

The registered refinement rule is **positive**, but this is a diagnostic result,
not evidence of generally improved probability calibration. In the coherent
condition, coordinate refinement improves Brier and reverse KL while worsening
held-out marginal log loss. The orthogonal control also shows substantial
undertraining of the amortized gate, even though its architecture can express
the exact answer.

## Scope and execution

Known dictionary, amplitude 1, Bernoulli prior .15, Gaussian noise standard
deviation .2, beta 25; d = K = 8. The two dictionaries are identity and identity
with a single atom pair at cosine .95. Paired data use one support/noise seed,
8192 training samples and 2048 independent test samples. Observed test active
prevalence is .146912. Only the actual VG-SAE linear gate is trained, for the
registered 1500 AdamW steps. The three-start coordinate solver uses exactly 100
sweeps, selecting the smallest final per-sample free energy.

Computation used the existing environment and physical GPU 2 (RTX A6000).
Measured training/evaluation time was 8.68 seconds, excluding interpreter import
and final file export; peak allocated CUDA memory was 67.67 MiB. No dependencies
were installed and no training budget or inference setting was adjusted after
the results were observed.

## Main measurements

All Brier and marginal MSE values are per atom. KL is `KL(q || exact posterior)`
in nats per sample; the exact-joint row uses the true joint distribution.

| Dictionary | Method | Brier | KL | Marginal MSE to exact | Marginal NLL / atom |
|---|---|---:|---:|---:|---:|
| Orthogonal | Exact joint | .002778 | 0 | 0 | .010676 |
| Orthogonal | Trained gate | .015857 | 3.464034 | .013404 | .082113 |
| Orthogonal | Refined mean-field | .002778 | approximately 0 | approximately 0 | .010676 |
| Cosine .95 | Exact joint | .011172 | 0 | 0 | .036569 |
| Cosine .95 | Trained gate | .033355 | 3.630812 | .022442 | .130688 |
| Cosine .95 | Refined mean-field | .015027 | .067174 | .004292 | .160296 |

The coherent refinement reduces mean free energy by 3.563638 nats/sample and
marginal MSE by 80.88%. Brier improves by .018328, with active-coordinate Brier
.160493 → .050192 and inactive-coordinate Brier .011460 → .008971. Exact values
are .037121 and .006704, respectively. The maximum refined fixed-point residual
is 1.11e-15. These pass the preregistered positive rule and achieved mean-field
limitation thresholds.

However, coherent marginal NLL worsens by .029608 nats/atom. Product-distribution
NLL for the sampled support rises from 1.045507 to 1.282370 nats/sample; the exact
joint NLL is .193573. Thus optimizing reverse KL does not give a uniform
improvement across proper probability scores in this sample. The exact joint's
support NLL and the sum of its marginal NLLs differ because its factors are
dependent; they must not be compared as the same likelihood.

## Reconstruction and ambiguity

These are posterior expected clean-signal squared errors per input dimension.
The mean decision is `D m`; the hard decision is `D 1[m > .5]`.

| Dictionary | Method | Mean reconstruction risk | Hard reconstruction risk |
|---|---|---:|---:|
| Orthogonal | Exact joint | .002781 | .003554 |
| Orthogonal | Trained gate | .016185 | .013122 |
| Orthogonal | Refined mean-field | .002781 | .003554 |
| Cosine .95 | Exact joint | .003232 | .004379 |
| Cosine .95 | Trained gate | .016934 | .017622 |
| Cosine .95 | Refined mean-field | .003461 | .004231 |

Thresholding exact marginals minimizes atomwise classification error, not clean
signal error for correlated atoms; this explains why a different hard decision
can have slightly lower clean-signal risk. The exact posterior mean is the
Bayes squared-error decision. Sampled clean-signal and noisy reconstruction
MSEs, q-expected noisy reconstruction energy, and per-sample scores are also
retained in `summary.csv` and `samples.csv`.

Exact Bayes atomwise support error increases from .003554 (orthogonal) to
.017098 (coherent), demonstrating finite-noise ambiguity under the known model.
The coherent pair's mean posterior covariance is -.036817. Both dictionaries
are full rank, so this pilot does not establish structural nonidentifiability.

## Validity and limits

The orthogonal analytic marginal error is 1.44e-15. Refined mean absolute KL in
that control is 7.52e-16; tiny negative reported KL values are floating-point
roundoff. Four deterministic tests passed, including a separately enumerated
variational-energy comparison and coordinate monotonicity witness. Raw-array
recalculation reproduces all summary Brier and exact-marginal MSE values.
`verification.json` records the independent checks and artifact hashes.

The fitted orthogonal gate's diagonal weights are approximately 4.85–4.93,
whereas its exact solution has diagonal weight 25. Training free energy was
still decreasing at the fixed final checkpoint. Therefore most of the observed
trained-gate gap cannot be attributed to insufficient architectural capacity.
It is a finite-budget training/amortization gap. The coherent refined residual
is an achieved mean-field gap; three starts do not certify a global optimum,
even with a tiny fixed-point residual.

The result supports retaining B05 as an **oracle diagnostic benchmark**. It
does not yet support promoting coordinate refinement as a calibrated SAE
method. Any follow-up should first use a separately registered stronger gate
optimization control, then test learned amplitudes/dictionaries and multiple
seeds. Existing outputs are retained without tuning or rerunning this pilot.

## Evidence files

- `results.json`: configuration, environment, source/protocol hashes, final gate
  weights, start diagnostics, summaries, preregistered verdict.
- `summary.csv`, `samples.csv`, `marginals.csv`: aggregate, per-sample, and
  per-atom evidence, including sampled support truth and separate exact,
  amortized and refined marginal probabilities.
- `training.csv`: fixed-interval training trajectory through the final step.
- `raw_arrays.npz`: dictionaries, training/test draws, all 256-state exact test
  posteriors, marginal predictions, all-start final energies, selected starts.
- `verification.json`: test/kernel witnesses and independent result checks.
