# P3 conditional VG refinement protocol

Preregistered 2026-09-21, before refinement calibration/test outcomes. This is a
bounded method-development pilot for the fixed VG-SAE research goal in
`RESEARCH_BRIEF.md`, not a replacement research topic or a new coordinate theorem.

## Frozen scope

- No training and no added parameters. Reuse exponential Stage1 seed0, d=128,
  width=1024 checkpoints selected by P1 **calibration**, at L0 caps 4 and 8:
  VG gamma3 (deduplicated), BatchTopK k4, JumpReLU coefficient .5, TopK k4/k6.
  All checkpoint paths/hashes and inherited candidate counts are serialized.
- Use the P1 frozen original dictionary/probabilities and original amplitude,
  noise and scale config. Fresh calibration 512 with seed **2026092192**, fresh
  test 512 with seed **2026092193**, source offset0 and P1's independent support,
  amplitude/noise SeedSequence components. Test is generated only after the P3
  calibration choice has been written and hashed. These draws are independent of
  P1 calibration/test; training/dictionary seed is still one.
- Original TRAIN is regenerated using P1's `make_train_test(cfg, 0, 'cpu')` and
  its dictionary checked against frozen source. No precision fitting on cal/test.
  For a profiled VG checkpoint fix beta = N*d/(2*sum TRAIN E), E being the
  encoder's analytic expected half-squared reconstruction error, including the
  Bernoulli variance with actual decoder-column squared norms. Clamp only the
  denominator by original loss_eps. Learned checkpoints would use learned beta.

## Inference and controls

Hold encoder amplitudes a, decoder D, bias b and beta fixed. Starting at encoder
m, run deterministic sequential coordinate sweeps in latent index order 0..1023,
with sweep counts **0, 1, 3**. Let r_minus exclude the current coordinate:

`m_j = sigmoid(beta * (a_j * D_j.T @ r_minus - .5 * a_j**2 * ||D_j||**2) - gamma)`.

Update the residual immediately after every coordinate. Retain each coordinate's
pre-sigmoid score for ranking to avoid artificial ties from saturated sigmoid.
Use original gate logits for the zero-sweep ranking control. Stable ties prefer
lower feature index. Native hard mask uses `m >= .5`, matching P1 Stage1 metric
convention. K_i is each input's **base native gate count**, including zero.

For every VG sweep count report all five prespecified readouts:

1. `native_a`: native threshold mask times unchanged original a (L0 can change).
2. `count_a`: top K_i gate scores times unchanged original a; exact per-input
   candidate counts isolate the mask change. This is an experimental readout
   using the encoder's own count, not a globally thresholded deployment claim.
3. `count_ma`: the same top K_i support times current m*a; amplitude control only.
4. `native_nnls`: nonnegative amplitude refit on the native candidate support.
5. `count_nnls`: nonnegative amplitude refit on the top K_i candidate support.

Every baseline receives `native_a` and **the identical support-fixed NNLS**.
NNLS uses only x-b and learned D, CPU float64 scipy.optimize.nnls,
`maxiter=max(3*K, 1)` per sample, no ground-truth input, no regularization. Any
failure aborts that arm (no silent fallback). Candidate support is fixed; fitted
zeros reduce actual nonzero L0 and are reported separately. This is a strong cheap
amplitude control, not an equal-wall-clock inference claim. No unsupported claim
of generic matched-compute superiority is permitted.

## Calibration, measurement, decision

- Choose only the `count_a` sweep count using lowest calibration hard latent
  relative error among {0,1,3} whose F1 >= base native F1 - .01. Ties prefer fewer
  sweeps. Sweep0 is a guaranteed fallback. Other readouts remain mechanistic
  controls, not additional selected policies. Serialize choice before test.
- All predefined arms are evaluated once on the fresh test. Report hard latent
  error sqrt(sum squared aligned error / sum squared true latents), actual
  nonzero-support F1, EV, MSE, actual nonzero L0 and candidate-support L0. Reuse
  P1 Hungarian matching from frozen true dictionary; truth enters evaluation
  only, never refinement/refit. Actual decoder norms and bias are retained.
- Report conditional VG free energy with beta fixed as above, including Gaussian
  and Bernoulli prior constants, entropy computed via stable xlogy at 0/1;
  posterior-mean residual risk and analytic stochastic risk remain separate.
  Float64 metrics evaluate the float32 inference state. An exact float64 small
  correctness test checks each coordinate's objective decrease and residual.
- Latency is warmed median on calibration inputs, batch128, **1 warm-up + 5
  timed repetitions** per arm. Include encode, correction, readout, decode, CPU
  NNLS host/device transfers and synchronization; exclude loading/matching/truth
  evaluation and one-time TRAIN beta fit. Report all individual latency samples,
  parameter counts and extra decoder products. NNLS CPU runtime is included;
  this heterogeneous path is not claimed to have matched GPU compute.
- An extension needs same-count hard error gain >=5% or EV gain >=.01, F1 loss
  <=.01, original fixed conditional objective decrease, latency <=3x native
  encoder/readout/decode and benefit beyond generic inference to justify a
  deployable follow-up. Quality gain with >3x latency supports only teacher
  exploration. Free-energy-only gain or refit-only gain does not support adopting
  the gate correction. No result rejects the underlying VG-SAE research goal.
- GPU2 preferred, target <.5 GPU-hour, hard stop 2h. Persist status, elapsed,
  environment, protocol/script hashes and checkpoint hashes. No outcome-driven
  scope changes, additional modules, tests, or training in this pilot.

## Correctness gates before the run

Float64 coordinate update must reduce the exact enumerated Bernoulli expected
energy/free energy at each coordinate and preserve the explicitly reconstructed
residual. Exact top-K_i includes K=0, full width, tied/saturated probabilities and
stable ordering. NNLS must keep code inside candidate support and decrease SSE
from a feasible source code, including an empty support and a fitted zero.
