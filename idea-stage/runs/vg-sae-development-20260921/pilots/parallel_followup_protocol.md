# P3b exploratory parallel correction follow-up

Preregistered 2026-09-21 before P3b calibration/test. **Exploratory follow-up after
observing P3**, within the same finite conditional VG correction development
route. It is not an independent fourth hypothesis, a confirmatory replication,
or evidence that the original P3 decision was preplanned for this implementation.
This is the last extra experiment in the present run; no recursive outcome-driven
variants follow it. The fixed research goal remains developing VG-SAE.

## Scope and frozen inputs

- Reuse exactly the five P1-CAL-selected exponential checkpoints used by P3:
  VG gamma3, BatchTopK k4, JumpReLU .5, TopK k4/k6. No training or parameters.
- Reuse original frozen dictionary, data-generation config, and P3's original
  TRAIN-only VG beta18.15495570109264, read from its hashed manifest. Do not fit
  beta on any new examples. Verify checkpoint and source manifest hashes.
- Fresh cal512 seed2026092194 and test512 seed2026092195, same P1 SeedSequence
  support/amplitude/noise components and source_offset0. Test generation occurs
  after the calibration decision is serialized and hashed. Do not modify P3.

## Prespecified inference arms

Starting from encoder m,a and decoder D,b, form
`r=x-b-(m*a)@D.T`, `c=r@D+m*a*sum(D**2,dim=0)`,
`target_logit=beta*(a*c-.5*a**2*sum(D**2,dim=0))-gamma`,
`m_eta=(1-eta)*m+eta*sigmoid(target_logit)`.

One simultaneous/Jacobi update only, eta in **{.25,.5,1}**. There is **no
Gauss-Seidel/coordinate free-energy monotonicity guarantee**. Report observed
fixed conditional free energy and posterior-mean/stochastic risk using original
a, independent of hard readout/NNLS. Actual decoder norms are used.

VG arms:

- Source `native_a`, source `native_pg`, source `native_nnls`.
- `count_a` at eta0 (base ranking control/fallback), .25, .5 and1.
- `count_nnls` at eta.25, .5 and1.

For count readouts fix each input's candidate count K_i to its source native
`m>=.5` count, preserve original a, and rank corrected gate scores. Stable ties
prefer feature index. Original scores are used at eta0 and target scores at eta1.
For intermediate damping preserve a stable mixture logit using logaddexp and
logsigmoid; do not collapse saturated probabilities into arbitrary ties.

All four baseline checkpoints receive the same source `native_a`, `native_pg`,
`native_nnls` arms. Total **22** predefined arms. Original source amplitudes must
be nonnegative. Refit may produce zeros; actual nonzero L0 and candidate L0 are
reported separately. Same candidate count after NNLS/PG is not assumed to imply
same actual L0.

`native_pg` is one common support-fixed projected-gradient amplitude step. From
native hard source code z and mask S, set tau=1/sum_j(S_j*||D_j||^2),
`z_new=relu(z+tau*((x-b-z@D.T)@D))*S`.
Use tau0 for empty support. The trace of support Gram bounds its largest
eigenvalue, hence this step decreases hard reconstruction SSE in exact arithmetic
on a fixed support. It has no claimed VG free-energy monotonicity. It uses two
extra dense decoder products, matching the parallel gate arm's product count;
actual measured time is reported rather than claiming equal wall-clock budget.

NNLS is exactly P3's common scipy float64 support-fixed solver with
maxiter=max(3*K,1), x-b and learned D only, no truth. CPU/GPU transfers are included
in its timing. No NNLS arm is selected as the VG gate policy.

## Selection, evaluation and decision

Choose eta from {0,.25,.5,1}, **count_a only**, with lowest cal hard latent error,
subject to cal F1>=base-.01 and cal mean fixed conditional free energy<=base
(with only1e-10 absolute numerical tolerance); ties prefer smaller eta. Eta0 is
the guaranteed fallback. Freeze before generating test. Report every prescribed
arm on test once, including failed-cal eta choices; never select on test.

Use P3 metrics/matching: hard aligned latent relative error, actual-support F1,
EV/MSE, candidate/actual L0. VG conditional metrics refer to m_eta with unchanged
original a; NNLS and PG coefficients do not enter those probability diagnostics.
Ground truth is evaluation-only.

Warmed latency uses common inference wrapper, calibration batch128,
**2 warm-up +11 timed repetitions**, CUDA synchronization, complete encode,
postprocessing/readout, final decode and any transfers. Exclude setup/loading,
matching, truth evaluation. Save every timing sample. Main native reference is
VG's same-wrapper native_a; also report base count_a and generic native_pg/NNLS.
No repeated benchmarks or algorithm changes after outcomes to cross a threshold.

A useful deployable follow-up needs cal-selected test gate policy to decrease
fixed conditional free energy, retain actual L0, improve error>=5% **or** EV>=.01
with F1 loss<=.01, and median time<=3x source-native reference **in this
implementation**, with residual benefit after generic controls. Controls having
the same two decoder products do not establish equal total compute/time. A
failure to meet one condition is reported without reframing it as a full method
win. Results after observed P3 remain exploratory even if all thresholds pass.

## Correctness and resource gates

Before data evaluation: float64 eta0 identity; independent explicit-loop Jacobi
conditional calculation; a counterexample showing simultaneous updates need not
reduce fixed free energy; generic PG nonnegative/support preservation and SSE
nonincrease using actual non-unit/correlated D and empty support.

GPU2, target few minutes, <=.25 GPU-hour (900-second hard limit). Persist hashes,
versions, seeds, selection and elapsed. No old outputs, source APIs, checkpoints,
project-external files, new training, additional modules or commits are changed.
