# Finite-beta native bias: derivation and frozen numerical witness

2026-10-03. Pre-execution analytic specification. No local computation, training,
numerical test or scientific execution was performed when this file was written.
The accompanying script is to run on Colab CPU only. It uses the existing
`src.sae_model.VariationalGarroteSAE.free_energy` directly.

## Claim and limits

For a noiseless independent Bernoulli source coordinate with a supplied unit
decoder, finite positive precision beta and positive sparsity field gamma, the
profiled native conditional objective need not prefer zero centering bias. The
zero-bias profiled point has a strictly descending positive-bias direction. A
deterministic finite positive-amplitude construction preserves a quantitative
objective decrease. The decoder remains exactly the teacher dictionary.

This refutes exact zero-bias recovery for this restricted profile. It does not
refute a dictionary-recovery gap, an epsilon-accurate recovery guarantee or the
full native model's best solution. It does not fit generating q, explain the
retained noisy support failures, change BR5, or justify another training sweep.
The old hard oracle-subset study lacked a direct native amortized parameter
witness; this construction supplies one on two noiseless input values. The
positive-noise model requires a separate argument.

## 1. Scalar objective and exact closure profile

Let X=S with S~Bernoulli(q), 0<q<1. Fix decoder D=1, centering/reconstruction bias
b in [0,1], nonnegative amplitude a_s and gate m_s. The standard normalized
native prior is pi=sigmoid(-gamma); entropy coefficient and variance correction
are both one. Omitting the Gaussian normalizer, which is constant at fixed beta,
the conditional expected objective at a state s is

```text
f_s(m,a;b) = beta/2 * [(s-b-m*a)^2 + m(1-m)*a^2]
             + KL(Bernoulli(m) || Bernoulli(pi))
           = beta/2 * [(s-b)^2 - 2(s-b)m*a + m*a^2]
             + KL(Bernoulli(m) || Bernoulli(pi)).
```

For finite beta/gamma the optimal gate is interior. Minimizing over a>=0 gives
a0=0 and a1=1-b. The first value is in the softplus encoder closure, not attained
at finite weights. Optimizing the two gates gives

```text
m0 = pi,
m1(b) = sigmoid(beta*(1-b)^2/2-gamma).
F(b) = beta/2 * [(1-q)b^2 + q(1-b)^2]
       + q*softplus(-gamma)
       - q*softplus(beta*(1-b)^2/2-gamma).
```

F is the scalar population profile after exactly minimizing those amplitudes and
gates; it is not an observed-data likelihood. The q here weights the declared
population and is not an estimator input inferred from samples.

## 2. Derivatives, minimizer location and hard-support boundary

Direct differentiation yields

```text
F'(b) = beta * [b-q+q(1-b)m1(b)],
F''(b) = beta * [1-q*m1(b)
                 - q*beta*(1-b)^2*m1(b)*(1-m1(b))] <= beta.
```

At b=0, `F'(0)=-beta*q*(1-m1(0))<0`. For q<=b<=1 the derivative is positive:
the term b-q is nonnegative and at b=q the remaining term is strictly positive.
By continuity, every global minimum of F on [0,1] is therefore in (0,q). This
does not establish uniqueness, convexity or a particular optimization trajectory.

If beta/2<=gamma, then m1(b)<=1/2 throughout [0,1], while m0=pi<1/2. The strict
native threshold H=1[m>1/2] is thus all-off over this restricted profile even
though generating q>0. Equality at b=0 yields m1=1/2 and is still inactive. It
does not follow that an unrestricted native global optimum is all-off.

## 3. One fixed candidate, no search

Define

```text
delta = sigmoid(gamma-beta/2) = 1-m1(0),
b_w = q*delta,
G = beta*b_w^2/2.
```

Since F''<=beta, Taylor's integral remainder gives
`F(b_w)<=F(0)+F'(0)b_w+beta*b_w^2/2=F(0)-G`.
Thus the predeclared candidate is strictly better in exact real arithmetic.
No stationary root, optimizer, seed, sweep-dependent candidate or best point is
selected. For beta=400 and gamma=16, delta is of order exp(-184), so this
guaranteed decrease can be far below float64 resolution. Such cases must remain
visible rather than prompting a new candidate.

## 4. Finite-amplitude witnesses and closure defect

Use the same strictly positive inactive amplitude `epsilon=b_w/16` in both the
baseline at b=0 and candidate at b=b_w. Leave m0=pi, a1=1-b and m1(b) as above.
The scalar finite construction is exactly

```text
F_epsilon(b) = F(b) + (1-q)*beta*pi*(b*epsilon+epsilon^2/2).
```

The finite baseline closure defect is `(1-q)*beta*pi*epsilon^2/2`. The candidate
defect also includes `(1-q)*beta*pi*b_w*epsilon`. Because `(1-q)*pi<1`,

```text
F_epsilon(0)-F_epsilon(b_w)
  >= G - beta*b_w^2/16
   = (7/8)*G > 0.
```

Moreover, `F_epsilon(b_w)<=F(0)-(223/256)*G`: the candidate finite construction
is also strictly below the baseline's exact closure value. These mathematical
inequalities concern the exact constructed parameters. Rounding those parameters
into float64 creates a separate numerical issue; the script records that issue
instead of asserting an outward-rounded certificate.

## 5. Exact native affine parameterization

Lift the scalar construction to d=K=2, D=I, iid S1,S2~Bernoulli(q) and b vector
equal to b times the all-ones vector. The exact four input states and weights are

| State | Weight |
|---|---|
| (0,0) | (1-q)^2 |
| (0,1) | q(1-q) |
| (1,0) | q(1-q) |
| (1,1) | q^2 |

Let `k=beta*(1-b)^2/2`, `l0=log(expm1(epsilon))`,
`l1=log(expm1(1-b))`, and `w=l1-l0`. Since the native encoder receives X-b,
use the following finite parameter tensors:

```text
decoder.weight       = I_2
pre_bias             = b * ones(2)
gate_encoder.weight  = k * I_2
gate_encoder.bias    = (-gamma+k*b) * ones(2)
amplitude_encoder.weight = w * I_2
amplitude_encoder.bias   = (l0+w*b) * ones(2)
log_beta             = log(beta)
```

At input coordinate zero this gives gate logit -gamma and amplitude epsilon;
at input coordinate one it gives gate logit k-gamma and amplitude 1-b. Both
models have the exact same teacher decoder. All weights are finite because
0<b_w<q<1 and epsilon>0. The total two-coordinate objective is twice the scalar
objective plus the unchanged Gaussian normalizer.

The model uses its existing learned-beta branch, but log_beta is frozen and no
parameter receives an optimizer step. The branch choice is necessary because
the profiled-beta branch would change the target being compared. All tensors
are float64 CPU. Construction on the Torch meta device avoids random
initialization draws, followed by assigning every actual CPU parameter.

## 6. Frozen executable panel and reporting contract

The only scientific cases are the full Cartesian product:

- q in {0.1, 0.3};
- beta in {1, 4, 40, 400};
- gamma in {1, 16}.

All 16 cases must be retained. There are 32 finite models, four exact input
states per model, zero sampled observations, zero random seeds and zero
optimization/training steps. A technical threshold-tie unit test may use
beta=2,gamma=1; it is not an additional research-panel case.

For each case the script saves the actual config and all state tensors as JSON,
four per-state native forwards and free energies, exact population weights,
soft/hard densities, reconstruction components, analytic floor defects, and
baseline-minus-candidate objective differences. It also evaluates the saved
float tensors independently with Python's standard Decimal exp/ln at 240 digits.
No new high-precision dependency is required.

The engineering roundoff guard is frozen as
`8192 * eps_float64 * max(1, weighted absolute objective-component scales of both models)`.
Native/reference parity must hold within that guard. A float64 decrease is
resolved only when the native difference exceeds the guard and the saved-tensor
Decimal difference is positive. Otherwise the status is
`unresolved_float64_difference`; the exact analytic positive bound remains
separately recorded. The guard is not a proved error bound and Decimal is not
directed interval arithmetic. No result is called a formal numerical certificate.

All output files go only under `COLAB_TASK_OUTPUT`:

- `native_bias_witness_results.json`: complete case records, protocol and source hashes;
- `native_bias_witness_summary.csv`: all 16 cases and resolution status;
- `native_bias_witness_report.md`: compact counts and claim limits.
- `native_bias_witness_runtime_receipt.json`: whole-process CPU/wall accounting,
  including imports and principal artifact serialization.

The entrypoint first requires `VG_SAE_COLAB_CPU_ONLY=1` and rejects a runtime
where Torch reports CUDA available. It refuses an existing
`native_bias_witness_results.json`, preserving an earlier incomplete or failed
attempt. Every artifact uses exclusive creation so a concurrent file appearance
also cannot be overwritten.

The total process CPU limit is 60 seconds and the wall deadline is 120 seconds.
Scientific work stops at 55 CPU seconds or 115 wall seconds, reserving five
seconds for output. CPU usage and the Linux resource limit include imports.
Wall accounting reconstructs process age from Linux `/proc` with a conservative
0.05-second guard; a disclosed fallback starts before the Torch import. The
final hard CPU limit and wall alarm remain active during serialization. No
memory limit is imposed on Torch's virtual address space; one CPU thread is used.

A scientific timeout does not resume or replace a case. Any interrupted case and
all remaining frozen cases receive explicit `budget_unresolved` statuses, with
no unavailable numeric value presented as a result. Finished rows remain intact,
and all 16 IDs stay in the output. An operating-system hard kill during the
reserved finalization interval can leave a partial exclusive-created result;
that file must be preserved and the process outcome reported as incomplete.

Run on the already authorized Colab CPU runtime from the repository root:

```text
python -m pytest tests/test_native_bias_witness.py -q
python -m scripts.check_native_bias_witness
```

The worker supplies `COLAB_TASK_OUTPUT` and `VG_SAE_COLAB_CPU_ONLY=1`. Tests check the nonuniform population
weights, finite parameter realization without RNG draws, direct enumeration of
the Bernoulli reconstruction risk, Gaussian normalization, analytic descent,
strict threshold ties, honest unresolved status, runtime authorization, output
preservation and retention of all rows on an artificial timeout. Safety tests
use helpers and do not install real process budgets in the test process. They
must not be executed on
the host. Source/config hashes and remote runtime receipts distinguish this
bounded calculation from historical evidence.

The expected scientific contribution is a precise finite-precision contract
qualification for the native objective. No result from this panel clears the
original measurement/identification gates or the user's overall pre-LLM goal.
