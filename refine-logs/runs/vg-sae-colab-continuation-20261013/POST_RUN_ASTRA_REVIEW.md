# Matched-amplitude v1: independent post-run review

2026-10-13. **Verdict: the two frozen finite-native pair certificates and the
three scoped implementation tests support the stated mechanism claim. No
material scientific or provenance discrepancy was found.** This review does
not complete the original density estimator, clear BR5, or establish a
three-model consensus.

Only this new review file was written. The protocol, code, results, and initial
review were not changed. Local work consisted of reading, archive-byte
comparison, and cryptographic hashes; no scientific numerical recomputation,
tests, model execution, or training was performed.

## Evidence chain

The review read the frozen protocol, source freeze, executed runner, reused
arithmetic/resource helper, native source, focused tests, result JSON, JUnit,
stdout logs, environment, worker state, remote plan, and archive receipt.

All nine files named by `MATCHED_SOURCE_FREEZE_v1.json` matched both the current
workspace bytes and their entries in the preserved source archive. All eight
files mapped by `ARCHIVE_RECEIPT.json` were byte-identical to their preserved
result-archive members. The result's eight start/end scientific-source hashes
were equal and matched the freeze. `src/__init__.py`, absent from that narrower
result hash dictionary, was independently covered by the nine-file source
freeze and source archive.

The source-archive hash matched both the archive receipt and remote worker
state. The result-archive hash matched the receipt. The canonical remote plan
hash matched worker state, and the local and remote plans were semantically
identical.

| Item | Verified SHA-256 |
|---|---|
| Executed runner | `8e755a3265650520dd85ca03837e36d39b104f8e03cc10c4781d36a56b7b2c21` |
| Frozen protocol | `76a31ba69c6bbfcf055c3f0c569375c7a568766632fa3ef524f0bdb034477635` |
| Source archive | `6f9fc4a941a31543a22814206a09f45c899d8f1e2dd5328e24a42d26fd0902a1` |
| Result archive | `50f1bf90c3672d4cf445f00c416f59fef95ac806ef4c32036183951880a04d9f` |
| Canonical remote plan | `9ce74cff414cea1daf67e9ffb30ba63f527c694b8b745671924137f86a486a7a` |

The initial review intentionally records the earlier runner hash
`db0ef559...`. Before the source freeze, the runner separated the primary
`pair_certified` status from the auxiliary `.02` scale comparison as requested
in that review. Inspection of the executed runner confirms that primary status
requires the premises and both strict scientific signs; the auxiliary check
has its own field. The frozen protocol bytes are unchanged. This is an
explicitly tracked pre-execution source update, not an unexplained mismatch.

## Execution evidence

Both planned tasks have `completed`, attempt 1, return code 0 in remote worker
state. Both prescribed q values appear in the scientific JSON and certificate
stdout, each with `pair_certified`; no case was substituted or omitted.

JUnit reports exactly three tests, zero errors, zero failures, zero skips, and
7.488 seconds. The stdout's `3 passed in 7.48s` is consistent with that scoped
report. These tests cover the exact fixed roster and the two native parity
cases. They are not a full repository test-suite result or three independent
population experiments.

The remote environment records Python 3.13.16, Torch 2.11.0+cpu, CPU-only
execution, and runtime storage. The arithmetic result records official
python-flint 0.8.0, 512-bit Arb, and one arithmetic thread. The scientific
protocol has no samples, quadrature, optimizer steps, or training.

The certificate receipt records 0.105746705 process CPU seconds and
0.15547782199996618 process wall seconds, within the declared 60 CPU /120 wall
second limits. These values include imports and principal result
serialization, exclude the receipt's own serialization, and use the existing
process-age wall-accounting convention. They do not measure the full Colab
session, test task, provisioning time, or billed compute units. Runtime release
and account-balance state are outside the supplied evidence reviewed here.

## Mathematical and numerical assessment

The executed case contract is exactly K=d=2, sigma=1/20, beta=400, t=1/2,
q in {1/10,3/10}, and supplied gamma=log((1-q)/q). These gamma values differ
from the old gamma=1/16 panel: this run deliberately supplies the correct prior
as well as the correct noise precision. It does not retrospectively resolve
the old panel's inconclusive matched rows.

The two models have D=I, b=0, the same exact posterior affine sigmoid gates,
and amplitudes 1 versus softplus(log(expm1(1))+(X-1)/2). They are finite
members of the native architecture. Normalized gate KL, entropy coefficient
one, and the full Bernoulli reconstruction variance remain included.

The earlier derivation was re-examined against the executed formula expressions:

- The observed stochastic-risk gain is the gate probability times the
  per-coordinate residual-square improvement. Mean-only reconstruction is not
  substituted for stochastic risk.
- The exact posterior identity removes the gate probability from the active
  population expectation. The lower bound uses a fixed |Z|<=1 event and
  retains the objective's factor 1/2.
- The hard coefficient-error increase subtracts a valid inactive-tail upper
  bound. The active event is admissible because the certified threshold is
  below 1-sigma.
- The absolute hard-NMSE upper bound uses the true denominator Kq, the
  amplitude Lipschitz bound, and the inactive Gaussian tail second moment.
- The runner compares exact rational outward endpoints. Its displayed decimal
  strings are explanatory approximations to the bounded expressions, not the
  actual objective gaps or attained NMSE values.

The retrieved values are:

| q | Objective-improvement lower bound, approximately | Hard-NMSE-increase lower bound, approximately | Hard-NMSE upper-bound expression, approximately |
|---:|---:|---:|---:|
| .1 | .0104984446445053 | .0000487226667540462 | .000625000000000000000051751 |
| .3 | .0314953339335159 | .0000487226667540462 | .000625000000000000000026464 |

Every lower-bound sign and the auxiliary upper-bound comparison is true in the
stored certificate. The exact rational endpoints in
`results/matched-amplitude-v1/matched_amplitude_results.json` are authoritative.
The upper-bound expressions contain small positive tail terms: neither row
certifies `NMSE <= .000625` exactly. A compact conservative report may say
`NMSE < .0006250001`, or describe the bound as approximately .000625 with a
positive tail correction. The bound is not an estimate of the attained NMSE.

No fresh interval arithmetic was performed by this reviewer. The scientific
assessment combines the independently checked analytic inequalities and source
expressions with the preserved Colab outward-arithmetic evidence. This is not
an independent arithmetic-backend replication or proof-assistant verification.

## Strongest defensible new claim

> With correctly supplied noise precision and Bernoulli prior, the native
> VG-SAE objective is not an order-preserving proxy for population hard
> coefficient recovery error: these finite native pairs strictly lower the
> objective while strictly increasing hard coefficient NMSE. The teacher
> decoder and every native hard-support decision remain exactly unchanged.

The exact derivative at the unit-amplitude reference,
`G'(0)=-Kq(1-exp(-1))`, additionally shows that this correctly specified
unit-amplitude solution is not stationary in the full native amplitude family.
The finite arithmetic certificates quantify one fixed direction at t=1/2;
they do not find its optimum.

This result isolates amplitude's response to observation noise. It demonstrates
a disagreement in objective ordering and coefficient recovery, even with the
favorable supplied geometry and prior. It does not establish that the method
has practically poor recovery: both perturbed models retain the stated small
absolute coefficient-error upper bound and the same support.

The following remain separate and unproved by this run:

- The sign of the change in **clean stochastic reconstruction risk**. That
  quantity includes Bernoulli mask variance and is not the hard coefficient
  NMSE measured by these bounds.
- A comparison against **every** recovering model, global-optimum failure,
  approximate-optimizer failure, or behavior of an actual trained checkpoint.
- A failure of dictionary identity, hard support, or unknown-q estimation.
  Dictionary and support are held fixed, and q is supplied to construct gamma.
- A C2b selector failure. Its .02 practical margin is a paired selector
  contrast with confidence and control requirements, not an absolute error
  threshold. The two constructed models are not the original selector bank.
- A C2a generating-count accuracy/reporting result or the cross-fit
  hard-between selection law. Neither has been measured here.

## Precise next scientific blocker

The immediate blocker is still the bridge from a finite matched-precision
objective and observable model/data quantities to a **class-level** recovery
statement. This pair disproves monotone ordering but cannot disprove a positive
gap outside a recovery neighborhood containing both members.

At matched precision the exact identity remains

    G=beta C/2 + J - E div f + d/2.

Thus a proposed certificate must quantitatively control the response/divergence
term, as well as J, conditioning, bias, and the source structure. A finite
population divergence contract is a concrete candidate because the derivative
of the native mean reconstruction is observable from the model and noisy input.
It must be justified with independently supplied noise calibration and an
evaluation design; merely rewriting the identity is not a useful bound.

The next useful analytic deliverable is a single complete implication at
sigma=.05/beta400: explicitly stated, independently checkable conditions imply
the imported geometry/bias and coefficient-risk premises, with a resulting
hard coefficient bound that retains the original C2 interpretation. If such
conditions cannot be made nonvacuous, identify the failed quantitative premise
or exhibit a valid all-recovering-class lower comparison. Another selected pair
or stricter convenient clean-risk tolerance would not fill this gap.

Even a successful recovery certificate would leave the distinct measurement
blocker: no identified law yet links the original fixed-data optimizer
ensemble's hard-between observable to generating q. A source-moment estimate
from X or a normalized BSC control is a comparison under its declared
information assumptions, not evidence that the original native estimator
succeeded. Do not reopen training grids or untouched confirmation worlds on
the strength of these two mechanism certificates alone.

BR5 therefore remains `measurement_unresolved`; W2 remains gated; C2a/C2b
remain unentered. The broad pre-LLM research objective remains incomplete.

## Review availability and workflows

This is the independent Astra review requested through the root agent. The
run's frozen ledger records the requested Opus 5.5 xhigh route as unavailable
because no usable Claude credential was present, and Gemini 3.1 Pro high as
unavailable because the saved Google OAuth grant was rejected. This reviewer
did not inspect credentials or independently rerun those connection attempts.
No Opus/Gemini review or three-model consensus is claimed; those requested
reviews remain outstanding.

The already-read project-local ARIS formula-derivation,
scientific-critical-thinking, physics-guide-solver, and physicist-analyst
workflows were applied to invariant-object preservation, matched dimensionless
precision, endpoint separation, and claim/evidence boundaries. They supply a
review procedure, not evidence for the research conclusions.

Workflow attribution uses the metadata verified earlier in this same review
session: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M.
Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge
for Research Agents*, [arXiv:2609.00065](https://arxiv.org/abs/2609.00065).
