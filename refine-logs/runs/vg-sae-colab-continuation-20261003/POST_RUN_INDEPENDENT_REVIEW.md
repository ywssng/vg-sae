# Independent review of the three completed Colab jobs

Date: 2026-10-03. Reviewer: independent code-audit agent; semantic review is same-family/provisional. Scope: `results/population-terminal-v1/`, `results/finite-noise-v1/`, `results/native-bias-v1/`, their frozen sources/protocols, and the associated remote validation receipts. A separate noisy native-obstruction theorem is outside this review.

**Verdict: the qualified claims below agree with the preserved artifacts and reviewed source.** No claim-level contradiction requiring a scientific rerun was found. This review read saved fields and checked byte/hash bindings; it did not rerun interval arithmetic, Decimal evaluation, tests, model forwards, or experiments locally.

## Provenance and execution checks

- All three downloaded `results.zip` SHA-256 values match their `ARCHIVE_RECEIPT.json`. Every copied scientific result/report file matches its corresponding member in that archive. The terminal `START.json`, `TERMINAL_AUDIT.json`, and `VERIFIED_PREFIX.jsonl` match `SEAL.json`.
- Each archived `source.zip` and canonical saved plan match the source/plan hashes in `remote-worker-state.json`. All three plans explicitly select CPU-only execution and runtime storage. Their tasks have one attempt, return code 0, and completed state. The results were downloaded, so they no longer depend solely on the ephemeral VM.
- Scientific source hashes match the pre-execution freezes: terminal verifier `2c861c08...`, finite-noise consumer `27dbbd74...`, native witness `ecdf3cb8...`. The finite-noise output binds the exact terminal audit, seal, and verifier source. Native initial/final source dictionaries match and report `sources_unchanged=true`.
- The environment receipts record Python 3.13.15, torch 2.11.0+cpu, NumPy 2.1.3, `cpu_only=true`, and runtime storage. Population arithmetic additionally records python-flint 0.8.0 at 512 bits. This is a new Colab evaluation environment, not an exact replay of the older dot-cloud runtime.
- Remote JUnit/log/artifact evidence supports **51 focused population/finite-noise tests with zero failures/skips**, and **10 native tests with zero failures/skips**. Those claims refer to their archived preflight source versions. The current finite-noise test file subsequently changed only its optional-flint import policy to `pytest.importorskip`; it no longer has the frozen test hash `38abef31...`. That later test-only version is not retrospectively covered by the 51-pass receipt. Scientific code remained unchanged.

Recorded resource receipts are within the stated limits: terminal verification used 354.994138368 process CPU seconds against 600, 358.137542449 wall seconds against 900, and 182,460,416 peak RSS bytes against 512 MiB. The finite-noise calculation records 0.153325362 CPU seconds against 60. The native whole-process receipt records 3.880381578 CPU seconds and 3.839969289 wall seconds against 60/120, including imports and principal output serialization. These are the stored scopes; they are not GPU-credit or billing measurements.

## Accepted population-terminal claim

The new verifier completed its declared **terminal-partition** task for all 20 fixed cells: four exact lambda-zero anchors and 16 positive-crossover cells. The receipt says `passed=true`, `completed_cells=20`, and `failure=null`. The original fixed interval, IDs, row order, source-manifest/run-seal identities, consumed result/trace hashes, and full terminal domain coverage are bound by the reviewed verifier.

For every positive cell, the derived record reports all three declared targets met: objective gap, each retained activity image, and each retained variance image. This follows a new, explicitly separate strict-pruning/re-enclosure pass on stored terminal certificates; no optimizer restart, new subdivisions, sampled inputs, or new data worlds occurred.

**Required qualification:** image targets are checked for each surviving component. Multiple disconnected components remain—for example, 51 in `q1_10_d2_l1_2`, 72 in `q1_10_d2_l8`, and two in `q3_10_d2_l16`. The recorded flags do not by themselves assert a total-hull diameter, optimizer uniqueness, one connected interval, or strict variance-bias sign. At lambda 16, some variance-sign labels remain `unresolved` despite the image-width targets being met.

The historical distinctions remain necessary:

| Evidence layer | Preserved status and meaning |
|---|---|
| Original primary population run | 12 positive cells had unresolved parameter images; four positive cells met targets; four anchors were exact. These statuses remain in `original_status`. |
| Original bounded full-tree numerical replay | Only three rows were completely replayed; 17 remained audit-unresolved. This old audit was not reopened or rewritten. |
| New terminal-partition verifier | All 20 declared rows completed the new verification; 16 positive rows have derived component-target success. This is a different, narrower arithmetic-verification algorithm than replaying every historical intermediate box. |

The new verifier explicitly records `all_archival_dependencies_rehashed=false`; it binds the fixed archival seals and rehashes artifacts it consumes. It shares official Arb elementary primitives with the original engine. “All original arithmetic and dependencies independently replayed” would exceed this evidence.

## Accepted finite-noise claims

The finite-noise calculation consumed the fully verified terminal rows, retained all 20 original specifications, and evaluated the prescribed sigma grid. The output agrees with its reviewed one-sided bound and strict comparison rules:

- `q < q0` is certified at a positive sufficient grid threshold in **16 of 20 cells**, including all four exact-cancellation anchors.
- The strict bracket `q0(1-q0) < q < q0` is certified in **10 of 16 positive-crossover cells**.
- All 20 original-noise entries at sigma `1/20 = 0.05` pass the elementary applicability prerequisites but certify neither side. They are insufficient-margin results, not evidence that the bias theorem is false at that noise.
- Both `q0=0.1, lambda=0.5` cells retain an unresolved lower exclusion. All four lambda-16 cells retain an unresolved upper exclusion. These were not replaced or omitted.
- At lambda zero, the lower strict-bracket field remains absent/inapplicable, as required.

The largest *tested sufficient* thresholds vary substantially. For example, the recorded strict-bracket threshold is `1/100000` for `q0=0.1, lambda=2` in either dimension, `1/10000000000` for `q0=0.3, lambda=0.5`, and `1/100000000` for lambda 8. They are conservative sufficient thresholds, not estimates of the largest true admissible noise, and they do not establish practical usefulness at sigma 0.05.

Every threshold refers to **all smaller positive sigma on the exact path `h=(lambda*sigma/2)^2`**, with supplied dictionary, unit exogenous amplitudes, fixed finite crossover, globally optimized product inference, globally fitted scalar variance, and population expectations. It is not a fixed-dictionary noise sweep, finite-sample coverage guarantee, amortized/native-VG theorem, or operational generating-density estimator. An overall `positive_noise_exclusion_certified` cell status can mean only one side; the explicit upper/lower/bracket fields must govern wording.

## Accepted native finite-beta witness claim

All 16 predeclared cases and their two finite models are preserved. The protocol records four exact support states, no sampled observations or random seeds, zero optimizer/training steps, the unchanged identity teacher decoder, and the frozen bias/floor rules. The result contains ten `finite_native_decrease_resolved_float64` cases and six `unresolved_float64_difference` cases, with no implementation failures or budget-unresolved rows. Every recorded analytic-descent/parity check passes.

The six unresolved cases are both q values at `(beta,gamma)=(40,1),(400,1),(400,16)`. The saved native float64 differences are slightly negative for the two beta-40/gamma-1 cases and zero for beta 400; their statuses correctly remain unresolved. Positive ideal-construction or saved-tensor Decimal values do not silently replace those native results.

A concrete supported comparison is `q=0.1, beta=4, gamma=1`: the recorded native objective decrease is `0.00324218764069939`, beyond its `5.430132131175163e-12` engineering guard. Both decoder tensors are exactly the identity teacher. The candidate's recorded hard reconstruction squared error is `0.0013019307863132389`, while its expected hard density remains approximately 0.1. Thus this is evidence about the native objective's restricted bias preference, not an unqualified reconstruction, support, or dictionary-recovery improvement.

The exact analytic decrease applies to the ideal finite construction; rounded model tensors have a separate 240-digit Decimal reference. The guard is explicitly not a proved numerical error bound, and Decimal is not directed interval arithmetic. None of the native rows is labeled a formal numerical certificate. Tiny parameter-rounding effects may dominate an extremely small ideal gap, which is why the saved-tensor and ideal-family fields must stay separate.

## Claims still unsupported by these jobs

These results do not resolve native generating-q measurement, a noisy native finite-beta recovery theorem, a global dictionary-recovery gap, learned-dictionary inference, finite-sample calibration, optimization convergence, novelty, or an LLM activation result. They do not clear BR5, W2, C2a, or C2b. Further claims require their own exact model, source review, and evidence; success in the supplied population model is not a substitute.

The three completed jobs need no repetition to support the qualified statements above. Later optional-dependency test maintenance or a separately scoped theorem does not alter these preserved scientific outcomes.

## Addendum: fourth job, positive-noise finite-beta obstruction

The review scope was subsequently extended to `results/native-noise-v1/`, `NATIVE_NOISE_SOURCE_FREEZE_v1.json`, its analytic protocol/review, and `validation/native-noise-preflight-cpu-20261003-v1.*`. The same restriction applied: artifact reading and byte/hash checks only, with no local numerical recomputation or tests.

**Verdict: the fourth job supports its stated finite-beta clean-risk obstruction in the six certified cases, with the remaining six explicitly inconclusive.** The result archive, copied result/CSV/report/runtime receipt, source bundle, and plan all match their recorded hashes. All files in the new source freeze match current bytes. Its one-attempt worker state is completed with return code 0, and its initial/final scientific sources agree. The driver hash is `30d75112ee17dfc8e912939331ccbb44ccc70572c07f3fc7f2d205f71ec28768`.

The complete 12-case product is retained: q in `{1/10,3/10}`, beta in `{400,1600,25600}`, gamma in `{1,16}`, always sigma `1/20`, full squared clean stochastic risk threshold `epsilon=1/20000`, the fixed rational orthogonal decoder, center shift 10, and finite gate logit 32. The stored protocol says no samples, quadrature, optimizer steps, training, adaptive choices, or Torch use in the arithmetic runner. Official python-flint 0.8.0 uses 512 bits and one thread. The whole-process receipt records 0.150671795 CPU seconds and 0.215966755 wall seconds within 60/120, with no budget stop.

For each of the two q values, the certified cases are `(beta,gamma)=(1600,1),(25600,1),(25600,16)`. The outward upper endpoint for the explicit native witness is strictly below the exact rational lower bound for **every** model meeting the specified clean-risk threshold. The finite X-only clean-class witness is separately enclosed and verified below epsilon in every row, so the comparison class is nonempty. The output preserves the normalized prior cost and nonzero Bernoulli gate-deficit risk; it does not treat the almost-on gate as exactly deterministic.

**All four matched-noise-precision cases beta=400 remain `bound_inconclusive`.** The two beta-1600/gamma-16 cases are also inconclusive. Negative sufficient margins are retained. These rows do not establish absence of an obstruction, correctness of the objective, or an explanation for the old beta-400 training states. The six successes occur at the predeclared overprecision ratios 4 or 64, not at the matched ratio 1.

The accepted claim concerns the exact real-valued native architecture and fixed-beta normalized population objective, with full stochastic reconstruction risk. It can separate the objective infima without assuming a global minimizer exists, but it does not certify SGD convergence or its optimization error. The rational decoder and analytic log-beta/softplus parameter expressions are explicit. No formal certificate for float64-rounded Torch tensors is asserted; the optional native parity test is a separate tolerance-based implementation check. A clean-risk obstruction is not a dictionary-recovery-gap theorem, a generating-q estimator failure, or a learned-beta training conclusion. No BR5/W2/C2 gate is cleared by this addendum.

The fourth preflight archive and JUnit report substantiate **28 passed, zero failures/errors/skips**: nine new obstruction tests, two native optional-import tests, and 17 tests for the maintained finite-noise consumer test file. The tested finite-noise test hash is now `83c4c7cffcc2d6e9e5a3a965ba3d0f2fac08ca9ed35d9b2b3814bffc8a3b2e4a`, resolving the later test-version coverage noted above without changing or rerunning the already completed scientific jobs.

The minimal core-source adoption is also byte-bound. Current `src/__init__.py` (`326cbd6c53c894d637aa8661b45fc0e0bd33650e6536417efb1ab0c1c25023bd`) and `src/sae_model.py` (`f88524123b3d278b6dd0b6200a0058595d113f0d6fbe7eb2fdceb9b930821f3f`) are identical to the merged import, prior native scientific source seals, and the successful new preflight snapshot. The new `tests/test_native_optional_imports.py` snapshot (`87db73772441ca5623c622f3761f593aff1f9c0f7e35766e38a5f6637d877a47`) passed both checks: public/native imports and a deterministic backward calculation with SAELens blocked, and preservation of lazy baseline alias identities. It does not depend on the untracked Phase 2 density module. These checks support adopting the already verified lazy-import bytes for a fresh checkout; they do not imply that unrelated working-tree changes were adopted or that the full SAELens stack was tested.

No scientific rerun is required by this review. All four jobs retain their original results, source versions, numerical limitations, and distinct evidence scopes.
