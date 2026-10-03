# Code and Colab continuation provenance audit

Date: 2026-10-03. Scope: static source, dependency, manifest, and file-layout inspection. Findings describe the state before the launcher extension below. No training, numerical experiment, test execution, or Colab allocation was performed by this auditor. Semantic judgments are same-family/provisional.

## Findings that affect continuation

1. **The current core model already matches the import.** Of 28 top-level Python modules in `research-import-20261002T131813Z/project/src/`, 27 have byte-identical counterparts in the current root. The only new module is `src/orthogonal_density.py`. This includes the pre-existing dirty `src/__init__.py` and `src/sae_model.py`, and the untracked `src/phase2_density.py`. Do not overwrite or silently stage those earlier changes.
2. **Continuation scripts and tests have not been integrated.** The import has 25 top-level scripts; `scripts/run_phase2_bridge.py` matches the root byte-for-byte, while the other 24 are absent from the root. All 11 imported top-level tests are absent from the root. The import itself is not a complete standalone checkout: `scripts/analyze_phase2_common_field_20261001.py:25` imports `scripts/analyze_phase2_bridge.py`, available only at the root.
3. **The existing Colab source bundle is smoke-specific.** `scripts/colab_experiment.py` packages `src/*.py`, `configs/base.yaml`, `scripts/colab_worker.py`, and `scripts/colab_smoke.py`. Plans that name imported scripts, tests, protocol files, or population engines will therefore fail unless those files are explicitly packaged. Root package installation is unnecessary for the minimal path and would bring optional SAELens/LLM dependencies and the project CUDA index into scope.
4. **Historical one-shot runners are not portable continuation commands.** Their fixed output directories, source seals, exact runtime checks, and closed attempt records intentionally prevent replacement runs. Use a new versioned runner and manifest for newly authorized work; preserve the old source and guards.
5. **Core source compatibility does not establish scientific readiness.** The import's `START_HERE_KO.md` explicitly retains native q measurement as unresolved, W2 blocked, C2 unentered, and numerical replay incomplete. Passing portable unit tests cannot clear those gates.

## Exact source versions

SHA-256 values were computed from file bytes without importing the experiment modules.

| File | Relation to root | Imported SHA-256 |
|---|---|---|
| `src/__init__.py` | identical; pre-existing dirty | `326cbd6c53c894d637aa8661b45fc0e0bd33650e6536417efb1ab0c1c25023bd` |
| `src/sae_model.py` | identical; pre-existing dirty | `f88524123b3d278b6dd0b6200a0058595d113f0d6fbe7eb2fdceb9b930821f3f` |
| `src/model.py` | identical | `a5c9ee7ff56a4b1d7391f3ee6f7fab00c5b75dbb83f738ee141011fd44d44836` |
| `scripts/run_phase2_bridge.py` | identical; untracked | `742faaaa5c89274f3d028db44a95f72a9d5ef9c06a84caba4e02370fdf27686e` |
| `scripts/run_phase2_density_probe_v2c.py` | new | `4f2f33aa01345f90d096b6ef5614da9ca29de44621737222e93a1d3731ba7cdf` |
| `scripts/run_phase2_paired_continuation_20261001.py` | new | `22787e8eb98c5b6042eed33ef49fa62123b8940340a5d16960af2512d3e26aad` |
| `scripts/run_phase2_fresh_precision_reset_replication_20261001.py` | new | `b5ebad8d97b9d81af9f8d3611627e6de9e3b0c8d435ee6a7b436c67de9a36037` |
| `scripts/phase2_fresh_precision_reset_adapter_20261001.py` | new | `1c38eb203502d96b957c79d14eb20472e3b8e0efe61a3b250f566cd54e307397` |
| `scripts/run_native_retention_g2_20261002.py` | new | `1cc4d5ce2c584a36d97ee3fabbc62c92c77f6508e909020abd2b0880d1a0d29b` |

The definitive historical G2 list is the imported `refine-logs/runs/vg-sae-native-retention-20261002/FINAL_SOURCE_HASHES.json`. It includes the two `colab/phase2_density_probe_v2c/` JSON files, G2 protocol, recovery/timing receipts, and G2 test. Include them only if a new job actually verifies the historical freeze; ordinary pure-function tests need fewer files.

## Minimal source closures

These are per-task closures, not a recommendation to rerun the historical experiments.

| Task | Required sources in addition to worker/new task wrapper | Python dependencies |
|---|---|---|
| Native VG objective and new deterministic fixtures | `src/__init__.py`, `src/model.py`, `src/sae_model.py`; relevant tests | torch, NumPy, pytest; SciPy only if the new fixture uses it |
| Existing Phase 2 bridge functions | previous row plus `scripts/run_phase2_bridge.py` | torch, NumPy |
| Existing G1 pure likelihood functions | `src/__init__.py`, new `src/orthogonal_density.py`, `tests/test_orthogonal_density.py` | NumPy; SciPy for test enumeration |
| Existing G2 pure functions/tests | native row plus bridge, density-probe-v2c, paired-continuation, fresh-reset-replication, fresh-reset-adapter, G2 runner and `tests/test_native_retention_g2_20261002.py` | torch, NumPy, SciPy, pytest |
| Existing common-field functions/tests | `src/phase2_density.py`, both `analyze_phase2_bridge.py` and `analyze_phase2_common_field_20261001.py`, `tests/test_phase2_common_field_20261001.py` | NumPy, SciPy, pytest; plotting requires matplotlib |
| New population replay using mathematical engine | versioned engine/replayer, exact input specification and referenced saved traces/results | python-flint; no torch or GPU needed |

The lazy exports in `src/__init__.py` and `src/sae_model.py` prevent native VG imports from eagerly loading SAELens. Preserve this property. The root `requirements.txt` is a broad environment file, and `pyproject.toml` requires Python >=3.14 with a CUDA-specific torch source; neither represents the small historical CPU environment. Record actual Colab Python/package versions separately rather than claim an exact historical replay from broad minimum requirements.

## Historical execution and raw-data obstacles

- G2 `run_native_retention_g2_20261002.py:55` requires torch `2.14.0+cpu` and NumPy `2.3.5`. Lines 155/161 expect `ROOT.parent/G1_private_preservation_receipt.json`, absent from both the merged project and its parent. A successful transport audit does not supply that path automatically.
- G2 loads its prior source checkpoints from `phase2_fresh_precision_reset_replication_20261001/`, directly below the project, while the original fresh runner's default is under `outputs/`. The direct-root imported directory exists and contains 183 files (3,508,015 bytes), including all 12 source/retained/reset groups and evaluator-only artifacts.
- Paired continuation's two fixed checkpoint paths at `outputs/phase2_dotcloud_density_probe_v3_20261001/.../gamma_15/checkpoint_8000.pt` and `outputs/phase2_dotcloud_density_probe_20261001a/.../gamma_16/checkpoint_8000.pt` are absent from the merged project. Historical raw snapshots are indexed in `base/history/SNAPSHOT_INDEX.json`; restore only the required snapshot into a new directory if needed.
- The merged project has 474 files (60,873,154 bytes) under `outputs/`. G1's 125 files total 18,917,966 bytes; G2's 88 files total 19,676,887 bytes. This inventory is file presence/size evidence, not an independent verification of array contents.
- The population engine inserts sibling `vg-sae-certified-projection-20261002/dependencies/` into its import path. That vendored directory is absent from the merged project. Its historical manifest hashes CPython 3.12 Linux binaries; `input_spec.json` pins Python `3.12.14` and python-flint `0.8.0`.
- Population `run_locked.py` has closed one-shot output, exact manifest/reviewer bindings, and an expired shared wall deadline. Its unexecuted stored-leaf draft additionally expects an absent approval receipt and the old deadline. Neither is a continuation entry point. A new bounded replay must explicitly preserve the old unresolved statuses and distinguish new evidence.

## Test scope and remaining verification

The 11 imported tests cover orthogonal likelihood, G2 objective/update/retention bookkeeping, oracle envelopes, exact conditional models, common-field observables, and precision-reset/continuation integrity. Several tests perform small optimizer updates or numerical enumeration; run them only on Colab CPU under the current user constraint. No test was executed during this audit.

Before new work is claimed validated, the Colab bundle should be inspected for its exact file list and hash; copied or reused sources should be bound by a manifest; only the necessary dependency set should be present; and CPU allocation should be checked before task execution. A new runner must send durable results to `COLAB_TASK_OUTPUT` or another explicitly preserved Drive path. The historical scripts' fixed root-relative output directories otherwise remain inside the ephemeral source extraction tree.

Do not rerun closed historical studies merely to turn missing runtime dependencies into a green status. The actionable next step is an explicit source-manifest extension to the existing Colab launcher and a versioned, bounded research task selected from the current evidence.

## Launcher implementation handoff

Following this audit, `scripts/colab_experiment.py`, `tests/test_colab_experiment.py`, and `docs/colab.md` were extended with an additive explicit source manifest, a CPU-only runtime guard, and optional runtime storage when Drive mount fails. Default source archive ordering/bytes and default Drive storage are retained. CPU-only/runtime policy is included in the saved plan identity, while source identity remains the archive hash. Runtime outputs require an immediate fetch before VM release. Regression tests were added for source containment and unchanged source identity, GPU refusal, plan-change refusal, and runtime status/fetch. Only diff whitespace was checked locally; the coordinator must run these tests on Colab CPU.
