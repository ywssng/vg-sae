# Colab CPU continuation protocol, version 1

Frozen scientific scope before new outputs, 2026-10-03.

## Resource and execution contract

Use the authenticated official Colab CLI and a Standard CPU runtime only.
All tests, numerical evaluations and model forward passes run there. The local
checkout is used for reading, editing, packaging, source hashing, transfer and
Git. No host training, GPU allocation, LM download, activation collection or
confirmation world access is authorized in this version.

Drive mount on the current CPU runtime terminated with `ValueError: mount
failed`. Short jobs therefore use explicitly selected runtime storage and
immediate result download. A missing result or observation timeout is not an
instruction to resubmit; inspect the live session and saved task state first.

1. Validate the Colab controller and recovery contracts remotely (180-second
   observation window). This is infrastructure validation, not a VG result.
2. Install the official `python-flint==0.8.0` only in the Colab runtime; record
   Python, torch, NumPy, SciPy, flint and platform versions. No local dependency
   installation is needed.
3. Run focused new numerical/provenance tests before consuming imported
   evidence. Cap this step at 120 wall seconds.
4. Independently verify the existing population terminal partitions, all 20
   fixed cells in their original order. Allow at most 600 process CPU seconds,
   one process/core, with a 512 MiB memory bound and a 900-second wall limit.
   Every uncompleted cell remains audit-unresolved; retain the completed prefix
   and failure/cap reason. No original optimizer restart or new subdivisions.
5. Evaluate the explicit finite-noise bound only from fully verified cells.
   Fixed grid and endpoints are in `FINITE_NOISE_DERIVATION.md`. Allow 60 CPU
   seconds and 120 wall seconds. Keep all nonpositive margins and inconclusive
   original-noise checks.
6. A separate native finite-beta analytic witness check uses 16 fixed
   q/beta/gamma cases and the exact four-state noiseless population, with finite
   sigmoid/softplus parameter tensors. Allow 60 CPU seconds, 120 wall seconds.
   This is a restricted-family identity/witness study with no fitting. Its
   analytic contract and numerical representability limits must be reviewed
   before execution.

These are distinct versioned jobs with content-hashed source, plans and input
receipts. Technical corrections get a new version and retain the failed
attempt; they do not erase outcomes or silently enlarge a scientific budget.
Successful tests do not authorize repeated testing without a change or an
unresolved issue. Additional research requires its own concrete scope and
budget within the user's broader continuation authorization.

## Input preservation and interpretation

The merged import's original raw files, historical source/approvals and result
seals remain unchanged. Required raw inputs may be losslessly packed for
transfer; each consumed member is checked against its original seal in Colab.
Raw checkpoints and large traces are not added to Git. Small source, config,
receipts and new research conclusions are committed and pushed after review.

Old primary image resolution, old full-tree replay coverage, new terminal
certificate coverage and new finite-noise exclusion are distinct fields.
The new terminal audit may strengthen evidence without rewriting any old
status. G1 learns D/q/v under known structural assumptions; the handoff's
“known dictionary” shorthand conflicts with that detailed protocol and is not
used as the new claim. Known-D mixture controls stay separate.

BR5 remains measurement_unresolved, W2 remains gated, and C2a/C2b remain
unentered. These values change only with evidence for the original defined
requirements, not with successful supplied-model mathematics.
