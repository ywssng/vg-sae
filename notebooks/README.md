# VG-SAE Experiment Notebooks

These notebooks are split by proposal experiment rather than merged into one file.

1. `01_non_amortized_vg_sanity_check.ipynb`: direct per-sample VG support inference.
2. `02_synthetic_sparse_coding_transition.ipynb`: synthetic sparse coding,
   all-official SAELens baselines, and variance-term ablation. Embedded outputs
   predate the L1 migration; reruns write to
   `outputs/notebooks/exp02_synthetic_sparse_coding_all_official/`.
3. `03_toy_superposition_phase_diagram.ipynb`: phase diagram over feature-frequency skew and lambda.
4. `04_gpt2_small_layer8_residual_stream.ipynb`: GPT-2 small layer-8 residual-stream lambda/width sweep.
5. `05_feature_uncertainty_quality.ipynb`: feature uncertainty versus recovery and seed stability.
6. `06_ioi_causal_control_case_study.ipynb`: first IOI causal-control patching scaffold.
7. `07_synthetic_sparse_coding_rho_model_comparison.ipynb`: six-way VG-SAE,
   L1-ReLU, TopK, BatchTopK, JumpReLU, and Gated SAE comparison by measured
   rho_model. Full mode is a seed-0/1000-step calibration with paired
   per-method initialization and empirically calibrated coverage grids; use
   `VGSAE_NOTEBOOK_FAST_DEV_RUN=1` for a smoke run. It writes to
   `outputs/notebooks/exp07_saelens_v647_all_official/`; embedded outputs and
   the older `exp07_saelens_v647_six_method/` and
   `exp07_synthetic_sparse_coding_rho_model_comparison/` artifacts are
   preserved pre-migration runs. L1 thresholds are calibrated on the training
   split only, and the full run uses a 100-step dead-feature window. This
   single-seed calibration does not establish cross-seed robustness.
8. `08_synthetic_sparse_coding_vg_sparsity_sweep.ipynb`: VG-SAE-only nonnegative gamma sparsity sweep.
9. `09_saelens_synthsaebench_rho_model_comparison.ipynb`: six-model comparison
   on the official SynthSAEBench generator. Training uses reproducible fresh
   activation streams and evaluation uses SAELens' native synthetic metrics;
   the default FAST mode is an end-to-end smoke run, while full mode is an
   exploratory sweep rather than the 200M-sample leaderboard protocol.
10. `10_exp07_parallel_sweep_results.ipynb`: plot-only reproduction of notebook
    07 from saved parallel train/eval artifacts under `outputs/runs/`. It
    defaults to `last` checkpoints for fidelity and writes collision-free
    figure names; set `VGSAE_SWEEP_DIR` or `VGSAE_CHECKPOINT_KIND` to select a
    different sweep or the separately tracked `best` results.

Run notebooks from the project root or from this directory. Outputs are written under `outputs/notebooks/`.
