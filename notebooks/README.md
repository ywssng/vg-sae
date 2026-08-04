# VG-SAE Experiment Notebooks

These notebooks are split by proposal experiment rather than merged into one file.

1. `01_non_amortized_vg_sanity_check.ipynb`: direct per-sample VG support inference.
2. `02_synthetic_sparse_coding_transition.ipynb`: synthetic sparse coding, baselines, and variance-term ablation.
3. `03_toy_superposition_phase_diagram.ipynb`: phase diagram over feature-frequency skew and lambda.
4. `04_gpt2_small_layer8_residual_stream.ipynb`: GPT-2 small layer-8 residual-stream lambda/width sweep.
5. `05_feature_uncertainty_quality.ipynb`: feature uncertainty versus recovery and seed stability.
6. `06_ioi_causal_control_case_study.ipynb`: first IOI causal-control patching scaffold.
7. `07_synthetic_sparse_coding_rho_model_comparison.ipynb`: VG-SAE and baseline synthetic sparse-coding comparison by measured rho_model.
8. `08_synthetic_sparse_coding_vg_sparsity_sweep.ipynb`: VG-SAE-only nonnegative gamma sparsity sweep.

Run notebooks from the project root or from this directory. Outputs are written under `outputs/notebooks/`.
