# Variational Garrote Sparse Regression and VG-SAE

`vg-sae` is a cleaned-up implementation that combines the strongest parts of
`vg-sae-x` and `vg-sae-e`.

It implements the Variational Garrote model from Soh et al. (2025), including the
beta-eliminated free-energy objective, AdamW training, synthetic spike-and-slab
data generation, sparsity metrics, gamma sweeps, and equation-level tests.

It also includes a first VG-SAE research stack: variational Bernoulli support
gates, nonnegative amplitudes, unit-norm decoder dictionaries, synthetic
sparse-coding experiments, baseline SAEs, transition diagnostics, and a small
GPT-2 residual-stream activation pipeline.

SAELens is pinned to v6.47.0 commit `8be1408`. `StandardSAE`, `TopKSAE`,
`BatchTopKSAE`, `JumpReLUSAE`, and `GatedSAE` are identity aliases of its
official training classes; their configs, losses, optimizer steps, and inference
export are not locally reimplemented. The project's `StandardSAE` name aliases
upstream `StandardTrainingSAE`; `to_inference_sae` returns the corresponding
upstream inference architecture.

## Quick Start

```bash
python -m src.train --config configs/base.yaml
```

Run the synthetic VG-SAE transition smoke experiment:

```bash
python -B scripts/run_synthetic_sweep.py \
  --output-dir outputs/synthetic_first_pass \
  --input-dim 8 \
  --widths 16 \
  --n-samples 256 \
  --support-density 0.125 \
  --lambdas 0.0,0.5,1.0,2.0 \
  --steps 80 \
  --beta-mode profiled \
  --include-no-variance \
  --include-baselines
```

Run the Stage-1 custom baseline through parallel, resumable jobs instead of
training in Jupyter:

```bash
# WANDB_API_KEY is read from the ignored project-root .env.
uv run python -B runs/run_saes_sweep.py \
  --methods all \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-per-device 16

# Evaluate both `last` and `best`; plotting still defaults to Notebook 07's `last`.
uv run python -B runs/run_saes_sweep_eval.py \
  --methods all \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-per-device 16
```

The Stage-1 data is a noiseless linear mixture of an overcomplete random unit
dictionary. Marginal firing probabilities are rank-skewed from the requested
mean `support_density` with the default Stage-1 exponent `frequency_skew=0.5`
while preserving that mean; settings that would require probability clipping
are rejected. Supports are independent, and active
amplitudes are nonnegative `Exponential(scale=1)` draws. There is no added
dictionary coherence, correlated firing, or hierarchy.
`ground_truth_num_features` controls the data dictionary while `sae_width`
independently controls the learned latent width. Their defaults are
`input_dim=128`, `ground_truth_num_features=1024`, and `sae_width=1024`, with
`support_density=0.01` (expected true L0 `10.24`).

The default output directory and W&B group are derived from the resolved data
config, for example
`stage1_din128_gt1024_sae1024_sd001_seed0`. An explicit `--output-dir` still
overrides it. The no-argument eval command resolves the same default config;
when training with CLI overrides, pass its resolved directory to eval with
`--sweep-dir`. W&B also stores this ID as the top-level `exp_id` config and
summary field, so runs can be filtered independently of their output path.

One output directory represents one fixed data condition. Sweep seeds with
`--seeds 0,1,2` and add repeatable controls such as
`--model-sparsity-control topk=1,2,4`. Data dimension and density changes get a
new automatic directory; for other config changes, use an explicit
`--output-dir`. Use `--methods vgsae` for one architecture, or
`--fast-dev-run --devices cpu --no-wandb` for a small local smoke run. Each run
stores its resolved config, training history, `best.pt` and `last.pt`, plus
checkpoint-specific metrics and mask arrays. Sweep-level CSVs are written under
`summary/`. Evaluation processes both checkpoints by default; pass
`--checkpoint last` or `--checkpoint best` to select one. Open
`notebooks/10_exp07_parallel_sweep_results.ipynb` after evaluation; it only
loads artifacts and draws the notebook-07 figures. `best` means the lowest
full-training objective observed at a history step and is intentionally not
mixed into the default `last` reproduction.

```text
outputs/runs/stage1_din128_gt1024_sae1024_sd001_seed0/
├── sweep_config.json, manifest.json
├── runs/<method>/<run_id>/
│   ├── config.json, training_history.csv
│   ├── checkpoints/{best,last}.pt
│   └── eval/<checkpoint>/{metrics.json,cache.npz}
└── summary/
    ├── training_curves.csv, data_preview.npz
    └── <checkpoint>/{final_metrics.csv,final_metrics_seed_mean.csv,summary.json}
```

Cache GPT-2 small layer-8 residual-stream activations, then sweep VG-SAE:

```bash
python -B scripts/cache_gpt2_activations.py \
  --output outputs/gpt2/gpt2_layer8_resid.pt \
  --max-tokens 4096

python -B scripts/run_gpt2_sweep.py \
  --cache outputs/gpt2/gpt2_layer8_resid.pt \
  --output-dir outputs/gpt2_sweep \
  --expansion-factors 4 \
  --beta-mode learned \
  --lambdas 0.0,0.5,1.0,2.0,3.0,5.0
```

Minimal Python usage:

```python
import torch

from src.model import VGConfig, VariationalGarrote
from src.loss import vg_free_energy

model = VariationalGarrote(VGConfig(n_features=8, gamma=1.0))
x = torch.randn(16, 8)
y = torch.randn(16)
loss = vg_free_energy(model, x, y)
loss.backward()
```

VG-SAE supports `beta_mode="profiled"` (the legacy beta-eliminated objective),
`"fixed"`, and `"learned"`. In profiled mode `--beta` is only the configured
initial/reporting value; the minibatch optimum is used in the loss. In learned
mode it initializes the trainable positive precision. The sparsity field
`lambda_sparsity` may be any finite real value: positive values favor sparse
supports and negative values intentionally favor dense supports.

The SAELens-native VG architecture lives in `src.saelens_vg` and registers the
architecture name `"vg"` for both training and inference. Use `beta_mode="fixed"`
or `"learned"` when the experiment must not impose the profiled
`partial F / partial beta = 0` stationarity condition. Its public SAELens code is
hard-thresholded for firing metrics, while the training objective reconstructs
with the variational expected code; the two metric families are named separately.

`src.saelens_data` converts the existing synthetic tensors and local activation
caches into deterministic SAELens data providers. Normalization is a single
SAELens-compatible scale fitted on training data only. Arrow activation-cache
rows can be kept together during splitting and selected token IDs can be removed;
for Pythia-scale runs, use a bounded cache slice or a native streaming store.
The parallel sweep keeps data selection behind a serialized `kind` boundary so
SynthSAEBench streams or real activation stores can be added without changing
the scheduler, checkpoint layout, or plot-only notebook.

## What Is Improved

- Uses trainable mask logits with `sigmoid`, avoiding zero gradients at the mask boundary.
- Uses the prior-corrected gamma penalty `+ gamma * sum(m)`, so larger positive gamma encourages sparsity.
- Generates synthetic data with exact finite-`N` active counts and SNR-calibrated additive noise.
- Keeps sklearn/scipy imports lazy so core VG code imports without optional baseline dependencies.
- Includes tests for the paper equations and the implementation choices above.
- Imports the exact pinned SAELens Standard/L1, TopK, BatchTopK, JumpReLU, and
  Gated training implementations and delegates optimization to the official
  trainer.

## Verify

```bash
python -m compileall -q src tests
python -m pytest tests -q
```

The same compile and test checks run automatically in GitHub Actions for every
push to `main` and every pull request.
