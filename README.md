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

SAELens is pinned to v6.47.0 commit `8be1408`. `TopKSAE`, `BatchTopKSAE`,
`JumpReLUSAE`, and `GatedSAE` are identity aliases of its official training
classes; their configs, losses, optimizer steps, and inference export are not
locally reimplemented. L1-ReLU remains the small project-local reference model.

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

## What Is Improved

- Uses trainable mask logits with `sigmoid`, avoiding zero gradients at the mask boundary.
- Uses the prior-corrected gamma penalty `+ gamma * sum(m)`, so larger positive gamma encourages sparsity.
- Generates synthetic data with exact finite-`N` active counts and SNR-calibrated additive noise.
- Keeps sklearn/scipy imports lazy so core VG code imports without optional baseline dependencies.
- Includes tests for the paper equations and the implementation choices above.
- Imports the exact pinned SAELens TopK, BatchTopK, JumpReLU, and Gated training
  implementations and delegates their optimization to the official trainer.

## Verify

```bash
python -m compileall -q src tests
python -m pytest tests -q
```

The same compile and test checks run automatically in GitHub Actions for every
push to `main` and every pull request.
