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
uv run python -B runs/run_CustomData_sweep.py \
  --beta-mode profiled \
  --methods all \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-per-device 2

# Evaluate both `last` and `best`; plotting still defaults to Notebook 07's `last`.
uv run python -B runs/run_CustomData_sweep_eval.py \
  --beta-mode profiled \
  --methods all \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-per-device 2

# Repeat both commands with --beta-mode learned; it gets its own root/group.
```

The Stage-1 data is a noiseless linear mixture of an overcomplete random unit
dictionary. Marginal firing probabilities are rank-skewed from the requested
mean `support_density` with the default Stage-1 exponent `frequency_skew=0.5`
while preserving that mean; settings that would require probability clipping
are rejected. `--frequency-skew 0` is the homogeneous-frequency ablation.
Supports are independent, and active amplitudes default to nonnegative
`Exponential(scale=1)` draws. `--amplitude-mode constant` uses
`sqrt(2) * scale`, while `--amplitude-mode uniform` uses
`Uniform(scale, ((sqrt(21)-1)/2) * scale)`, whose upper multiplier is about
`1.7913`. All three laws have the same active second moment
`E[amplitude^2] = 2 * scale^2`. There is no added
dictionary coherence, correlated firing, or hierarchy.
`ground_truth_num_features` controls the data dictionary while `sae_width`
independently controls the learned latent width. Their defaults are
`input_dim=128`, `ground_truth_num_features=1024`, and `sae_width=1024`, with
`support_density=0.01` (expected true L0 `10.24`).

The default output directory and W&B group are derived from the resolved data
config, for example
`stage1_beta_profiled_din128_gt1024_sae1024_sd001_seed0`. The learned-beta
counterpart uses `stage1_beta_learned_...`. An explicit `--output-dir` still
overrides it. The no-argument eval command resolves the same default config;
when training with CLI overrides, pass its resolved directory to eval with
`--sweep-dir`. Training sweeps require authenticated W&B online logging. W&B
stores this ID as the top-level `exp_id`, the configured stage as `stage`, and
the actual sweep-directory basename as `sweep_root`; the latter is also the W&B
group. `amplitude_mode`, `amplitude_scale`, and `frequency_skew` are stored as
top-level filterable fields and tags alongside stage, method, and beta mode. The
full sweep root stays in config/group rather than a tag because W&B limits
individual tags to 64 characters, so long custom output directories remain
directly filterable.

One output directory represents one fixed data condition. Sweep seeds with
`--seeds 0,1,2` and add repeatable controls such as
`--model-sparsity-control topk=1,2,4`. Data dimension and density changes get a
new automatic directory; for other config changes, use an explicit
`--output-dir`. Use `--methods vgsae` for one architecture, or
`--fast-dev-run --devices cpu` for a small local smoke run. Each run
stores its resolved config, training history, `best.pt` and `last.pt`, plus
checkpoint-specific metrics and mask arrays. Sweep-level CSVs are written under
`summary/`. Evaluation processes both checkpoints by default; pass
`--checkpoint last` or `--checkpoint best` to select one. Open
`notebooks/10_exp07_parallel_sweep_results.ipynb` after evaluation; it only
loads artifacts and draws the notebook-07 figures. Set
`VGSAE_DENSITY_MODE=hard` for a fully hard-code comparison: the x-axis,
reconstruction/recovery/support metrics, masks, and empirical test-set reference
line then use the same thresholded inference codes. `best` means the lowest
full-training objective observed at a history step and is intentionally not
mixed into the default `last` reproduction.

```text
outputs/runs/stage1_beta_profiled_din128_gt1024_sae1024_sd001_seed0/
├── sweep_config.json, manifest.json
├── runs/<method>/<run_id>/
│   ├── config.json, training_history.csv
│   ├── checkpoints/{best,last}.pt
│   └── eval/<checkpoint>/{metrics.json,cache.npz}
└── summary/
    ├── training_curves.csv, data_preview.npz
    └── <checkpoint>/{final_metrics.csv,final_metrics_seed_mean.csv,summary.json}
```

Run Stage 2 on the pinned official SynthSAEBench generator with streamed data:

```bash
# Short coefficient/L0 calibration. --test-samples defaults to train/8.
uv run python -B runs/run_SynthSAEBench_sweep.py \
  --calibration-grid \
  --beta-mode profiled \
  --output-dir outputs/runs/stage2_synthsaebench16k_calibration \
  --training-samples 1048576 \
  --history-every 64 \
  --methods all \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-per-device 2

uv run python -B runs/run_SynthSAEBench_sweep_eval.py \
  --sweep-dir outputs/runs/stage2_synthsaebench16k_calibration \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-per-device 2

# Full default: about 200M train samples and exactly one-eighth as held-out test.
# beta_mode selects both the VG objective and its calibrated gamma grid/root.
uv run python -B runs/run_SynthSAEBench_sweep.py \
  --beta-mode profiled \
  --methods vgsae \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-per-device 2

uv run python -B runs/run_SynthSAEBench_sweep_eval.py \
  --beta-mode profiled \
  --methods vgsae \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-per-device 2

# Repeat the two commands with --beta-mode learned for the learned-beta grid/root.
```

This runner always loads `decoderesearch/synth-sae-bench-16k-v1` at revision
`b2efd8b919ae46d6d487c73d46db5ee52813621d`; it does not build benchmark
variants. The fixed dimensions are 768 input units, 16,384 ground-truth
features, and SAE width 4,096. Training is an online stream rather than a finite
dataset, so there is no epoch axis. The batch-aligned defaults are 199,999,488
train samples (195,312 optimizer updates at batch size 1,024) and 24,999,936
fresh held-out samples. The released experiment configs use constant Adam
learning rate `3e-4`; pass `--lr-decay-fraction 0.3333333333` to reproduce the
final-third linear decay described in the paper instead.

The pretrained artifact records `scale_children_by_parent=false`, while the
paper's generator description and creation script use `true`. Manifests store
the pinned revision, model-config SHA-256, and SAELens source revision so
fixed-artifact results are not mistaken for a regenerated paper variant.
Evaluation accumulates the official
Mean Correlation Coefficient (MCC), uniqueness, classifier, L0, dead-latent,
shrinkage, and explained-variance
metrics in streaming form. Only a small preview is cached for heatmaps. Point
`VGSAE_SWEEP_DIR` at a completed Stage-2 directory and run notebook 10 to draw
the SynthSAEBench-specific panels with the same artifact-only workflow.
`vg_posterior_diagnostics.png` compares VG hard inference with
the posterior expectation; Stage-1 CustomData artifacts skip that unavailable
panel without error. For Stage-2 artifacts, notebook 10 and `plot_all` also
write four additive `stage1_style_*.png` diagnostics using the Stage-1 panel
columns. These retain Stage-2 matching semantics: decoder recovery is the MCC
alias, classifier metrics are per-latent macro averages, and unmatched
ground-truth features are not scored through Stage-1's rectangular union. For
a VG-only mode root, also set
`VGSAE_BASELINE_SWEEP_DIR` to the preserved 35-run non-VG Stage-2 root; the
notebook fills only methods absent from the primary root and resolves each
heatmap from its source root.

The default one-seed method grid has seven controls per method. TopK and
BatchTopK use target `k=[15,20,25,30,35,40,45]`. Full 200M-sample calibration
runs were used to invert the calibration-stream hard-L0 curves toward the same
targets.
Mode-specific 20M sweeps plus 50M anchor runs select VG-SAE gamma
`[1.64,1.72,1.84,2.01,2.26,2.84,6.12]` for `profiled` and
`[1.63,1.71,1.82,1.99,2.22,2.80,6.00]` for `learned` for the definitive 200M
runs; choosing `--beta-mode` selects the matching grid as well as a separate
sweep root. The other calibrated
grids are L1
`[0.99,1.07,1.17,1.36,1.69,2.42,4.26]`, JumpReLU
`[0.41,0.46,0.52,0.61,0.78,1.16,1.80]`, and Gated
`[1.07,1.10,1.21,1.38,1.70,2.17,3.28]`. These are empirically calibrated static
controls, not the official L0 autotuner; rounded interpolation means achieved
L0 remains the authoritative comparison axis. The same fixed evaluation stream
was used for calibration and the final tables, so it is a calibration stream,
not an untouched test set. The removed-mode VG grid
`[0.77,0.82,0.88,0.96,1.07,1.17,1.39]` is historical only and is not used by
either supported mode. VG-SAE therefore reports and plots
hard and posterior-expected L0/reconstruction separately; a short pilot showed
a large hard/expected gap. Interrupted jobs write an exact rolling resume
checkpoint every 10,000 updates. The official study used five seeds; the local
default is seed 0, and `--seeds 0,1,2,3,4` requests the full five-seed repeat at
five times the compute.

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

VG-SAE experiment interfaces expose `beta_mode="profiled"` (the legacy
beta-eliminated objective) and `"learned"`. In profiled mode `--beta` is only
the configured initial/reporting value; the minibatch optimum is used in the
loss. In learned mode it initializes the trainable positive precision. The
sparsity field `lambda_sparsity` may be any finite real value: positive values
favor sparse supports and negative values intentionally favor dense supports.

The SAELens-native VG architecture lives in `src.saelens_vg` and registers the
architecture name `"vg"` for both training and inference. Use
`beta_mode="learned"` when the experiment must not impose the profiled
`partial F / partial beta = 0` stationarity condition. Its public SAELens code
is hard-thresholded for firing metrics, while the training objective
reconstructs with the variational expected code; the two metric families are
named separately.

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
