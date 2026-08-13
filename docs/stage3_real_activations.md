# Stage 3: real-model activation SAE sweep

Stage 3 trains the same four comparison families on pinned residual-stream
activations from the model/layer conditions used in the main experiments of
*Sparse but Wrong*:

| target | hook used by the HF proxy | `d_in` | tokenized artifact | requested/effective train tokens |
| --- | --- | ---: | --- | ---: |
| Gemma-2-2B layer 5 | `model.layers.5` | 2,304 | Gemma `abbrv-2B` | 500,000,000 / 500,002,816 |
| Gemma-2-2B layer 12 | `model.layers.12` | 2,304 | Gemma `abbrv-2B` | 1,000,000,000 / 1,000,001,536 |
| Llama-3.2-1B layer 7 | `model.layers.7` | 2,048 | Llama `abbrv-1B` | 500,000,000 / 500,002,816 |

The effective values reproduce SAELens' complete-4096-token-batch accounting.
Each SAE has width 32,768 and is trained from a tokenizer-specific, pinned
`chanind/pile-uncopyrighted-*-1024-abbrv-{1B,2B}` artifact. Gemma uses the
published `abbrv-2B` artifact and Llama uses `abbrv-1B`. The next 1,048,576 tokens
form a disjoint held-out activation range. No activation cache is materialized;
the pinned language model produces activations online and the rolling
checkpoint includes the activation shuffle-buffer state for exact stream-position
and buffered-batch continuation. As in the paper runner, resident language-model
weights are float32 and CUDA forwards use BF16 autocast; future activations are
regenerated from the pinned source/model and remain subject to GPU backend
determinism. The pretokenized rows already contain the paper-configured BOS
token; the live source never inserts a second BOS. The loader seeks directly to
the containing immutable Parquet shard, so held-out evaluation and resume scan
at most one shard prefix rather than replaying the entire 500M/1B token prefix.

## Run

Both model repositories require accepting their Hugging Face licenses and an
authenticated local Hugging Face session. Training also follows the existing
sweep policy and requires authenticated online W&B logging.

```bash
# Full paper-main seed scope. Evaluation starts automatically only
# after every selected training job has a valid final checkpoint.
uv run python -B runs/run_RealActivation_sweep.py \
  --targets all \
  --methods all \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-per-device 1
```

The full command is intentionally large. Runs are independent and resumable,
and the target-specific grids total 390 train/eval jobs (162 Gemma L5, 84
Gemma L12, and 144 Llama L7). Gemma L5 and Llama L7 use seeds `[0,1,2]`, as in
the main Figure 9; the public Gemma L12 artifacts expose seed 0, so its default
is `[0]`. Pass `--seeds 0,1,2` to extend L12 to the same robustness repeat.
Float32 width-32,768 final checkpoints alone are
on the order of hundreds of GB for the full grid; each active run also keeps one
atomic multi-GB optimizer/buffer resume checkpoint, which is deleted after
successful completion. A target or method can be completed incrementally:

```bash
uv run python -B runs/run_RealActivation_sweep.py \
  --targets gemma-2-2b-layer5 \
  --methods vgsae,batchtopk \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Use `--skip-eval` to stop after training. To evaluate an already completed
target root explicitly:

```bash
uv run python -B runs/run_RealActivation_sweep_eval.py \
  --sweep-dir outputs/runs/<stage3-target-sweep-id> \
  --methods all \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

`--fast-dev-run --seed 0 --targets <one-target>` reduces every selected method
to one control and one complete 4,096-token optimizer batch. It still downloads
and loads the real gated language model, so it is a pipeline smoke test rather
than a CPU unit test.

## Methods and controls

- The target-specific BatchTopK grids are Gemma L5
  `[10,40,80,120,160,200,240,260,300,400,500,750,1000,1500,2000]`, Gemma L12
  `[10,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,350,400,450,500,750,1000,1500,2000,2500]`,
  and Llama L7
  `[10,20,40,60,100,150,200,250,500,750,1000,1500,2000]`.
- `VG-SAE` always uses `beta_mode="learned"`. Its sparsity coefficients are
  preregistered hypotheses obtained by linear interpolation/extrapolation of
  the Stage-2 learned-beta control curve against `log10(density)`, evaluated at
  each target's paper BatchTopK `K / 32768` density.
- `L1/ReLU SAE` uses the analogous Stage-2 L1 interpolation so that all
  coefficient-controlled models receive a target-density hypothesis without a
  new real-activation calibration experiment.
- `BatchTopK SAE` uses the target-specific K coordinates recovered from the
  paper's main vector figures. It uses learning rate `3e-4`, constant LR, and
  decoder-norm-aware activation ranking. Its decoder initialization norm is
  `0.1`, matching all three published target artifacts.
- `JumpReLU SAE` uses coefficients
  `[0.125, 0.25, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75]`, learning
  rate `2e-4`, a 100M-token L0 warmup, tanh scale 4, bandwidth 2, and pre-act
  coefficient `3e-6`, following the paper text. Threshold and decoder
  initialization both start at `0.1`, matching the public Gemma layer-12
  artifacts. Those public JumpReLU artifacts are labeled 500M tokens and
  learning rate `7e-5`, whereas the main paper comparison describes the 1B
  suite and Appendix H gives `2e-4`; Stage 3 follows the paper-text budget/LR
  and stores that choice in each resolved config.

The comparison x-axis is always achieved held-out hard L0. A configured K,
lambda, or interpolated density is never presented as an achieved sparsity.
CLI `--model-sparsity-control METHOD=...` overrides a grid without changing the
other method families.

## Automatic evaluation

Activation evaluation covers the exact held-out token budget and records hard
reconstruction MSE/relative error/cosine/explained variance, L0/L1/density,
dead-latent fraction, norm and bias diagnostics, VG posterior/expected-code
metrics, and the exact blockwise
`mean_{i<j} |cos(W_dec[i], W_dec[j])|` diagnostic from *Sparse but Wrong*.

The first 16,384 held-out tokens also receive model-intervention evaluation:
CE loss with the original model, SAE replacement, and zero ablation; KL with
SAE replacement and ablation; and their normalized CE/KL scores. These tokens
are a prefix of—not an additional split from—the activation-evaluation range.

Real activations have no known generating dictionary, sparse coefficients, or
support labels. Stage-1/2 ground-truth recovery, support, MCC, uniqueness,
classifier, and clean-latent generalization fields therefore remain present as
JSON `null` values with explicit availability reasons. They are not replaced
with proxy values.

Each target root contains per-run configs, histories, rolling/final
checkpoints, held-out metrics and bounded previews. Evaluation aggregates
`summary/last/final_metrics.csv`, seed means, training curves, and three
headless figures for reconstruction, sparsity/VG diagnostics, and decoder
pairwise cosine.
Incremental evaluation rewrites this shared summary from every current,
completed manifest artifact; adding one method does not discard earlier ones.

## Pinned references

- *Sparse but Wrong*, arXiv v4: <https://arxiv.org/pdf/2508.16560v4>
- Paper implementation at the inspected commit:
  <https://github.com/chanind/sparse-but-wrong-paper/tree/d5886b540dc5b9cac4f76e6db2b0cce1b0b7c585>
- Paper training/evaluation notebook:
  <https://github.com/chanind/sparse-but-wrong-paper/blob/d5886b540dc5b9cac4f76e6db2b0cce1b0b7c585/notebooks/train_and_eval_llm_sae.ipynb>
- Published SAE artifacts and resolved configs:
  <https://huggingface.co/chanind/sae-l0-exploration/tree/main>
- Pinned SAELens source used by this repository:
  <https://github.com/decoderesearch/SAELens/tree/8be14080485952f729ed58d674bcddf9778e0aa4>
