# Reproduce the first-principles campaign

Run every command from the repository root. Use the existing `.venv`; no environment rebuild is needed. Fresh output directory is recommended for an independent reproduction. Check `nvidia-smi` first and select an idle physical GPU. The commands below use GPU1; the original execution used GPUs1/2/3 for worlds2401/2402/2403 in parallel.

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/outputs/.cache"
export HF_HOME="$PWD/outputs/.cache/huggingface"
export TORCH_HOME="$PWD/outputs/.cache/torch"
export MPLCONFIGDIR="$PWD/outputs/.cache/matplotlib"

.venv/bin/python -m pytest tests -q

for task_seed in 2401 2402 2403; do
  .venv/bin/python -B scripts/run_first_principles.py \
    --block exact --seed "$task_seed" \
    --output-dir outputs/first_principles_reproduction
done

for task_seed in 2401 2402 2403; do
  CUDA_VISIBLE_DEVICES=1 timeout --kill-after=10s 720s \
    .venv/bin/python -B scripts/run_first_principles.py \
    --block frozen --seed "$task_seed" --device cuda:0 \
    --output-dir outputs/first_principles_reproduction
done

for task_seed in 2401 2402 2403; do
  CUDA_VISIBLE_DEVICES=1 timeout --kill-after=10s 1440s \
    .venv/bin/python -B scripts/run_first_principles.py \
    --block joint --seed "$task_seed" --device cuda:0 \
    --output-dir outputs/first_principles_reproduction
done

.venv/bin/python -B scripts/run_first_principles.py \
  --block analyze --output-dir outputs/first_principles_reproduction
.venv/bin/python -B scripts/plot_first_principles.py \
  --output-dir outputs/first_principles_reproduction
```

`--smoke` with a separate output directory runs small execution checks; it does not provide research results. Baseline framework variant, environment revision, config, seed roles, checkpoint calibration history and worker status are recorded with each run. Frozen/main workers enforce their predeclared wall budget in training callbacks; the external timeout covers evaluation too. A timeout is an incomplete worker, not a negative research result.

The code allows completed fits to be skipped in an existing directory, but wall-time accounting in an independent reproduction should use a fresh directory. Raw checkpoints/NPZ files remain local under outputs and are not added to Git. Small final JSON/CSV summaries and selected figures are retained as tracked evidence.
