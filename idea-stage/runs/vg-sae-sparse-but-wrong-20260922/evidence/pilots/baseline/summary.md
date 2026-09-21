# P1 results — fixed current-VG baseline pilot

Completed 225 models; 225 final-grid evaluations.
Wall time including train/calibration/I/O/test: 0.777 hours.
Group means below use only covered worlds and are descriptive. Use the paired screening rows for method comparisons; coverage sets can differ.

## Primary target L0 = 1.8

|rho|Method|Covered worlds|Signed cosine|Absolute cosine|Support F1|Native L0|Signed-assignment leakage|Hard EV|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|-0.4|vg|0/3|—|—|—|—|—|—|
|-0.4|l1|1/3|0.9178|0.9178|0.8538|1.7457|0.1060|0.7746|
|-0.4|gated|1/3|0.9623|0.9623|0.9546|1.8218|0.0728|0.8801|
|-0.4|jumprelu_anthropic|0/3|—|—|—|—|—|—|
|-0.4|batchtopk|0/3|—|—|—|—|—|—|
|+0.0|vg|0/3|—|—|—|—|—|—|
|+0.0|l1|0/3|—|—|—|—|—|—|
|+0.0|gated|3/3|0.8977|0.9497|0.9289|1.8918|0.0502|0.8748|
|+0.0|jumprelu_anthropic|0/3|—|—|—|—|—|—|
|+0.0|batchtopk|0/3|—|—|—|—|—|—|
|+0.4|vg|0/3|—|—|—|—|—|—|
|+0.4|l1|1/3|0.8051|0.8051|0.8635|1.7081|0.1547|0.8221|
|+0.4|gated|1/3|0.7578|0.7578|0.8215|1.8976|0.1768|0.8755|
|+0.4|jumprelu_anthropic|0/3|—|—|—|—|—|—|
|+0.4|batchtopk|0/3|—|—|—|—|—|—|

## Paired screening

|Target|rho|Comparator|L0-matched worlds|Joint-pass worlds|
|---:|---:|---|---:|---:|
|1.8|-0.4|l1|0/3|0/3|
|1.8|-0.4|gated|0/3|0/3|
|1.8|-0.4|jumprelu_anthropic|0/3|0/3|
|1.8|-0.4|batchtopk|0/3|0/3|
|1.8|+0.0|l1|0/3|0/3|
|1.8|+0.0|gated|0/3|0/3|
|1.8|+0.0|jumprelu_anthropic|0/3|0/3|
|1.8|+0.0|batchtopk|0/3|0/3|
|1.8|+0.4|l1|0/3|0/3|
|1.8|+0.4|gated|0/3|0/3|
|1.8|+0.4|jumprelu_anthropic|0/3|0/3|
|1.8|+0.4|batchtopk|0/3|0/3|
|2.0|-0.4|l1|2/3|0/3|
|2.0|-0.4|gated|2/3|0/3|
|2.0|-0.4|jumprelu_anthropic|0/3|0/3|
|2.0|-0.4|batchtopk|3/3|1/3|
|2.0|+0.0|l1|2/3|0/3|
|2.0|+0.0|gated|1/3|0/3|
|2.0|+0.0|jumprelu_anthropic|0/3|0/3|
|2.0|+0.0|batchtopk|2/3|1/3|
|2.0|+0.4|l1|1/3|0/3|
|2.0|+0.4|gated|1/3|0/3|
|2.0|+0.4|jumprelu_anthropic|0/3|0/3|
|2.0|+0.4|batchtopk|2/3|1/3|
|2.2|-0.4|l1|1/3|0/3|
|2.2|-0.4|gated|1/3|1/3|
|2.2|-0.4|jumprelu_anthropic|0/3|0/3|
|2.2|-0.4|batchtopk|0/3|0/3|
|2.2|+0.0|l1|0/3|0/3|
|2.2|+0.0|gated|0/3|0/3|
|2.2|+0.0|jumprelu_anthropic|0/3|0/3|
|2.2|+0.0|batchtopk|1/3|0/3|
|2.2|+0.4|l1|0/3|0/3|
|2.2|+0.4|gated|0/3|0/3|
|2.2|+0.4|jumprelu_anthropic|0/3|0/3|
|2.2|+0.4|batchtopk|0/3|0/3|

## Scope

- Three paired worlds are screening replication units, not definitive significance.
- Primary comparisons require both target coverage and pair L0 distance <= .10.
- Calibration true-dictionary assisted oracle selection is a separate ceiling.
- Decoder and amplitude gains never refit on test; full final grid was preregistered.
- mixing_energy is signed-assignment leakage and can include pure sign/mapping mismatch; it is not alone evidence of multi-feature mixing. Interpret beside absolute cosine and decoder cosine matrices.
- VG beta/floor is available at saved evaluation checkpoints, not every training update.
- Official baseline total training-loss trajectory was not persisted; convergence is not established.
