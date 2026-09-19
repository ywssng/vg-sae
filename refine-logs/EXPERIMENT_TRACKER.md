# Experiment Tracker

2026-09-19. DONE은 실제 실행된 작업만 뜻한다. R001 이후는 계획이다.

| Run ID | Purpose | System / Variant | Split | Metrics | Priority | Status | Evidence |
|---|---|---|---|---|---|---|---|
| P001 | 알려진 replication 구성의 구현 확인 | gamma2 / prior-count1, r4/16/64/256, learned/profiled | 단일 constructed input | energy, KL, mean/hard error | MUST | DONE | idea-stage/evidence/pilots/replication_results.json |
| P002 | width/prior 및 readout screening |2seeds, widths32/64/128,1200steps,10models |8192 train /2048cal unused/2048test | three risks, L0, F1, variance | MUST | DONE: simple fixes unsupported | idea-stage/evidence/pilots/training_seed0.json 및 seed1 |
| P003 | near-prior group 원인분해 |10models, random/small-ma controls | train mean replacement/test outcome | raw vs mean-preserved deletion | SUPPORT | DONE: exploratory | idea-stage/evidence/pilots/exploratory_group_ablation.csv |
| P004 | 학습 예산 영향 |2seeds,width32/128,6000steps,6models | P002와 같은 분할 | risk gap, beta/loss change | MUST | DONE: gap reduced | idea-stage/evidence/pilots/convergence_seed0.json 및 seed1 |
| P005 | B05 oracle posterior | exact/amortized/refined, orthogonal/coherent095 |8192train/2048test | Brier, marginal NLL, joint KL | BACKUP | DONE: mixed scores, undertraining | idea-stage/evidence/pilots/posterior_results.json |
| V001 | 새 pilot 회귀 검사 |3test modules | deterministic |28 tests | MUST | DONE:28 PASS | idea-stage/evidence/validation.json |
| R001 | clone/지표 구현 검사 |ideal/feasible, grouped AP, symmetry | tiny deterministic | loss/metric invariants | MUST | TODO | EXPERIMENT_PLAN B1/B2 |
| R002 | C1 screening |3data×3train sources, cal matched pair | new train/cal/test |ΔF,S,drift,prevalence | MUST | TODO | EXPERIMENT_PLAN B1 |
| R003 | C2 continuation |high/low clone×variance on/off | same new splits |grouped AP interaction | IF R002 GO | NOT STARTED | EXPERIMENT_PLAN B2 |
| R004 | 외적 반복 |pinned SynthSAEBench |newcal64k/test64k |C1/C2,coverage | IF R003 GO | NOT STARTED | EXPERIMENT_PLAN B3 |

현재 지지되는 것은 구성의 수치 일치와 제한된 탐색 관측이다.
실제 learned replication preference(C1), 인과적 feature 손상(C2), baseline 우월성은
아직 결과가 없다. calibration 전체 개선도 P005로 주장할 수 없다.
