# P2: 현재 VG-SAE의 beta 초기화·최적화 경로

2026-09-21. 결과 확인 전에 고정한 개발 파일럿이다. 연구 목표는 기존 VG-SAE
개발이며, 별도 진단 논문이나 새 architecture를 선정하는 실험이 아니다.

## 질문과 비교

현재 learned-beta 경로가 beta=1에서 시작하는 최적화 지연을 겪는지, train-only
moment 초기화가 같은 목적함수의 학습을 개선하는지 검사한다.

1. `learned`: 현재 full VG objective, beta 초기값1.
2. `learned_moment`: 같은 full objective; 동일 초기 encoder/decoder의 train 전체
   energy E로 beta0=d/(2*mean(E))를 계산하고 [1e-4,1e4]로 수치 제한.
   이후 log_beta를 평소처럼 학습한다. test/cal 통계는 초기화에 쓰지 않는다.
3. `profiled`: 기존 minibatch-profiled 구현의 별도 reference.
   E_batch log(mean E_batch)는 global learned-beta objective를 전체 데이터에서
   profile한 것과 일반적으로 같지 않다. 세 arm 전체를 동일 objective라고 부르지 않는다.

## 고정 설정

- 기존 `VariationalGarroteSAE`, `fit_sae`를 그대로 사용한다. 새 모듈/학습 loss 없음.
- d16, true/learned width64, mean true L0=4, independent Bernoulli supports,
  noise SD.05, frequency_skew0. Exponential 및 RMS-matched constant amplitude.
- data seeds20261021/22/23, optimizer seeds0/1/2. 같은 seed/amp의 세 recipe와
  gamma2/6은 동일 train/cal/test와 같은 network 초기화 및 batch 순서를 사용한다.
- seed마다 한 번 생성한 같은 dictionary의 train2048/cal2048/test4096 분할.
- gamma2/6, LR.003, batch256, weight decay0, max6000updates,
  decoder normalization, entropy weight1, variance on, native threshold>.5.
- pre-bias를 모든 arm에서 동일하게 train mean으로 초기화한다.
- `fit_sae`의 zero-based logging 규칙 때문에 early는1001updates,
  final은6000updates다. 실제 update 수를 기록한다. 중간 model checkpoint 보존.
- 전체36models. 기존 pilot 기준 model당 약30초로 총.3GPUh 추정,
  import/evaluation 포함 상한1GPUh로 관리한다. seed별 GPU0/1/3을 사용한다.
  한 process2시간 timeout. 실제 시간을 결과에 기록한다.

## 선택·판정과 한계

먼저 모든 cal 결과를 저장한다. 동일 amplitude/gamma별로3개 seed의 final cal
hard latent relative error 평균이 가장 작은 recipe를 선택하고 JSON을 동결한다.
동률은 learned → learned_moment → profiled 순서다. test는 그 뒤에만 평가한다.
모든 predefined arm의 test 결과를 보고하되 test winner로 recipe를 바꾸지 않는다.

세 risk(mean/sample/hard), hard L0, aligned support F1, signed coefficient relative
error, decoder recovery, beta/beta_train_stationary와 full objective history를 기록한다.
기본 gamma와 amplitude별 결과를 유지하고 가장 좋은 점만 골라 보여주지 않는다.

- 초기화 개선 신호: learned_moment가 learned보다 early cal coefficient error를
  평균.02 이상 줄이고, final test error 악화가.01 이하인 같은 gamma/amp 조건.
  실제 hard L0가20% 넘게 다르면 sparsity 변화가 섞인 신호로 구분한다.
- final에서도 안정적 개선이 있으면 후속 full-run 학습 recipe 후보로 채택한다.
  최종 차이가 없어지면 학습 속도에 관한 선택이며 새 statistical method 이득이 아니다.
- 한 gamma에서만 신호가 있거나 마지막 objective/beta가 계속 변하면 조건부/미수렴으로
  기록한다. 6000updates가 전역 최적성이나 충분한 학습을 보장하지 않는다.
- profiled 비교는 구현 절차 선택을 돕는 reference다. warmup, 새로운 posterior,
  baseline superiority, low-data/noise 전반의 강점을 이 파일럿으로 주장하지 않는다.

## 실행

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -B scripts/run_vg_development_training.py --seed 0 --device cuda:0
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -B scripts/run_vg_development_training.py --seed 1 --device cuda:0
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -B scripts/run_vg_development_training.py --seed 2 --device cuda:0
```

위 명령은 calibration과 checkpoint까지만 생성한다. 세 seed가 끝난 뒤 별도
selection JSON을 작성한 후 `--evaluate-frozen-selection <path>`로 test를 평가한다.
큰 checkpoint는 ignored outputs에만 두고 작은 수치·설정·선택 기록을 공유한다.
