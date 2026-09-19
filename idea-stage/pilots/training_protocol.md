# 학습 폭·추론 분해 파일럿 사전 계획

작성: 2026-09-19. 결과를 보기 전에 고정한 탐색용 계획이다.
관련 후보는 decoder replication, prior scaling, expected-to-hard decision gap이다.
이 세 질문 중 첫째는 별도 `replication_protocol.md`의 수치 실험으로 검사한다.
아래는 나머지 두 파일럿이며, 같은 학습 모델을 재사용한다.

## P2: 실제 학습에서 폭과 사전확률의 영향

- 기존 `VariationalGarroteSAE`, `fit_sae`, synthetic generator를 그대로 사용한다.
- d=16, 생성 feature=32, Bernoulli density=2/32, exponential amplitude,
  noise SD=0.05, 학습/보정/평가 표본=8192/2048/2048.
- 데이터 seed=20260919로 하나의 dictionary와 전체 표본을 만든 뒤 분할한다.
  학습 seed=0,1, 폭=32,64,128. 데이터 dictionary를 seed마다 바꾸지 않는다.
- 1200 AdamW updates, LR=0.003, batch=256, zero weight decay,
  decoder normalization, learned beta initialized 1, 기본 entropy/variance 유지.
- fixed-pi: pi=2/32, gamma=log(15). fixed-prior-count: pi=2/width.
  폭 32에서는 두 설정이 같으므로 한 번만 학습한다.
- 같은 폭/seed의 두 설정은 동일 초기화를 쓴다. gate bias=-2를 공통 유지한다.
  마지막 checkpoint를 사용하고 결과를 보고 학습 길이나 계수를 바꾸지 않는다.
- full test의 hard/expected EV, 실제 L0, expected count, KL, weighted variance,
  decoder 최대 양의 cosine, cosine>0.95 pair 비율, hardening distortion을 기록한다.
- fixed-pi에서 폭 증가와 함께 duplicate geometry 및 hard/expected gap이
  함께 늘고, fixed-count에서 모두 줄면 후속 연구를 지지하는 약한 신호로 본다.
  사전 기준: 두 seed에서 gap 감소 >=0.02 EV이고 expected EV 악화 <=0.02.
  이 기준은 확증적 유의성 기준이 아니다.
- 반대 결과는 이 설정에서 단순 prior scaling이 유효한 해결책이라는 가설에
  불리한 근거다. 수식적 replication 가능성을 반박하지는 않는다.
- 고정 pi와 고정 count는 prior strength도 달라진다. 효과가 있더라도
  실제 sparsity-matched gamma sweep과 충분한 학습 없이 원인을 확정하지 않는다.
- 데이터 seed 하나, optimizer seed 둘, 작은 폭/짧은 학습으로 실제 모델
  activation 및 Stage-2의 원인을 확정하지 않는다.

## P3: 학습된 dictionary에서 추론 규칙만 바꿨을 때

- P2 모델에서 default `1[m>0.5]*a`, 동일 support의 `m*a`, 동일한 표본별
  support 개수를 유지한 top-(m*a), fixed-K=2 top-m과 top-(m*a)를 비교한다.
- support 선택 효과와 선택 후 amplitude 효과를 분리한다.
- full test에서 EV와 Hungarian alignment 기반 support F1을 기록한다.
  alignment는 입력/label을 보지 않고 dictionary만 사용하며, unmatched learned
  atoms를 FP에 포함한다. 이 값은 실제 모델 activation의 semantic score가 아니다.
- 동일 표본별 L0에서 top-(m*a)가 default 대비 두 seed 모두 EV +0.02 이상이면
  decision gap의 회복 가능성을 지지한다. F1이 0.02 이상 하락하면 복원-회복
  상충으로 표시한다. fixed-K도 별도로 보고한다.
- topK/readout 자체는 신규 방법으로 주장하지 않는다. gain이 없어도
  단순 readout으로 해결되지 않는다는 정보를 얻는다. NNLS, pursuit, 독립적
  threshold calibration 및 baseline에 같은 후처리는 후속 필수 대조군이다.

## 자원·재현

두 seed를 A6000 0/1에 나눠 실행한다. 사전 추정 5–15분/seed,
총 0.17–0.5 GPU-hour, timeout 1시간/seed. 실제 시간을 JSON에 저장한다.
큰 checkpoint는 `outputs/idea_discovery_20260919/`에만 두고 Git에 넣지 않는다.
새 방법의 우월성이나 통계적 확증을 주장하지 않는다.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -B scripts/idea_discovery_training_pilot.py --seed 0 --device cuda:0
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -B scripts/idea_discovery_training_pilot.py --seed 1 --device cuda:1
```

## 결과 확인 뒤 추가한 탐색 분석

폭별 학습/단순 readout 결과를 확인한 뒤, near-prior gate의 집합 제거를
추가한다. 이 부분은 위 사전 계획에 포함됐던 확증 실험이 아니다.
near-prior는 `|gate_logit + gamma| <= 0.25`로 정하고, 같은 표본별 개수의
무작위 group, 작은 ma group 및 모든 m<=0.5 group을 비교한다.
제거된 decoded vector의 크기와 입력에 따른 변화량을 구분하고,
원시 제거와 학습 split에서 추정한 평균으로 치환하는 제거를 함께 기록한다.
이 분석도 복제 경로의 학습발생을 단독 증명하지 않는다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -B scripts/analyze_idea_discovery_pilots.py
```
