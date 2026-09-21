# P2 사전 프로토콜 — 정규화된 scalar prior 적응

2026-09-22. Training/test 결과를 보기 전에 작성했다. Jury S02를 구체화한다.
목표는 true L0나 support label 없이 prior를 학습해 유용한 feature 복원 operating point를
얻는지 검사하는 것이다. `pi=mean(m)` 정지조건을 correct L0 발견으로 해석하지 않는다.

## 모델과 자료

- 현재 VG의 d20/J5, full variance/entropy, minibatch-profiled beta. 새로운 scalar gamma만
  run-local subclass에 둔다. gamma*sum(m)+5*softplus(-gamma)의 정규화를 보존한다.
- World210/211/212, QR D seed=world. Train32768/cal8192/test16384의 sample RNG는
  각각10000/20000/30000+world, model init=world, batch stream=40000+world.
- p=.2/.4/.6; Gaussian hub correlation ±.4; max(0,Normal(1,.15)) magnitude.
  p/rho 조건 간 common random numbers, split 간 independent RNG. Copula correlation과
  실제 Bernoulli correlation은 구분한다. Width는 충분하며 true average L0는1/2/3이다.
- 새 density마다 새 데이터로 재학습한다. 'Transfer'는 고정 sparsity recipe의 전이이며
  source에서 학습한 weights를 target에 zero-shot 적용한다는 뜻이 아니다.

## 사전 고정 arms와 학습

- 각 p/rho/world에서 fixed gamma2, learned gamma 초기값−2/2/6의4 arms.
- Primary는 learned init2 대 fixed2. 나머지 initializations도 모두 평가하며
  가장 좋은 초기값을 골라 primary로 바꾸지 않는다.
- 총72 models. 현재 GPU2/3의 unrelated 작업 때문에 P2는 CPU2 threads를 우선 측정한다.
  CPU 실행 시 source p=.4 fixed2 6개도 같은 CPU에서 다시 학습한다(원래 reuse 가능했던
  66개 신규+6reuse 대신72개 신규). Hardware가 다른 fixed control과의 비교를 피하는
  실행상 변경이며 grid나 선택 기준을 늘리지 않는다.
- Adam lr.003, beta(.9,.999), wd0; gamma는 같은lr 별도 parameter group.
  3000 updates, batch256; 매 epoch shuffle without replacement, cycle하여32768 train재사용.
  Gradient global norm1, VG decoder unitnorm 및 radial gradient projection.
- Gamma를 update 뒤[−8,8]로 project하고 clip/boundary 횟수 기록. 명시한 safeguard이며
  gamma/clamp에 지배되는 결과는 자동 sparsity 성공으로 세지 않는다.
- Checkpoints completed updates500/1500/3000. Final만 primary; 중간은 trajectory.
  Core bias0, gatebias−2. Fresh test는 모든 primary-final 목록 동결 후에만 생성한다.

## 판정과 기록

두 target densities .2/.6와 두 correlation 부호 각각에서 2/3 worlds 이상이
fixed2 대비 signed-cosine+.05 또는 recovered-fraction+.20, F1+.03,
mixing-energy−.01을 함께 달성하면 후속 조사 가치가 있다. Source p=.4 cosine
악화가 .03보다 크면 넓은 전이를 주장하지 않는다. 초기값3개의 cosine 범위≤.05,
clamp boundary occupancy≤1%도 요구한다. n=3 screening이며 확증 통계가 아니다.

단순히 prior probability 또는 expected L0가 true 평균에 가깝다는 것은 성공이 아니다.
Actual hard L0/count MAE, per-feature geometry, F1, coefficient error, c_dec, beta,
mean/sample/hard reconstruction을 함께 보고한다. L0의 변화는 적응의 의도된 total effect다.
같은 실제 L0에서의 intrinsic mixing 개선은 이 grid로 보장하지 않는다.

## 예산·검증·실행

Numerical gates: 기존 fixed-gamma loss 및 theta gradient 일치, normalized prior γgradient의
self-consistency identity, finite-difference 검증. CPU100-update timing을 먼저 재고
72×3000 updates와 snapshot 비용에25%여유를 적용한다. CPU wall 상한2시간,
GPU 배정이 나중에 가능해져도 기존 CPU run을 결과에 따라재시도하지 않는다.
GPU0/2/3의 무관한 작업을 변경하거나 종료하지 않는다.

출력: `outputs/sbw_20260922/prior/`. 모든 실패/숫자/seed를 보존한다.
`training_results.json` 이후 최종 checkpoint manifest를 동결하고 test를 한 번 평가한다.

CPU timing 결과: 100 updates 최대 .5823초, 72×3000 및25%여유의 예상합계
1572.14초(약26.2분). World3개를각CPU2threads로병렬실행한다. GPU사용0.
실측소요시간은결과에따로기록하며예상을완료사실로사용하지않는다.
