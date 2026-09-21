# VG-SAE: L1 대안에서 실제 feature 복원으로

2026-09-22. Initial anchored proposal. P2/P3 완료, P1 직접 baseline 비교 진행 중.
주방법의 선택은 사용자가 시작한 VG-SAE 개발이며 pilot이 없는 대체 주제를 고르지 않는다.

## Immutable Problem Anchor

Sparse but Wrong가 제기한 잘못된 희소성 수준과 feature 혼합 문제를 출발점으로,
L1 진폭 벌점에 의존하지 않는 Variational Garrote 기반 새로운 SAE를 개발한다.
확률적 support 선택과 amplitude 추정이 실제 feature 복원을 개선하고, 올바른 활성 수를
모르는 상황에서 희소성 설정에 따른 오류를 줄이는지 검증한다. 주 연구는 VG-SAE 방법
개발이며, 재구성–희소성 곡선·decoder 진단은 성공의 대리 목표가 아니라 이를 검증하는 도구다.

## 무엇을 만드는가

1. 입력 activation에서 feature가 쓰일 확률 m과 nonnegative amplitude a를 각각 예측한다.
2. Binary support의 quadratic reconstruction risk를 정확히 평균한 VG objective와
   normalized support prior/entropy로 encoder와 linear dictionary를 함께 학습한다.
3. 학습 뒤 명시한 hard support×amplitude로 코드를 만들고, synthetic truth에서는
   실제 feature 방향·혼합·발화 패턴을 평가한다. LLM에서는 sparse probing과 downstream
   동작을 제한된 외적 검증으로 쓰며 ground truth가 있다고 주장하지 않는다.

방법의 중심 가설은 **L1 진폭 축소에 의존하지 않는 VG의 support 선택이 실제
feature recovery를 개선하는 SAE 학습 원리가 될 수 있다**는 것이다.
Sparse but Wrong의 core warning을 성공 기준에 반영한다. 낮은 MSE와 적은 L0만으로
성공을 선언하지 않고, 적절한 활성 수가 알려져 있지 않을 때의 실패를 함께 드러낸다.

## 논문에서 받은 문제와 우리가 검증할 delta

참조 논문은 잘못된 L0의 feature mixing과 reconstruction 중심 평가의 한계를 보인다.
주 비교는 BatchTopK/JumpReLU다. L1 amplitude shrinkage의 직접 근거는 Gated SAE
등에서 따로 인용한다. 이미 존재하는 selection/magnitude 분리, analytic VI,
spike-and-slab 또는 adaptive k 자체를 새로운 성과로 주장하지 않는다.

VG의 구체적 조합은 Bernoulli support-only variational risk, 입력 의존 point amplitude,
learned linear dictionary와 normalized prior/entropy다. Variance와 prior가 있는
확률적 학습 원리가 실제 dictionary/support 품질에 어떤 영향을 주는지가 delta의 실증 부분이다.
AFA/SoftSAE/AdaptiveK/MDL/iSAE도 직접 경쟁이며, 각각의 budget·orthogonality·stability
조건이 true features를 보증하는 것은 아니듯 VG의 gamma도 정답 활성 수를 보장하지 않는다.

## 정확한 모델과 학습 의미

y=x−b, m=sigmoid(W_g y+c_g), a=softplus(W_a y+c_a), h=m*a,
x_hat=b+D h, ||d_j||=1을 기본 구현으로 둔다.

`E=.5||y−D(m*a)||²+.5 sum_j m_j(1−m_j)a_j²||d_j||²`.
`pi=sigmoid(−gamma)`, unit entropy이면 prior−entropy가 Bernoulli KL이다.
Learned global beta의 objective는 `beta E−d/2 log(beta/(2pi_math))+KL`이며,
minibatch profiling은 `d/2 log(2 mean(E)/d)+KL` 형태다. 둘은 서로 다른 stochastic
학습 절차다. P3의 fixed beta10은 이 차이를 제거한 작은 controlled mechanism 실험이다.

Amplitude는 point estimate이고 별도 normalized prior/entropy가 없다. 따라서
full generative-model ELBO, calibrated semantic posterior, unbiased finite encoder,
noise variance의 정확한 추정을 이 수식으로 주장하지 않는다.

고정 a,D,b,beta에서 gate의 좌표 stationary equation은
`m_j=sigmoid(beta*a_j*d_j^T r_minus_j−.5*beta*a_j²||d_j||²−gamma)`다.
이 식은 support에 대한 조건부 evidence comparison을 설명하지만, feed-forward gate가
그 최적점에 도달하거나 learned dictionary가 정답이라는 보장은 아니다.
Unit-norm D에서 고정 codes의 variance gradient는 radial이므로 직접적인
off-diagonal mixing 억제항이라고 부르지 않는다. Gate/amplitude를 통한 간접 효과를 검증한다.

## 이번 개발 선택과 complexity

기본 feed-forward VG를 중심으로 비교한다. P2 scalar prior와 P3 joint posterior는
주방법의 필요한 수정을 찾는 판별 실험이며 결과에 따라 무조건 결합할 component가 아니다.
어떤 reviewer 점수를 올리기 위해 head/teacher/prior를 추가하지 않는다.

P2는 normalized scalar gamma 하나만 학습해 true L0 없이 조정할 수 있는지 검사했다.
P3는 g=5에서32 support states를 정확히 계산하고 최적화한 product posterior와 비교했다.
이는 exponential cost를 가진 기전 실험이지 32k-latent에 배포 가능한 새 SAE가 아니다.
대규모 block posterior 또는 distillation은 이득과 비용이 실제 확인됐을 때의 후속 선택이다.
LLM teacher, RL, 새 backbone은 필요하지 않다. LLM은 activation과 evaluation task를 제공한다.

## 현재 근거와 해석

### P1 — 직접 비교: 아직 실행 중

Paper-inspired d20/true5/SAE5, Gaussian-copula hub−.4/0/+.4, p=.4, 3worlds,
VG/L1/Gated/Anthropic-tanh JumpReLU/BatchTopK 각5controls의225fits다.
각3000updates와500/1500/3000checkpoints, 독립cal/test를 사용한다.
Signed Hungarian recovery, mixing, per-feature support를 우선하고 MSE/L0/c_dec는
각각 별도로 읽는다. Native code와 L1 calibration-only amplitude rescale control을 구분한다.
같은 width/data/HPO/updates지만 trainable parameter counts는 VG330, L1225,
Gated235, Jump230, BTK225로 달라 C1 차이를 VG objective만의 인과 효과로 부르지 않는다.

기존09-21의 독립-support one-world 결과는 이번 correlated-feature 주장의 증거가 아니다.
P1 완료 뒤 모든 seed·coverage·negative 결과를 여기에 통합하며 test를 보고 grid를 바꾸지 않는다.

### P2 — scalar-prior adaptation: 지지되지 않음

72models를 CPU에서3worlds×p(.2/.4/.6)×rho(−.4/+.4)×fixed2/learned-init(−2/2/6)로
학습했다. Actual labels 없이 gamma를 update하며 전체 모델의final checkpoint를 동결한
뒤 독립 test를 평가했다. Primary learned-init2가 fixed2 대비 사전 joint criterion을
통과한 paired cell은0/18, 초기값별 signed-cosine 범위≤.05인 cell도1/18이다.
단순 scalar-prior 학습을 기본 개선안으로 채택하지 않는다.

예를 들어 p=.2,rho+.4의3world 평균에서 fixed2 cosine .99998/F1 .99999/L0 .99563이고,
learned-init2는 .99941/.83398/1.49744다. p=.4,rho−.4에서는 두 recipe 모두
hard L0=5, EV>.999이지만 signed cosine은 약.78이고 F1 .57078이다.
Reconstruction이 좋다는 것이 올바른 feature/support를 뜻하지 않는다는 직접 관찰이다.

Gamma의 stationary equation `sum mean(m)−J*pi=0`는 자기 일관성이다. 이것이
true density를 식별한다는 결론을 내리지 않는다. Target density마다 재학습한 recipe
transfer이며 zero-shot weight transfer가 아니다. Target-density fixed-gamma curve가
없으므로 intrinsic matched-L0 mixing 개선을 이 파일럿으로 판정하지 않는다.

### P3 — joint posterior: density-matched 기전 귀속은 미해결

Fixed beta10, gamma0/4/8,3worlds×rho±.4에서 amortized MF / optimized product MF /
exact32의54models를 CPU에서 학습했다. 선택된 MF는3starts, cyclic coordinate50sweeps,
residual검사로 얻었다. 비수렴 sample도 제거하지 않고 fixed-q partial gradient로 유지했다.
수렴 시에만 envelope interpretation을 쓴다. Runtime과solverfailure를 모두 기록한다.

Current summary: matched-L0 exact-vs-MF pairs는4/18 target/world/sign이며 사전 joint
success는없다. 한 positive-correlation world에서 exact가 dictionary를 크게 개선했지만
native L0가 MF와달라 covariance-specific benefit으로 귀속할 수 없다. 다른5 gamma0
world의 dictionary 차이는매우작다. MAP와 marginal readout이support점수를바꾸는사례도
있으므로 읽기 방식과학습된dictionary의효과를 구분한다.

## 두 가지 최종 검증할 주장

- **C1, 아직 검증 필요:** VG-SAE가 correlated/anticorrelated features에서 L1 대안으로
  유용한 signed dictionary 및 support recovery를 얻으며, 그 이득의 범위·fidelity·비용을
  강한 Gated/AnthropicJump/BatchTopK와 비교해 설명할 수 있다.
- **C2, 아직 검증 필요:** 관찰된 VG 이득 중 일부가 단순 amplitude rescale 또는 generic
  gate 분리로 설명되지 않는 VG support-risk 학습의 역할이다. 같은 architecture의
  적절한 대조와 finite precision/mean-field/amortization 구분이 필요하다.

Wrong-L0 전반의 견고성, 자동 true-L0 발견, posterior covariance의 고유 이득은
추가 조건부 가설이다. 파일럿 결과가 없거나 coverage가 실패한 주장을 C1/C2의
확인된 supporting contribution으로 합치지 않는다.

## 최소 후속 실험 스케치

1. **참조 문제에 대한 method evidence:** P1의 coverage와 convergence를 확인하고,
   충분한 near-correct operating region과 실제 low/high region을 독립 cal에서 확보한
   뒤 새 holdout에 고정한다. Paper-compatible50feature setting 및 strong baseline
   recipe를 작은 수로 확인한다. Oracle truth selection과 deployment selector를 분리한다.
2. **같은 VG architecture의 최소 mechanism:** P1/P3가 드러낸 특정 병목 하나를 고른다.
   Profiled/finite/global precision이나 amplitude/gate amortization을 동시에 복잡하게
   결합하지 않는다. 같은samples·updates·initialization에서 기여를 판별한다.
   No-variance deletion은 m/a compensation을 허용하므로 collapse를 확률 모델의 우위로 쓰지 않는다.
3. **현실적 source에서 feature usefulness:** 먼저 저장된 SynthSAEBench와 Gemma의
   checkpoint를 fresh data에서 평가한다. c_dec는단독성공기준으로쓰지않고 sparse probing,
   reconstruction/CE/KL와사용가능한task검증을함께보고한다. Real semantic truth 주장은하지않는다.

상세 run order/budgets는 P1과전체 비판 검토 뒤 동반 EXPERIMENT_PLAN에서 고정한다.
이전486-run plan을 새 연구 질문의실행계획으로그대로돌리지않는다.

## 실패와 남은 불확실성

Toy5features의 좋은 결과는 LLM monosemanticity를 보장하지 않는다. 나쁜짧은pilot이
VG원리를 전부반박하지도않는다. Wrong-L0 matching이나baseline수렴이부족하면
유보하고 최소 후속으로확인한다. 실제효과가없으면 sameVG개발안에서학습원리와표현을
수정하며, 주제를별도의진단논문으로대체하지않는다.
