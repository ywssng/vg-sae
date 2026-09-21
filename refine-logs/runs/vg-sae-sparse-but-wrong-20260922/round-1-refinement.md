# VG-SAE: L1 대안에서 실제 feature 복원으로

2026-09-22. Full revision 1. 세 파일럿 완료; 현재 근거와 다음 개발 대조를 통합했다.
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

### P1 — 직접 비교: 방법의 가능성, 우위와 coverage는 구분

Paper-inspired d20/true5/SAE5, copula rho−.4/0/+.4, p=.4, 3 worlds에서
VG/L1/Gated/Anthropic-tanh JumpReLU/BatchTopK 각5 controls, 총225 fits를 완료했다.
3000 updates, checkpoints500/1500/3000의675개 cal 후보와254개 고유 test checkpoints를
평가했다. 전체225 final grid가 포함되고 선택은 test 전에 동결했다. 모든 fit이 완료됐다.

사전 primary target L0=1.8에서는 VG coverage가0/9여서 비교는 미해결이다.
이를 robustness의 증거나 방법 자체의 실패로 바꾸지 않는다. Target2에서는7/9의
covered worlds에서 VG의 signed dictionary cosine과 support F1가 거의1이었다.
L1/Gated도 좋은 covered points가 있어, 어느 strong comparator에 대해서도
각 상관 부호의2/3 worlds에 요구한 joint 우위 기준을 만족하지 못했다.

Anthropic JumpReLU는 세 target 모두에서 coverage가0이었다. 가장 작은 coefficient .03도
평균 native L0가 약.67–1.00이고 더 큰 penalty는 더 sparse했다. 이는 제대로 조율된
JumpReLU를 VG가 이겼다는 결과가 아니다. 짧은 budget·normalization·warmup·coefficient
range를 포함한 실제 recipe의 비교 범위가 부족한 것이다. Convergence도 확증되지 않았다.

Target2에서 L0가 맞은5개 VG/L1 pairs 모두 calibration-only L1 amplitude rescaling이
계수 NMSE와 MSE를 줄였고 support/dictionary는 같았다. 이 control 없이 amplitude bias
차이를 VG 고유의 feature-demixing 효과로 쓰지 않는다. 예: rho−.4/world211의 L1
coefficient NMSE .06972→.01328. NMSE와 square-root relative error를 혼동하지 않는다.

Gamma2 VG는 rho−.4/0/+.4에서 평균 native L0가5/5/4.798인데 EV는 .999 이상이었다.
Truth-free min-c_dec도 rho+.4/world212에서 gamma2를 골라 cosine .99971,
L0 4.39465/F1 .62633을 냈다. 좋은 decoder만으로 support 결정을 인증할 수 없다.

Width/data/HPO/updates는 맞췄지만 parameter counts는 VG330, L1225, Gated235,
Jump230, BTK225이며 normalization/parameterization도 다르다. Recipe 수준 비교이고
VG objective만의 인과 효과나 같은 capacity의 승리라고 부르지 않는다.

### P2 — scalar-prior adaptation: 지지되지 않음

72models를 CPU에서3worlds×p(.2/.4/.6)×rho(−.4/+.4)×fixed2/learned-init(−2/2/6)로
학습했다. Actual labels 없이 gamma를 update하며 전체 모델의final checkpoint를 동결한
뒤 독립 test를 평가했다. Primary learned-init2가 fixed2 대비 사전 joint criterion을
통과한 paired cell은0/18, 초기값별 signed-cosine 범위≤.05인 cell도1/18이다.
단순 scalar-prior 학습을 기본 개선안으로 채택하지 않는다. Full-train 추가 진단에서
primary18 cells 중14개는 최종 |dF/dgamma|>2였다. 따라서 이번 결과는 유한 예산의
gradient recipe 비지지이며, 잘못된 empirical-Bayes 정상점으로 수렴한 증거가 아니다.
Optimizer coupling, duration, precision policy는 원인이 확인되지 않은 다음 질문이다.

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
native L0가 MF와 달라 density 차이와 분리된 covariance-specific benefit으로 확정할 수 없다.
그러나 같은 gamma/beta/초기화의 전체 효과는 실제 신호이며 L0 불일치가 이를 없애지는 않는다.
다른5 gamma0 world의 dictionary 차이는 매우 작다. MAP와 marginal readout이support점수를바꾸는사례도
있으므로 읽기 방식과학습된dictionary의효과를 구분한다.

## 부호·방향·support를 구분한 해석

Frozen `mixing_energy`는 signed Hungarian 할당 뒤의 leakage다. 순수 부호나 mapping
오류도 포함하므로 모든 감소를 다중 feature의 기하학적 분리로 표현하지 않는다.
P3의 rho−.4/world210 amortized model은 signed cosine .60075지만 absolute cosine
.99966, sign-invariant multicomponent energy .00049다. 주 실패는 이 경우 방향의 부호다.
반면 rho+.4/world212의 MF→exact는 absolute cosine .96034→.99959와
multicomponent energy .07763→.00081도 좋아져 실제 혼합과 부호 모두의 변화가 있다.
두 분석은 모든 frozen model에 적용한 사후 설명용 분해이며 성공 기준을 바꾸지 않는다.

## 선택한 다음 개발 대조: precision을 명시적으로 통제

추가 head나 joint module을 먼저 채택하지 않는다. 같은 VG 구조의 fixed/global-learned/
minibatch-profiled precision을 분리해, 재구성 압력이 support 선택을 지배하는 조건을
검증하는 것이 다음 최소 개발 단계다. Fixed finite beta는 **검증할 후보**이며
현재 public default를 이미 바꿨거나 성능 이득을 확인했다고 표현하지 않는다.

이 선택에는 별도의 constructed witness가 있다. Orthonormal D, x=Dz, z≥0에서
`b_t=−t D1`, `W_a=D^T`, `c_a=0`, `W_g=0`, `c_g=(4t+4)1`로 두면
`a_t=softplus(z+t)`, `m_t=sigmoid(4t+4)`다. Expected risk는0으로 가고 hard L0는
항상J지만 decoder와 c_dec는 정답 그대로다. 고정 gamma의 KL은 유한한 값으로 간다.
따라서 epsilon 없는 ideal profiled/global-optimized-precision objective에는 이 family를
따라 유한 하한이 없다. 실제 코드의 epsilon은 이 발산을 제한한다. 이를 생략해 구현이
그대로 무한 발산한다고 쓰지 않는다. 숫자와 production-epsilon tests를 보존했다.

Fixed finite beta에서는 E≥0와 KL≥0로 Gaussian 상수의 유한 하한이 있다.
이는 해당 escape를 막는 수식 성질이며 feature/support recovery의 보장이 아니다.
이 family가 실제 SGD의 원인이었다고 확인한 것도 아니다. 단지 같은 구조에서
precision policy를 분리할 구체적인 근거이며 독창적인 정리로 내세우지 않는다.
자세한 가정과 수치는 newrun의 `evidence/dense_offset_witness.md/json`에 있다.

## 두 가지 최종 검증할 주장

- **C1, 아직 검증 필요:** VG-SAE가 correlated/anticorrelated features에서 L1 대안으로
  유용한 signed dictionary 및 support recovery를 얻으며, 그 이득의 범위·fidelity·비용을
  강한 Gated/AnthropicJump/BatchTopK와 비교해 설명할 수 있다.
- **C2, 아직 검증 필요:** 같은 VG architecture에서 precision 정책을 명시적으로
  통제하면, near-perfect reconstruction과 잘못된 support가 공존하는 영역을 줄이고
  실제 feature/support 복원이 좋은 operating region을 더 안정적으로 확보할 수 있다.
  고정 finite beta의 수학적 하한과 이런 경험적 개선은 별개의 주장이다.
  L1·Gated 등보다 좋은 성능이 VG의 추가 encoder capacity와 무관하다고 주장하지 않는다.

Wrong-L0 전반의 견고성, 자동 true-L0 발견, posterior covariance의 고유 이득은
추가 조건부 가설이다. 파일럿 결과가 없거나 coverage가 실패한 주장을 C1/C2의
확인된 supporting contribution으로 합치지 않는다.

## 최소 후속 실험 스케치

1. **비교 가능 구간과 선택 규칙부터 확보:** development-only calibration에서 native L0
   범위와 학습 길이를 확인한다. 특히 JumpReLU의 더 낮은 penalty, VG/BTK의 실제 low/correct/
   high 영역을 좁게 찾은 뒤 새로운 main worlds/holdout에 고정한다. Oracle 선택과 실제
   사용 가능한 selector를 따로 보고한다. 유효 grid 안의 성공은 전체 density robustness가 아니다.
2. **동일 구조의 precision 대조 하나:** fixed beta10 / global learned beta(init10) /
   minibatch-profiled beta를 같은 initialization·samples·updates·gamma와 hard readout으로
   비교한다. Gamma는 고정해 P2의 미수렴 prior 문제를 동시에 섞지 않는다. Bias와 amplitude
   offset, native/expected support, objective decomposition을 관측하고 원인을 미리 확정하지 않는다.
   같은 gamma의 전체 효과와 별도의 L0-controlled view를 함께 보고한다.
3. **방법을 고정한 뒤 범위 검증:** paper-compatible50-feature correlated setting으로
   한 번 확장하고, 기존 SynthSAEBench/Gemma 자료로 평가 경로를 먼저 확인한다. 새 방법의
   효과는 해당 새 recipe로 실제 학습한 모델에서 검증한다. LLM에서는 c_dec만이 아닌 sparse
   probing/task utility와 CE/KL·활성 수를 함께 보고한다. 완전한 semantic ground truth는 없다.

계산량과 첫 실행 순서는 동반 실험 계획에서 제한한다. 이 단계에서는 새 큰 sweep을
추가하지 않았고 이전486-run plan을 그대로 실행하지 않는다. Simple learned prior와
exact32를 자동 채택하지 않으며, 원래 VG-SAE를 새 진단 논문으로 바꾸지도 않는다.

## 실패와 남은 불확실성

Toy5features의 좋은 결과는 LLM monosemanticity를 보장하지 않는다. 나쁜짧은pilot이
VG원리를 전부반박하지도않는다. Wrong-L0 matching이나baseline수렴이부족하면
유보하고 최소 후속으로확인한다. 실제효과가없으면 sameVG개발안에서학습원리와표현을
수정하며, 주제를별도의진단논문으로대체하지않는다.
