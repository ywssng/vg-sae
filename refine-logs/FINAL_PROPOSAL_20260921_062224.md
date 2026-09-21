# VG-SAE: 진행 중인 방법 개발과 검증 파이프라인

2026-09-21. 기존 VG-SAE가 주방법이며, 아래 개선 선택지는 같은 연구 안에 있다.
현재 방법의 수식과 파일럿을 바탕으로 한 후속 검증 제안이다. C1/C2 확증은 아직 남아 있다.

## Immutable Problem Anchor

현재 구현된 Variational Garrote 기반 SAE를 출발점으로, VG의 확률적 support
선택과 amplitude 추정이 SAE 학습에 제공하는 이점을 명료화하고 필요한 방법·학습·
추론·평가 절차를 개선한다. 합성 ground truth, 현실적인 synthetic benchmark,
실제 모델 activation의 단계적 검증으로 새로운 SAE 방법으로서의 기여를 평가한다.
주 연구 목표는 VG-SAE 개발이며, 진단 실험은 그 방법을 개선하고 검증하기 위한 도구다.

## 방법을 먼저 설명하면

1. 입력에서 support 확률 m과 nonnegative amplitude a를 예측하고, learned linear
   dictionary D로 복원한다. 현재의 두 affine encoder와 unit-norm decoder를 기준으로 둔다.
2. Bernoulli support의 기대 재구성 손실을 표본 추출 없이 정확히 계산하고,
   normalized Bernoulli prior/entropy 비용과 함께 학습한다. noise precision을
   학습하는 경로와 minibatch에서 profile하는 기존 경로를 구분해 기록한다.
3. 실제 사용 code는 hard support와 amplitude로 만든다. calibration에서 정한
   operating point를 test에 그대로 적용하며 coefficient/feature 복구와 입력
   fidelity를 함께 평가한다. 이 결과로 학습·추론 변경의 필요성을 결정한다.

주 기여 후보는 **원래 VG의 support 선택–진폭 추정 결합을 amortized SAE로 구현해,
제한된 활성 예산에서 유용한 sparse feature/code recovery를 얻는 방법**이다.
generic 확률적 SAE 최초성이나 새로운 clone 정리를 주 기여로 내세우지 않는다.

## 정확한 방법과 해석 범위

중심화된 입력 y=x−b에서 m=sigmoid(W_g y+c_g), a=softplus(W_a y+c_a)이고
q(s|x,a(x),D)=product Bern(m_j)다. amplitude는 point estimate다.

E=.5||y−D(ma)||² + .5 sum_j m_j(1−m_j)a_j²||D_j||².

pi0=sigmoid(−gamma), entropy coefficient1에서 full loss는
beta E −d/2 log(beta/(2pi_math)) + sum_j KL(Bern(m_j)||Bern(pi0))다.
이 **support expectation**의 학습에는 latent MC나 threshold STE가 필요하지 않다.
이는 변분 sparse coding 전체에서 최초인 성질이라고 주장하지 않는다.

learned mode는 global log_beta를 학습한다. profiled mode는 각 minibatch의
beta*=B*d/(2 sum E)를 사용한다. minibatch log-energy 평균은 full-data profile
log-energy와 일반적으로 다르므로 두 경로를 완전히 같은 최적화 문제로 부르지 않는다.
현재 두 모드의 public 의미를 보존하고 실험의 mode와 tuning 예산을 명시한다.

a(x)의 normalized stochastic prior/entropy는 없으므로 완전한 생성모델 ELBO,
amplitude posterior, calibrated semantic probability를 이 수식만으로 주장하지 않는다.
주로 **conditional support variational objective를 사용하는 SAE**로 기술한다.
free per-example amplitude의 좌표 최적점에 additive L1 shrink가 없는 것과,
실제 한 번의 affine-softplus encoder가 bias-free인 것은 다른 주장이다.

## 지금까지의 근거

- P1:기존546checkpoint를 새calibration에서 선택하고 새test에서 평가했다.
  Exponential/L0cap8의 VG hard latent error .69878, F1 .32709, input EV .47784,
  실제L0 3.002. 비교 TopK k6은 error .73694, F1 .29710, EV .51410, L0 5.879.
  즉 coefficient/selection에서 유리한 신호와 입력 fidelity의 손실이 함께 있다.
  같은 L0 상한 비교이며 exact-L0 비교·새training-seed 반복·SOTA가 아니다.
  Constant도 낮은 cap에서는 VG가 좋지만 cap16에서는 L1이 더 낮은 error를 냈다.
- P2:3개의 paired seed,2gamma,2amplitude,3beta recipe로36개 작은 모델을 학습했다.
  train-energy beta 초기화는 gamma/seed에 따라 개선·악화가 섞였다.
  새 기본 초기값으로 채택하지 않는다. 이 실험은 LR decay나 gamma warmup을 시험하지 않았다.
  더 중요한 신호는 learned/gamma2에서 1,001→6,000 updates로 EV가 오르는 동안
  exponential의 hard latent error가 .90063→1.12136, F1이 .28779→.21238로
  악화됐다는 것이다. Constant에서도 같은 방향이었다. 작은 조건의 관찰이며
  순수 과적합이나 full-size의 보편적 실패로 단정하지 않는다. 다만 reconstruction
  plateau를 primary recovery의 checkpoint 선택 기준으로 쓰지 않는다.
- P3:고정된 VG checkpoint의 conditional gate sweep은 같은 입력별 L0에서
  coefficient error를 약5% 줄였지만 현재 순차 구현은 약992배 느렸다.
  support-fixed NNLS가 큰 개선을 주고, NNLS 이후 gate correction의 추가 coefficient
  이득은 cal/test에서 방향이 달랐다. 순차 solver나 teacher를 바로 채택하지 않는다.
- P3b: 같은 선택지의 병렬 후속은 새 cal/test 각 512개에서 완료했다.
  Calibration으로 고른 eta=.5는 error .706038→.678962, F1 .338201→.347255,
  EV .444171→.502693, 같은 actual L0 3.25를 얻었다. 그러나 native 대비
  latency 4.3687배로 사전 3배 기준에 못 미쳤다. 공통 amplitude PG는 더 빠르고
  error도 .675913으로 낮다. Gate 보정에는 EV/F1 측면의 다른 절충이 남는다.
  P3 뒤에 추가한 탐색이며 기본 추론이나 teacher로 채택하지 않는다.

수치·설정·원본은 저장소의 `idea-stage/runs/vg-sae-development-20260921/evidence/pilots/`에 있다.
2026-09-19의 별도 진단 주제 선택은 이 연구의 목표나 기여로 사용하지 않는다.

## 기존 방법과 구별할 지점

Gated의 selection/magnitude 분리, JumpReLU/TopK/BatchTopK의 hard sparse code,
VSC/S3C/entropy-ELBO/VAEase의 변분 sparse coding은 직접 선행이다.
현재의 구체적인 차이는 Bernoulli support만 변분화하고 입력별 amplitude는 점 추정하여
선형 dictionary에서 기대 reconstruction risk를 정확히 계산하는 amortized 구성이다.
이 조합의 이름이나 요소 나열만으로 충분한 기여가 되지는 않는다.

신규성의 실질적 근거는 **현재 조합이 어떤 예산·데이터 조건에서 더 유용한 code를
학습하는지**, 그리고 **그 차이가 objective의 support-risk 항에서 오는지**다.
기존 학습 토큰·매개변수·추론 시간·tuning 횟수를 함께 공개한다.
VAEase 등의 실험을 직접 재현하지 않았다면 그 계열보다 우수하다는 표현은 쓰지 않는다.

## 개발 결정과 복잡도 예산

즉시 적용할 것은 독립 cal/test 선택, mean/sample/hard 지표 분리, 원래 VG
objective의 정확한 기술과 단계별 검증이다. 기본 두-head/linear-D 구조와
unit entropy/full variance를 유지한다. P2가 일관된 개선을 보이지 않았으므로
beta 초기값을 자동 변경하지 않는다. P3의 느린 solver도 기본 추론에 넣지 않는다.

새 trainable module 수를0으로 강제하는 제약은 없다. 다만 작은 수치 차이나
신규성 우려만으로 추가하지 않는다. 병렬 보정, gate/amplitude teacher 또는
residual-conditioned head는 품질·비용·generic control을 통과할 때의 후속 선택지다.
LLM teacher/RL/새 backbone을 붙일 필요는 없다. LLM은 application activation 제공 모델이다.

## 두 가지 검증할 주장

- **C1:** VG-SAE는 사전 지정한 sparsity 예산 아래 강한 baseline과 비교해
  의미 있는 coefficient/feature recovery tradeoff를 제공한다.
  현재 근거는 한 training seed의 resampling이므로 새 training seeds 확인이 필요하다.
  입력 EV·실제L0·runtime을 숨기지 않으며 모든 metric의 dominance를 요구하거나 주장하지 않는다.
- **C2:** 현재 parameterization과 **minibatch-profiled beta recipe** 안에서,
  variance와 entropy가 support–amplitude 결합 및 유용한 hard code를 유지하는 데
  기여하는지 각각 검증한다. 같은 architecture에서 항을 제거하고 m/a scale과
  세 readout risk를 검사한다. 판정은 `variance-supported`, `entropy-supported`,
  `both-supported`, `neither-supported`로 나눈다. 실제로 지지된 항만 결론에 쓴다.
  이 내부 주장은 미검증이며 learned-beta recipe에 자동으로 일반화하지 않는다. 추가 encoder capacity와 무관한 확률 방법 전반의
  우월성까지 주장하지 않는다. 그런 확장이 필요할 때만 좁은 deterministic
  capacity control을 추가하고, 현재 최소 실험에는 강제하지 않는다.

## 최소 실험 블록

### B1: 원래 Stage1 크기의 확인

d128/true1024/width1024, density.01, skew.5, exponential을 primary로 둔다.
constant는 범위 대조다. 새 training seeds100/101/102, 기존 n_train8196과
동일 형식의 source를 만들고, 독립cal4096/test8192를 별도 RNG로 생성한다.
기존 방법6개(VG,L1,TopK,BatchTopK,JumpReLU,Gated)를 동일 데이터에 비교한다.
같은 latent 폭 외에 parameter수·tokens·walltime·inference latency를 보고한다.

Primary는 L0 cap8, coefficient relative error와 support F1 guardrail이다.
Cap4/16과 input EV/MSE는 secondary tradeoff로 함께 보고한다.
모든 method가 cap 안에서 cal latent error 최소점을 고를 수 있게 하고, test는 선택에 쓰지 않는다.
실제 L0가 다르면 정확히 같다고 쓰지 않는다. exact-L0 주장은 별도의 가까운 점 비교가 있을 때만 한다.
기존 불균등 grid를 최종 비교에 그대로 쓰지 않고 method별 tuning trial·data budget을 같게 한다.

각 world seed는 dictionary·train data·optimizer initialization을 함께 바꾸되 방법 간
같은 dictionary와 split, minibatch 순서를 공유한다. 모델마다 구조가 다르므로 모든
방법의 초기 tensor가 같다고 주장하지 않는다. L1의 주 비교는 native ReLU code이며,
원래 train-GMM threshold를 적용한 버전은 같은 checkpoint의 별도 secondary readout이다.
GMM readout으로 native L1의 primary 결과를 교체하지 않는다.

방법별 동일한 수의 control/recipe trial과 사전 지정 checkpoint 후보를 제공한다.
Checkpoint와 control은 cal hard latent error 최소화로 함께 선택하되 cal mean L0가
상한 이하여야 한다. Tie는 cal MSE→actual L0→고정 trial/checkpoint 순서로 푼다.
Test는 선택 동결 뒤 한 번 평가한다. Training loss나 input EV plateau로 대체하지 않는다.
구체적인 grid, 업데이트 수, 후보 수와 GPU 상한은 동반 EXPERIMENT_PLAN의 B1에서
고정하며, 그 계획과 함께 검토한다. Baseline의 native auxiliary loss와 schedule을 보존한다.

Operational go의 상대 baseline은 각 seed에서 non-VG 후보의 cal error가 가장 낮은
방법으로 test 이전에 고정한다. 모든 baseline과의 차이도 따로 보고한다.
세 seed의 paired relative error 개선 평균 5% 이상, 평균 F1 하락 .02 이내,
어느 seed도 error 5% 초과 악화가 없을 때 후속 범위 확장을 우선한다.
이는 작은 표본의 실행 우선순위 기준이며 통계적 확증이 아니다. Test actual L0가 cap을
넘는 점은 cap 충족 실패로 표시하고 비교 성공으로 세지 않으며 test 재선택하지 않는다.
Test exact-L0 비교가 아니면 cap-constrained tradeoff라는 범위를 유지한다.

### B2: 원래 objective의 필요성

같은 VG encoder/decoder에서 full, no-variance, no-entropy, 둘 다 제거의 4 variants.
모든 arm은 minibatch-profiled beta와 LR .01을 사용한다. 각 arm의 energy에 따른
beta 재조정까지 포함한 recipe 비교이며 동일 수치 beta를 고정한 원인 분리는 아니다.
gamma는 각각 같은 HPO budget으로 cal에서 고르고 실제 L0와 beta를 기록한다.
Full 대비 no-variance와 no-entropy를 각각 별도 판정한다. 각 비교에서 세 world의
cal/test cap 충족, 2/3 이상 낮은 hard latent error, 평균 상대 개선 3% 이상,
평균 F1 하락 .02 이내 및 m/a·risk 해석의 일관성을 요구한다. Variance만 통과하면
variance만, entropy만 통과하면 entropy만 지지한다. 둘 다 통과할 때만 둘 모두를
지지한다고 쓰며, 둘 다 제거한 arm과 factorial interaction은 기술적 paired 결과로
보고한다. C1에서 learned beta가 선택돼도 이 C2 결과를 그 recipe의 검증으로 옮기지 않는다.
손실값은 서로 다른 objective이므로 직접 우열을 판정하지 않는다.
native hard coefficient error/F1와 mean/sample/hard reconstruction을 사용한다.
같은 architecture/parameter count의 deletion check로 좁은 C2를 겨냥한다.
Variance를 없애면 h=m*a를 고정한 채 m을 낮추고 a를 키우는 scale compensation이
가능하다. Entropy가 있으면 m=prior로 가는 동안 amplitude가 정보를 나를 수 있고,
둘 다 없으면 gamma>0에서 m→0, a→infinity 방향이 생긴다. Full variance는 같은
h에서 .5*((1−m)/m)*h²*||D_j||²여서 이 방향을 억제한다. 이는 조건부 자유 변수의
성질이며 affine-softplus encoder에서 실제로 발생했는지는 별도 측정해야 한다.
따라서 deletion collapse만으로 경쟁 SAE나 변분 방법 전반에 대한 우위를 주장하지 않는다.
m quantile, amplitude와 code norm, mean/sample/hard risk, beta, gradient norm을 함께
기록한다. Beta mode와 checkpoint/HPO 기회를 맞추고 test를 본 뒤 새 clamp나 penalty를
넣지 않는다. 서로 다른 목적함수의 raw loss 크기는 성능 비교에 쓰지 않는다.

### B3: Stage2·실제 activation으로 옮길 범위 확인

우선 기존 pinned SynthSAEBench checkpoint를 새cal/test stream에서 좁게 평가한다.
기존 calibration 재사용 표는 확증 test로 쓰지 않는다. Stage2의 MCC/classifier 정의와
Stage1의 rectangular-union latent error/F1를 같은 숫자로 합치지 않는다.
선택한2–3 operating point가 유지되면 추가 training seeds를 실행한다.

실제 activation은 저장된 Gemma L5 seed0 완료 모델 중 VG와 baseline의 소수점을
평가하는 것이 먼저다. 문서 단위로 분리한 fresh cal/test text, native hard L0,
x fidelity, CE/KL를 본다. 이 단계에서는 latent ground truth가 없으므로 원하는 L0
아래 calibration reconstruction/CE 등 실제 사용 가능한 지표로만 선택한다.
Synthetic의 oracle-assisted 선택과 real activation의 operational 선택을 구분한다.
real support 정답이 없으므로 semantic calibration·true recovery를 주장하지 않는다.
이 단계에서 이득이 없으면 적용 범위를 제한하며 주 연구를 다른 진단 논문으로 바꾸지 않는다.

## 다음 실행과 예산

현재 run의 새 pilot budget은 8 GPUh 이내이며, P1/P2/P3/P3b의 주요 타이머 합계는
약 0.2883 GPUh다. Import와 개별 smoke/검증 시간은 포함하지 않는다.
대규모 새 학습은 이번 run에서 하지 않았다.

다음 순서는 원래 크기의 training recipe/동일 tuning budget 고정 → B1+B2의
좁은 multi-seed 비교 → 이미 저장한 Stage2/3 모델의 fresh evaluation이다.
Stage2 기존200M VG는 중앙 약4,900초/run 기록이 있으므로 전체 grid를 즉시
5seeds로 반복하지 않는다. 별도 throughput 측정과 단계별 budget cap을 먼저 둔다.
정확한 job수·데이터 분리·go/no-go는 통합 EXPERIMENT_PLAN에서 고정한다.

## 미해결 문제

독립 training seeds·공정한 tuning·objective ablation·실제 activation 평가는 남아 있다.
현재 방법을 새 SAE로 연구할 근거가 있다는 것과 그 기여가 이미 완성됐다는 것은 다르다.
부정 결과는 같은 VG-SAE 개발 안에서 다음 선택을 바꾸는 근거이며,
원래 아이디어를 임의의 별도 연구 주제로 대체하는 이유가 아니다.


## 구현 인계와 실패 처리

현재 core API는 `src/sae_model.py`, `src/sae_loss.py`, `src/sae_train.py`를 재사용한다.
이번에 만든 네 pilot runner는 새 heldout 선택, beta 초기화, 조건부 gate 및 공통
amplitude control의 재현용이다. B1의 다중 checkpoint/HPO 선택을 지원하는 runner는
다음 구현 단계이며, pilot runner만으로 B1이 이미 실행 가능하다고 표시하지 않는다.

필수 변경은 사전 정의한 checkpoint 저장, split provenance, cal-only selection JSON,
train-only L1-GMM fit, 원자료 전체 curve와 비용 로그다. 원래 모델에 별도 teacher나
trainable head를 추가하지 않는다. 필요성이 나중에 확인되면 같은 anchor 안에서 검토한다.

Cap feasible 모델이 없으면 그 method/seed는 미달로 보고한다. NaN/OOM은 실패와 예산에
포함하고 동일 조건의 기술적 재시도만 별도 기록한다. C1이 약하면 불균등 tuning과
readout 및 학습 경로를 먼저 확인한다. C2가 붕괴만 보이면 coupling에 관한 좁은 결론으로
남긴다. 현실 데이터 전이가 약하면 적용 범위를 줄인다. 어느 경우도 원래 연구를 별도의
복제 진단 주제로 자동 전환하는 규칙은 아니다.


상세 grid·업데이트·예산은 [실험 계획](EXPERIMENT_PLAN.md), 실제 실행 상태는
[tracker](EXPERIMENT_TRACKER.md)에 있다.
