# 연구 제안: VG-SAE의 복제 유인과 희소 feature 선택의 관계

2026-09-19. 연구 계획이며, 새 방법의 성능이나 실제 복제 손상을 입증한 논문이 아니다.

## Problem Anchor

VG-SAE의 Bernoulli gate와 입력별 amplitude를 함께 학습할 때, 목적함수가
feature를 선택하는 표현과 동일 정보를 여러 latent에 나누는 표현 중 무엇을
선호하는지 밝히고, 그 선호가 실제 경성 feature 품질에 영향을 주는지 검증한다.
기존 VG-SAE 코드와 synthetic/실제 activation 평가 구조를 사용한다.
일반적인 dropout 복제 정리, 확률적 SAE 최초성, 보편적 calibration 또는 SOTA는
주장하지 않는다. 새 학습 모듈 없이 원인 하나를 판별하는 것이 목표다.

## 방법을 먼저 설명하면

1. 학습이 충분히 진행된 VG checkpoint에서 각 feature의 gate KL와 확률적
   재구성 variance 기여를 측정한다. posterior mean, sampled reconstruction,
   hard reconstruction을 각각 평가한다.
2. feature 하나를 두 개로 나누는 개입이 loss를 낮추는지 먼저 계산한다.
   동일한 평균/경성 출력을 유지하는 이상적 개입과 실제 softplus encoder에
   구현 가능한 개입을 구분한다.
3. 실제 개입에서도 loss 선호가 확인된 경우에만, 같은 폭·계산량의 대조군과
   짧게 추가 학습하여 분산된 표현 증가가 support 회복에 손상을 주는지 본다.
   복제 선호가 없거나 손상이 없으면 그 가설을 기각한다.

## 출발 근거와 미입증 부분

`idea-stage/evidence/pilots/`의 초기 1200-step 실험은 optimizer seed 2개와
data seed 1개다. 폭32→128의 posterior-mean/hard EV 차이는 약 .025→.051로
늘었지만, 실제 stochastic risk는 모든 설정에서 hard risk보다 나빴다.
fixed total prior와 단순 ma readout은 개선하지 않았다. 따라서 mean-hard 차이를
학습 objective 실패나 calibration 실패의 증거로 사용하지 않는다.

6000-step 후속에서는 mean-hard 차이가 폭32 약 .018, 폭128 약 .024로 줄었다.
마지막 약200 step beta 변화는 .4–.9% 수준이지만 완전 수렴을 보장하지 않는다.
width 관찰은 인과 근거가 아니며, 초기 격차에는 학습 예산의 영향이 있었다.

상수 입력·bias off의 replica 구성은 mean reconstruction exact, KL0, hard0,
variance∝1/r을 허용한다. 이는 알려진 dropout 원리의 VG 구현 확인이고
nonzero amplitude에서 stationary point도 아니다. 실제 학습발생은 미입증이다.

## 기술적 핵심

단위 decoder atom d_j, q(s_j|x)=Bern(m_j), amplitude a_j를 사용한다.

V_j(x)=0.5 m_j(1−m_j) a_j² ||d_j||²,
K_j(x)=KL(Bern(m_j)||Bern(pi)), pi=sigmoid(−gamma).

현재 beta를 고정하고 atom j를 r개로 복제하면서 각 amplitude를 a_j/r로
바꾸면 mean과 hard reconstruction이 모두 동일하다. gate를 복사했으므로
V_j는 V_j/r, KL는 rK_j가 된다. 따라서 입력 평균에서

ΔF_j(r)=(r−1) E[K_j] − beta(1−1/r) E[V_j].

r=2의 ideal loss preference는 beta E[V_j] > 2 E[K_j]로 판별한다.
이 identity 자체를 새 일반 정리로 주장하지 않는다. 실제 learned checkpoint에서
어느 feature/조건에 이 유인이 존재하는지와 그 후속 결과가 검증 대상이다.
gamma, entropy weight=1, beta를 고정한 local 비교다. profiled beta에 이 선형식을
그대로 쓰지 않고 aggregate energy를 넣어 전체 objective를 다시 계산한다.

### 실제 architecture에 넣는 위치

`VariationalGarroteSAE`의 decoder column과 gate encoder row를 복제한다.
amplitude encoder row도 복사하고 두 복제의 bias에서 log(2)를 뺀다.
softplus(t−log2)는 softplus(t)/2와 같지 않다. 따라서 이 구현 가능한 개입은
평균/경성 출력을 정확히 보존하지 않는다. loss 변화는 Δrecon, Δvariance,
ΔKL로 분리하고, mean drift·hard drift·L0 변화를 보고한다.
새 public config나 기본 inference를 변경하지 않고 별도 experimental runner로 만든다.

선택 후보는 calibration split에서 전체 atom에 대한 KL/V, 평균 amplitude,
firing frequency로 정한다. test에 보고할 atom은 결과 확인 전에 고정한다.
비교는 같은 폭 증가량을 가진 random-eligible atom 및 amplitude/frequency가
비슷한 atom 복제로 제한한다. match가 되지 않으면 causal 비교라고 부르지 않는다.

## 기여와 선행연구 차이

dominant contribution 후보는 **학습된 VG의 복제에 대한 loss 선호와 실제
희소 feature 선택의 관계를 반증 가능한 개입으로 검증하는 것**이다.
가정만 바꾼 새 Bayesian SAE, 새 prior, readout trick의 우수성을 목표로 하지 않는다.

- Cavazza et al., AISTATS2018: duplication와 width-dependent dropout은 알려짐.
  [원문](https://proceedings.mlr.press/v84/cavazza18a.html).
- SoftSAE: tiny soft weights의 정보 누출과 hard stabilization은 알려짐.
  [원문](https://arxiv.org/abs/2605.06610).
- Plascencia2026: ReLU/L1의 diffuse phase 존재도 알려짐.
  [원문](https://arxiv.org/abs/2609.10299).

차별점은 conditional gate KL와 learned amplitude가 있는 이 objective에서
실제 loss 선호가 나타나는 조건, 그리고 feature recovery 손상과의 연결이다.
그 연결이 없으면 단순 known-mechanism application만 남으므로 논문 중심 주장을
철회한다. 이 novelty는 현재 검색에 대한 잠정 평가다.

## 복잡도 예산

재사용: 데이터 generator, VG/SAELens baseline trainer, dictionary matching,
SynthSAEBench fixed generator, Stage3 activation hooks. 새 trainable module=0.
수정하는 것은 실험 전용 clone intervention과 진단 집계뿐이다.
LLM/VLM teacher, auxiliary calibration net, joint gate blocks는 사용하지 않는다.
LLM은 후속 실험의 activation 제공 모델이며 새 teacher/reward 역할을 맡지 않는다.

## 두 가지 조건부 주장과 최소 실험

- C1: 충분히 학습된 VG checkpoint 일부에는 실제 encoder가 표현할 수 있는
  복제가 loss를 낮추는 유인이 있고, 그 효과는 재구성 drift만으로 설명되지 않는다.
  B1: analytic identity 검사 + held-out parameter-feasible cloning; effect components와
  baseline-matched controls. negative이면 C1 기각.
- C2: C1에서 찾은 유인은 짧은 재학습 뒤 latent 분산과 support recovery 손상으로
  연결된다. B2: clone warm start와 동폭 random expansion을 같은 예산으로 학습,
  measured hard-L0를 calibration에서 맞추고 fresh test F1/AP/recovery/EV 측정.
  B3: B1/B2를 SynthSAEBench 한 조건에서 반복해 단순 독립 synthetic 한계를 검사.
  C2가 실패하면 C1의 local loss 결과만 남기며 harmful mechanism이라고 부르지 않는다.

support ground truth가 없는 실제 activation에서는 calibration을 주장하지 않는다.
Stage3 CE/KL/probing은 synthetic에서 효과가 확인된 뒤의 확장이고 현재 필수블록이 아니다.

## 비용·실행 중단 조건

현재 작은 학습은 1200steps 모델당 약6초, 6000steps 약28–34초였다.
다음 B1은 기존 checkpoint 평가로 <0.1 GPU-hour 예상, B2는 synthetic 3data×3train
seed와 paired controls를 최대2 GPU-hours 안에서 먼저 측정한다. B3은 warm cache와
generator throughput을 먼저 측정한 뒤 4 GPU-hour 이내 pilot을 설계한다.
이 시간은 예산 상한/추정이며 아직 측정한 full benchmark 시간이 아니다.

test atom selection 금지, prototype throughput이 예상보다 느리면 scale을 줄인다.
C1 유인이 없거나 matching/mean-drift control에서 사라지면 복제 주장을 중단한다.
C2가 없으면 local objective geometry와 실용적 해석을 분리해 기록한다.
현재 연구 상태는 탐색 계속 가능, empirical mechanism 및 submission readiness 미확정이다.
