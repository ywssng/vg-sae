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

선택 후보는 calibration split에서 signed ideal margin
M_j=beta E[V_j]/2−E[K_j]로 순위를 정한다. 주 비교는 high-margin atom과
amplitude/frequency 및 feasible drift가 비슷한 low-margin atom의 복제다.
random expansion은 주 비교에 넣지 않는다. test는 선택에 사용하지 않는다.
아래의 실행 규칙에 따라 대조군이 없으면 판단 유보로 남긴다.

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
  B1: analytic identity 검사 + held-out parameter-feasible cloning. ΔF와
  variance–KL 기여가 모두 음수인지 확인한다. 충분한 eligible 후보에서 실패하면
  tested configuration에 대한 비지지이며 보편적인 부재를 증명하지 않는다.
- C2: 사전에 정의한 clone 개입 뒤, high/low-margin 간 feature 품질의 학습 변화
  차이가 variance term on/off에 따라 달라지는 조건이 있다. B2는 주입 직후와
  추가 학습 후를 비교하고 duplicate-invariant 지표를 쓴다. 이것은 주입한 개입의
  국소 효과이며, 일반 학습에서 자발적 복제가 발생한다는 주장이 아니다.
  B3: B1/B2의 동일 규칙을 SynthSAEBench 한 조건에서 반복한다.
  C2가 지지되지 않으면 local loss preference만 남기며 harmful mechanism이라고
  부르지 않는다. semantic 지표 보존/raw L0 증가만 있으면 비용 증가라고 기록한다.

support ground truth가 없는 실제 activation에서는 calibration을 주장하지 않는다.
Stage3 CE/KL/probing은 synthetic에서 효과가 확인된 뒤의 확장이고 현재 필수블록이 아니다.

## 비용·실행 중단 조건

현재 작은 학습은 1200steps 모델당 약6초, 6000steps 약28–34초였다.
다음 B1은 기존 checkpoint 평가로 <0.1 GPU-hour 예상, B2는 synthetic 3data×3train
seed와 paired controls를 최대2 GPU-hours 안에서 먼저 측정한다. B3은 warm cache와
generator throughput을 먼저 측정한 뒤 4 GPU-hour 이내 pilot을 설계한다.
이 시간은 예산 상한/추정이며 아직 측정한 full benchmark 시간이 아니다.

test atom selection 금지, prototype throughput이 예상보다 느리면 scale을 줄인다.
C1에서 유효한 대조군이 없거나 drift 한계를 넘으면 판단 유보이고 C2를 실행하지 않는다.
유효한 대조군이 충분하지만 양성 기준에 못 미치면 tested protocol에서 비지지다.
C2가 없으면 local objective geometry와 실용적 해석을 분리해 기록한다.
현재 연구 상태는 탐색 계속 가능, empirical mechanism 및 submission readiness 미확정이다.


## 실행을 고정하는 세부 규칙

### B1: 표본 분할, atom 선택과 수용 기준

새 confirmatory-small 데이터는 generator seeds 100/101/102와 각 optimizer
seeds 0/1/2를 사용한다. d16, truth32, width128, density2/32, noise .05,
exponential amplitude를 공통 유지한다. train8192/cal2048/test8192를 하나의
같은 dictionary에서 분할한다. 현재 탐색 파일럿과 다른 표본이며 test는 끝까지
atom 선택·수렴 결정·threshold 결정에 쓰지 않는다. test8192는 B1 gate용4096과
B2 최종 평가용4096으로 미리 나눈다. B1의 go/no-go에 본 표본을 B2의 새로운
확증 표본으로 재사용하지 않는다.

source training의 checkpoints 6000/8000/10000에서 cal R,V,K,beta와 signed margin을
기록한다. 마지막 두 checkpoint의 각 component/beta 변화가 5% 이하이고,
margin rank Spearman>=.9이며 M>1e-4의 부호 일치율>=.9면 operationally stable로
표시한다. 분모는 max(|이전값|,1e-3), absolute numerical tolerance=1e-6이다.
이 기준을 통과하지 않으면 14000까지 한 번만 연장하고 여전히 불안정하면 미수렴
조건으로 보고하여 C2로 넘기지 않는다. stationary/global optimum 보장은 아니다.

B1은 source beta와 gamma, entropy weight1을 고정한다. ideal function-level
clone은 해석식 검증용이고 feasible clone과 별도 행으로 보고한다.
feasible atom j의 변경은 gate row/decoder column 복제, amplitude bias−log2다.
R,V,K는 width로 평균내지 않는 현재 코드의 sample 평균을 그대로 사용한다.

ΔF=beta ΔR+beta ΔV+ΔK, S=beta ΔV+ΔK를 모두 저장한다.
실용적 탐색 cutoff는 epsilon_F=epsilon_S=1e-4 nat/sample,
mean/hard decoded drift energy는 각각 cal input centered energy의1e-4 이하로 둔다.
이 값은 선행연구가 정한 보편 상수가 아니라 test를 보기 전 정한 작은 개입 기준이다.
수치 검증은 float64에서1e-8 abs/relative로 먼저 한다.

cal에서 feasible drift 기준을 통과하고 M>1e-4인 atom을 큰 M 순서로 본다.
low control은 M이 아래 사분위에 있는 별도 atom이며, 두 atom의
log(mean amplitude+1e-8) 차이<=.25, firing-frequency 차이<=.01,
normalized mean/hard drift energy 차이<=5e-5로 match한다.
margin은 mechanism 대비이므로 match하지 않는다. 동일 후보가 있으면 낮은 index를
택한다. 첫 번째 match된 한 쌍을 각 source checkpoint의 primary pair로 고정한다.
cal pair가 없으면 해당 checkpoint는 inconclusive다. 높은 test 이득으로 바꾸지 않는다.

각 atom의 전체 cal/test 표는 descriptive prevalence로 보존하되 성공 사례만
분모에서 골라내지 않는다. primary C1은 사전에 선택한 high atom의 held-out
ΔF<−epsilon_F와 S<−epsilon_S 및 drift 기준을 모두 충족하는지다.
3개 data world 모두에서 optimizer-seed 평균이 같은 부호/크기 기준을 통과해야
C2 go로 둔다. 각 world의3개 값과 범위를 모두 보고하며, 9개를9개 독립
데이터셋으로 취급하거나 이 operational gate를 유의성 검정으로 부르지 않는다.
불확실성은 data world를 최상위 cluster로 요약하며3개 world라는 한계를 명시한다.

### B2: 대칭, optimizer, 주입 직후 기준선

한 source checkpoint에서 high/low-margin 두 feasible clone을 만든다.
각 clone에서 기존 variance term on/off로 분기해 총4개 arm을 만든다.
variance-off는 같은 posterior의 더 정확한 구현이 아니라 명시적인 objective ablation이다.
width는 네 arm 모두129, gamma/beta는 source 값으로 고정하고 두 encoder와
unit-normalized decoder만 학습한다. source Adam state는 네 arm 모두 reset한다.
LR3e-4, batch256, 동일 minibatch 순서, 2000steps, weight decay0을 사용한다.

analytic loss와 동일 Adam state는 복제 대칭을 보존하므로, 복제 쌍의 decoder
접공간·gate row·amplitude row에 서로 반대인 seeded unit-vector perturbation을
넣는다. epsilon 후보1e-5/1e-4/1e-3 중 위 drift 기준을 high/low 모두 통과하는
가장 큰 값을 cal에서 한 번 고른다. 같은 값·방향seed를 on/off arms에 복사하고
decoder를 재정규화한다. epsilon 선택에 test나 최종 AP를 사용하지 않는다.
표현 가능한 비대칭이 없거나 drift 기준을 통과하지 못하면 continuation은 판단 유보다.

원 source t=0−, perturbation을 포함한 clone 직후 t=0+, 추가 학습 t=T를
각각 평가한다. primary 학습 효과는 Q(T)−Q(0+)이며 t=0−와의 차이는 즉시
개입 효과로 별도 보고한다. C1의 ΔF/S/sign/drift gate를 이 실제 perturbed
t0+ 초기상태에서 cal 및 B1 holdout에 재적용한다. perturbation 없는 이상적
clone이 통과했다는 이유로 넘기지 않는다. 실패하면 test를 보고 epsilon이나
pair를 바꾸지 않고 해당 사전 계획을 비지지/유보로 기록한다. 동일 atom이 남는 것 자체는 양성 기준이 아니다.

### B2: duplicate-invariant primary ranking endpoint

known true dictionary에서 learned atom을 **가장 큰 양의 cosine**의 true atom에
할당한다. cosine cutoff .8, 동률은 작은 true index, 미달은 unmatched다.
각 true feature의 score는 할당된 learned gate들의 max로 정의한다.
이는 평가용 score이며 posterior probability라고 부르지 않는다.
표현된 true feature의 AP를 계산하고, 할당된 latent가 없는 feature의 AP는0으로
둔 뒤 true feature 전체를 평균한다. support가 test에서 전혀 관찰되지 않은
feature는 AP undefined로 남기고 고정 평가 집합의 누락률을 함께 보고한다.
독립 synthetic에서는 모든 feature를 사전에 평가 집합으로 정한다.
SynthSAEBench에서는 cal positive>=20인 feature 집합을 test 전에 고정하고,
frequency 구간·포함률을 보고한다. 미관찰 test feature가 많으면 의미 비교를 보류한다.

AP는 threshold-free ranking metric이다. native threshold .5는 별도로 계산하는
hard EV/raw L0 및 grouped support F1에만 적용한다. AP가 낮아졌다는 사실을
0.5 경성 support가 나빠졌다는 뜻으로 바꾸지 않는다.

cosine cutoff .8이 tiny direction drift를 큰 AP 변화로 바꿀 수 있으므로,
같은 saved scores에서 .75/.80/.85를 모두 사전고정 계산한다.
세 cutoff 각각에서 아래 C2의 세 손상 조건을 모두 통과해야 robust ranking
손상이라고 부른다. .8만 통과하면 cutoff-sensitive/inconclusive로 기록한다.
각 true-feature의 best positive cosine, 세 cutoffs의 coverage 및 cutoff±.01
구간의 atom 수를 함께 보고해 alignment 변화와 threshold crossing을 구분한다.
새 calibration network나 새로운 posterior metric을 도입하는 것이 아니다.

경성 support 손상이라는 문구는 같은 max score를 .5에서 자른 grouped F1에서도
high/on change, D_on, I가 각각−.01 이하이고 동일 세 cutoff/data-world 기준을
통과할 때만 사용한다. AP만 통과하면 feature-ranking 결과로 제한한다.
F1은 zero-denominator를0으로 정의하고 test positive가 없는 feature는 앞의
AP eligibility/누락 규칙에 따라 undefined로 남긴다.

정확한 duplicate를 넣어도 mapping 및 max score가 같아 primary AP가 같다는
sanity test가 필수다. raw latent L0, hard EV, unmatched firing은 별도 보조
지표다. 기존 Hungarian/union 지표도 보존하되 그것만으로 semantic harm을 판정하지 않는다.

hard EV/L0/grouped F1에서는 native threshold .5가 primary다. 추가로 cal에서 true 평균 support count에
가장 가까운 공통 budget을 찾은 hard-L0-matched 평가를 보조로 한다.
각 achieved L0는 target의2% 또는0.05 중 큰 허용치 안이어야 한다.
동일 gate ties로 공통 budget이 없으면 비교불가로 남기며 보간해 일치시켰다고 하지 않는다.

### B2: mechanism interaction, 비지지와 유보

각 source pair에 대해 D_on=[Q_H(T)−Q_H(0+)]−[Q_L(T)−Q_L(0+)]를 계산하고
D_off도 같은 방식으로 계산한다. I=D_on−D_off가 primary mechanism interaction이다.
negative I만으로 harm이라고 부르지 않는다. high/on 자체의 AP change<=−.01,
D_on<=−.01, I<=−.01을 모두 만족해야 '이 국소 clone 개입과 variance term에
의존하는 feature-ranking 손상'을 지지한다고 본다. 경성 support 손상은
위 grouped F1 corroboration까지 충족해야 한다. 역시3 data world 모두의
train-seed 평균에서 같은 방향이어야 go다. 세 값이 보존되고 raw L0만 늘면
semantic harm이 아니라 representation cost 증가다.

variance-off는 다른 objective를 만들므로 이 결과는 고정 beta/gamma와 주입된
초기조건에 대한 paired intervention의 결론이다. 자연 학습의 자발적 복제,
다른 beta 최적화, semantic real-model feature에 대한 보편 인과로 확장하지 않는다.
효과가 작거나 방향이 섞이면 비지지; matching·stability·metric coverage가
성립하지 않으면 유보한다. 근거가 없다고 새 module을 더하지 않는다.

B3는 이 규칙을 같은 형태로 작은 SynthSAEBench 조건에 옮기는 외적 반복이다.
B3 test64k도 B1용32k/B2용32k로 미리 분리한다.
새 learned method의 superiority comparison은 현재 claim map의 필수가 아니다.
나중에 이를 주장할 경우 Gated/JumpReLU/BatchTopK의 동일 계산·sparsity 대조를 추가한다.
