# VG-SAE 후속 실험 계획 — Sparse but Wrong

2026-09-22. **미실행 계획**. 완료한 P1/P2/P3를 다음 개발 단계와 구분한다.
사용자가 시작한 VG-SAE를 L1 대안으로 개발하고 실제 feature 및 support 복원을
개선하는 것이 목표다. 진단 지표는 이 목표를 검증하는 도구다.

## 주장과 현재 출발점

| 주장 | 필요한 근거 | 연결 block |
|---|---|---|
| C1: VG가 상관·반상관 feature에서 유용한 dictionary/support recovery를 제공한다 | 조율과 수렴을 확인한 L1, rescaled L1, Gated, Anthropic JumpReLU, BatchTopK와 새 worlds 비교. Oracle 선택과 사용 가능한 선택을 분리 | B1, 조건부 B3 |
| C2: 같은 VG 구조에서 precision 정책을 통제하면 잘못된 dense support 영역을 줄일 수 있다 | 고정 gamma, 동일 초기화·표본·budget에서 fixed / global learned / minibatch profiled 대조. Signed identity와 support의 개선 및 비용을 확인 | B2 |

C1의 유용성, L1 대비 개선, strong baseline 대비 개선은 각각 판정한다. 같은 폭이
같은 parameter count라는 주장, calibrated semantic posterior, 자동 true-L0 발견,
parameter-count와 무관한 VG objective의 우월성을 주장하지 않는다.

- P1: 225 fits 완료. Primary target 1.8의 VG coverage는 0/9이고 target 2에서는
  7/9. 좋은 복원점은 있지만 joint superiority는 확인되지 않았다. JumpReLU는
  모든 target에서 coverage 0이므로 강한 baseline 비교가 아직 열려 있다.
- P2: learned gamma의 joint success 0/18, primary 18 cells 중 14개의 full-train
  gradient residual이 2 초과. 유한 예산의 recipe 실패이며 잘못된 EB 정상점으로
  수렴한 증거가 아니다. Gamma adaptation은 현재 기본 개선안에 넣지 않는다.
- P3: 동일 gamma의 한 world에서 exact posterior의 total-effect rescue가 있었다.
  제한된 L0 matching, sign/mapping과 실제 다중 방향 혼합, readout 효과를 구분한다.
  Joint module은 복제·비용 근거가 추가되기 전까지 후속 선택으로 남긴다.
- Dense-offset witness는 true decoder와 dense code가 공존할 수 있음을 보인다.
  Ideal profiled / globally optimized beta의 목적함수는 해당 family에서 하한이
  없지만 production profiled branch는 epsilon으로 제한된다. Global learned
  branch에는 이 energy clamp가 없다. Finite beta의 수학적 하한은 경험적 recovery
  보장이 아니며 이 family가 실제 SGD 실패 원인이었다는 증거도 아니다.

## 공통 데이터·선택·판정 계약

새 small toy는 d=20, true features=5, SAE width=5, nonnegative magnitude
`max(0, 1 + .15 epsilon)`, QR dictionary, Gaussian-copula star rho=±.4다.
QR와 finite training set은 paper-inspired 변경이며 공식 generator의 완전 재현이라고
부르지 않는다. 기존 rho=0 pilot은 음성 대조 관찰로 보존하며 새 주 비교의 반복 수에
포함하지 않는다. 실제 binary firing correlation과 native true L0도 저장한다.

개발 world는 310, confirmation worlds는 311/312/313을 제안한다. 아직 사용되지
않았는지 실행 전 manifest를 검사한다. 충돌하면 결과를 보기 전에 다음 연속 미사용
seed를 manifest에 고정한다. Dictionary/init, train, calibration, test, minibatches는
각각 `(run_id, stage, world, rho, p, stream_role)`에서 독립 생성한다.
B1 main과 B2 confirmation은 동일한 canonical stage `main_confirmation`을 사용한다.
B1/B2라는 analysis label은 RNG key에 넣지 않는다. Development의 B1/B2도 canonical stage
`development`와 동일한 source data/init/batch streams를 사용한다. 같은 VG policy/gamma/world fit은
같은 초기화·data/batch stream·checkpoint를 참조해야 18 fits 공유로 계산한다. 동일 cell의
방법들은 같은 train/cal/test samples와 batch 순서를 공유한다. 각 cell은 train
32,768, calibration 8,192, test 16,384 examples다. Development에는 test를 만들지 않는다.

**Source truth와 target truth의 역할을 분리한다.**

1. `S_source`: p=.4 development calibration의 dictionary/support truth를 써서
   방법별 하나의 coefficient와 checkpoint fraction을 고정한다. 두 부호에서
   `min(signed cosine, support micro-F1)`의 최솟값을 최대화하고, EV가 각 방법의
   최고 EV보다 .02 넘게 낮은 후보는 제외한다. 동점은 작은 계산비용, 작은 control
   index 순서다. 다른 density에서는 이 recipe로 새로 학습하되 target의 true L0,
   support labels, dictionary를 selection API에 주지 않는다. Synthetic source의
   감독을 사용하므로 완전한 truth-free 학습이라고 부르지 않는다.
2. `S_blind`: main p=.4에서 calibration input과 learned decoder만 쓰는 기존
   minimum-c_dec selector를 평가한다. Final-checkpoint grid에서 c_dec 최소,
   동점이면 calibration MSE 최소, 이후 control index 순으로 선택한다. P1의 실패를
   숨기지 않으며 새 score를 test 뒤에 발명하지 않는다. 이것만으로 support를
   인증하거나 unknown-L0 문제를 해결했다고 하지 않는다.
3. `S_oracle`: 각 main cell의 calibration truth를 써서 같은 recovery score로
   선택한 상한이다. Main target L0를 사용한 matching도 oracle diagnostic으로
   표시한다. 이 두 결과는 실제 사용 가능한 selector의 성능으로 합치지 않는다.

**C1의 primary는 `S_source`다.** Source-supervised recipe를 새 main world와 target
밀도에 적용한 결과로 정의한다. `S_blind`는 별도 truth-free secondary, `S_oracle`과
matched-L0는 별도 diagnostic/ceiling이다. Selector 간 가장 좋은 test 결과를 골라
primary로 바꾸지 않는다. 해당 method에 eligible `S_source`가 없으면 선택 실패로
표시하고 그 비교의 C1 결론을 유보한다.

Main/test 개봉 전에 method policy, grid, training length, checkpoint fractions,
alignment, selection IDs와 calibration-only amplitude gains를 선택 명세에 기록해 고정한다.
모든 final grid와 세 checkpoint의 fresh-test 평가는 한 번에 수행하고 전체 곡선을
보존한다. Test를 읽고 control, threshold, seed, training length를 변경하지 않는다.

주 지표는 signed Hungarian dictionary cosine, cosine≥.95인 true feature 비율,
support micro/macro F1 및 feature별 recall이다. Native hard L0와 VG expected L0,
hard/mean/expected-risk reconstruction, centered EV, coefficient NMSE, runtime을
같이 보고한다. Nonnegative code에서 sign reversal은 identity 실패다. Absolute
Hungarian cosine, sign-invariant multicomponent energy, outside-span energy와
전체 cosine matrix는 실제 다중 feature 혼합을 따로 설명한다. 기존 `mixing_energy`는
signed-assignment leakage라는 이름으로 유지한다.

L1의 nonnegative per-latent gains는 calibration input reconstruction만으로
적합하고 decoder/support는 고정한다. Rescaled L1는 추가 training/HPO fit이 아니다.
Native inference를 주 평가로 쓰고 VG gate threshold=.5를 사전 고정한다. Soft
amplitude가 양수라는 이유로 expected code의 nonzero 수를 hard L0로 보고하지 않는다.

World가 통계 단위다. 모든 paired differences, 실패 world, coverage 분모를 공개한다.
동일 checkpoint가 여러 target에 포함돼도 독립 반복으로 세지 않는다. 세 seeds는
방향성 검증 규모이며 정밀한 population uncertainty나 보편적 우위를 보장하지 않는다.

## B1 — 먼저 조율한 직접 비교와 실제 선택 규칙

**목적:** C1. Main paper의 recovery/support table과 realized-L0 curves에 사용한다.
새로운 큰 HPO 구현을 만들지 않고 기존 P1 runner를 좁게 확장하는 계획이다.

### Calibration-only range / convergence scout

개발 world 310, p=.4, rho ±.4에서 다음 초기 3점만 실행한다. 총
`5 methods × 3 controls × 2 signs = 30 fits`다.

| 방법 | 최초 control grid | Recipe 계약 |
|---|---|---|
| 현재 profiled VG | gamma={0, 2, 4} | 기존 nonnegative two-head VG, unit decoder norm |
| L1/ReLU | coefficient={.03, .1, .3} | 공식 baseline norm/parameterization 유지 |
| Gated | coefficient={.03, .1, .3} | 공식 gate/magnitude와 auxiliary loss 유지 |
| Anthropic JumpReLU | coefficient={.0003, .003, .03} | expected_average_only_in normalization, tanh penalty, bandwidth 2, threshold .1, tanh scale 4, pre-act coefficient 3e-6 |
| BatchTopK | k={1.8, 2.0, 2.2} | fractional batch allocation, 공식 decoder 처리 유지 |

Jump의 pilot 최솟값 .03은 너무 sparse했다. 두 decade 아래부터 다시 bracket하고
공식 input normalization을 적용한다. Normalizer는 train 입력으로만 적합하고
inverse transform 후 원 단위의 reconstruction을 평가한다. Folded weights와
threshold가 같은 native inference를 주는지 확인한다. VG decoder 제약을 baseline에
강제하지 않는다. 세 methods의 regularization 숫자가 같은 penalty budget이라는
가정도 하지 않는다. 변경된 input scale에서 과거 .03 결과를 직접 연결하지 않는다.

Small-toy 초기 공통 budget은 Adam lr=3e-4, batch=256, T=16,000 updates다.
Jump L0 warmup은 10,000 updates를 유지한다. 이 batch/sample budget은 공식 15M
fresh-example setting의 재현이 아니다. Development logs는 4k/8k/12k/16k에서
기록하며 method selection 후보는 warmup 후 12k/14k/16k 세 개다.

두 sign의 calibration에서 native L0 target {1.8, 2.0, 2.2}, 허용차 ±.15의 coverage와
실제 endpoint bracketing을 계산한다. 이것은 비교 가능한 곡선을 얻는 설계 도구이고
VG를 일부러 undersparse하게 만드는 것이 주 목표는 아니다. Main 유용성 gate는
true-L0 근처 복원과 selector 성능이다. 1.8/2.2는 wrong-setting 보조 분석이다.

**Freeze 전 허용된 보충은 방법별 control 한 개뿐이다.** 각 control은 두 sign 모두에서
실행하므로 최대 10 fits다. 누락된 target 2.0을 먼저 채우고, 그다음 1.8/2.2 중
현재 endpoint에서 가까운 것을 채운다. 단조 구간에서는 VG gamma의 산술 중점,
양수 penalty의 기하 중점, BTK k의 산술 중점을 쓴다. 전 범위가 너무 sparse하면
penalty 최솟값을 10으로 나누거나 VG gamma를 2 줄인다. 너무 dense하면 penalty
최댓값을 10배 또는 gamma를 2 늘린다. BTK는 목표값 자체를 쓴다. 비단조면 target에
가장 가까운 두 관측 control 사이의 중점을 쓰고 비단조임을 기록한다. 보충 control은
두 sign의 최대 target-distance를 가장 줄일 것으로 예상되는 한 개로, 동점이면
작은 값을 고른다. 실제 new result를 보기 전 그 값을 manifest에 기록한다.

수렴은 warmup 후 두 마지막 calibration checkpoints 사이의 normalized reconstruction
risk 변화≤1%, native L0 변화≤.1을 둘 다 만족하는지 본다. Identity/support trajectory도
공개한다. 이는 stationarity 증명이 아니다. 어느 방법의 유효 영역에서 두 조건을
만족하지 못하면 **모든 방법의 개발 fits**를 동일하게 T=32,000까지 한 번 연장한다.
새 eligible selection 후보는 24k/28k/32k 세 개로 교체한다.
기존 12k/14k/16k checkpoints는 superseded development evidence로 보존하며 삭제하지 않는다.
이전 checkpoints를 새 eligible 후보에 합치지 않는다. B1의 최종 T를 확정한 뒤 B2
개발을 시작하므로 B2가 나중에 별도의 horizon 확장 기회를 얻지는 않는다. 32k에서도 부족하면 unconverged로
표시하고 그 comparator에 대한 확정 우위를 보류한다. 추가 LR×warmup factorial은 없다.

### 새 held-out 직접 비교와 density transfer

B2 development를 마친 뒤 방법을 고정한다. 방법별 최종 3 controls에는 `S_source` 우승 coefficient를 반드시 포함한다.
VG는 B2의 same-gamma risk control gamma2도 포함한다(우승값과 같으면 한 슬롯).
이 mandatory set을 포함하는 세 값의 조합 중 target2.0 coverage를 우선하고 이어
1.8/2.2 coverage를 최대화한다. 동률이면 두 sign의 최대 target-distance가 작은 조합,
이후 원래 control-index 사전식 순서로 고른다. `S_source` 우승 checkpoint fraction도
그대로 main의 같은 coefficient에 적용한다. 3개 controls와 후보 수를 늘리지 않는다. 따라서 세 target를 동시에 못 채울 수도 있으며 이를 숨기지
않는다. 모든 방법에 main 3 controls, checkpoint 3개만 허용한다.

- Main p=.4: `3 worlds × 2 signs × 5 methods × 3 controls = 90 fits`, calibration
  candidates 270개. 새로운 rho=0 main sweep은 이번 핵심 budget에 넣지 않는다.
- Unknown-count transfer: p={.2,.6}, rho ±.4의 새 training cells에서 방법별
  `S_source` recipe 하나만 사용한다. `3 worlds × 2 signs × 2 densities × 5 methods
  = 60 fits`, calibration snapshots 180개. 이 snapshot의 truth로 재선택하지 않는다.
  같은 target에서 재학습한 recipe transfer이며 weight zero-shot transfer가 아니다.
- Same-L0 보조 비교: main calibration에서 target ±.15, pair difference≤.10을
  만족한 native readout끼리만 paired differences를 계산한다. Covered worlds와
  unmatched worlds를 모두 보인다. Matching 실패는 미해결이며 우위나 견고성의 근거가 아니다.

**판정(`S_source` primary):** useful region은 signed cosine≥.95, support F1≥.95인 점과 EV를 함께
제시한다. L1 대비 identity improvement는 rescaled L1와 비교해 signed cosine
차이≥.02, F1 차이≥.03, EV 손실≤.02가 두 부호 각각 3 worlds 중 2개 이상인지를
사전 판정한다. Gated/Jump/BTK도 같은 기준으로 각각 보고하되, 모두를 이겨야만
L1 대안으로서 유용하다는 뜻은 아니다. Ceiling parity는 VG 고유 개선을 입증하지
못한다. Unknown-count transfer에서 같은 joint 기준을 density/sign별로 검사하며,
좋은 oracle 결과만 있고 `S_source`/`S_blind`가 실패하면 usable selection은 미해결이다.
좋은 control 개수의 비율을 서로 다른 scale의 parameter-robustness 점수로 비교하지 않는다.

## B2 — 같은 VG의 precision 정책 한 가지 대조

**목적:** C2. 추가 prior, teacher, slab, posterior family를 교차하지 않는다.

Arms는 fixed beta=10, global learned beta(init=10), minibatch profiled beta의
세 개다. Gamma는 각 fit에서 고정하고 amplitude/gate/decoder/bias 초기 상태,
mini-batch 순서, optimizer, norm 제약과 hard readout을 공유한다. Learned beta는
production parameterization을 유지한다. Profiled arm의 epsilon과 floor-hit rate,
global arm의 beta trajectory와 overflow/gradient 상태를 별도로 기록한다.

Development는 world 310, rho ±.4, gamma={0,2,4}로 18 fits다. B1의 profiled
중복 fits도 별도 budget 상한에 포함해 보수적으로 계산한다. B1이 gamma 하나를
추가했으면 나머지 두 policies에서 두 signs씩 최대 4 fits만 더 한다. B1의 T와
checkpoint 규칙을 그대로 적용한다. 개발의 precision policy 채택 비교는 gamma2, **공통 final T**에서만 한다.
다른 gamma와 중간 checkpoint는 효과 곡선과 진단이다. 선택된 policy 안에서 C1의
`S_source` coefficient/checkpoint를 이후 고정한다. Precision 후보는 signed cosine≥.95를 유지하면서 profiled 대비 cosine 손실≤.01,
F1 개선≥.03, EV 손실≤.02일 때만 경험적 개선 후보로 표시한다.
어느 후보도 통과하지 못하면 현재 profiled를
주 baseline VG로 유지하고 C2를 미확인으로 둔다. 후보 간 동점이면 fixed를 우선한다.

Main은 B1의 고정 gamma 3점, p=.4, rho ±.4, worlds 311/312/313에서 확인한다.
세 policies의 전체는 `3 × 3 × 2 × 3 = 54 fits`지만 선택된 VG policy의 18 fits는
B1과 완전히 공유한다. **추가 fits는 36개**, 후보 snapshots는 108개다. 다른
checkpoint/config로 재학습하고도 공유했다고 계산하지 않는다.

주 대조는 gamma=2에서 **세 policy 모두 공통 final T**의 fixed/global 대비
profiled same-gamma total effect다. Policy별 `S_source` checkpoint fraction을 이
C2 대조에 사용하지 않는다. C1의 선택된 시점과 C2의 동일-update 비교를 분리한다.
P1에서 dense 문제가 드러난 gamma를 test 전에 지정한 것이다. 각 부호 3 worlds 중
2개 이상에서 signed cosine≥.95, cosine 차이≥−.01, F1 차이≥+.03, EV 손실≤.02이면
해당 policy의 개선 신호로 판정한다. Dictionary가 이미 맞고 support만 나쁜 경우도
포함하는 기준이다. Cosine까지 +.02 이상 좋아지는 더 강한 결과는 별도 표시한다.
추가 gamma들은 effect curve와 failure region으로 공개한다.
같은 gamma에서 더 적절한 L0를 택해 복원이 좋아지는 것도 실용적 개선이다.
별도 matched-L0 view는 “activity 차이를 넘어선 개선”의 좁은 가설만 검사한다.
평균 L0 matching만으로 직접적인 covariance/objective 인과 효과를 식별했다고 하지 않는다.

매 250 updates마다 mean reconstruction energy, Bernoulli variance, KL/prior/entropy,
beta, expected/native L0, ||bias||, 각 decoder 방향의 bias projection, inactive-truth
조건의 amplitude 평균과 gate saturation을 기록한다. Truth 조건부 진단은 선택
API와 분리한다. 각 final checkpoint에서 calibration mean으로 입력을 설명하는
비율, `b + D E[h]`의 상쇄 크기, cosine matrix를 export한다. 이를 통해 dense-offset
witness와 비슷한 현상이 나타나는지 관찰한다. 큰 bias 하나만으로 witness 도달이나
원인을 확정하지 않는다. Bias 고정/제거라는 추가 training factorial은 이번 block에 없다.

Finite beta의 하한만 있고 empirical recovery가 개선되지 않으면 C2는 실패/미확인이다.
그 경우 주방법을 자동 prior나 exact32로 교체하지 않고 어떤 operating regime에서
precision이 부족했는지 기록한 뒤 같은 VG 개발 안에서 다음 질문을 선택한다.

## B3 — 방법 고정 뒤 한 번의 확장과 실제 source 평가

**조건부:** B1의 유용한 영역과 baseline coverage, B2의 정책 선택이 고정된 뒤 진행한다.
진단만으로 external usefulness를 대신하지 않는다. 아래 단계마다 비용/실행 gate를
통과해야 하며 현재 실행한 것으로 기록하지 않는다.

1. **한 개의 paper-compatible 50-feature setting.** g=h=50, d=100;
   `p_i=.345*(49-i)/50+.05`, 합 10.9525. Source commit d5886b5의 correlation
   generator(seed 42, positive ratio .5, range .3–.9, sparsity .3, eigenvalue clip과
   diagonal renormalization) 및 1,000-step feature-direction optimization을 따른다.
   공통 correlation world 하나 안에서 initialization/train/cal/test seeds 411/412/413을
   구분하므로 independent correlation-world 3개라고 부르지 않는다. 목적은 5-feature에서
   50-feature로의 한 번의 범위 확장이다. 원 논문의 전 width/correlation 설정 재현이 아니다.
2. Recipe는 B1에서 고정하고 scale에 따른 **range calibration만** development seed 410에서
   허용한다. 5 methods × 3 controls=15 fits, 3 checkpoint 후보=45개. 초기 controls는
   VG/L1/Gated의 고정 3점을 사용하고, 공식 normalized Jump coefficient {.05,.1,.3},
   BTK k={5,11,18}을 쓴다. 각 방법 최대 한 개 control 보충=5 fits만 허용한다. 여기서도
   비교 영역을 못 찾으면 missing coverage로 멈춘다. Main은 5 methods × 3 controls ×
   3 seeds=45 fits, 후보 135개다. 목표 budget은 fresh 15M examples, batch1024,
   Adam lr3e-4, checkpoints 12M/13.5M/15M samples다. 실제 마지막 partial batch와
   updates를 기록한다. 이를 채우지 못하면 limited-budget extension으로 표시한다.
3. **기존 realistic/source 경로 확인을 먼저 한다.** Stage2 SynthSAEBench와 Stage3
   Gemma artifacts의 config, split, checkpoint 유무 및 평가 코드부터 inventory한다.
   기존 saved checkpoint는 기존 recipe의 근거이며 새 precision 정책의 결과가 아니다.
   Code/path inspection과 기존 checkpoint read-only 평가만으로 신방법 효과를 주장하지 않는다.
4. 첫 새 realistic 비교는 Stage2의 기존 SynthSAEBench source 한 개와 기존 width 하나에서
   final VG, rescaled L1, development에서 고정한 strong comparator 하나의 3 systems ×
   seeds 511/512/513=9 fits, final checkpoint 각 1개로 제한한다. Source checksum,
   exact width, token/example budget와 데이터 offset은 inventory와 throughput 뒤 실행
   manifest에 고정한다. 아직 확인하지 않은 값을 만들어 실행 가능한 확정 config라고
   부르지 않는다. Ground truth가 제공되면 C1 지표, 그렇지 않으면 source의 실제 task
   labels와 utility만 사용한다. 이 단계가 유익할 때만 Gemma layer 5의 기존 source로
   같은 3 systems × 3 seeds=9 fits를 별도 확장한다. 새 train/cal/test text의 비중복과
   sparse probing labels를 확보하지 못하면 보류한다. CE/KL, reconstruction, activity,
   sparse probing 또는 실제 task utility, cost가 필수다. c_dec 단독 통과는 없다.

Human annotation, 새 backbone, LLM teacher, RL, joint-posterior distillation은 이 계획에
필요하지 않다. 기존 평가 path가 새 method 효과를 검증하는 데 적합한지가 B3의 첫 판단이다.

## 실행 순서·정확한 작은 실행 수·비용 gate

| 순서 | 작업 | 새 training fits 상한 | 최종 eligible candidates 상한 | 진행 조건 |
|---|---|---:|---:|---|
| M0 | 후속 runner 구현, 5 methods × 500-step throughput probe, metric/readout/normalization 검증 | 5 probes | scientific 후보 0 | config hash, 시간/메모리, baseline native inference 확인 |
| M1 | B1 source scout와 coefficient 한 번 보충 | 30+10=40 | 120 | coverage 부족/비수렴 상태를 명시 |
| M2 | B2 precision development, 보충 gamma 대응 | 18+4=22 | 66 | policy/grid/T/selector 동결 |
| M3 | B1 p=.4 main | 90 | 270 | 새 test 열기 전 selection IDs 동결 |
| M4 | B2 confirmation의 추가 두 policies | 36 | 108 | M3의 VG 18 fits 공유 |
| M5 | B1 fixed-source density transfer | 60 | 180 | target labels로 재선택 금지 |
| M6 | 조건부 B3 50-feature scout/main | 20+45=65 | 60+135=195 | 방법 고정, throughput gate |
| M7 | 조건부 realistic source 한 개 | 9 | 9 final | inventory 후 exact execution manifest |
| M8 | 더 뒤의 선택적 Gemma 한 layer | 9 | 9 final | M7의 유용성과 별도 비용 gate |

M1–M5는 최대 **248 scientific fits**, 최대 **744 최종 eligible selection candidates**다.
이는 물리적으로 보존할 checkpoint의 상한과 다르다. B1 source scout가 T=32k로
연장되면 최대40 fits의 이전 3개씩, **최대120개 superseded checkpoints**가 추가로 남는다.
따라서 저장 checkpoint는 최대864개이고, 비용에도 연장분을 포함한다. B2는 B1에서
확정한 T로 시작하므로 별도 extension checkpoints는 없다. 주기적 숫자 로그는
checkpoint/candidate와 별도로 센다.
그중 development 62 fits/186 eligible candidates에는 test가 없고, main/transfer 186 fits의
558 checkpoints를 fresh test에서 동결된 방식으로 평가한다. Throughput probes는 별도다.
한 training의 3 checkpoints와 같은 checkpoint의 여러 selectors를 새 fit으로 세지 않는다.
Development 연장은 fit 수를 늘리지 않지만 updates와 비용은 늘린다.

**다음 실행의 첫 세 작업:** (1) 데이터/metric 및 Jump normalization 동등성 확인,
(2) 5방법 throughput probe, (3) development의 Jump 3 coefficients × 2 signs부터
range scout 시작. 이 순서는 이전에 coverage가 전혀 없었던 baseline을 일찍 검증한다.

P1의 225 fits × 3,000 updates 전체 runner는 약 .777 GPUh였다. 이를 단순 비례하면
248 fits × 16,000 updates는 약 4.6 GPUh, 32,000은 약 9.1 GPUh다. 새로운
normalization, data generation, optimizer와 평가 비용이 달라 **예측 예시일 뿐 약속이
아니다**. 실행 전 M0의 방법별 train/eval/I/O timing으로 다시 산정하고 setup·실패·retry도
합계에 넣는다. GPU 점유 상태를 다시 확인하며 기존 GPU0 작업을 건드리지 않는다.

미래 실행의 **제안 hard cap**은 M0–M5 합계 8 GPUh, CPU fallback을 쓸 경우 별도
8 CPU process-hours다. 이 수치는 사용자가 승인한 장기 계산 예산이나 이미 소비한
예산이 아니다. 예상치가 cap을 넘으면 전체 main을 시작하지 않고 개발 단계 결과와
미완료 block을 남긴다. 중간 cap에 도달해도 complete로 표시하지 않는다. 품질이 나쁜
seed만 삭제하거나 methods별 budget을 몰래 줄여 맞추지 않는다.

B3 50-feature 제안 cap은 별도 8 GPUh, M7은 별도 12 GPUh다. 예측상 15M×65 fits가
8 GPUh에 안 들어가면 source-faithful 실행을 그 cap에서 보장하지 않고 limited-budget
extension 또는 미실행으로 남긴다. M8의 장기 budget은 미지정이다. 현재 task는
계획 작성이며 어느 미래 sweep도 launch하지 않는다.

## M0의 미구현 전제

기존 P1/P3 runner는 현재 world, device, step 수와 grid가 고정된 pilot 코드다.
이 계획이 그 CLI로 이미 실행 가능하다고 표시하지 않는다. 후속 opt-in runner에서
world/device/T/checkpoint/grid/precision을 명시적으로 받아야 한다. 기존 public config의
의미와 현재 결과를 덮어쓰지 않는다.

Jump의 train-only normalization을 적합하는 prepass는 공통 minibatch stream을
소비하지 않도록 별도 provider를 쓰거나 state를 복원한다. Fold 전후 native inference와
원 단위 reconstruction, 같은 main cell의 첫 batch 및 전체 샘플 순서를 확인한다.
Signed matching, hard support, selector 입력의 source/target truth 분리,
no-feasible 상태, source-winner 포함, final-T C2 비교는 작은 수치/회귀 검사 대상이다.
B1/B2 공유 fit은 `main_confirmation`의 같은 실제 state를 참조한다.

50-feature 방향이 충분히 orthogonal한지 검증하기 전에 작은 toy의 에너지 분해식을
그대로 쓰지 않는다. Signed/absolute cosine은 유지할 수 있지만 nonorthogonal 또는
unequal-width source에서는 해당 source에 맞는 projection/coverage 지표가 필요하다.
현재 shared helper의 equal-width/orthonormal 가드를 무시하지 않는다.

## 산출물과 중단 해석

표 1은 selector별 paired dictionary/support/EV/activity/cost와 coverage를 함께 싣는다.
그림 1은 native L0에 따른 실제 recovery curve, 그림 2는 precision별 same-gamma
trajectory와 signed/absolute geometry 분해다. B3 결과는 통과 시 범위 확장 표가 된다.
Gamma adaptation, exact32, 추가 modules, 폭 전 범위 sweep은 appendix 후보 목록에만
남기고 must-run에 끼워 넣지 않는다.

Baseline의 coverage 또는 수렴 실패는 해당 비교의 미해결이다. Recovery가 oracle에만
있으면 usable selector는 미해결이다. Precision의 이론적 하한만 확인되면 C2의 empirical
부분은 미확인이다. 정확한 dictionary와 낮은 F1가 공존하면 decoder proxy의 한계를
보고한다. 이런 결과를 VG-SAE 개발을 중단하고 별도 진단 주제로 바꾸라는 판정으로
사용하지 않는다. 새로운 방법의 유용성을 확보하는 데 필요한 다음 최소 변경을 고른다.

근거: `RESEARCH_BRIEF.md`; 이번 run의 `round-2-refinement.md`;
`outputs/sbw_20260922/{baseline/interpretation.md,prior/findings.md,joint/RESULTS.md}`;
이번 run의 `evidence/ref_paper_analysis.md` 및 preliminary scientific review.
