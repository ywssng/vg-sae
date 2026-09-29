# VG-SAE: 변분 선택모형에서 희소 밀도 추정까지

작성: 2026-09-29. 상태: 계획 초안, Phase 2 미실행. 원래 원고와 2026-09-24 Phase 1을 한 논문으로 연결한다. Phase 1/2는 연구 단계이며, 기존 Stage 1/2/3는 데이터 검증 규모이므로 구분한다.

## Problem Anchor

- 연구 문제: Variational Garrote 기반 SAE를 명시한 통계물리적 모형과 변분원리에서 유도·이해하고, 그 모형의 밀도 응답과 독립 재학습 간 선택 불확실성으로 적절한 sparse operating density를 추정할 수 있는지 검증한다.
- 핵심 병목: 재구성 오차와 sparsity만으로 올바른 feature 분해나 밀도를 선택할 수 없다. 한 모델 내부의 posterior uncertainty, 모델 간 선택 불안정성, 실제 feature recovery 사이의 연결은 별도로 검증해야 한다.
- 유지할 목표: VG-SAE 자체의 개발과 이해를 중심에 둔다. Phase 1은 모형·추론·목적함수의 근거, Phase 2는 정답을 보지 않는 밀도 추정과 그 유용성 검증이다. Phase 1 성공이 Phase 2 성공을 함의한다고 가정하지 않는다.
- 비목표: 별도 복제 진단 주제로 전환, 모든 SAE를 VG의 특수형으로 선언, SOTA를 유일한 성공 조건으로 강제, 파라미터 없는 보편적 true-L0 발견, semantic posterior calibration 또는 thermodynamic phase transition의 무근거 주장.
- 제약: 이번 요청은 계획 작성·검토다. 새 학습·GPU 실험은 실행하지 않는다. 기존 결과와 원고는 보존한다. 후속 비용은 제안이며 9월 24일의 2 GPUh 승인 범위를 확장한 것으로 취급하지 않는다.
- 완료 기준: 원래 밀도 추정 목표를 포함한 하나의 논문 서사, 실행 가능한 추정 절차, 주장별 실험·반증 기준, Phase 1의 실제 근거와 Phase 2의 미검증 가설을 구분한 계획을 제공한다.

## 하나의 논문 주장

**논문 가설:** VG의 명시적 선택모형에서 서로 다른 불확실성을 구분하고, 독립 재학습의 선택 곡선을 이용하면 일부 명시된 조건에서 feature recovery에 유용한 밀도를 정답 없이 추정할 수 있다. 그 적용 조건과 실패도 함께 밝힌다.

- C1(기반 설명): 조건부 자유에너지의 항과 posterior 근사오차를 구분하면 선택확률·밀도 신호가 무엇을 뜻하는지 설명할 수 있다. Phase 1의 제한적 실제 근거가 존재한다.
- C2(주된 새 검증): 정답을 보지 않는 선택 불확실성 곡선 추정기가 지정 synthetic 조건에서 생성 밀도를 추정하고, 별도로 recovery가 좋은 operating point를 선택하는가? 아직 결과가 없다.
- C2a 생성 밀도 오차와 C2b recovery regret는 서로 다른 endpoint다. 어느 하나가 실패했다고 다른 것으로 사후 교체하지 않는다.
- 알려진 free-energy identity, variance decomposition, Soh의 곡선 자체를 새로운 정리로 주장하지 않는다. 학습되는 SAE의 feature 좌표에 이 추정 논리를 옮길 조건·절차·실패 범위가 추가 연구 대상이다.

## Phase 1에서 가져오는 것

원본은 `refine-logs/runs/vg-sae-first-principles-20260924/`의 plan/results이고, config는 `configs/first_principles_20260924.json`이다. 135 exact cells, 9 frozen fits, 54 joint fits의 기존 결과를 재사용하되 새 seed로 세지 않는다.

1. 고정 amplitude=1 모형: orthogonal 조건의 exact=MF, overlap에 따른 posterior dependence와 factorization gap.
2. 같은 target의 encoder/MF/exact 비교: 유한 학습의 추가 오차. Orthogonal 잔여 오차를 구조적 표현 한계로 부르지 않는다.
3. joint SAE: variance·entropy·global learned vs minibatch profiled beta가 mean/stochastic/hard readout에 미치는 역할.
4. 한계: F/reverse-KL 개선과 Brier/marginal accuracy 개선은 같지 않다. Variance 삭제 결과는 beta도 반응하는 total effect다.

### 두 단계를 잇는 수식 설명

정렬된 입력별 gate 확률을 m[r,x,j]라 쓰면, finite ensemble의 평균에 대해 다음은 대수적 항등식이다.

`mean_xj mbar(1-mbar) = mean_rxj m(1-m) + mean_xj Var_r(m)`.

좌변의 across-realization 평균을 취한 soft selection uncertainty, 첫 항의 within-model uncertainty, 둘째 항의 between-fit variation을 나눠 기록한다. 이 식 자체는 새 정리가 아니며 semantic correctness를 보장하지 않는다. Hard masks의 between-fit 불일치도 별도 지표다.

고정 x,a,D,beta의 field response 항등식과 gamma별로 D,a,encoder까지 재학습한 곡선의 도함수는 구분한다. 후자에는 학습된 파라미터의 변화가 섞이므로 Phase 1 응답식을 그대로 적용하지 않는다.

## Phase 2의 추정 대상

1. `L_gen = sum_j p_j`, `rho_gen=L_gen/K`: 생성 dictionary 폭과 학습 폭이 같은 synthetic에서 정의된 expected generating count. Primary family는 constant amplitude, 낮은 noise로 작은 amplitude의 관측 불가능성을 줄인다. 모델 자체의 식별성을 증명했다고 하지는 않는다.
2. `L_oper`: 고정 SAE family/폭/학습 예산에서 recovery가 좋은 hard-code count. Test에서만 계산하는 coefficient NMSE oracle와 비교하여 regret를 구한다. Selector는 이 oracle를 보지 않는다.
3. SynthSAEBench처럼 K_true>K_SAE인 경우와 real LM에서는 `L_gen=correct SAE L0`를 주장하지 않는다. Capacity-conditional recovery/utility로 범위를 바꾼다는 규칙을 미리 정한다.

## 최소 방법: 기존 VG-SAE + 학습 후 곡선 추정기

새 neural component는 없다. Primary는 global learned beta VG-SAE, native hard gate `m>0.5`, 고정 final checkpoint다. Profiled beta는 Phase 1과 연결하는 ablation으로 남긴다. Gamma는 외부 sweep이며 sparsity estimator를 parameter-free라고 부르지 않는다.

### 입력과 데이터 역할

- world w: 생성 dictionary와 train/align/select/test 데이터. 모든 방법이 같은 world를 공유한다.
- repeat r: 같은 world의 같은 training samples에서 initialization과 minibatch order만 독립. 같은 r의 gamma ladder는 초기 상태와 batch stream을 공유한다.
- `X_train`: SAE 학습, `X_align`: feature 정렬 보조·native density calibration, `X_select`: 불확실성 곡선과 selector, `X_test`: 한 번의 최종 평가. 모든 입력 split 독립.
- truth dictionary/support/amplitude는 selector 함수의 인자가 아니다. Phase 2의 ground-truth 접근은 dev 단계와 최종 evaluator에만 허용한다.
- 독립 world가 일반화 평가의 단위다. Gamma 점, feature, input sample, optimization repeat를 독립 world 수로 세지 않는다.

### 곡선 구성 초안

각 gamma에서 R개의 학습된 decoder를 signed cosine Hungarian으로 공통 medoid reference에 정렬한다. 비음수 code에서는 sign flip이 동치가 아니므로 primary는 absolute cosine으로 sign을 지우지 않는다. Reference는 pairwise matching cost의 합이 가장 작은 run으로, 동률은 사전 run ID 순서로 정한다. 정답 dictionary는 쓰지 않는다. 각 gamma의 평균은 feature permutation에 불변이므로 gamma 사이 동일 reference를 강제하지 않는다.

Native hard mask `h[r,x,j]=1[m>0.5]`를 정렬하고 다음을 계산한다.

`rho_g = mean_rxj h[r,x,j]`

`U_g = R/(R-1) * mean_xj hbar[x,j]*(1-hbar[x,j])`.

유한 R correction은 feature correspondence가 고정이라는 조건에서 seed 분산의 보정이며, 추정한 alignment의 오차를 없애지 않는다. Alignment는 bootstrap마다 다시 계산한다. Soft total/within/between 지표도 저장한다.

동일 gamma에서 density가 너무 다른 run을 비교하면 density mismatch를 instability로 셀 수 있다. 따라서 native curve의 repeat density spread를 기록하고, 정답을 보지 않는 common-density checkpoint 선택을 비교한다. 구체적인 primary matching 규칙과 허용오차는 dev에서 고정하고 confirmatory 결과를 보고 바꾸지 않는다.

### 추정과 abstention

원고의 nonnegative mixture를 primary로 바로 사용하지 않는다. 후보 rho* 수가 curve 점 수보다 많으면 mixture가 비식별적일 수 있고 정규화한 NNLS weight는 Bayesian posterior가 아니다.

Primary는 단일 template `U(rho)=A f(rho;q)+b`, A>=0, b>=0를 constrained least squares로 fit한다.

`f(rho;q)=rho/q*(q-rho)` for rho<=q;
`f(rho;q)=(rho-q)*(1-rho)/(1-q)` for rho>q.

q를 사전 후보 grid에서 profile하고, 각 q에서 A,b를 NNLS로 풀어 전역 grid minimum을 얻는다. 반환값은 q_hat, L_hat=K*q_hat, fit residual, seed/bootstrap sensitivity와 abstention reason이다. NNLS mixture 및 원래 argmax-weight 추정기는 attribution/identifiability sensitivity로만 비교한다.

Endpoint(all-off/all-on)의 낮은 U를 정답으로 선택하지 않는다. 최소 7개 서로 다른 interior density 점, q 양쪽 최소 3점, 충분한 density coverage, 음성이 아닌 signal amplitude와 residual, alignment/repeat 안정성을 확인한다. 식별 불가·flat curve·single-sided coverage면 숫자를 강제로 출력하지 않는다. Threshold는 dev worlds에서 고정하고 test에서 조정하지 않는다. Stable-but-wrong representation은 모든 진단을 통과할 수도 있으므로 abstention을 correctness certificate라고 부르지 않는다.

Bootstrap은 repetitions를 cluster 단위로 재표집하고 alignment/curve fit를 재계산한다. Curve 점별 독립 bootstrap은 금지한다. World-level 성능 구간은 worlds를 재표집한다. Bootstrap 출력은 sampling/optimization sensitivity이며 density의 Bayesian credible interval이 아니다.

### Density를 실제 SAE 선택으로 변환

각 repeat에서 `X_select`의 native achieved rho가 q_hat에 가장 가까운 이미 학습된 checkpoint를 선택한다. 동률은 sparse 쪽, 이후 gamma/run ID 순서로 고정한다. 새로운 threshold나 보간 code를 만들지 않는다. Quantization error와 deployment density/test drift를 별도 보고한다. 모든 repeat의 선택된 checkpoint를 평가하여 가장 좋은 seed를 고르지 않는다.

## 비교군과 실패를 구분하는 분석

- 같은 VG checkpoint bank의 selector 비교가 primary: uncertainty fit, min held-out hard reconstruction MSE, min decoder-pair cosine(c_dec), raw interior min-U, uniform random control(expected result over finite bank). Reconstruction-only가 dense 끝점을 택하는 것도 그대로 결과다.
- 같은 bank에서 truth를 사용하는 coefficient-NMSE oracle는 evaluator 전용이다. 선택 regret는 oracle 대비의 차이이며 개선이 반드시 positive여야 한다고 강제하지 않는다.
- TopK는 separate training family로서 같은 training-step/token budget의 comparator이고 U selector를 같은 방식으로 적용한다. Target K 자체를 true count oracle로 고르지 않는다. 추가 L1과 BatchTopK는 확장 단계의 선택적 baseline이다.
- Phase 1 연결 ablation: global learned vs profiled beta, variance 삭제, soft within uncertainty만 사용, truth alignment oracle, fixed-density nuisance control. 삭제와 parameter 반응은 total effect로 해석한다.
- 실패: permutation-only, all-off/all-on, duplicate atoms, independent random masks with same marginal density, stable but wrong/mixed atoms. 이들은 지표 단위 검증·failure diagnosis이며 별도 연구로 목표를 대체하지 않는다.

## 실험의 큰 세 블록

1. **기반 설명(Phase 1 재사용):** B1–B4와 두 불확실성의 수학적 연결. 수식 재현과 경험적 설명을 구분한다.
2. **밀도 추정의 핵심 검증(Phase 2 synthetic):** dev/calibration 이후 freeze; 서로 다른 generating density의 fresh worlds에서 생성 밀도 오차와 recovery regret를 별도 평가. primary estimator와 같은 bank selector, TopK 및 소수 ablation을 비교한다.
3. **적용 범위(Phase 2 transfer):** 작은 controlled exponential/skew/overlap 확장 후, 사전 gate를 만족할 때 고정 SynthSAEBench artifact와 real LM으로 확대한다. 현실 데이터의 true-L0는 주장하지 않는다.

### 제안 설정과 비용

Primary generator: K_true=K_SAE=16, d=16 orthogonal 또는 d=8 random unit dictionary, independent homogeneous support p in {1/16,2/16,4/16}, constant amplitude1, noise std .05. 각 family×density에 8 fresh worlds와 R=6 optimization repeats를 제안한다. Train8192/align2048/select4096/test8192, lr .003, batch256, 4000 updates, beta initial400, no input rescaling/weight decay, clip1. Controls는 dev에서 native density coverage를 보고 method별 고정한다. 모든 primary fits는 final checkpoint를 사용한다.

개발 단계는 각 family에 2 worlds, p=3/16, R=4, 최대17 controls. Confirmatory는 최대 17 controls에서 global learned VG와 TopK를 비교한다. 대략 10k 미만의 작은 fits 범위이나 실제 수량은 experiment plan에서 산술로 확정한다. 9/24 tiny run의 속도를 더 큰 설정으로 무조건 환산하지 않는다. 첫 timing 이후 GPUh를 다시 산정하며, 이번에는 실행하지 않는다. 확장 benchmark는 별도 conditional budget이다.

### 사전 성공 기준 초안

C2a: all scheduled confirmatory worlds를 denominator로 하여 report-rate>=80%, accepted estimates의 median relative count error<=20%, all-world within-20%-and-reported rate>=70%. 실패·abstention과 conditional accuracy를 함께 보고한다. Seed sensitivity 구간의 coverage와 폭도 보고하며 높은 abstention으로 accuracy를 꾸미지 않는다.

C2b: 같은 bank의 reconstruction-only와 c_dec 대비 paired world-level coefficient-NMSE regret 감소를 검증한다. Bootstrap 95% interval이0을 넘지 않는 방향과 사전 최소 effect를 함께 요구한다. 생성 밀도가 틀려도 recovery가 좋을 수 있고 그 반대도 있으므로 둘을 분리한다. 8 worlds/cell은 예비 확증 규모이며 power를 주장하지 않는다.

확장 전에 primary의 실패를 해석하고, unseen test에 맞춰 q grid나 fit family를 고쳐 같은 결과를 확증으로 재사용하지 않는다. Negative 결과도 원래 추정 가설에 대한 결과로 남긴다.

## 논문 구성 초안

작업 제목: **Variational Garrote Sparse Autoencoders: From Selection Free Energy to Density Inference**. 원래 **L0 Is Not a Free Parameter**는 C2 결과가 뒷받침될 때 유지할 수 있는 제목 후보다. 투고 venue는 확정하지 않는다. 기존 PRE 스타일 원고를 존중하고 임의로 ICLR로 바꾸지 않는다. 내부 구성 예산은 본문10쪽(공식 venue 제한 아님), references/appendix 별도.

1. Introduction1.25쪽: reconstruction으로 density를 선택할 수 없는 문제와 VG 접근. 하나의 질문, 두 검증 단계.
2. Related work1쪽: VG/Sparse but Wrong, variational sparse coding, stability와 feature alignment. 기존 이론과 본 작업을 분리.
3. Model and inference1.75쪽: conditional free energy, exact/MF/encoder, 두 uncertainty와 assumptions. 새 정리 여부를 구별.
4. Density inference1.75쪽: ensemble, truth-free alignment, achieved density, template fit, deployment와 failure conditions.
5. Experiments3쪽: Phase 1 근거 요약, Phase 2 main table/hero curve, 필요한 ablation/transfer.
6. Discussion and limitations.75쪽: stable-wrong, capacity, amplitude/noise, optimization and failed transfer.
7. Conclusion.5쪽: 실제 C1/C2 결과 범위에 맞춘 결론.

Hero figure는 같은 synthetic world에서 achieved hard density를 x축으로 U curve와 fitted q를 표시하고, 별도 panel에 recovery/MSE curves를 보인다. Truth line과 recovery optimum은 evaluator overlay이며 selector input이 아니다. 대표 예시를 test-best로 고르지 않고 사전 지정 world와 전체 world 요약을 함께 보여준다. Negative/abstaining example도 포함한다. Phase 2 수치는 아직 없으므로 숫자를 채우지 않는다.

## 구현 인계의 빈틈

- 기존 `src/evaluate.py`는 piecewise kernel와 NNLS를 제공하지만, normalization된 weights를 posterior로 부르는 현재 출력을 primary interval로 재사용하지 않는다.
- `src/sae_sweep_eval.py`의 `paper_style_sigma_sel`은 input axis를 평균한다. 이것은 fixed input의 cross-run instability가 아니므로 historical field를 Phase 2 evidence로 사용하지 않는다.
- 새 `[world,repeat,control,input,feature]` schema, truth-free matcher, native density matching, abstaining single-template fitter, independent selector/evaluator, run manifest가 필요하다. 이번 계획 작업에서는 구현하지 않는다.
- 공개된 Phase 1 결과는 초기 설명 근거다. 이후 test plan은 기존 pilot에 맞춘 calibration이며 새 confirmatory worlds가 필요하다.

## 알려진 위험과 판단 원칙

Soh의 exchangeability/no-FP 또는 no-FN 가정은 SAE에 자동 성립하지 않는다. Template transfer가 최우선 미검증 가정이다. Stable-but-wrong은 U가 낮아도 feature가 틀릴 수 있는 반례다. 따라서 성능 우위를 강제하거나 실패 시 다른 주제로 바꾸지 않고, 어떤 조건에서 추정이 가능했는지와 실패했는지를 같은 논문의 결과로 보고한다. Known identity 확인만 남으면 새 기여가 완성됐다고 주장하지 않는다.
