# Phase 1–2 통합 실험 계획

2026-09-30 문헌·토론 반영판. **계획만 작성. Phase 2 학습·평가 결과는 없다.** 아래 수치 기준은 제안된 실행 프로토콜이며, 관측된 성능이나 power 계산이 아니다. Phase 1 원본 계획·결과는 `refine-logs/runs/vg-sae-first-principles-20260924/`에 보존한다.

## 이번 개정의 우선순위

기존7,240–10,376 fits를 최초 필수 묶음으로 실행하지 않는다. Phase 1 물리 설명과0-fit 대조 → 두 teacher-world의238–350 fits bridge → 나머지 개발 및 protocol lock → 조건부36-world 본검증으로 순서를 바꾼다. Core 연구 질문은 유지하고 estimator가 측정하는 ensemble과 observable를 먼저 확인한다.

현재 operational primary는 `E_opt + rho_hard + U_hard`다. 원래 VG 논문의 data-realization/soft-total과 같다고 부르지 않는다. `E_joint`와 soft 관측량은 사전 지정 challenger이며 모든 결과를 보존한다. 어떤 조합이 좋은지 아직 모르고, confirmatory 결과를 보고 primary를 바꾸지 않는다.

판정·방법론 근거는 `refine-logs/runs/vg-sae-physics-literature-20260930/PI_SYNTHESIS.md`, 상세 수식은 같은 run의 `PHASE1_PHYSICS_BRIDGE.md`를 따른다. 9월29일 버전과 과거review는 해당run에 보존되며 당시READY 점수를 새설계의 검증으로 재사용하지 않는다.

## 연구 목표와 주장

VG-SAE의 선택 자유에너지와 추론 근사를 이해하는 Phase 1을 바탕으로, 독립 재학습 간 선택 불확실성에서 sparse operating density를 추정하는 Phase 2를 검증한다. 연구 단계 Phase와 데이터 규모 Stage를 혼동하지 않는다.

| 논문 주장 | 최소 근거 | 상태 | 블록 |
|---|---|---|---|
| C1: 조건부 모형에서 objective 항과 exact/MF/encoder 오차의 역할을 구분할 수 있다 | 기존 exact·frozen·joint 결과와 명시적 모형 가정 | 지정 조건의 초기 근거 있음 | P1 |
| C2a: truth-free uncertainty curve가 지정 matched-capacity synthetic의 expected generating density를 추정한다 | fresh worlds의 오차·보고율·실패율, 여러 generating densities | 미검증 가설 | P2-A/B |
| C2b: 그 추정으로 선택한 SAE가 recovery가 좋은 operating point에 도달한다 | 같은 bank의 selector 비교, all-world policy 결과와 oracle regret | 미검증 가설, C2a와 별개 | P2-B/C |

C1은 기반 설명, C2는 주된 후속 검증이다. 모든 SAE 통합, universal feature count, semantic confidence, parameter-free 추정, SOTA를 주장하지 않는다. Known identity나 unit test 통과는 신규성 증거가 아니다.

## P1 — 기존 근거와 두 단계의 연결 / 본문, 재사용

- 기존 B1 audit, B2 exact135 cells, B3 frozen9 fits, B4 joint54 fits를 원래 seed/config와 함께 재사용한다. 이번에 재실행하지 않았다.
- Figure: exact/MF/encoder 차이 + mean/stochastic/hard risk 비교. 3 worlds는 초기 evidence이고 새 confirmatory worlds에 합치지 않는다.
- 유지할 해석: 높은 overlap에서 F 감소와 marginal accuracy 개선은 일치하지 않을 수 있다. Variance 삭제는 beta 반응을 포함한 total effect다.
- 추가할 이론 설명: `mean mbar(1-mbar) = mean m(1-m) + mean Var_r(m)`의 finite-ensemble 항등식. 각 평균 축과 추정한 feature correspondence 조건을 명시한다. 새 정리가 아니다.
- 고정 모형의 field derivative와 gamma마다 다시 학습한 SAE response는 같다고 가정하지 않는다.
- Phase 1에서 사용하는 exact conditional posterior를 Phase 2의 생성 support 정답 또는 semantic label로 대체하지 않는다.

### Phase 1 추가 해석과 최소 대조

- Fixed-support energy의 off-diagonal Gram interaction, exact count covariance, stable MF Hessian response를 명시한다. 기존 exact field-response 검증을 새 결과로 세지 않는다.
- Fixed-model ambient orthogonal rotation을0-fit 대조로 추가한다. 입력·bias·decoder와 필요한encoder까지 함수 보존변환하고 Adam 재학습 invariance는 주장하지 않는다.
- 기존 orthogonal frozen gate의 analytic logits를 parameter witness로 확인한다. 추가 continuation이 필요하면3fits×최대6000updates=18000updates를 별도계상하며 blanket retraining하지 않는다.
- Geometry×support co-firing의 matched/mismatched prior24-target oracle는 SHOULD 후속이다. 새prior 구현을 density pilot 착수 gate로 강제하지 않는다. Correlated prior가 독립성에 미치는 영향은 해당target에서 검증한다.
- PR3는 amplitude stress의 보조 concentration 지표이며hard L0나semantic compositionality를 대신하지 않는다.

## P2-A — 추정기의 정의·실패 검사·개발 calibration / MUST

### A0. 학습 전 작은 수치·계약 검사

1. `[repeat,input,feature]` 축을 명시한 mask tensor를 사용한다. 두 world를 같은 repeat 축에 섞으면 오류를 내도록 구현한다.
2. 동일 decoder/code에 순열을 적용한 경우 수치상 같은 curve가 나와야 한다(tie 없는 test). Sign flip은 비음수 amplitude의 대칭이 아니므로 보존하지 않는다.
3. All-off/all-on, stable-wrong와 **stable-correct deterministic hard masks**를 모두 넣는다. 같은 H가 반복되면 U_hard는0이고 추정기는 보류한다. 이는 correctness 실패판정이 아니다. 동일soft m이 반복되어도 soft-total에는 m(1-m)가 남을 수 있으므로 같은 반례를 soft-total 전체에 적용하지 않는다.
4. 같은 marginal density의 독립 random masks, 일부 atom의 복제, stable mixed dictionary를 포함한다. 이들은 지표의 한계 검사이며 모든 잘못된 representation을 abstention으로 탐지할 수 있다고 주장하지 않는다.
5. 원래 template의 no-FP/교환가능 under-selection, no-FN/교환가능 over-selection mask generator에서 곡선과 fitting을 검사한다. 위 가정을 깨는 unequal feature frequency와 input별 support count 변화도 검사한다. Ideal test 통과는 실제 SAE 성능이 아니다.
6. `E_x f(rho_x;q_x)`와 `f(E_x rho_x;E_x q_x)`가 일반적으로 다름을 예시와 식으로 명시한다. 원래 global variable-selection template를 input-dependent SAE에 옮기는 것은 검증 대상이다.
7. NNLS mixture의 비유일성/flat profile를 검사한다. weight 합이0이면 uniform 'posterior'를 출력하는 기존 fallback을 새 estimator에 쓰지 않는다.

### A1. 데이터·반복·split 계약

- **teacher_world_id**: D와 생성분포, 공통align/select/test streams를 고정한 상위 독립 단위. **training_dataset_id**와 **optimizer_repeat_id**를 그 아래 별도로 둔다. E_opt는 dataset0을 공유하고 E_joint는 repeat별 다른dataset을 쓴다. 같은 teacher의 dataset draws를 독립 world n에 더하지 않는다.
- **reference run**: repeat0. 독립 initialization/batch stream으로 학습하며 alignment anchor로만 사용한다.
- **ensemble repeats**: confirmatory는 repeat1–5(R=5), reference 제외. 개발은 repeat1–3(R=3). 매 world 총6개 또는4개 학습 궤적이다.
- 현재primary E_opt는 teacher별 모든repeats가 dataset0의 같은training examples를 사용한다. 바꾸는 것은 initialization과 batch order다. Bridge의 E_joint만 독립dataset1–3을 쓰며 아래BR 계약을 따른다. 같은 repeat의 controls는 초기 parameter state와 batch stream을 공유하고 각각 처음부터 학습한다. Warm start와 gamma continuation은 primary에서 쓰지 않는다.
- Train8192 / align2048 / select4096 / test8192. `align`은 native density matching과 feature reference 진단, `select`는 U curve와 모든 label-free selector, `test`는 최종 평가에 쓴다. Input splits는 독립 RNG다.
- Truth dictionary/support/amplitude와 generating p는 selector API에서 제외한다. 개발 world truth는 설계 진단에만 사용하고 confirmatory truth는 evaluator가 selector output을 저장한 뒤 연다.
- World IDs: 개발31000–31007(2 dictionary families×2 densities×2 worlds), 본검증32000–32035(2 families×3 densities×6 worlds). 순서는 family O/R, p 오름차순, 개발world0–1/본검증world0–5다. Mechanism은 본검증의 p=.125인 각 family 첫3worlds(32006–32008,32024–32026)를 사전에 선택해 재사용하며 새로운 독립 world로 세지 않는다. 아래의 data/optimizer별 seed 규칙을 사용하고 전체 mapping을 저장한다.
- Data seed는 `SeedSequence([20260930,teacher_world_id,data_role_id,training_dataset_id])`, optimizer seed는 `SeedSequence([20260930,teacher_world_id,optimizer_role_id,optimizer_repeat_id])`로 분리한다. Arm이나 control을 optimizer seed에 넣지 않아 paired initialization을 유지하며 모든 control에서각repeat의dataset을 고정한다. Control을 seed에 넣지 않아 같은 repeat의 parameter/batch pairing을 유지한다. Method가 서로 다른 구조이면 동일 parameter state를 주장하지 않고 동일 data/compute recipe만 보장한다.

### A2. Primary 생성모형과 학습

- K_true=K_SAE=16. 두 family: (O) d16 orthonormal D(QR, 부호 convention 고정), (R) d8 independent random unit atoms. Random D는 orthogonal이나 identifiability 보장 모형이 아니다.
- 독립 homogeneous Bernoulli support. 개발 p={.09375,.1875}, 본검증 p={.0625,.125,.25}; 개발 expected count={1.5,3}, 본검증={1,2,4}. 한 정답을 항상 출력하는 추정기를 걸러낸다.
- Primary amplitude=1, noise std=.05. 이 모형에서도 generative support의 완전 식별성은 가정하지 않는다. 작고 약한 exponential activity의 관측 한계는 P2-C에서 별도로 다룬다.
- Primary VG: global learned beta, initial400, native hard gate m>.5, lr.003, Adam, batch256, no weight decay/input scaling, clip1. Main의 제안 fixed horizon H_star=4000은 bridge 안정성 점검 뒤 잠근다. Bridge만 전 fit8k 및2k/4k/8k snapshots를 사용한다. Main에서snapshot을test-best로 선택하지 않는다. H_star 변경은 새로운version/예산 및 untouched 확인을 요구한다.
- Basic gamma grid17개: `[-4,-2,-1,0,.5,1,1.5,2,2.5,3,3.5,4,5,6,8,10,12]`.
- 모든 method와 control에서 고정 final budget를 사용한다. 좋은 test 결과를 얻은 fit만 연장하거나 고르지 않는다.

### A3. Native hard-density matching — primary 규칙

Target density grid는 `[.02,.03,.045,.0675,.10,.15,.225,.3375,.50625,.675,.85]`의11점이다. Generating p에 맞춰 target을 옮기지 않는다.

각 run에서 align split의 native hard rho가 target t에 가장 가까운 checkpoint를 고른다. 허용차는 `delta(t)=min(.015,max(.003,.05*t))`. 동률은 achieved rho가 작은 쪽, 이후 control/run ID 순서다. Reference와 모든 R repeats가 허용차를 만족한 target만 curve에 사용하며, 제외된 target·부족 coverage도 기록한다. Mask interpolation, feature threshold 변경, 참값으로 candidate를 고르는 동작은 없다.

Density coverage를 위한 사전 정의된 label-free 추가 학습은 run당 최대 8개 controls다. 각 반복에서 미충족 target의 `거리/delta`가 가장 큰 것을 고른다(동률 target 오름차순). 기존 gamma 순으로 인접하고 target을 bracket하는 쌍 중 gamma 간격이 가장 작은 쌍의 중간값을 추가한다(쌍이 동률이면 작은 gamma 쌍). Bracket이 없으면 그 target은 이 pass에서 추가 불가로 표시하고 다음 target을 본다. 이미 시도한 control/동일 midpoint는 재시도하지 않는다. 한 pass에서 추가 가능한 새 midpoint가 없으면 즉시 종료한다. 새 control을 실제 학습했을 때만 budget을1차감하고 다음 pass에서 모든 미충족 target을 재검사한다. 최대25개가 되면 중단한다. 범위 밖 extrapolation은 없다. Nonmonotonic density를 정렬해 감추지 않는다. 각 fit마다 dev/confirm 모두 같은 adaptive rule을 적용하고 truth/test는 보지 않는다.

공통 target이7개 미만이면 `insufficient_coverage`. 모든 비교 selector에는 실제 학습한 동일 bank 전체를 제공한다. Adaptive coverage overhead도 compute에 포함한다.

### A4. Truth-free feature correspondence

각 retained target에서 reference의 unit decoder와 각 repeat의 decoder를 signed cosine Hungarian으로 일대일 정렬한다. 모든16개 슬롯과 dead features를 유지하며 matching이 나쁜 슬롯을 골라 버리지 않는다. Absolute cosine matching과 sign flip은 primary에서 사용하지 않는다. 동일 총cosine의 assignment가 여러 개면 original column-index 순 lexicographically smallest assignment를 선택한다. Tie가 없는 permutation test와 tie/duplicate sensitivity는 구분한다.

Norm convention: decoder norm을 float64로 계산하여 n_j>0이면 unit direction d_j/n_j와 effective coefficient n_j*z_j를 써서 reconstruction을 보존한다. 정확히0인 atom은 direction0, matching cosine0, effective coefficient0으로 둔다(0으로 나누지 않는다). 0<n_j<=1e-12인 near-zero atom도 정규화하되 그 비율과 matching 민감도를 별도 보고한다. Native hard masks/count는 export된 모델의 실제 inference에서 별도로 측정해 zero-atom normalization 때문에 조용히 바꾸지 않는다. Recovery NMSE에는 effective coefficients를 쓴다. VG의 unit-decoder constraint 위반이나 nonfinite 값은 구현/학습 failure로 검사한다.

Geometry matching에는 ground truth가 들어가지 않는다. 각 target에서 따로 reference를 선택하되 reference run ID는 항상0이다. Across-control global feature identity는 요구하지 않는다. Mean U는 각 target의 공통 permutation에 불변이다.

Active-slot frequency-weighted matching cosine, negative matches, dead slots, top1–top2 cosine gaps를 저장한다. True-D alignment는 evaluator의 oracle diagnostic일 뿐 primary가 아니다. Main estimate를 반복한 alternative-reference sensitivity에서 q 변화도 저장한다. Subspace reproducibility와 individual feature identity가 다를 수 있으므로 matching 실패를 feature 존재의 부정으로 읽지 않는다.

### A5. 관측량과 추정기

Select split에서 aligned hard masks H[r, x, j]의 mean Hbar를 얻는다.

VG의 `H[r,x,j]=1[m[r,x,j]>.5]`, `z_hard=H*a`, `rho_hard=mean(H)`를 명시적으로 사용한다. 이는 gate의 hard occupancy다. Floating-point underflow나 작은 amplitude로 생기는 epsilon-based effective-code L0와 별도로 기록하며 서로 대체하지 않는다. L1/TopK의 native counts도 각 모델의 실제 추론 정의를 명시한다.

`rho_t = mean_rxj H`, `U_hard(t)=R/(R-1)*mean_xj Hbar*(1-Hbar)`.

Finite-R correction은 고정 correspondence 조건에서만 분산 보정 의미를 갖는다. 학습된 alignment의 오차까지 제거하지 않는다. Soft gate의 total/within/between decomposition도 **보정 없는 finite-ensemble identity**로 따로 저장한다. U_hard와 soft total의 식을 섞지 않는다.

Actual select-split rho를 x축으로 사용한다. Reference 및 각ensemble repeat별로 align/selection native density를 대조하고, 하나라도 drift의 절댓값이 `2*delta(t)`를 넘는 target은 `density_drift`로 제외한다(평균drift만으로 서로 상쇄하지 않는다). 남은 distinct points가7개 미만이면 보류한다. Pairwise target rho 차이가1e-4 미만이면 작은 target 하나만 남기고 제외 기록을 쓴다.

Primary fit: `U_hard(rho)=A*f(rho;q)+b`, A>=0, b>=0. q 후보는 `.020,.021,...,.800`. 각 q에서 two-column `[f,1]` constrained LS를 풀고 unweighted SSE를 profile한다. q 양쪽에 최소3개의 실제 curve points가 있는 후보만 허용한다. Pointwise error가 독립이라고 가정한 likelihood를 사용하지 않는다. Exact tie는 더 작은 q를 반환하지만 tie/profile spread는 아래 보류 기준에 포함한다.

`f(rho;q)=rho/q*(q-rho)` for rho<=q, `(rho-q)*(1-rho)/(1-q)` otherwise. 전체 nondegenerate curve를 맞추며 raw minimum을 선택하지 않는다.

다음 중 하나라도 해당하면 추정값 대신 `abstain`과 reason을 출력한다.

1. 유효 common points<7 또는 양쪽3점 조건을 만족하는 q가 없음.
2. `max(U)-min(U)<.01`, A<=1e-6, 또는 SST<=1e-10: flat/zero signal.
3. SSE_min/SST>.25, 또는 q_hat가 admissible q grid의 첫/마지막 점: single-template fit 부적합 또는 경계 해.
4. Near-optimal set `SSE(q)<=SSE_min+.05*SST`의 폭이 `max(.04,.5*q_hat)`보다 크거나, 후보 grid에서 두 개 이상의 분리된 component: shape 비식별성.
5. 아래 bootstrap의 valid refits<80%, 또는 90% stability-band 폭이 `max(.05,.75*q_hat)`보다 큼.
6. 사전 대체 reference1을 사용하고 원래 reference0을 ensemble로 넣은 민감도 fit이 유효할 때 |q_alt−q_hat|>.05. 대체 reference fit가 불가능하면 `alignment_sensitivity_unresolved`로 보류한다.

이 thresholds는 계획상의 선택이며 자연법칙이나 검증된 calibration 값이 아니다. 개발에서 실행 불가능한 것으로 드러나면 계획 버전을 수정하고 **confirmatory seeds를 열기 전에** 다시 고정한다. Confirmatory 결과를 보고 바꾼 버전은 새 연구 라운드와 새 worlds를 요구한다.

Bootstrap B=200: reference는 고정하고 R repeat IDs를 cluster resample하여 같은 repeats의 전 control bank를 함께 재표집한다. Align/matching/density mapping/profile를 재계산한다. 서로 다른 repeat가3개 미만인 draw는 invalid로 기록한다. Bootstrap refit와 alternative-reference fit의 유효성에는 조건1–4만 적용한다(bootstrap 안에서 bootstrap을 재귀 실행하지 않는다). Input 표본은 primary bootstrap에서 고정하고, input-resampling sensitivity는 보조다. Band는 5–95 percentile의 seed sensitivity이며 calibrated confidence/credible interval이라고 부르지 않는다. 전체world 성능의95% interval은 paired world bootstrap B=2000이다. 개발의 R3에서는 이 bootstrap gate를 확증 gate로 판정하지 않고, R5를 사용하는 본검증에서만 계획된 rule을 적용한다.

### A6. 실제 checkpoint 선택과 비교 정책

추정 성공 시 각 repeat에서 align split의 native rho가 q_hat에 가장 가까운 **실제 final checkpoint**를 선택한다(같은 tie rule). 모든repeat가 `|rho_selected−q_hat|<=delta(q_hat)`이면 `deployment_status=available`, 아니면 `deployment_status=coverage_missing`이다. **이 deployment 판정은 유효한 q_hat나 estimate_status를 지우지 않는다.** C2a는 A5에서 나온 estimate_status로만 평가하고, C2b에서만 bank quantization/deployment 부족을 처리한다. Select split에서 fit한 q를 test에 맞춰 수정하지 않는다. Reference run은 성능 평균에서 제외한다.

**추정 보류와 배포 fallback은 다른 출력이다.** C2a에서는 estimate_status=abstain만 실패/미보고로 계산한다. C2b의 all-world operational policy는 estimate abstention 또는 deployment coverage_missing 시 world 전체에서 각 repeat의 select hard-MSE가 최소인 bank checkpoint를 선택한다. 이를 estimator가 지정한 밀도의 recovery 성공으로 세지 않는다. Estimate-accepted diagnostics, deployment-available diagnostics, 이유별 fallback fraction, all-world policy 결과를 모두 낸다. Control·seed를 test-best로 고르지 않는다.

같은 VG bank의 필수 비교:

| Selector | Selection input/rule | 역할 |
|---|---|---|
| Template estimator + 명시적 fallback | 위 q_hat 또는 select hard-MSE fallback | primary policy |
| Reconstruction-only | 각 repeat의 select hard-MSE 최소; tie sparse/control 순서 | primary paired comparator |
| Decoder-only | 각 repeat의 내부 achieved-rho .02–.85 후보 중 c_dec 최소; 없으면 MSE | 저비용 geometric comparator; SbW의 real-LM elbow와 동등하다고 하지 않음 |
| Raw stability minimum | 같은 retained common target에서 U 최소; tie 낮은 rho; target0개면 MSE fallback | whole-curve fit 필요성 |
| Posterior-uncertainty peak | 같은 target의 mean m(1-m) 최대; tie 낮은 rho; target0개면 MSE fallback | Phase 1 quantity만 쓸 때와의 비교 |
| Uniform random control | 각 repeat bank 모든 후보의 metric 산술평균 | label-free trivial reference, draw 운에 의존하지 않음 |
| Recovery oracle | test coefficient NMSE가 최소인 bank 후보 | evaluator 전용 descriptive attainable bound, deployable selector 아님 |

Oracle는 finite test와 bank에서 최적을 고른 optimistic benchmark다. All-selector의 같은 oracle를 빼는 regret와 direct paired metric 차이를 함께 보고한다. Decoder cosine는 한 geometry 내에서만 비교한다.

## BR — 대형 실행 전 ensemble·observable bridge / 첫 학습 wave

### BR1. 사전 지정된 두 teacher와 공유 reference

- Teacher-world31000,31004: 두 family의 첫 개발world, p=.09375. 같은D·생성분포·align/select/test를 유지한다.
- E_opt: dataset0을 공유하는 reference0+optimizer repeats1–3. Reference는 ensemble 평균에서 제외한다.
- E_joint: 같은reference0 bank를 재사용하고 repeats1–3만 새로 학습한다. 각repeat는 서로 독립인 training_dataset_id1–3, 같은size8192를 사용한다. 각r의initialization/batch-index seed는 E_opt의r와 pairing한다.
- 이 대조는 optimizer-given-data 대 joint-data-and-optimizer다. Initialization pairing만으로 순수 data variance를 추정했다고 하지 않는다.
- Counts: E_opt136/최대200 + E_joint102/최대150 = 총 238/최대350 distinct fits. 모두8k이므로4000-step equivalents476/700. 이 중E_opt136/200은 기존개발bank의부분집합이다.
- Reference는 E_opt와training data를공유하므로 양arm의alignment 조건이완전히대칭이라는보장은없다. 각 arm에서coverage와matching을보고한다.

### BR2. 전체 곡선과 optimization horizon

기본 17gamma를모든 run에서8k까지학습하고2k/4k/8k를저장한다. 같은controls에서horizon별native density mapping과fit를다시계산한다. 전체eligible status와동일retained-target 교집합결과를함께보고한다. 교집합만으로좋은사례를골라전체안정성이라고하지않는다.

Adaptive controls는A3의label-free rule을 **4k native hard-density에서만** 적용하여run당최대 8개고른다. 새 control도8k까지학습하고모든 snapshots를저장한다. Reference bank는한번만만들어공유한다. 이bank를freeze한후2k/4k/8k를평가하며8k의 hole을보고추가 control을붙이지않는다. Basic17이primary horizon check이고extended는4k coverage에조건부로선택된보조결과다. Soft challenger를위한추가 학습은없다.

### BR3. 두 ensemble × 두 readout의사전비교

모든 arm/readout/horizon/basic-or-extended 조합을 저장한다. Shape/profile 진단에A5의조건1–4를적용하되R3에서R5용bootstrap80%gate를적용하지않는다. `pilot_fit_status`와`stage=development`를써서confirmatory estimate_status/coverage와구분한다.

| Readout | Density matching과x축 | y/분해 | 해석 |
|---|---|---|---|
| hard primary | 각model의native hard rho로A3 target/tolerance matching; actualselect rho | corrected R/(R−1)*mean Hbar(1−Hbar) | training procedure가유도한hard selection 변동 |
| soft challenger | 같은target/tolerance를native mean m에별도적용; actualsoft rho | U_total=mean mbar(1−mbar) | within-gate와between-fit를포함 |

Soft의matching/coverage가안되면그상태를보고한다. Hard로맞춘run의soft-U 그림은mixed-axis diagnostic으로별도표시하고soft-density-matched estimator라고부르지않는다. Soft matching뒤그target의reference/model tuple을다시정렬한다.

`U_within=mean_rxj m(1−m)`, `V_between=mean_xj mean_r(m−mbar)^2`, `U_total=U_within+V_between`를finite-ensemble uncorrected identity로기록한다. Hard의finite-R correction을soft-total에그대로곱하지않는다.

현재primary는E_opt+hard다. Challengers(E_joint+hard, E_opt+soft, E_joint+soft)는모두보고하고순위를보고실패한arm을삭제하지않는다. Primary변경은과학적근거와개발결과를공개하고새 version을고정한뒤새 확증에서만평가한다.

### BR4. Shared-reference sensitivity / 추가 training0

기존E_joint repeat1 bank를alternative common reference로쓴다. E_opt ensemble1–3은유지한다. E_joint에서는새reference가된repeat1을빼고기존reference0을넣어R3과서로다른dataset을유지한다. Matching/coverage/q와그차이를모두보고하고유리한anchor를고르지않는다. 이대조는reference와membership이함께바뀌는sensitivity이지순수anchor effect가아니다. 별도의통과gate나추가reference fit을만들지않는다.

### BR5. 진입 gate와 실패 시 행동

1. Measurement/axis/normalization sanity가 실패하면 구현 문제로 기록하고 대형 학습을 시작하지 않는다.
2. Coverage/profile가 부족하면 왜 q가 없었는지 보고한다. Stable-correct와 stable-wrong의 flat-hard를 구분하는 것은 evaluator의 역할이며 U만으로 구분한다고 하지 않는다.
3. Gate의 primary 비교는 **E_opt+hard의 basic17 bank**다. 4k→8k에서 shape-valid 상태가 바뀌거나 양쪽 valid q가 `max(.02,.25*q_4k)` 이상 달라지면4k recipe를 안정적이라고 확정하지 않는다. 공통 eligible target 결과와 각 전체 status를 함께 보고한다. 두 worlds의 통과도 보편 안정성 증거는 아니다. Challengers와 extended 결과는 모두 공개하지만 이 gate를 대신하지 않는다.
4. Basic17은 coverage가 부족하고 extended에서만 valid라면 `basic_coverage_unresolved`로 기록한다. Extended의 성공만으로 basic17 gate를 통과시키지 않는다. 개발 density 자료로 새로운 고정 basic grid를 제안할 수 있으나 새 protocol/count version을 먼저 기록하고 남은 개발 teacher에서 horizon check를 확인한다. Confirmatory world는 그 전까지 열지 않는다. 이 조건에서 기존 W2 비용을 추가 검증 비용 없이 그대로 확정하지 않는다.
5. Primary가 측정 가능하고 위 horizon 문제가 없으면 나머지6개 개발 world에서 같은 protocol을 확인하고 H_star 및 전체 threshold를 잠근다. Truth error/recovery 우위를 진입 조건으로 쓰지 않는다.
6. Hard primary가 flat한데 challenger만 정보를 보이면 두 가설의 결과를 보존한다. Best challenger를 대입해 원래 방법의 성공이라고 하지 않는다. Primary/horizon 변경이 필요하면 개발 전용 새 version으로 명시하고32000–32035 confirmatory를 열기 전에 동결한다.
7. Pilot 수치만으로3-replicate confidence calibration, C2a/C2b 성공, thermodynamic phase를 주장하지 않는다. 모든 정의된 arm이 무정보면 대형 sweep을 자동 실행하지 않고 동일 밀도 추정 목표 안에서 수정할 가정을 명시한다.

## P2-B — 잠근 synthetic estimator의 본검증 / CONDITIONAL CORE, 핵심 본문

- 2 dictionary families×3 generating densities×6 fresh worlds=36 independent worlds. R5+reference1. Native VG 최대25controls. **기본3672 fits, coverage 추가 포함 최대5400 fits**.
- 각 world에서 estimator output(JSON)을 저장·hash한 뒤 evaluator가 synthetic truth와 final test를 읽는다. 모든 예정 world를 결과표에 포함한다.
- C2a primary: `abs(K*q_hat−L_gen)/L_gen`, report rate, all-world error<=20%-and-reported rate. Generating expected count와 finite-test empirical count를 모두 저장하되 primary target은 expected count로 유지한다.
- C2b primary: 모든 repeat의 hard coefficient NMSE 평균, same-bank oracle 대비 regret, reconstruction-only와의 paired difference. NMSE는 `sum_x ||z_hat_hard_aligned(x)-z_true(x)||² / sum_x ||z_true(x)||²`다. Unit-norm decoder로 함수 보존 folding한 뒤 signed cosine Hungarian으로 D_true와 대응하며 amplitude/code의 sign flip은 하지 않는다. Ktrue=Ksae라 모든 슬롯을 계산하고 어려운 feature를 제외하지 않는다. Zero/nonfinite denominator는 evaluator failure다. Oracle는 각 repeat의 동일 bank에서 이 test NMSE 최소인 후보이며 tie는 native density/control 순서다. 각 selector regret=선택 NMSE−oracle NMSE는 수치오차 외 음수가 될 수 없다. 두 selector의 regret 차이는 음수일 수 있다. Support F1/dictionary cosine/MSE/achieved test L0는 secondary다.
- 전체36world 집계와 각6world/cell를 함께 보고한다. Cell을 숨기고 pooled success만 주장하지 않는다. 같은 world 내 repeat average를 한 관측치로 쓴다. CI bootstrap은 family×density strata 안에서 worlds를 재표집한다.

### 사전 판정과 주장 범위

1. C2a의 넓은 지정범위 지지: 전체 report rate>=80%, reported estimates의 median relative error<=20%, all-world within20%-and-reported>=70%, 각cell report>=4/6. 한 cell만 체계적으로 실패하면 그 cell을 포함한 일반화를 철회하고 제한적 결과로 쓴다.
2. C2b의 선택 효용: all-world policy의 mean coefficient NMSE가 reconstruction-only보다 최소.02 낮고 paired95% CI의 upper<0이며, c_dec 대비 악화의 upper bound<=.02. .02는 NMSE의 사전 practical threshold이며 검증된 power 근거가 아니다. 충족하지 않으면 superiority를 주장하지 않는다.
3. 보조 usable-regime 기준: all-world same-bank oracle regret<=.05인 world가70% 이상인지 보고한다. 이 기준으로 C2a 실패나 C2b 비교 실패를 덮지 않는다.
4. Template가 맞고 q가 정확해도 recovery 효용은 없을 수 있고, q가 틀려도 recovery가 좋을 수 있다. 결과표의 열을 분리한다.
5. 6world/cell은 고정된 초기 확증 예산이며 통계적 power를 확보했다고 표현하지 않는다. 넓은 구간은 불확실한 결과다. 사후 seed 추가로 유의성이 나올 때까지 반복하지 않는다.

Primary comparison은 reconstruction-only다. c_dec는 사전 지정한 non-inferiority guard이며 두 조건을 모두 만족해야 강한 C2b 주장을 한다. 나머지 comparator 및 cell별 intervals는 탐색적 설명으로 보고하고 여러 검사 중 유리한 것만 골라 성공을 선언하지 않는다. Fit infrastructure failure와 과학적 abstention은 다르다. 모든36world가 프로토콜대로 평가 가능해야 all-world C2b 결론을 내리며, 실행 실패가 남으면 incompleteness를 명시한다.

Figure: 전체world q_hat vs L_gen 및 abstention, curve 예시, recovery regret의 paired distributions. Example world는 각cell ID가 가장 작은 것으로 사전 고정하고 실패해도 교체하지 않는다.

## P2-C — 기전·적용 범위 / L1은main진입시고정, 넓은확장은조건부

### C0. Timing·optimization 민감도 / BR로 대체

기존 24개의세gamma 연장계획은폐기하지않고9월29일기록에보존하되, 현재실행에서는BR의전체17-control horizon screen으로대체한다. 추가24구간을새예산에중복 계상하지않는다. Timing은BR 첫fits에서측정하고8k continuation과평가비용을포함한다.

### C1. Phase 1 연결 ablation / CONDITIONAL EXPLANATION

사전 선택한 본검증 subset: 2 families×p=.125×3worlds=6worlds. R5+reference1, 같은 split·seeds·gamma bank·density adaptation. Primary learned VG bank는 재사용하고 추가 methods는 profiled VG, learned-beta no-variance, learned-beta no-entropy다. 삭제 objective는 동등 ELBO가 아니다. 각각 같은 initialization state/batch stream을 쓴다. Subset 선택은 main 결과를 보기 전에 고정하며 별도 독립 확증으로 세지 않는다.

- 추가 기본1836 fits, 최대2700 fits(6×6×17/25×3). Primary bank612/900 fits는 W3에 이미 포함되어 있다.
- Purpose: q/report rate/regret 변화가 within uncertainty, mean/stochastic/hard risk, precision 정책과 어떻게 연결되는가? Primary와 ablation의 hard density coverage를 따로 보고한다. Profiled 모드의 변화를 variance 단독 직접효과라고 부르지 않는다.
- Mask permutation/duplicate/alternative reference/true-D oracle sensitivity는 같은 bank에 대한 재평가로 수행한다. True alignment에서만 성공하면 배포 가능한 estimator의 성공으로 세지 않는다.

### C2. 다른 SAE family와의 구분 / L1은main진입시필수, TopK는조건부

같은6 mechanism worlds와 잠근 H_star에서 official SAELens RI-L1을 main 결과와 무관하게 비교한다. VG-specificity를 판단하는소규모L1은사전에 포함하고불리한결과도보고한다. Norm-aware TopK는recovery-family비교가필요할때사전선택한조건부블록이다. Original-paper-exact 또는 parameter-count matched라 부르지 않는다.

- L1: R5+reference1, coefficient logspace(1e-5,10,17), native density coverage에 대한 log-coefficient midpoint 추가 최대 8개. 위 gamma adaptive rule을 log coefficient축에서 사용한다. 동일 template/selector를 적용해 VG 고유성과 일반적인 ensemble 안정성 효과를 구별한다. 기본612 fits, 최대900.
- TopK: R5+reference1, k=1,...,16(고정16개), 576 fits. 주 역할은 recovery comparator다. ReLU 때문에 actual count가 k보다 작을 수 있으나 낮은 generating count 양쪽에3개의 native density를 보장하지 못한다. Template eligibility를 보고할 수는 있지만 구조적 coverage 실패를 VG의 density 추정 우위로 세지 않는다. 같은 MSE/c_dec/oracle selection을 적용한다.
- 이런 작은 비교로 전 family SOTA를 주장하지 않는다. L1에서도 같은 selector가 잘 작동하면 generic stability-based selector 성분을 인정한다.

### C3. 구조·분포 transfer / SHOULD, Phase 2 synthetic 성공 뒤

동일 d8/K16 family, p=.125에서 세 개의 단일 변경을 각각3freshworlds로 시험한다: (i) exponential amplitude scale1/sqrt2, (ii) skew exponent.5로 mean p를 보존, (iii) 첫 두 atom cosine.9. Noise는 primary와 같이.05 유지. 각각R5+reference1, primary VG만, 최대25controls: 기본918/최대1350fits.

Support co-firing correlation과 dictionary overlap은 서로 다른 조작이다. Firing correlation은 후속으로 남긴다. Exponential 조건에서는 작게 활성화된 feature의 관측 가능성이 달라지므로 expected L_gen와 recovery optimum의 불일치를 별도 분석한다. 이9worlds는 primary36worlds의 sample size에 더하지 않는다.

### C4. SynthSAEBench / CONDITIONAL 확장

Core 결과와 budget 측정 이후에만 새 run plan을 확정한다. 기존 fixed artifact/revision, d768/Ktrue16384/Ksae4096을 유지한다. 기존200M one-seed calibration은 개발 자료로만 쓰고 fresh align/select/test RNG streams를 둔다. Ktrue/Ksae mismatch에서 L_gen를 유일한 correct SAE L0로 쓰지 않는다.

최소 staged pilot 제안: VG만 3ensemble+1reference×9controls×20M samples=36 fits, 총720M generated training samples. Final200M recipe의 재현이라고 부르지 않는다. 예산은 `36*t_20M` 실측 후 결정한다. Expansion 성공 조건은 label-free selector의 native-density coverage, independent-stream recovery/coverage regret이며 full dictionary recovery가 아니다. 큰폭 Hungarian O(K^3) cost를 먼저 profile하고 sparse candidate matching 근사가 필요하면 method-version을 새로 검증한다. Tiny exact Hungarian과 같다고 주장하지 않는다.

### C5. 실제 LM / OPTIONAL 외적 검증

기존 Gemma/Llama390-job grid를 자동 실행하지 않는다. 첫 제안은 Gemma-2-2B layer5 하나, width4096, 20M training tokens, 3ensemble+1reference×9controls=36 fits다. 기존 width32768/500M protocol과 다른 pilot로 명시한다. HF 접근권한·W&B 정책·모델과 corpus revision·fresh token splits를 실행 전에 확인한다.

지표는 held-out native density, SAE replacement CE/KL와 reconstruction, feature 재현성이다. True density error는 측정 불가다. Sparse probes/IOI causal patching은 기존 scaffold에서 별도 task·negative controls를 완성한 경우에만 추가한다. CE/KL만으로 semantic feature correctness를 주장하지 않는다. 이 LM 실험은 core 논문 성립의 필수 gate가 아니다.

## 실행 순서·총량·제안 비용

| Wave | 범위 | 기본/최대 새 fits | 상태/전이 |
|---|---|---:|---|
| W0 | P1 reanalysis·rotation·A0 counterexamples | 0 | 기존exact response재사용, 신규학습없음 |
| W1a | BR E_opt,2teachers×4runs×17/25 | 136 / 200 | 전부 8k,2k/4k/8k저장 |
| W1b | BR E_joint,2teachers×3runs×17/25 | 102 / 150 | 동일reference재사용, 전부 8k |
| W2 | 남은6개개발teachers×4runs×17/25 | 408 / 600 | BR후H_star/protocol잠금 |
| W3 | Main36teachers×6runs×17/25 | 3672 / 5400 | untouched확증, 모든selector결과보고 |
| W4 | Small L1, main의6teacher subset×6runs×17/25 | 612 / 900 | W3진입시사전포함, 결과 무관보고 |
| X1 | Profiled/no-variance/no-entropy3추가methods | 1836 / 2700 | 설명할hypothesis와진입 조건사전 고정 |
| X2 | TopK k1–16,6teachers×6runs | 576 / 576 | 별도recovery-family비교, 조건부 |
| X3 | 작은amplitude/skew/overlap transfer | 918 / 1350 | primaryscope확장시조건부 |

W1은238/350 distinctfits이고모두8k이므로4k-step equivalents476/700이다. Saved snapshots는추가fits가아니다. W2–W4가전부진행되면총4930/7250 distinctfits다. H_star/4000=h라두면core step-equivalents는 `476+4692*h` / `700+6900*h`이다. H_star4000이면5168/7600,8000이면9860/14500이다.

X1/X2/X3는모두별도조건부이며각 fit×h를더한다. H_star4000에서세확장을모두추가하면distinct8260/11876, step-equivalents8498/12226이다. 이를현재필수 예산으로부르지않는다. 선택적frozen3fit continuation은추가18000updates=4.5개의4k-equivalent이며해당선택시만더한다. Prior24-target, rotation, metric reanalysis는학습fits0이지만실제CPU/evaluation시간을로그에포함한다.

비용은method별실측t_4k를쓸때 `sum(4k-equivalents*t_4k)/3600`에I/O/evaluation여유를더한다. **미측정 가정**10–40초/4k-fit와25%overhead라면W1은약1.7–9.8 GPUh다. 제안W1상한10GPUh이며timing이이를넘길것으로예상되면규모를몰래늘리지않고계획버전을재검토한다. 이번 문서 작업은이비용을사용하거나승인한것이아니다.

같은가정에서조건부W1–W4 core는H_star4000일때약18–106GPUh,8000일때약34–202 GPUh다. 기존제안160GPUh를자동으로상향하지않는다. 실제 timing과H_star가잠긴뒤계획한world/precision을보존할수있는지검토한다. 축소가필요하면targetscope·추론정밀도·budget변경을같은version에명시하고main test를열기전에고정한다.

Hypothesis별조건부 분기: W3/W4진입후same-bank selectors/L1은모든 결과를보고한다. X1은prior/precision/objective에대한설명주장을할때비교할methods/subset/endpoints를추가 학습전에고정한다. Negative main을본뒤시작한진단은exploratory로표시하며기존comparator나success기준을바꾸지않는다. X2/X3를실행하지않으면그성능/범위주장을하지않는다. 기존Phase1결과의total-effect해석으로충분하면큰X1을필수로강제하지않는다.

대형SynthSAEBench/LM은C4/C5의조건부별도계획이다. 이번작업에서어떤학습도실행하지않았다. 전체일정은W0→W1→W2→lock→W3/W4→필요한X의순서이며, 실제throughput이없으므로고정완료일을약속하지않는다.

## 필요한 구현과 output 계약

- 새 estimator 입력은 tensor schema와 truth-free run metadata만 받는다. 필수metadata는 teacher_world_id, training_dataset_id, optimizer_repeat_id, reference_id, ensemble_policy, readout_kind, density_axis, horizon, bank_kind다. Bank array의각축을명시하고teacher/dataset/optimizer를같은seed로뭉치지않는다. `src/evaluate.py`의 kernel는 재사용 가능하지만 기존 NNLS posterior API와 별도 결과 schema가 필요하다.
- 기존 `src/sae_sweep_eval.py`의 `paper_style_sigma_sel`은 input-axis aggregate다. Cross-run instability로 재사용하지 않는다. 새 names: `within_gate_uncertainty`, `between_run_soft_variance`, `ensemble_soft_uncertainty`, `between_run_hard_variance`.
- 결과 root 제안 `outputs/phase2_density_<frozen_run_id>/`; world/role/repeat/control/config/status/checkpoint, alignment maps, calibration mapping, curve/fit/profile/bootstrap/abstention, selector policy, test metrics를 저장한다.
- CSV는 `curve_points.csv`, `density_estimates.csv`, `selection_metrics.csv`, `world_summary.csv`; 현재 이 파일들은 아직 생성되지 않았다.
- 계산된 q가 없어도 row를 유지한다. 필수 fields는 `estimate_status`, `estimate_reason`, `q_hat`, `deployment_status`, `deployment_reason`, `fallback_used`, `fallback_reason`이다. Estimate와 deployment 상태를 한 status로 합치지 않는다. Fit/worker failure와 과학적 추정 보류는 구분한다. Historical metrics를 새 schema로 이름만 바꾸지 않는다.
- Lock receipt에는 code/config hash, world IDs, all selectors/thresholds, planned counts, endpoint hierarchy, data access separation을 담는다. Confirmatory 변경은 새 version으로 분리한다.
- Nonfinite 학습은 scientific/implementation failure로 남기고 성공할 때까지 seed를 바꿔 재시도하지 않는다. Infrastructure interruption은 동일 checkpoint/seed/config에서1회 resume만 허용한다. Checkpoint 손실로 처음부터 다시 학습해야 하면 자동 restart하지 않고 incomplete로 남긴다. Retry wall time도 비용 cap에 포함한다.

## 논문에 포함할 것과 미룰 것

본문 필수: P1 설명의 핵심, BR ensemble/readout 진단, 실행진입한 P2-B density error+recovery regret+report rate와small L1 비교. 광범위P2-C ablation은주장에필요한범위만포함한다. 부록: 수식·axis 검증, 모든 cell/seed/control, optimizer budget sensitivity, mixture identifiability, 추가 baseline recipe. 큰 SynthSAEBench/LM, full phase diagram, 새 teacher/encoder, learned gamma는 현재 core에서 제외하되 후속 경로를 보존한다.

Phase 2가 실패하면 같은 질문에 대한 실패 범위를 보고한다. Phase 1만 완료했다는 이유로 원래 밀도 추정 목표까지 완성됐다고 쓰지 않는다.
