# 두 과학자의 공동 권고안

2026-09-30. 작성: ARIS scientist. Scientific scientist의 직접 교차 응답과 PI의 중간판정을 통합했다. [Scientific 최종 확인](../scientific/03_joint_response.md)은 운영 이견 없이 동의하며 경험적 불확실성을 보존한다. Same-family provisional 연구 설계 권고이며 실험 결과·신규성 인증·실행 승인이 아니다.

## 유지할 중심 질문

**VG-SAE의 조건부 선택모형과 추론 가정을 명시하면, 그 불확실성을 이용해 적절한 활성 밀도를 추정하고 feature recovery에 도움이 되는 선택을 할 수 있는가?**

Phase 1은 모형·추론·readout의 설명 근거이며, Phase 2는 density inference와 recovery 선택의 검증이다. 생성 밀도 추정 C2a와 operating-point 선택 C2b는 별개 endpoint다. SOTA나 큰 LM 실험을 논문의 필수 성립 조건으로 두지 않는다. 어느 쪽이 실패해도 다른 쪽의 성공으로 대체하지 않는다.

## 문헌에서 채택한 내용

| 출처 | 채택할 통찰 | 옮기지 않을 주장 |
|---|---|---|
| Klindt et al. Nature2026의 공개 preprint §§4–5.3 | Generative assumptions, representation identifiability, sparse inference, dictionary learning, human interpretability의 구분 | 일반 SAE/LM의 무조건적 식별성; 알려진 D의 compressed-sensing 보장을 joint SAE에 자동 적용 |
| Tubiana–Monasson PRL2017 Eq. (1)–(2), supplement §III | Connection sparsity, code occupancy, dominant-amplitude count의 구분; 모형에 맞는 order parameter | RBM=VG-SAE, `L~1/p`의 density 추정식 전용, 작은 곡선의 thermodynamic phase 해석 |
| Hou–Huang PRL2020 Eq. (1)–(6), App. E | Prior/likelihood matching, truth overlap 대 self-overlap, hyperparameter self-consistency의 조건 | Receptive-field correlation q=활성 밀도, learned beta=Nishimori 보장, SGD repeats=posterior replicas |
| Soh VG §III.1.4 Eq. (18)–(19) | 고정 변수에 대한 data-realization ensemble과 soft-total selection quantity의 출발점 | 현재 fixed-data hard-between quantity로의 검증 없는 전이 |

정확한 URL·버전·접근 깊이는 ARIS의 SOURCE_MANIFEST.json과 두 독립 입장문을 따른다. Nature 출판본 전체는 읽지 못했고 공개 저자 preprint의 수식·가정을 확인했다. 이 한계를 최종 문헌표에서도 유지한다.

## Phase 1에 먼저 반영할 작은 보강

1. 기존 exact covariance와 field-response 결과를 재사용한다. `-∂gamma E[N]=Var(N)`는 이미 Phase 1에서 검증됐다. 새 필수 실험이나 새 정리로 세지 않는다.
2. 고정 x,D,a,beta에서 Gram의 off-diagonal interaction과 `Var(N)=sum Var(s_j)+2sum Cov(s_i,s_j)`를 연결해 설명한다. Interior stable MF branch의 `-d sum(m)/d gamma=1ᵀH⁻¹1`과 diagonal gate variance를 구분한다. Joint retraining/profiling의 total derivative에 그대로 적용하지 않는다.
3. Ambient orthogonal rotation은 training 없는 sanity로 둔다. x,b,D를 함께 회전하고 a를 고정하거나 encoder를 conjugate해야 조건부 posterior가 보존된다. Decoder의 좌표별 sparsity가 달라져도 latent support energy가 보존될 수 있다는 대조다. Adam 재학습 invariance를 주장하지 않는다.
4. Stable-correct deterministic hard masks와 stable-wrong masks가 모두 flat hard-between curve를 만들 수 있음을 명시한다. Flatness는 feature correctness 판정이 아니다. 모든 run의 soft m이 같더라도 soft-total에는 m(1−m)가 남을 수 있다는 점도 함께 쓴다.
5. Geometry×co-firing의 matched/mismatched correlated-prior 2×2는 SHOULD 후속 기전검사다. 새 posterior target 구현이 density pilot의 착수를 막는 첫 gate가 되어서는 안 된다. Trainable correlated-prior SAE는 필수로 추가하지 않는다.

PR3는 exponential-amplitude stress가 dominant feature count를 설명할 필요가 있을 때만 선택적으로 사용한다. Generating support count·hard L0·semantic compositionality를 대신하지 않는다.

## 최초 학습 pilot와 비용

현재 M1의 development world 중 두 family에서 사전 지정한 하나씩, 총 2 worlds로 시작한다. 두 world의 선택·seed·stream은 결과 전에 고정한다. Sample size와 데이터 분포는 기존 development 계약을 우선 유지한다.

| Bank | Reference/ensemble 구성 | 기본 17 controls | 최대 25 controls |
|---|---|---:|---:|
| `E_opt` | world당 reference 1 + 동일 training data의 optimizer repeats 3 | 136 fits | 200 fits |
| `E_joint` challenger 추가분 | 같은 reference를 공유; repeat마다 독립 training samples를 쓰는 새 repeats 3 | 102 fits | 150 fits |
| 합계 | reference bank를 중복 학습·계상하지 않음 | **238 distinct fits** | **350 distinct fits** |

먼저 E_opt의 136/200 fits만으로 full-curve horizon screen을 시작할 수 있다. E_joint까지 모두 8k로 실행하면 **4k-step equivalents는476/700**이다. 저장한 2k/4k snapshot을 각각 새 fit으로 세지 않는다. 실제 wall/device time, exact/re-evaluation 비용, coverage 실패도 기록한다. 이 수는 GPU-hour 실측이나 power 보장이 아니다. 이번 문서 작업에서 학습은 실행하지 않았다.

모든 기본 17 controls는 8k까지 학습하고 2k/4k/8k snapshot을 보존한다. Adaptive controls는 사전 지정한 **4k primary native hard-density rule에서만** 최대 8개를 선택한다. 그 control bank를 고정한 뒤 추가 controls도 동일 원칙으로 처음부터 8k까지 학습해 중간 snapshot을 남긴다. Basic17 horizon 비교를 항상 먼저 제시한다. Extended bank는 4k coverage에 조건부로 선택된 보조 결과이며 horizon에 대칭적으로 선택된 bank가 아니다.

Reference는 두 arm의 ensemble 평균에 포함하지 않는다. 두 arm은 같은 held-out align/select/test bank를 쓰고 initialization ID를 pairing한다. Reference는 E_opt와 training examples를 공유하지만 E_joint repeats와는 독립 training samples 관계이므로, 그에 따른 alignment 차이도 treatment contrast의 일부다. Coverage·matching·alternative-reference sensitivity는 각 arm에서 별도로 판정한다. Shared reference와 두 arm을 독립 worlds로 세지 않는다.

Zero-training sensitivity로 기존 E_joint repeat1을 alternative common reference로 사용한다. E_opt는 원래 ensemble repeats 3개를 유지한다. E_joint에서는 새 reference가 된 repeat1을 빼고 기존 reference0을 넣어 R=3과 독립 training datasets를 유지한다. Primary와 alternative 결과를 모두 보고하고 더 유리한 anchor를 사후 선택하지 않는다. 이 대조를 위해 reference fit을 추가하거나 별도 통과 gate를 만들 필요는 없다.

`E_joint`는 **data+optimizer 결합 변동**이다. Initialization pairing만으로 순수 data variance가 분리되지 않는다. 이를 분해하는 주장이 필요할 때만 작은 crossed/nested design을 별도 설계한다. 모든 controls에서 각 repeat의 training dataset은 일관되게 유지한다.

Schema의 상위 independent world는 teacher dictionary와 고정 생성 조건·평가 streams를 가리킨다. 그 아래 training-dataset realization ID와 optimizer-repeat ID를 분리하고 reference가 사용하는 dataset ID를 기록한다. 기존 `world=D+하나의 training stream` 정의를 그대로 쓰면 E_joint의 nested 구조를 표현하지 못한다. 같은 teacher 아래 여러 training draws를 새로운 독립 world n으로 늘리지 않는다.

## Observable과 primary 변경 계약

현재 operational primary는 `E_opt + native hard-density matching + hard-between U`로 명명해 유지한다. 원 VG와의 차이는 ensemble과 soft→hard observable 두 축이다. 이를 감춘 채 같은 이론식의 재현이라고 하지 않는다.

동일 checkpoint로 soft-total, soft-within, soft-between을 재평가한다. Soft 후보 estimator를 시험할 때에는 **native soft density축·mapping·coverage 계약을 별도로 정의**해야 한다. Hard로 맞춘 runs에 soft-U를 얹은 그림을 soft-density-matched estimator와 같다고 부르지 않는다. Soft challenger를 위해 추가 adaptive training을 자동으로 늘리지 않는다. 기존 bank로 coverage가 부족하면 그 한계를 결과로 보고한다.

모든 사전 지정 ensemble/readout 가설과 개발 결과를 함께 남긴다. Confirmatory truth error를 보고 잘 맞는 arm 또는 uncertainty 정의를 고르는 것은 금지한다. 미실행 계획의 primary를 과학적 근거로 바꾸는 일 자체는 가능하다. 변경 시 이유와 개발 결과를 공개하고 새 protocol version을 고정한 다음 untouched confirmatory worlds에서 검증한다. 다른 방법으로 바꾼 뒤 이전 primary의 성공인 것처럼 서술하지 않는다.

Pilot의 목적은 positive result 강제가 아니라 curve coverage, horizon dependence, ensemble/readout 차이와 failure의 의미를 구별하는 것이다. 두 worlds는 보편적 안정성을 증명하지 못한다. Broad claim은 이후 fresh worlds의 결과가 필요하다.

## 후속 본검증과 비교의 조건

36-world R=5 본검증은 위 bridge와 나머지 development를 거쳐 protocol을 잠근 뒤 수행하는 후속 단계로 유지할 수 있다. 예산·precision을 변경하면 기존 계획의 broad C2a 범위를 그대로 약속하지 말고 새 버전으로 명시한다. 새 결과를 보며 유의해질 때까지 seed를 추가하지 않는다.

동일 bank의 reconstruction/c_dec/raw-U/within-gate/random/oracle 비교는 main 결과와 무관하게 모두 보고한다. VG-specificity 주장을 유지하려면 작은 L1 comparison도 사전에 고정하고 primary 결과의 유불리와 무관하게 수행한다. TopK는 native density lattice의 한계를 인정하는 recovery comparator다.

광범위 precision/entropy/variance retraining ablation은 조건부로 미루되 **분기 해석을 사전에 적는다**. 예를 들어 q 신호가 유지되면 VG-specificity와 objective 역할을 검토하고, 신호가 없으면 coverage/optimization/ensemble/readout 중 확인된 원인을 가를 작은 대조를 선택한다. 이미 얻은 불리한 comparator를 숨기는 것은 조건부 실험 설계가 아니다.

## 보존한 불확실성

어떤 ensemble와 observable가 generating density에 유익한지는 아직 알 수 없다. 문헌은 그 답을 제공하지 않는다. Data-realization을 도입해도 prior/template의 exchangeability, learned feature correspondence, heterogeneity와 stable-wrong 문제가 자동으로 해결되지 않는다. 원래 density inference 목표를 유지하면서 그 적용 범위를 검증하는 것이 공동 권고다.
