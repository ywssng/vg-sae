# Scientific 독립 입장: VG-SAE의 식별 조건과 단계적 검증

작성: 2026-09-30. 역할: scientific-skills 기반 과학자. **상대 연구자의 입장을 읽기 전 작성한 독립 검토**다. PI의 문헌 식별 정보만 전달받았다. 이번 작업은 문헌·계획 검토이며 새 학습, GPU 실험, 모델 코드 수정은 수행하지 않았다.

## 판단

VG-SAE를 first principles에서 유도하고 불확실성에서 적절한 밀도를 추정하려는 원래 질문은 유지할 가치가 있다. 세 논문은 이를 자동으로 정당화하는 정리가 아니라, 어떤 변수를 구분하고 어떤 반증 실험을 먼저 해야 하는지 알려준다. 현 계획의 가장 큰 미해결점은 **같은 데이터에서 SGD를 반복해 얻는 불안정성이, 생성 밀도를 알려주는 재현 가능한 신호인가**다. 기존 9월 24일 결과는 objective와 inference의 설명에는 유효하지만 이 연결을 검증하지 않았다.

따라서 7,240–10,376 fits를 한 번에 core 의무로 삼기보다, 알려진 좌표에서의 정확검사 → 전체 곡선의 최적화 민감도 → 작은 학습된 좌표 pilot → 잠근 fresh-world 검증 순으로 조건부 진입해야 한다. 대규모 ablation은 selector가 작동하거나 실패 원인을 가를 가치가 확인된 후 수행한다. 이 단계화는 밀도 추정 목표를 포기하는 것이 아니라 그 목표가 어디에서 깨지는지 먼저 확인하는 설계다.

## 읽은 근거와 한계

저장소에서 `AGENTS.md`, `.agents/project-memory.md`, `PAPER_PLAN.md`, `refine-logs/FINAL_PROPOSAL.md`, `refine-logs/EXPERIMENT_PLAN.md`, `refine-logs/EXPERIMENT_RESULTS_20260924_000000.md`를 읽었다. 원본 summary CSV의 열과 row 수를 확인했다: exact 135, frozen 9, joint 54. 원본 checkpoint를 재평가하거나 모든 수치를 독립 재계산한 감사는 아니다. 하위 `AGENTS.md`/`AGENTS.override.md`는 대상 `refine-logs` 아래 검색에서 발견되지 않았다.

적용한 절차는 `scientific-scientific-critical-thinking/SKILL.md`와 `scientific-experimental-design/SKILL.md`다. 후자의 nested-design 및 sequential/adaptive 참고문서를 선택적으로 읽었다. 이번 연구에 필요한 효과 대상, 교란, 독립 반복 단위, 개발/확증 분리를 적용했고 임상·생물학적 판정 체계는 가져오지 않았다.

| 문헌 | 실제 확인 깊이 | 한계 |
|---|---|---|
| Klindt et al., *A unifying framework from neural superposition to sparse interpretable codes*, Nature Machine Intelligence 8, 1025–1037 (2026) | [출판사 metadata/abstract](https://www.nature.com/articles/s42256-026-01259-z), Perspective·2026-07-14 확인. [공개 저자 preprint](https://arxiv.org/html/2503.01824v1)의 §4, §5.1–5.3과 limitations, 관련 수식 읽음 | arXiv는 *From superposition to sparse codes: interpretable representations in neural networks*, 2025-03-03 v1이다. 2026 version of record 전체를 읽었다고 하지 않으며 수정 사항의 완전 일치도 미확인 |
| Tubiana & Monasson, PRL 118, 138301 (2017) | [arXiv v2 본문 전체](https://arxiv.org/html/1611.06759v2), [저자 공개 출판본 PDF](https://jertubiana.github.io/files/Tubiana%20and%20Monasson%20-%202017%20-%20Emergence%20of%20Compositional%20Representations%20in%20Rest.pdf)의 본문과 그림 설명 대조 | [APS supplement](https://link.aps.org/supplemental/10.1103/PhysRevLett.118.138301)는 접근 실패. 저자 publication page에는 본문 PDF만 찾음. 별도 supplement의 학습 상세·추가 수치까지 검증하지 않음 |
| Hou & Huang, PRL 124, 248302 (2020) | [arXiv v2](https://arxiv.org/html/1911.02344v2) 본문 Eqs. (1)–(6), Fig. 1–2, supplement Appendix D의 threshold 및 Appendix E의 hyperparameter derivation 읽음 | 공개 저자 manuscript와 supplement를 읽었으며, 구독 제한 APS version of record의 문장별 대조나 replica 계산 전체 재유도는 하지 않음 |

arXiv HTML의 생성 날짜처럼 보이는 2026년 표시를 원 논문의 출판 날짜로 쓰지 않았다. 2017/2020 출판 서지는 DOI 및 abstract metadata를 따른다.

## 논문이 직접 말하는 것

아래 문단은 원문의 결과·가정에 대한 짧은 기술이고, 이후 절은 우리 계획에 대한 별도 추론이다.

**Klindt 등:** 세 단계는 representation의 선형 식별성, sparse coding을 통한 분리, 해석가능성의 평가다. Preprint §5.1의 인용 정리는 특정 data-generating process와 global cross-entropy optimum을 가정한다. 같은 절의 limitations는 latent/representation 차원이 같은 정리와 과완전 superposition의 간극을 인정한다. §5.2는 알려진 projection의 compressed sensing과 학습할 dictionary를 구분하며 SAE의 amortized encoder가 recovery 한계를 가질 수 있다고 논한다. 따라서 일반 SAE의 전역 식별성을 보장하는 단일 정리로 읽을 수 없다. [§4–5.2, Theorem 1, Eqs. (6)–(9)](https://arxiv.org/html/2503.01824v1)

**Tubiana–Monasson:** 이론은 binary visible/비선형 hidden RBM의 quenched random-weight ensemble이며, 희소 연결, 문턱값, visible field, 낮은 effective temperature가 함께 중요하다. Compositional regime은 여러 hidden units의 강한 동시 활성과 약한 background를 구분한다. 본문의 participation ratio는 exponent 3을 쓰며 asymptotic strong/weak scaling에 근거한다. Connection sparsity p와 선택되는 강한 feature 수 L은 서로 다른 변수다. Replica-symmetric ground-state 계산과 MNIST 학습 비교가 근거이며 SGD 학습 과정의 보편적 상전이 정리는 아니다. [Eq. (1)–(2), Figs. 2–4](https://arxiv.org/html/1611.06759v2)

**Hou–Huang:** 두 hidden-neuron RBM에서 binary synaptic receptive fields를 추론한다. q는 두 receptive field의 prior correlation이고, beta는 teacher 생성 노이즈의 inverse-temperature다. Bayes-matched teacher–student 및 thermodynamic-limit RS 해석에서 data density M/N에 따른 symmetry-breaking threshold를 구한다. Eq. (6)의 noise/correlation 추정은 evidence의 stationary condition이며 Appendix E는 true q와 temperature가 모두 맞아야 해당 analytic-energy 관계가 성립함을 강조한다. 데이터가 부족할 때 hyperparameter landscape가 비볼록일 수 있다고 본문에서 명시한다. [Eqs. (1)–(6), Appendix D–E](https://arxiv.org/html/1911.02344v2)

## 연구 대상과 대칭을 먼저 고정해야 한다

우리의 표기에서는 support 평균 p, dictionary geometry c_D, support co-firing c_s, optimizer repeat r, sample size n을 분리하는 것이 좋다. RBM 문헌의 p/q를 그대로 가져오면 weight sparsity·RF correlation·support density가 충돌한다.

현재 SAE의 핵심 조건부 모형을 a, D, beta를 고정해 쓰면

`P(s|x,a,D,beta,gamma) ∝ exp[-beta/2 ||x-D(a⊙s)||² - gamma Σ_j s_j]`.

이 식을 전개한 pair term은 `-beta a_i a_j (d_iᵀd_j) s_i s_j`다. 양의 overlap과 양의 amplitudes는 대체 설명 사이의 경쟁을 만들 수 있다. Correlated support prior를 추가하면 그 prior의 log-interaction이 더해진다. 따라서 orthogonal D에서 MF=exact이라는 현재 결과는 **독립 support prior**까지 포함한 조건부 사실이다. Co-firing prior가 생기면 orthogonality만으로 factorization을 보장할 수 없다. 반대로 pair covariance가 작다고 독립임을 단정할 수도 없다. 서로 반대의 상호작용이 상쇄될 수 있기 때문이다. 이는 위 조건부 식에 대한 우리의 대수적 추론이지 세 논문의 새 결론은 아니다.

동일하게 `Var(N)=Σ Var(s_j)+2Σ_{i<j}Cov(s_i,s_j)`에서 exact susceptibility와 diagonal gate variance를 나누어 기록해야 한다. 이 항등식과 기존 `-dE[N]/dγ=Var(N)` 검증을 새 결과로 포장하지 않는다. 새 정보는 geometry와 prior가 off-diagonal 항을 어떻게 바꾸는가다.

비음수 amplitude SAE에서 decoder sign-flip을 동등 대칭으로 처리하지 않는 현재 계획은 옳다. Permutation은 nuisance지만 개별 feature가 중복·병합되어 있으면 one-to-one Hungarian의 좋은 평균 cosine만으로 latent identity를 인증할 수 없다. Signed matching, 모든 슬롯 유지, subspace와 individual recovery 구분은 유지한다.

## 핵심 위험 1: 동일 데이터 repeat의 의미

현재 Phase 2 repeat는 training examples를 고정하고 initialization/batch order만 바꾼다. 따라서 primary U는 `Var_optimizer(H | D, training data, budget, control-selection rule)`의 관측량이다. 학습 데이터 재표집 uncertainty나 Bayesian support posterior variance와 같지 않다. 현재 문서가 seed-sensitivity band로 한정한 점은 강점이다.

더 강한 반례가 있다. 한 문제에서 각 control의 학습이 같은 정렬된 해로 수렴한다고 가정하면 모든 repeat의 H가 같아지고 U는 모든 density에서 0이다. 올바른 dictionary와 support를 회복하는 경우에도 estimator는 flat-curve abstention을 한다. 이는 항상 발생한다고 예측하는 주장이 아니라, **추정기가 잘 되는 조건에 적당한 학습 불안정성이 필요할 수 있음**을 보여주는 논리적 반례다. Density를 물리적 속성처럼 부르려면 optimizer horizon을 바꿔도 q가 얼마나 보존되는지 먼저 알아야 한다.

현재 C0는 gamma {0,2,4}만 4k→8k로 연장한다. 이 세 점은 전체 curve의 shape, admissible q, density coverage와 q_hat 변화를 판정하지 못한다. 약한 C0를 두고 36-world 큰 검증으로 들어가면 실패가 template transfer 때문인지 불충분 학습 때문인지 판단하기 어렵다.

## 핵심 위험 2: support count, recoverability, compositionality

Generating expected count, finite sample support count, native hard count, effective participation count, reconstruction-optimal count, recovery-optimal count는 별개다. C2a와 C2b를 나눈 현재 계획은 유지한다. PRL의 강한 hidden-feature 수를 SAE hard L0와 같다고 하면 안 된다.

Primary는 amplitude=1의 Bernoulli 조합이다. 이때 finite set의 noiseless support sums는 d<K라도 random continuous D에서 서로 다를 수 있다. 따라서 d8/K16이라는 이유만으로 모든 support를 원리적으로 식별 불가능하다고 해석하는 것도 틀리다. 반대로 finite Gaussian noise에서 가깝게 겹치는 support sums는 통계적으로 구분하기 어렵다. Continuous-amplitude sparse coding의 worst-case 보장과 binary-amplitude finite-support 문제를 구분해야 한다.

이 때문에 작은 known-D exact control에서 Bayes-optimal support/count risk를 먼저 보는 것이 고정보다 좋다. Dictionary learning과 amortization 오차를 포함한 learned-SAE 실패를 정보 한계로 이름 붙이지 않는다. 그 control은 evaluator만 이용하며 blind selector에는 truth를 주지 않는다.

Compositionality를 논문 주장으로 추가할 필요는 없다. 낮은 L0, 높은 participation count, 재조합 가능한 정답 factor는 각각 다른 성질이다. 이미 학습한 code의 scale-preserving participation ratio와 concentration을 부록 진단으로 추가하는 것은 저비용이지만, 그것만으로 compositional regime을 주장하지 않는다. Novel-combination test는 그런 주장을 실제로 하려는 경우에만 별도 설계한다.

## 핵심 위험 3: prior와 finite sample의 역할

Hou–Huang의 prior q는 synaptic geometry에 관한 것이다. SAE support co-firing을 그 q와 같은 것으로 부를 수 없다. 다만 그 논문은 matched/mismatched prior와 sample size를 구분하는 설계가 왜 중요한지를 보여준다. 현재 8192 examples 한 값으로만 검증하면 알고리즘 변동이 sample-limited feature ambiguity에 반응하는지, optimizer-local minima에 반응하는지 구분할 수 없다.

또한 동일 dictionary에서 독립 training datasets를 만들어야 dataset variation을 dictionary-world variation에서 분리할 수 있다. Sample count 비교에서는 nested sample streams와 같은 optimizer budget를 쓰고, 별도로 fixed-epoch 결과를 해석해야 한다. Fixed updates는 총 gradient evaluations를 맞추지만 data passes를 바꾼다. 둘을 동시에 통제했다고 주장할 수 없다.

## 우선순위별 최소 구별 실험과 Phase 2 수정

아래는 **실행 제안**이다. 숫자는 최소한의 설계 선택이며 power가 검증된 기준이 아니다. 새 파일의 실험 결과처럼 읽으면 안 된다.

### S0 — 학습 없는 조건부 정확검사: 즉시 채택

- Tiny K=8에서 첫 pair의 geometry `c_D∈{0,.9}`와 support correlation `c_s∈{0,.6}`를 독립 조작한다. p=.25, a=1, beta=8을 고정해 기존 Phase 1의 matched 설정과 연결한다. 나머지 support는 독립이다.
- Bernoulli pair를 `P11=p²+c_s p(1-p)`, `P10=P01=p(1-p)(1-c_s)`, `P00=(1-p)²+c_s p(1-p)`로 생성하면 marginals와 expected count를 보존하면서 co-firing만 바꿀 수 있다.
- 같은 generated samples에서 matched prior exact, independent-prior exact, 각 target의 MF를 비교한다. 3 blocked worlds×4 teacher cells×2 student-prior choices=24 tiny target cells이며 중복되는 독립-prior cell은 표시한다. Training fit은 0이다. Matching/gamma sweep보다 이 작고 정확한 contrast가 먼저다.
- Readout: full reverse KL, marginal Brier/NLL, covariance matrix, count variance의 diagonal/off-diagonal, count prediction error, Bayes support risk. Equal mean count에서 분포·dependence의 변화가 무엇을 바꾸는지 판정한다.
- **Phase 2 업데이트:** correlated prior에서 template 편향이 분명하면 independent support를 C2a의 명시적 적용 조건으로 잠그고, co-firing stress를 최소 development stress로 승격한다. 곧바로 trainable correlated-prior SAE를 추가하지 않는다. 오류가 covariance 항으로 설명되지 않아도 그 실패를 보존한다.

### S1 — orthogonal frozen gate 잔여 오차의 분리: 즉시 채택

- 기존 a=1, unit D, 독립 prior orthogonal target에서 analytic logit `beta d_jᵀx - beta/2 - gamma`와 모델 output을 비교한다. Analytic parameters를 export할 수 있는지 확인하는 것은 새 architecture 실험이 아니다.
- 기존 3 orthogonal frozen trajectories를 고정된 4k/8k 시점까지 연장하는 계획을 세우고 exact KL 및 stationarity를 기록한다. 단, 이번 검토에서는 실행하지 않는다. 좋은 checkpoint만 고르지 않는다.
- **Phase 2 업데이트:** residual이 크게 줄면 9월24일 .36 nats를 optimization-limited로 계속 설명하고 full-curve horizon sensitivity를 의무화한다. 남는 잔차가 있을 때만 parameterization/data scaling 구현을 추적한다. 구조적 encoder 한계라는 이름을 먼저 붙이지 않는다.

### S2 — full-curve optimization pilot: 대규모 본검증 전 필수

- 현재 개발 world 중 family별 첫 한 world를 사전 선택하고, 기존 R3+reference1의 **전체 17-control bank**부터 만든다. 기본 136 fits, coverage 추가를 포함하면 최대 200 fits다. 기존 M1 544/800 fits의 부분집합이다.
- 모든 fit에 2k/4k/8k snapshot을 사용하며 각 시점에서 density matching, alignment, profile, q_hat/abstention을 다시 계산한다. Coverage가 부족하면 동일 사전 adaptive rule을 horizon마다 적용할지, control union을 고정할지 구현 전에 명시한다. 선택에 따라 비교 estimand가 바뀐다.
- 제안 primary sensitivity는 먼저 고정된 기본17 bank에 대해 같은 controls의 horizon 차이를 보고, 확장 bank 결과를 별도 기록하는 것이다. 특정 시점의 coverage 우위가 안정성으로 보이지 않게 한다.
- Pilot 기준 제안: 두 family 중 어느 하나에서 accepted↔abstained 상태가 반복적으로 바뀌거나, 4k→8k accepted q의 차이가 `max(.02,.25*q_4k)`를 넘으면 4k universal recipe를 확증에 사용하지 않는다. 이 두 worlds가 보편 안정성을 증명하지는 않는다. 기준 통과도 다음 개발로 갈 근거일 뿐 성공 판정이 아니다.
- **Phase 2 업데이트:** 현 C0의 세 gamma 연장을 이 full-curve 검사로 대체한다. Budget dependence가 크면 horizon과 claim을 새 버전으로 잠근 뒤 나머지 개발 worlds를 사용한다. 계속 신호가 소실되면 U estimator를 finite-budget heuristic으로 제한하거나 C2 hypothesis failure로 기록하고 큰 sweep을 중단한다.

### S3 — 데이터 uncertainty와 optimizer uncertainty의 분리: 작은 추가 pilot

- S2의 두 dictionary를 고정한 채 독립 training data seed 하나를 더 만든다. Evaluation streams는 같은 dictionary의 공통 held-out inputs를 써서 비교의 잡음을 줄인다. 기본136/최대200 추가 fits다. 새 dictionary-world 두 개처럼 독립 n을 늘려 세지 않는다.
- 별도로 원래 stream의 nested1024 examples와8192 examples 비교를 계획한다. 계산비가 허용하면 full bank를 추가하고, 아니면 결과를 본 뒤 gamma를 고르지 말고 사전 선택한3 control에서 기전 탐색만 한다. 후자의 경우 q_hat의 n-민감도를 검증했다고 쓰지 않는다.
- 같은 D/data에서 repeat variance, 같은 D/다른 data에서 mean predictor 차이, 다른 D/world 변이를 나누어 보고한다. 반복측정의 숫자를 독립 worlds로 세지 않는다.
- **Phase 2 업데이트:** primary output에 `ensemble_scope=optimizer_given_dataset`와 `optimizer_horizon`을 저장한다. Data-resampling ensemble을 secondary로 시험하더라도 primary estimator와 섞지 않는다. Sample size에 따라 q가 바뀌면 n-dependent claim으로 제한하고 8192만의 C2a 범위를 분명히 한다.

### S4 — 저비용 negative controls: 유지·강화

- 현재 all-off/all-on/stable-wrong/random masks/duplicates를 유지한다. 추가로 **oracle-correct deterministic masks**도 U=0, abstention임을 명시한다. Failure 검사가 correctness detector가 아니라 식별 가능한 shape detector임을 드러내는 가장 명확한 control이다.
- 같은 density라도 learned feature의 mixed/duplicate 구조와 coefficient risk가 달라지는 counterexample를 평가용으로 유지한다. Hungarian의 ambiguous ties를 숨기지 않는다.
- **Phase 2 업데이트:** abstention reason은 `no_retraining_variation`과 `fit_misspecification`, `coverage_failure`, `alignment_failure`를 구분할 수 있도록 raw diagnostics를 남긴다. Stable-correct abstention을 실패로 숨기거나 stable-wrong를 탐지한다고 과장하지 않는다.

## 큰 fit 계획의 단계화 제안

현 계획은 모든 실패를 남기고 fresh worlds에서 검증한다는 장점이 있다. 바꿀 것은 통계적 약속보다 진입 순서와 ablation의 의무 범위다.

1. **첫 단계:** S0/S4와 현재 A0. Training 0. Known identity를 다시 확인하는 목적보다 template의 필요 가정·반례를 확인한다.
2. **첫 학습 단계:** S1과 S2. 기존 M1 중136/최대200fits를 먼저 사용하고 전체 곡선 horizon을 검사한다. 8k 연장을4000-step equivalents에 별도로 센다.
3. **개발 단계:** 필요시 S3 뒤 기존 나머지 M1을 진행한다. Reusable fits와 새 fits를 별도 집계해 중복 계상하지 않는다. Threshold를 변경하면 confirmatory world를 열기 전에 freeze한다.
4. **핵심 검증:** C2a/C2b 절차와 36 fresh worlds를 유지하는 방안이 합리적이다. 먼저 pilot의 variance/abstention을 사용해 최종 precision과 비용을 검토하고, 변경 시 새 계획 버전으로 고정한다. 결과를 보며 유의해질 때까지 world를 늘리지 않는다.
5. **기전·baseline 단계:** C1의1836–2700 additional fits와 C2의1188–1476 fits를 모두 사전 의무로 묶지 않는다. 먼저 small matched subset에서 learned/profiled/fixed-beta 및 L1의 곡선을 비교한다. Primary가 성공하면 VG 특이성 대 일반 stability 효과를 가르기 위한 L1 비교를 우선한다. 실패하면 실패 원인을 실제로 구분할 ablation만 진행한다. TopK는 recovery comparator 역할을 유지하며 density lattice를 불공정하게 평가하지 않는다.

정해진 한 수치로 최종 총량을 줄였다고 주장하지 않는다. 첫 정보 획득 비용을 낮추고 후속 비용을 조건부로 만든다. 큰 확증을 축소할 경우 기존 broad C2a 범위·precision을 그대로 주장할 수 없다.

## 채택·유보·기각

| 제안 | 판단 | 이유 |
|---|---|---|
| First-principles 모형·추론 오차와 density inference를 한 논문 질문으로 유지 | 채택 | 현재 evidence와 원래 사용자 의도에 정렬 |
| Geometry, prior dependence, sample size, optimizer budget를 분리 | 채택 | 서로 다른 failure mechanisms를 구별 |
| Fixed known-D exact control과 full-curve horizon test를 앞당김 | 채택 | 대형 grid 이전에 주요 가설을 반증할 수 있음 |
| Cross-run U를 명시적 finite-budget/동일-data estimand로 기록 | 채택 | 측정한 randomness와 주장 일치 |
| Code participation/concentration을 보조 진단으로 기록 | 선택 채택 | 이미 있는 code로 계산 가능; L0와 다름을 보여줌 |
| 새 correlated-prior model, learned-gamma, EM를 즉시 도입 | 유보 | 기존 estimator의 실패 원인을 이해하기 전에 method를 늘림 |
| 고차원 finite-size scaling과 phase diagram | 유보 | thermodynamic claim을 할 때 필요한 별도 연구 |
| Novel-combination/causal semantic benchmark | 유보 | compositional/semantic claim을 하려는 경우에만 필요 |
| RBM의 compositional phase = 작은 SAE L0 | 기각 | 모형, order parameter, limit가 다름 |
| Learned beta/gamma = Nishimori 보장 | 기각 | Bayes matching 및 normalized generative evidence의 조건 미충족 |
| Stable feature = true feature, stable density = generating density | 기각 | stable-correct/incorrect counterexample 모두 존재 |
| 세 논문으로 VG-SAE 신규성·성공 확정 | 기각 | 문헌은 위치와 가정의 근거이며 현재 방법의 실증 아님 |

## 상대 연구자에게 제시할 논점

1. 성능이 좋은 수렴된 orthogonal learner에서 U가 사라지는 경우를 density estimator의 과학적 적용 조건에 어떻게 포함할 것인가? 내 입장은 full-curve budget test가 대형 검증보다 앞서야 한다는 것이다.
2. RBM의 Gibbs equilibrium/quenched disorder 평균과 SGD seeds의 분포를 연결하는 정리가 없는 상황에서, physics 해석은 어디까지 가능한가? 내 입장은 objective의 통계물리적 유도와 estimator의 algorithmic 측정을 명시적으로 구분하되 두 연결을 실험으로 검증하자는 것이다.
3. Prior dependence test를 넣는다면 latent co-firing을 바꾸는 것이 지금 밀도 추정에 가장 직접적이다. 그러나 Hou–Huang의 RF-correlation prior와 동일한 재현이라 부르지 않아야 한다. 더 직접적인 dictionary-prior 검사가 필요한가?
4. 36-world 본검증 이전에 mechanisms/baselines를 모두 크게 돌려야 하는가? 내 입장은 먼저 작은 VG bank의 shape와 failure 원인을 확인하고, L1의 generic stability test를 우선하며 광범위 objective ablation은 조건부로 두자는 것이다.

## 방법론 provenance

Scientific skills가 이 검토의 효과 대상·교란·반복 단위·단계화 구분에 실질적으로 기여했다. 2026-09-30에 [arXiv record](https://arxiv.org/abs/2609.00065)를 확인했다: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner; 2026; 현재 v2, 2026-09-02 개정. Journal reference는 표시되지 않았다. 참고: *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065 . 이 인용은 작업 절차의 provenance이며 VG-SAE·물리학 주장에 대한 실질적 근거로 사용하지 않았다.
