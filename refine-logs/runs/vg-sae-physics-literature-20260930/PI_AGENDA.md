# PI 검토 의제 — 2026-09-30

사용자 요청은 세 지정 논문 및 기존 문헌을 바탕으로 전체 논문과 Phase 1·2의 설계를 보완하는 것이다. ARIS scientist와 scientific-skills scientist가 독립 검토 후 실제 교차 논의를 수행하고 PI가 근거를 확인해 통합한다. 이번 작업은 문헌·수식·계획 검토이며 학습 실험을 시작하지 않는다.

## 유지할 연구 목표

VG-SAE의 명시적 통계물리 모형과 추론을 이해하고, 선택 불확실성에서 적절한 밀도를 추정해 recovery와 대조한다. Phase 1과 Phase 2를 별개 논문으로 갈라 목표를 바꾸거나, SOTA를 유일한 성공 기준으로 만들지 않는다. 세 논문의 물리학적 용어를 기존 loss에 붙이는 것만으로 새 기여를 주장하지 않는다.

## 독립적으로 확인한 원문 범위

- Nature2026는 Perspective. 저널 metadata/abstract와 저자의 공개 preprint2503.01824v1을 구분한다. Preprint §5.1의 supervised identifiability는 특정 DGP·global optimum·차원 조건을 전제로 한다. 일반 LLM의 undercomplete superposition을 자동 증명하지 않는다.
- PRL2017은 binary visible/graded hidden RBM의 compositional regime. Sparse weights와 소수의 강한 hidden activity는 서로 다른 양이며, 그 asymptotic result를 VG의 hard count에 동일시하지 않는다.
- PRL2020은 binary receptive-field weights의 posterior와 teacher/student prior·noise matching을 다룬다. Correlation q는 support activation density가 아니다.
- Soh §III.1.4의 Eq18 ensemble은 data realizations를 언급하며, §III.2에서는 training split을 재표집한다. 9월29일의 fixed-data optimizer repeats는 그와 다른 estimand다.

## 두 scientist가 논의할 질문

1. Phase 1에서 conditional support energy의 pair interaction과 MF response를 명시하면 기존 covariance 결과의 설명력이 어떻게 늘어나는가? 고정 D,a,beta와 joint/profiled 학습을 구분할 것.
2. Weight-coordinate sparsity와 code occupancy를 분리하기 위해 고정 모형의 ambient orthogonal rotation control이 필요한가? 동일 Gram/관측 likelihood를 보존하는 조작과 새 학습을 혼동하지 말 것.
3. RBM prior-on-weights에서 VG prior-on-support로 옮길 수 있는 최소 통찰은 무엇인가? Oracle matched/mismatched diagnostic과 새 learned correlated prior를 구별할 것.
4. Fixed-data SGD variation을 줄일수록 uncertainty curve가 평평해질 수 있다. 그렇다면 Phase 2의 density signal은 어떤 ensemble을 가리켜야 하는가? Soft/hard 및 data-resampling/optimizer 반복을 함께 보되 사후로 잘 맞는 정의만 고르지 말 것.
5. 7k–10k fits보다 앞에 필요한 최소 pilot는 무엇인가? Full density curve의 optimization budget 민감도, 같은 population의 독립 training samples, 작은 finite-input calibration이 충분한가?
6. Generating count, dominant-amplitude participation ratio, dictionary identifiability, recovery, human interpretability를 구분하면서 한 논문 안에서 무엇을 필수·후속으로 둘 것인가?

## PI의 잠정적 수식 점검 대상

아래는 논문으로부터의 자동 등가 주장이 아닌, 프로젝트 모형에서 직접 확인할 계산이다.

Fixed x,D,a,beta에서 b_i=a_i*d_i라 두면 support energy는 `constant + sum_i [gamma + beta*||b_i||²/2 - beta*b_i^T*x]*s_i + sum_{i<j} beta*(b_i^T*b_j)*s_i*s_j`로 전개된다. Positive dictionary overlap은 동일 관측을 설명하는 선택들 사이의 비용을 만든다. 이것이 RBM의 hidden conditional distribution과 같은 모형이라는 뜻은 아니다.

Interior locally stable MF stationary branch에서는 `H_ii=1/[m_i(1-m_i)]`, `H_ij=beta*b_i^T*b_j(i!=j)`, `-d sum(m)/d gamma=1^T H^{-1}1`. Positive-definite Hessian·고정 amplitude/dictionary/precision·분기 미변경 조건이 필요하다. Singular branch나 gamma마다 joint 재학습한 curve에는 그대로 적용하지 않는다.

Isotropic Gaussian likelihood에서 x→Rx,D→RD, R^TR=I이면 조건부 exact posterior와 Gram은 보존되지만 coordinate-wise dictionary sparsity는 변할 수 있다. 재학습 Adam trajectory까지 invariant라는 주장은 하지 않는다.

이 계산은 알려진 선형대수·변분원리의 적용이다. 신규 정리나 상전이·semantic feature 식별성의 증명으로 승격하지 않는다.

## 토론과 채택 원칙

독립 입장문 → 서로의 원문을 읽은 반론 → 핵심 쟁점별 응답 → 공동 권고와 남은 이견 → PI의 채택/수정/유보 판단 순서로 기록한다. 점수 상승이나 형식적 만장일치보다 출처, 반례, 구별 가능한 실험을 우선한다. 두 agent는 같은 모델 계열일 수 있으므로 외부 독립 실증 검증이라고 부르지 않는다.
