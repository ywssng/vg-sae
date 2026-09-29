# VG-SAE 문헌 검토와 통합 설계: PI 최종 판단

2026-09-30. 사용자 지정 세 논문과 기존 VG/SAE 문헌, Phase 1 실제 결과, 9월 29일 통합 계획을 검토했다. ARIS scientist와 scientific-skills scientist가 독립 입장문을 작성하고 서로 직접 반론·수정을 주고받았다. PI는 원문과 수식을 확인하고 아래 권고를 계획에 반영했다. **새 학습·GPU 실험이나 모델 코드 변경은 수행하지 않았다.**

## 결론

원래 목표인 **VG-SAE의 원리적 이해 → 불확실성에 기반한 밀도 추정 → feature recovery 검증**은 유지한다. 세 논문이 이 연구의 성공이나 신규성을 증명하지는 않는다. 가장 큰 보완은 **무엇의 불확실성을 측정하는가**를 고정하는 것이다.

9월 29일 계획의 주관측량은 같은 training data에서 초기화·batch order를 바꾼 **hard-code의 알고리즘적 변동**이다. 원래 Soh 식은 soft 선택확률의 ensemble average를 사용하며 데이터 실현도 바뀐다. 이전 계획은 그 전이를 검증하지 않은 채 큰 반복 학습으로 넘어갈 수 있었다. 이제 ensemble 축과 soft/hard 관측량을 각각 구분하는 작은 bridge를 먼저 수행한다.

이는 Phase 2를 포기하거나 다른 진단 논문으로 바꾸는 결정이 아니다. 추정기가 작동할 수 있는 조건과 실패를 먼저 확인하여 원래 밀도 추정 질문에 더 직접적으로 답하는 설계다.

## 세 논문에서 실제로 가져올 내용

| 논문 | 본 연구에 반영할 내용 | 그대로 가져오지 않는 내용 |
|---|---|---|
| Klindt et al., Nature Machine Intelligence2026 | 모형 가정, representation 식별성, sparse inference/dictionary learning, human interpretability를 나누는 서사 | 알려진 projection의 sparse recovery 조건을 학습된 VG-SAE의 일반 식별성 보장으로 쓰기 |
| Tubiana–Monasson, PRL2017 | 연결 가중치의 희소성, hard occupancy, 강한 coefficient의 effective count를 구분하고 모형에 맞는 관측량 사용 | RBM의 compositional phase, `L~1/p`, effective temperature를 VG에 그대로 대응시키기 |
| Hou–Huang, PRL2020 | matched prior/likelihood, truth-overlap와 self-consistency, 데이터 수에 따른 추론 조건 구분 | RF-weight correlation q를 support density로 읽기; learned beta를 Nishimori 보장으로 부르기 |

Nature의 [출판사 페이지](https://www.nature.com/articles/s42256-026-01259-z)는 metadata/abstract·Perspective 분류를 확인했다. 구체적인 가정과 수식은 [공개 저자본2503.01824v1](https://arxiv.org/html/2503.01824v1)의 §4–5.3을 읽었다. 출판본 전체와 저자본의 완전한 일치는 확인하지 못했다.

PRL2017은 [저자본1611.06759v2](https://arxiv.org/html/1611.06759v2), 저자의 출판본 PDF 및 [supplement §III](https://www.phys.ens.fr/~monasson/Articles/a105-si.pdf)를 대조했다. PRL2020은 [저자본1911.02344v2](https://arxiv.org/html/1911.02344v2)의 본문과 관련 supplement를 읽었다. 전체 replica 계산을 재증명한 것은 아니다. 개별 읽기 깊이는 두 입장문과 `aris/SOURCE_MANIFEST.json`에 남겼다.

문헌을 권위만으로 수용하지 않았다. 예를 들어 Perspective 저자본 Eq.(8)의 fixed-D squared-error+L1 문제는 coefficient에 대해 convex다. 인접한 non-convex/NP-hard 표현을 우리 원고에 옮기지 않으며, L0 combinatorial optimization과 joint dictionary learning을 구분한다. 이는 저자본의 해당 식에 대한 수학적 점검이고 최종 Nature 출판본의 같은 문장을 확인했다는 뜻은 아니다.

## Phase 1: 수식과 결과를 연결하는 보강

**1. 선택 변수의 interaction을 명시한다.** 고정 x, D, a, beta에서 `u_j=a_j d_j`, residual `r=x-b`라 두면 support energy의 pair coefficient는 `+beta*u_i^T*u_j`다. Log posterior에서는 부호가 반대다. 기존 high-overlap 결과를 단순한 관측으로 나열하는 대신, Gram off-diagonal이 posterior competition을 만드는 위치를 보여준다. 독립 support prior 아래 orthogonal columns이면 이 interaction이 사라진다.

**2. 세 가지 response를 구분한다.** Exact fixed-model count variance는 marginal variance의 합에 covariance 항을 더한 것이다. Locally stable MF stationary branch의 response는 `1^T H^{-1}1`이며, 일반적으로 factorized `sum m(1-m)`와 같지 않다. Gamma마다 dictionary/amplitude/beta를 다시 학습한 곡선의 기울기는 이 둘과도 다르다. 기존 135-cell exact field-response 검증은 재사용하며 신규 결과로 다시 세지 않는다.

**3. 좌표에 따른 weight sparsity와 support occupancy를 분리한다.** Isotropic Gaussian 모형에서 x, b, D를 같은 orthogonal R로 회전하고 a를 고정하거나 encoder를 함수 보존 변환하면 조건부 support posterior는 보존된다. 반면 decoder의 coordinate sparsity는 바뀔 수 있다. Fixed-model 0-fit 대조를 추가하며 Adam을 새로 학습한 궤적의 회전 불변성은 주장하지 않는다.

**4. Prior matching은 작은 oracle 비교로 다룬다.** Geometry와 support co-firing을 독립 조작하는 2×2 tiny exact reference를 SHOULD 후속으로 둔다. Marginal density를 유지한 채 prior dependence만 바꾸고 matched/independent student prior를 비교한다. 이는 Hou–Huang의 weight-posterior 재현이 아니며 trainable correlated-prior SAE를 추가하는 것도 아니다.

**5. Hard support 수와 강한 coefficient 수를 구분한다.** Amplitude stress에서 필요하면 `PR3(z)=(sum |z_j|^3)^2/sum |z_j|^6`를 보조 지표로 사용한다. Zero code는0으로 기록한다. PR3를 true L0나 semantic compositionality의 증거로 쓰지 않는다.

이 보강은 알려진 변분·선형대수 관계의 적용이다. 새로운 상전이 정리나 기존 SAE 전체를 통합하는 정리로 주장하지 않는다. 수식과 추가 평가 사양은 `PHASE1_PHYSICS_BRIDGE.md`에 기록한다.

## Phase 2: 실제로 바꿀 설계

### 확률 공간과 관측량

- `E_opt`: 동일 teacher와 동일 training dataset에서 optimizer initialization/batch order를 바꾼다. 현재 operational primary를 유지한다.
- `E_joint`: 같은 teacher에서 training data를 repeat마다 다시 뽑고 optimizer도 바꾼다. 초기화를 arm 사이 pairing하지만 **data+optimizer 결합 변동**이며 순수 data variance가 아니다.
- Hard-between, soft-total, soft-within, soft-between을 같은 bank에서 계산한다. Hard estimator는 hard rho축, soft challenger는 별도로 matching한 soft rho축을 사용한다.
- 현재 primary를 유지하는 것은 이론적 우월성을 인정한 것이 아니다. 개발 근거로 바꾸려면 모든 비교를 보존하고 새 protocol을 고정한 뒤 untouched worlds에서 검증한다.

특히 모든 repeat가 같은 정확한 hard code에 수렴하면 hard-between U는0이다. 모든 repeat가 같은 잘못된 code에 수렴해도0이다. 그러나 같은 soft m이 반복되어도 soft-total에는 `m(1-m)`가 남을 수 있다. 이 반례를 원래 soft 식 전체에 잘못 적용하지 않는다.

### 최초 bridge wave

기존 개발 world 중 두 family의 첫 world인31000,31004를 사용한다. Reference0는 공유하고 각 ensemble 평균에서 제외한다. 공통 held-out inputs와 paired initialization을 유지한다.

| 항목 | 기본 17 controls | 최대25 controls |
|---|---:|---:|
| E_opt:2worlds×(reference1+repeats3) | 136 | 200 |
| E_joint:2worlds×새repeats3, reference 재사용 | 102 | 150 |
| 고유 학습 fits | **238** | **350** |
| 모두8k일 때4k-step equivalents | **476** | **700** |

모든 기본 control을8k까지 학습하여2k/4k/8k의 전체 곡선을 비교한다. 기존 C0의 gamma{0,2,4} 일부 연장을 대체한다. 추가 control은4k hard-density coverage 규칙으로만 최대 8개 고르고, 새 control도8k까지 학습한다. Basic17 결과가 우선이며 extended bank는4k에 조건부로 선택된 보조 진단이다.

최초 wave에서 얻는 것은 성공 확증이 아니라 실행 가능성·최적화 의존성·ensemble/readout 차이의 구분이다. 두 worlds를 근거로 보편적 안정성을 주장하지 않는다. 공유 reference의 data 비대칭은 기존 E_joint run1을 대체 anchor로 재평가하는 0-fit sensitivity로 확인한다.

### 대형 계획의 상태

9월29일의7,240–10,376 fits는 최초 필수 실행 묶음에서 해제한다. 36개 fresh-world 본검증과 small L1 비교는 bridge·나머지 개발·protocol lock 이후의 조건부 후속이다. Same-bank selectors와 사전 지정 small L1은 main 결과가 좋거나 나빠도 모두 보고한다. 광범위 precision/entropy/variance와 TopK·분포확장 bank는 질문별 분기를 고정한 뒤 실행한다. 불리한 comparator를 보고서에서 지우는 것은 허용하지 않는다.

최초 wave의 비용은 원래의 미측정 가정10–40초/4k-fit와25% overhead를 적용하면 약1.7–9.8 GPUh다. 실제 timing 이전의 제안이며 이번 작업에서 사용하거나 승인받은 비용이 아니다. 고유 fit와 snapshot/continuation 비용을 따로 계산한다.

## 실제 토론에서 바뀐 판단

| 쟁점 | 초기 이견 | 토론·PI 판정 |
|---|---|---|
| Ensemble primary | ARIS는 source에 가까운 data challenger를 중시, scientific는 optimizer-given-data estimand의 명시를 중시 | 현재 primary는 이름을 명확히 해 유지, 독립data challenger 필수 비교, 과학적 근거에 따른 향후 변경은 허용하되 새 version+새 확증 |
| Flat uncertainty | 초기에는 양쪽 모두 converged-correct flatness 강조 | Scientific가 soft-total에는 within term이 남음을 지적; 반례를 hard-between으로 한정 |
| Correlated prior2×2 | Scientific는 첫0-fit gate, ARIS는 새target 구현의 우선순위에 반대 | 기존 exact-response 재해석 먼저, 새prior는 SHOULD 기전확장으로 이동 |
| Full-curve budget test | 세gamma 연장은 부족하다는 데 동의 | 같은17controls의2k/4k/8k,4k기준adaptive freeze로 구체화 |
| Shared reference | PI가 중복fit를 줄이는 공통anchor 제안 | 두 scientist 수용, reference data 비대칭과 alternative-anchor sensitivity 명시 |
| 대형ablation | 모두 필수인가에 대한 우선순위 차이 | 저비용 bridge 우선, 비교군은 결과 무관 공개, 광범위추가 학습은 사전조건부화 |

Independent positions는 `aris/01_position.md`, `scientific/01_position.md`; 실제 반론과 수정은 각 `02_response.md`; 공동안은 `aris/03_joint_recommendation.md`, 최종 endorsement는 `scientific/03_joint_response.md`다. 초기 이견을 지우지 않았다. Operational 합의는 이루어졌으나 어느 ensemble이 실제로 유익한지는 미해결이다.

## 논문 서사에 대한 PI 결정

문제 → 모형 가정 → conditional interaction/추론 → 어떤 ensemble의 uncertainty인가 → density estimator → 별도 recovery 평가의 흐름으로 갱신한다. Phase 1은 Phase 2를 자동 정당화하지 않고 추정 신호가 무엇을 의미하는지 규정한다. C2a 생성밀도 정확도와 C2b 선택효용, estimate 상태와 deployment 상태는 그대로 분리한다.

SOTA, human semantic calibration, RBM 상전이, 새로운 generative model을 필수 기여로 추가하지 않는다. 실제 LM의 CE/KL·probe·개입은 서로 다른 외적 지표이며 human interpretability와 동일하지 않다. 그 주장을 하려면 별도 평가가 필요하다.

기존 VG, Sparse but Wrong, VAEase/variational sparse coding, entropy ELBO, identifiability/stability 문헌은 계속 사용한다. 기존 source ledger를 이어 읽되 새 세 논문으로 출판 가능성·신규성을 확정한 것처럼 쓰지 않는다.

## 방법론 출처

요청에 따라 ARIS의 project-local research-lit/alphaxiv/research-review 절차와 scientific critical-thinking/experimental-design 절차를 서로 다른 agent에 적용했다. 과학적 근거는 원문과 프로젝트 자료이며, skill은 검토 절차의 출처다. 두 agent는 같은 모델 계열의 협업이므로 cross-family 검증이라고 하지 않는다.

Scientific 절차의 provenance: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [현재 arXiv record](https://arxiv.org/abs/2609.00065), [DOI](https://doi.org/10.48550/arXiv.2609.00065). Scientific agent가9월30일 metadata(v2)를 확인했다. 이 자료를 VG-SAE 물리학 주장의 증거로 인용하지 않는다.
