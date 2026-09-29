# VG-SAE: 선택 자유에너지에서 밀도 추정까지

2026-09-29. Phase 1의 실제 초기 결과와 Phase 2의 후속 가설을 묶는 통합 방법 제안. **계획 작성이며 새 실험을 실행하지 않았다.**

## Problem Anchor

- 연구 문제: Variational Garrote 기반 SAE를 명시한 통계물리적 모형과 변분원리에서 유도·이해하고, 그 모형의 밀도 응답과 독립 재학습 간 선택 불확실성으로 적절한 sparse operating density를 추정할 수 있는지 검증한다.
- 핵심 병목: 재구성 오차와 sparsity만으로 올바른 feature 분해나 밀도를 선택할 수 없다. 한 모델 내부의 posterior uncertainty, 모델 간 선택 불안정성, 실제 feature recovery 사이의 연결은 별도로 검증해야 한다.
- 유지할 목표: VG-SAE 자체의 개발과 이해를 중심에 둔다. Phase 1은 모형·추론·목적함수의 근거, Phase 2는 정답을 보지 않는 밀도 추정과 그 유용성 검증이다. Phase 1 성공이 Phase 2 성공을 함의한다고 가정하지 않는다.
- 비목표: 별도 복제 진단 주제로 전환, 모든 SAE를 VG의 특수형으로 선언, SOTA를 유일한 성공 조건으로 강제, 파라미터 없는 보편적 true-L0 발견, semantic posterior calibration 또는 thermodynamic phase transition의 무근거 주장.
- 제약: 이번 요청은 계획 작성·검토다. 새 학습·GPU 실험은 실행하지 않는다. 기존 결과와 원고는 보존한다. 후속 비용은 제안이며 9월 24일의 2 GPUh 승인 범위를 확장한 것으로 취급하지 않는다.
- 완료 기준: 원래 밀도 추정 목표를 포함한 하나의 논문 서사, 실행 가능한 추정 절차, 주장별 실험·반증 기준, Phase 1의 실제 근거와 Phase 2의 미검증 가설을 구분한 계획을 제공한다.

## Technical Gap과 Method Thesis

학습된 feature의 의미는 seed와 density에 따라 달라질 수 있다. 따라서 global regression 변수에 쓰던 VG의 선택 곡선을 SAE의 column index에 바로 적용할 수 없다. Reconstruction을 잘하는 밀도, 생성 support의 기대 count, recovery가 좋은 밀도도 서로 다른 대상이다.

**방법 가설:** 기존 VG-SAE의 학습 구조를 유지하면서, native density로 맞춘 독립 학습들의 feature를 정답 없이 대응시키고 그 선택 불확실성 곡선을 모형에 맞추면, 지정 조건에서 generating density와 유용한 operating point를 추정할 수 있다. 불확실한 경우는 추정을 보류하고 실패 범위를 결과로 남긴다.

Route A는 기존 VG-SAE+단일 곡선 추정기다. Route B는 posterior teacher, 새로운 encoder 또는 learned prior를 추가하는 것이다. 이번 병목은 관측량의 의미와 density 선택을 검증하는 것이므로 A를 채택한다. 새 neural component는0개다. Large LM, distillation, RL을 추가할 직접적 이유가 없다. Alignment·native hard code·held-out selection을 정확히 다루는 것이 필요한 기술 요소다.

## Contribution Focus

- **주된 후속 기여 C2:** learned SAE 좌표에서 uncertainty-informed density inference를 검증하는 구체적인 방법과 적용 범위. C2a generating density, C2b recovery selection을 별도로 평가한다.
- **기반 설명 C1:** conditional free energy와 exact/MF/encoder, objective/readout 차이를 통해 관측량을 해석한다. 기존 Phase1에서 제한적으로 지지됐다.
- **기여로 세지 않는 것:** 알려진 free-energy/variance identity, Soh kernel 자체, Hungarian alignment 자체, seed stability 자체, unit test 통과.

새 주장으로 확정하기 위한 실험은 아직 없다. 성능 개선을 연구 가치의 유일한 기준으로 삼지 않지만, density 추정 효용은 실제 density/recovery 검증 없이 선언하지 않는다.

## Phase 1: 재사용할 근거

9월24일 run의135 exact cells,9 frozen fits,54 joint fits를 seed/config와 함께 재사용한다. Orthogonal conditional model에서는 MF=exact이고 높은 overlap에서 dependence/gap이 관찰됐다. Encoder에는 유한 학습의 추가 오차가 남았고, F를 낮춰도 marginal accuracy가 개선되지 않는 조건이 있었다. Variance 삭제는 mean-code reconstruction을 개선하면서 full stochastic/hard risk를 악화시켰다. 이 결과는 profiled beta까지 반응한 total effect다.

이들은 작은 지정 조건의 설명 evidence다. 새로운 정리, global optimizer 인증, semantic posterior 또는 Phase2 성공을 의미하지 않는다. 원본 `refine-logs/runs/vg-sae-first-principles-20260924/EXPERIMENT_RESULTS.md`를 기준으로 인용한다.

## 두 단계 사이의 수학적 연결과 한계

정렬된 gate의 finite ensemble에서 `mbar(1-mbar)=mean_r m(1-m)+Var_r(m)`가 성립한다. Posterior 내부 uncertainty와 between-fit variation을 따로 기록한다. 이 대수 관계는 semantic correctness를 보장하지 않는다.

Truth-corresponding `(input,feature)` 슬롯 중 active 비율을 q라 하자. Repeat selection probability의 active/inactive 그룹 평균이 mu1/mu0, 분산이 v1/v0이면:

`rho=q*mu1+(1-q)*mu0`

`U=q*mu1*(1-mu1)+(1-q)*mu0*(1-mu0)-q*v1-(1-q)*v0`.

Under-selection의 no-FP와 active exchangeability, 또는 over-selection의 no-FN과 inactive exchangeability에서 기존 piecewise kernel가 나온다. Learned SAE의 정렬·빈도·input별 support는 이 충분조건을 깨뜨릴 수 있다. 일반적으로 input별 kernel의 평균과 평균density의 kernel는 다르다. 이를 설명하고 실제 편차와 추정 실패를 검사한다. 이 적용 계산을 새로운 universal identifiability theorem으로 부르지 않는다.

같은 gamma에서 parameters를 고정한 field response와 서로 다른 gamma에서 다시 학습한 model curve도 구분한다. Primary는 **native achieved density에 조건을 맞춘 재현성 곡선**이며 exact thermodynamic response가 아니다.

## Phase 2 방법

### 입력·출력과 구현 구조

`fixed training world → repeat/control bank → align-split native density matching → signed decoder correspondence → select-split curve → profile fit/abstention → native checkpoint selection → untouched-test evaluator`.

Truth metadata는 마지막 evaluator에만 들어간다. 개발 world는 protocol calibration 전용이다. World는 dictionary와 data의 독립 단위이며 repeat는 같은 world의 optimization 변화다. Controls나 inputs를 독립world로 세지 않는다.

출력은 q_hat, K*q_hat, shape profile, bootstrap sensitivity, coverage/alignment diagnostics, status/reason이다. 보류 시 추정값을 성공으로 채우지 않는다. 별도의 operational policy는 estimate abstention 또는 deployment coverage 부족 시 reconstruction selector로 fallback하고, 이유별 비율을 공개한다. Estimate validity와 bank quantization에 따른 deployment validity는 별도 status로 저장한다.

### 학습·선택 절차

1. 기존 global learned-beta VG, native m>.5, final checkpoint를 primary로 쓴다. 같은 world에서5ensemble repeats+independent reference1을 학습한다. Same-repeat controls는 초기상태와 batch stream을 공유한다.
2. 별도 align split에서 고정 density ladder의 각 target과 가장 가까운 **실제 checkpoint**를 각 run에서 고른다. Bounded label-free control 추가와 수치 tolerance는 EXPERIMENT_PLAN A3에 고정한다. Mask interpolation/threshold tuning은 없다.
3. 각 target에서 reference0에 signed cosine Hungarian으로 대응한다. Unit decoder norm folding을 적용하고 모든dead/active 슬롯을 유지한다. Original index나 true-D alignment를 primary에 사용하지 않는다.
4. 동일 select inputs에서 `R/(R-1)*mean_xj Hbar*(1-Hbar)`로 hard between-fit uncertainty를 계산한다. Soft total/within/between 항은 별도로 기록한다. 실제 achieved density를 x축으로 쓴다.
5. `A*f(rho;q)+b`, A,b>=0인 단일template를 q-grid에 profile한다. Whole-curve fit의 coverage, signal, residual, profile 폭, reference/seed 민감도를 검사하여 식별되지 않는 curve는 보류한다. Exact grid·모든 cutoff·bootstrap 수는 EXPERIMENT_PLAN A5의 단일 기준이다.
6. 추정 q에서 각 repeat의 가장 가까운 native checkpoint를 align split으로 고르고, test에서 NMSE·support recovery를 평가한다. Test-best seed나 threshold는 선택하지 않는다.

### 추정 대상과 비교

Matched-capacity synthetic의 expected generating density 오차(C2a)와 같은 SAE bank의 finite recovery-oracle 대비 regret(C2b)는 별개다. Oracle는 evaluator가 test를 본 hindsight bound이며 population optimum이 아니다. Ktrue>Ksae/real LM에서는 generating count를 유일한 correct operating density로 취급하지 않는다.

Primary 비교는 동일 VG bank에서 reconstruction-only, c_dec, raw-U minimum, within-posterior peak, uniform-control 정책이다. All-world policy와 accepted-only diagnostics를 분리한다. L1에 같은 template를 적용해 generic stability 효과를 검사하고, TopK는 native lattice/coverage 한계를 고려해 recovery comparator로 둔다. TopK의 template ineligibility를 VG density 우위로 계산하지 않는다.

### 원래 mixture 계획을 단순화한 이유

Piecewise template의 비음수 혼합 가중치를 정규화해도 Bayesian posterior나 신뢰구간이 되지 않는다. 많은 후보가 적은 관측점을 설명하면 mixture가 비유일할 수 있다. Primary는 단일 q를 profile하여 식별 가능한지 검사하고, 기존 mixture는 appendix sensitivity로만 남긴다. Bootstrap band도 conditional optimization sensitivity이며 calibrated confidence/credible interval이 아니다.

## Claim-Driven Validation

1. **P1:** 기존 theory/inference/objective evidence와 위 조건부 계산. Phase1 결과의 한계를 그대로 보존한다.
2. **P2-A/B:** 학습 전 축·순열·퇴화·template 가정 검사,8개 개발world에서 실행 calibration,36개 새 본검증world에서 C2a/C2b. World별 예측을 hash한 뒤 test truth를 연다.
3. **P2-C:** 사전 정한 main-world subset에서 precision/variance/entropy, L1/TopK와 단순 selector 비교. 작은 분포 변화와 Stage2/3는 계층적으로 확장한다.

Primary 설정은2dictionary families×3Bernoulli expected densities, K16, constant amplitude1/noise.05다. 새world/test로 검증한다. 생성밀도·관측밀도·recovery 최적점이 같다고 전제하지 않는다. Metrics와 성공/실패율을 cell 및 전체에 보고한다.

C2a는 충분한 report rate와 낮은 error가 함께 필요하다. C2b는 all-world NMSE에서 reconstruction-only 대비 실질적 개선과 c_dec non-inferiority를 사전에 지정한 기준으로 평가한다. Numerical thresholds는 protocol의 운영 선택이며 검증된 power 주장이 아니다. 여러 비교 중 유리한 하나만 골라 결론을 바꾸지 않는다.

## Failure Handling

- Flat/endpoint curve: 숫자를 강제하지 않고 abstain.
- Density coverage 부족/드리프트: native matching 실패와 estimator shape 실패를 구분.
- Reference dependence/duplicates: correspondence의 불안정성을 기록하고 sensitivity gate 적용. 낮은 instability가 정답의 인증은 아니다.
- Stable-but-wrong: recovery evaluator에서 드러날 수 있다. 이를 모든 경우 unsupervised하게 감지한다고 주장하지 않는다.
- Variable firing/noise/amplitude: 생성 count와 useful operating point의 불일치를 사후 목표 교체 없이 보고.
- Worker failure: scientific abstention과 분리, 계획된world를 조용히 제외하지 않음.
- Negative C2: 같은 질문의 실패 범위로 남기고 다른 주제로 전환하지 않는다. C1만으로 full paper contribution이 완성됐다고 선언하지 않는다.

## Novelty와 문헌 위치

출발점은 Soh의 VG/selection curve와 Sparse but Wrong의 density/feature identity 문제다. Variational sparse coding, entropy-based objectives, cross-seed Hungarian alignment와 stability 선행을 인정한다. 후보 차별점은 VG의 conditional-support 구조를 명시한 뒤 **학습된 SAE 좌표에서 native density를 맞추어 곡선 추정이 어디까지 성립하는지 실제 recovery와 연결하는 것**이다. 모형 이름·metric·matching 자체의 신규성은 주장하지 않는다. Primary-source 근거와 metadata는 저장소의 `refine-logs/runs/vg-sae-unified-paper-20260929/LITERATURE_GROUNDING.md`에 기록한다.

## Complexity·Compute·논문 인계

새 trainable component0개, 재사용 VG/SAELens/known kernel. 새 구현은 run-bank schema, blind matcher/selector, single-template profile 및 evaluator다. Historical input-averaged `paper_style_sigma_sel`을 cross-run 지표로 재사용하지 않는다.

Core base7240/max10376 small fits, budget sensitivity의 추가24구간 포함7264/10400 step-equivalents. Measured throughput 이전 가정으로25–145GPUh이며160GPUh의 제안 cap을 둔다. 이전2GPUh 승인에서 자동 확장한 실행이 아니며 이번에는 학습하지 않는다. Conditional SynthSAEBench/LM은 별도 timing/budget를 요구한다. 이것은 계획상 견적이지 사용자 확정 예산이 아니다.

논문은7절/내부10쪽으로 구성하며 C1을 앞부분, C2의 density/recovery 검증을 중심 결과로 둔다. 본문·그림·citation·결과별 주장표는 저장소 루트의 `PAPER_PLAN.md`, exact protocol은 `refine-logs/EXPERIMENT_PLAN.md`, 실행 상태는 `refine-logs/EXPERIMENT_TRACKER.md`를 따른다. 이 run 폴더에도 동일 사본을 보존한다. 원래 밀도 추정 질문을 paper title/intro에서 빼지 않고 아직 미검증임을 명시한다.
