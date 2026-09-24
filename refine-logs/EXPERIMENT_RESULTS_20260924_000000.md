# VG-SAE first-principles 초기 실험 결과

2026-09-24. 계획한 must-run 초기 캠페인을 완료했다. 수식·구현 감사 → exact response → frozen encoder → joint 학습 순서로 실행했다. 설정은 결과를 보기 전에 `configs/first_principles_20260924.json`에 고정했다.

## 핵심 결과

**분산항이 posterior-mean reconstruction과 실제 Bernoulli reconstruction risk를 연결하는 역할이 joint 학습에서 뚜렷하게 드러났다.** 분산항을 제거하면 평균 code의 재구성은 매우 좋아져도 stochastic/hard readout은 크게 나빠졌다. 이는 통계물리학적 유도에서 따라오는 항이 무엇을 담당하는지 설명하는 초기 evidence다. 새로운 free-energy 정리나 baseline 대비 보편적 우위를 입증한 결과는 아니다.

또한 dictionary가 orthogonal이면 exact posterior와 mean-field가 일치했고, 거의 겹치는 atom에서는 conditional dependence와 factorization gap이 나타났다. 학습된 gate encoder에는 그 gap과 별도로 유한 학습 예산의 추가 오차가 남았다.

## 완료 범위와 비용

| 블록 | 완료 | 근거 |
|---|---:|---|
| B1 구현 감사 | scalar/core/adapter +5 baseline 계열 검토 | IMPLEMENTATION_AUDIT, independent regression tests |
| B2 exact/MF response |135 cells =3 worlds×3 coherence×3 beta×5 gamma | exact_summary.csv, sample NPZ |
| B3 frozen encoder |9 fits, 각2000 updates | frozen_summary.csv, checkpoint/target checks |
| B4 joint 학습 |54 fits, 각2000 updates | joint_summary.csv, 각fit config/history/checkpoint |
| B4 profile 평가 |72 rows |36 VG fits×batch16/256 |

총63 학습 fit,126,000 optimizer updates. World는2401/2402/2403이며 반복 grid point를 독립 seed로 세지 않는다. Training+snapshot 합계0.14754 device-hours, GPU가 할당된 worker wall-time 합계0.14925 GPUh. 작은GPU smoke는 약4.53초 추가였다.2GPUh 상한 이내, failed worker0. 정확열거는 CPU에서 실행했다. 별도 대규모 LLM 실험은 수행하지 않았다.

## B1: 구현의 정확성

통상 범위 VG 수식은 논문이 명시한 prior와 일치했다. Soh의 일부 식과 prior 사이 부호 모순을 그대로 복제하지 않았다. 검토 중 아래 실제 오류를 수정했다.

1. BF16 gate/entropy와 scalar entropy tail에서 불확실성·gradient가 소실되는 문제.
2. Profiled log-risk와 beta_eff가 다른 floor를 사용해 같은 batch 복제만으로 beta가 변하는 문제.
3. Activation normalization을 fold하는 BatchTopK에서 threshold 단위가 바뀌지 않아 inference support가 달라지는 문제.

Gated는 문헌에 근거한 RI-L1 변형이므로 auxiliary decoder를 임의로 detach하지 않았다. JumpReLU는 원논문의 pre-ReLU/STE와 일부 조건에서 다르므로 paper-exact라고 하지 않고 주실험에서 제외했다. L1과 TopK도 pinned SAELens recipe 이름으로 기록했다.

변경 전 전체342 tests 통과. 최종 **378 tests 통과**, compileall과 git diff --check 통과. 기존 TransformerLens deprecation warning1개. 값·gradient의 독립 exact enumeration, finite difference, inference conversion 및 새 campaign evaluation을 검증했다.

## B2: conditional dependence와 factorized approximation

아래는 생성모형과 일치하는 beta8/gamma log3에서 seed별 결과의 평균이다. 다른 inference controls도 전체 CSV에 남아 있다.

| Dictionary pair cosine | MF reverse-KL to exact (nats/sample) | Marginal MSE | Exact pair covariance | MF residual≤1e−6 비율 |
|---:|---:|---:|---:|---:|
|0|1.96e−16|2.56e−32|1.19e−17|100%|
|.7|.04958|.00210|−.02797|100%|
|.95|.14112|.01009|−.06497|100%|

Orthogonal의 posterior factorization golden control은 통과했다. 전체135 cells에서 orthogonal analytic marginal 최대오차3.00e−15, 직접 KL와 F+logZ 차이 최대1.33e−14, `−dE[N]/dγ=Var(N)` 중앙차분 오차 최대1.16e−9였다.

높은 overlap에서 negative support covariance와 best-found MF의 gap을 함께 확인했다. 이는 관측을 설명하는 대체 가능한 atom 사이의 조건부 경쟁과 일치한다. Dictionary overlap을 firing correlation으로 표현하지 않는다. Best-found MF에 전역 최소 인증은 없으며, 수렴 잔차가 작다는 것과 전역 최적이라는 것도 다르다.

**C1: 지정한 작은 모형과 overlap 범위에서 지지.** 알려진 항등식의 수치 확인은 검증이고, 그 구조와 근사오차의 관계를 현재 모형에서 관찰한 것이 실험 내용이다. 상전이를 증명한 결과로 부르지 않는다.

## B3: encoder의 추가 오차

다음은 mean±sample SD over3 seeds이며 신뢰구간이 아니다.

| Pair cosine | Encoder reverse-KL | Best-found MF reverse-KL | Gate refinement F 감소 |
|---:|---:|---:|---:|
|0|.35982±.00493|≈0|.35982±.00493|
|.7|.45031±.01245|.04891±.00199|.40140±.01086|
|.95|.55967±.01269|.13587±.00419|.42380±.00852|

모든9 fits에서 감소량은 양수이며 사전.01 nats 기준을 넘었다. Coherence.95에서 MF residual 기준을 충족한 sample은 전체6144개 중6143개였다. 그1개도 제거하거나 성공으로 숨기지 않았다.

특히 orthogonal posterior는 linear gate로 정확히 표현할 수 있다. 따라서 그 조건의 약.36 nats를 encoder 구조의 본질적인 표현력 한계라고 해석하면 안 된다. 현재2000-update 학습/최적화의 잔여 오차가 포함된 결과다. Exact/MF의 차이와 trained encoder/MF의 차이를 구분해 보고할 수 있었다.

중요한 한계도 확인했다. Coherence.95에서는3 seeds 모두 refinement가 reverse-KL을 낮추지만 Brier, marginal NLL, exact marginal MSE는 악화했다. 평균 Brier는 encoder.07248 → MF.07471이며 exact posterior는.06509였다. 따라서 **변분 자유에너지를 낮추는 것과 marginal posterior 확률의 정확도를 개선하는 것은 같은 목표가 아니다.** Brier/NLL은 calibration만을 측정하지 않는다. 이는 feature의 의미적 calibration에 대한 주장이 아니라 known synthetic support와 exact marginals에서 직접 계산한 결과다.

**C2: frozen 설정의 구분 가능성은 지지; joint와 semantic posterior로의 확장은 미확인.**

## B4: 유도된 항의 역할과 readout 차이

다음은 사전에 지정한 gamma2에서3 independent worlds의 평균±sample SD다. 전체 gamma0/2/4, L1/TopK controls의 최종 결과는 모두 CSV에 보존했다.

| VG 설정 | Mean-code MSE | Full stochastic MSE | Hard-code MSE | Expected mask count | Hard code L0 |
|---|---:|---:|---:|---:|---:|
|Global learned β|.01646±.00053|.02962±.00047|.02312±.00054|2.759±.027|1.648±.074|
|Profiled β, full objective|.01805±.00037|.03222±.00053|.02472±.00053|2.811±.005|1.576±.047|
|Profiled, entropy 삭제|.03248±.00545|.04488±.00619|.04097±.00612|1.291±.103|1.018±.089|
|Profiled, variance 삭제|.000179±.000028|4.00493±.99187|1.05227±.28057|4.216±.216|1.092±.438|

Variance 삭제 모델의 stochastic MSE는 삭제된 training term을0으로 놓고 계산한 값이 아니다. 저장한 model의 `Σm(1−m)a²||d||²`를 재계산하여 **동일한 full Bernoulli risk**로 평가했다. 평균 reconstruction만 보면 유리한 모델이 실제 mask sampling이나 hard thresholding에서는 불리할 수 있음을 보여준다.

이는 **profiled β까지 함께 반응하는 objective 삭제의 total effect**다. Variance 항 하나의 직접 인과효과나 feature 혼합의 보편적 해법으로 과장하지 않는다. Entropy 삭제도 expected/hard count와 risk를 함께 바꿨으며 그 변화가 항상 개선이라는 사전 가정은 없었다.

Generating expected support count는2다. Expected mask count와 hard code L0가 서로 다르며 둘 다 정답2에 자동으로 일치하지 않았다. Input-dependent amplitudes의 선택확률을 calibrated semantic confidence라고 부를 근거는 없다.

Profile 평가72 rows는 모두 floor가 작동하지 않았고 Jensen gap은.003515–3.10934였다. 같은 checkpoint라도 full-data profile과 minibatch 평균 profile이 다르다는 수치 근거다. 이 결과만으로 실제 SGD 실패 원인을 입증한 것은 아니다.

Baseline 맥락: TopK k2의 native L0는2, MSE평균.02910, signed dictionary cosine.86184였다. L1 최소 coefficient .001에서는 dense L0≈15.58이었고 가장 큰 .1에서도 L0≈6.57이었다. 이 coarse grid는 matched-L0 우위 검증에 충분하지 않다. VG와 baseline의 parameter count·학습 objective도 같지 않으므로 SOTA 비교나 통계적 우위를 주장하지 않는다.

## 산출물과 재현

- 원본: `outputs/first_principles_20260924/{exact,frozen,joint}/seed_*`의 config, status, result, checkpoints 및 exact/frozen NPZ.
- 표: exact_summary.csv, frozen_summary.csv, joint_summary.csv, profile_summary.csv, paired_differences.csv.
- 그림: `figures/conditional_inference.{png,svg}`, `figures/joint_objectives.{png,svg}`.
- 실행: 같은 run 폴더의 REPRODUCE.md. 모든 final grid를 평가했고 test 기반 재조율은 없다.

별도 결과 검토 `/root/experiment_design_review`는 counts/config/steps/동일 batch stream 및 actual truth를 대조했다. No-variance 저장 checkpoint의 전체 test risk와 두 입력의65,536-state 직접열거를 독립 재계산해 analytic stochastic risk를 확인했다(최대오차1.78e−15). 검토자는 앞서 inference module을 구현했으므로 zero-context 독립 audit이라고 부르지 않는다. Same-family provisional 결과 판정은 C1 지지, C2 제한적 지지다.

초기 캠페인은 완료됐다. 결과는 VG-SAE의 **목적함수 항과 추론 근사를 설명하는 연구**에 사용할 수 있는 구체적 evidence를 제공한다. 새로운 이론적 기여의 완성, 실험 범위 밖 LLM 성능, 최적화 수렴 및 의미적 feature 복원 일반화는 아직 주장하지 않는다.
