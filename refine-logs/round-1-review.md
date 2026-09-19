# Research Refine — Round 1 Review

<details open>
<summary>Full raw reviewer response — gpt-6-astra / ultra</summary>

검토일: 2026-09-19
Reviewer model: `gpt-6-astra`
Reasoning effort: `ultra` — 사용자의 명시적 override
Canonical agent ID: `/root/refine_ultra`
Review provenance: **same-family provisional**. 별도 reviewer agent의 판단이며 cross-family acceptance가 아니다.
검토 대상: `refine-logs/round-0-initial-proposal.md` 전체. Problem Anchor를 먼저 읽었고, 문헌 기록, 실제 pilot JSON, `src/sae_model.py`, `src/sae_train.py`, `src/sae_evaluate.py`를 대조했다.

**Verdict: REVISE. OVERALL SCORE: 8.00 / 10.**

초기 연구 제안으로서 문제와 범위는 좋다. 고정 beta의 이상적 복제식은 맞고, softplus의 비동등성, 기존 dropout 선행, mean–hard gap의 한계를 이미 인정했다. 새 결과가 없다는 이유로 계획을 감점하지 않았다. 남은 문제는 C2의 양성 결과가 목적함수의 harmful preference를 판별할 수 있는지다. 현재 설계에서는 복제 주입, 대칭 유지, feature matching, L0 재조정만으로 일부 관측을 만들 수 있다. 이 경로를 닫으면 작은 실험으로도 의미 있는 진단 연구가 된다.

## 1. 고정된 Problem Anchor와 범위 판정

> VG-SAE의 Bernoulli gate와 입력별 amplitude를 함께 학습할 때, 목적함수가
> feature를 선택하는 표현과 동일 정보를 여러 latent에 나누는 표현 중 무엇을
> 선호하는지 밝히고, 그 선호가 실제 경성 feature 품질에 영향을 주는지 검증한다.
> 기존 VG-SAE 코드와 synthetic/실제 activation 평가 구조를 사용한다.
> 일반적인 dropout 복제 정리, 확률적 SAE 최초성, 보편적 calibration 또는 SOTA는
> 주장하지 않는다. 새 학습 모듈 없이 원인 하나를 판별하는 것이 목표다.

Anchor status: **preserved**. 목적함수의 국소 선호와 실제 feature 선택의 관계를 분리한 점이 옳다. 아래 수정은 이 관계를 식별하기 위한 것이며 새로운 SAE, prior, teacher, calibration network를 요구하지 않는다. 강제로 복제를 넣은 실험만으로 일반 학습에서 자발적인 복제가 발생한다고 주장하면 범위가 달라진다. 현재 제안은 그 주장을 하지 않으므로 이를 유지하면 된다.

## 2. 점수와 보정 상태

| Dimension | Score / 10 | Weight | Weighted contribution | 판단 |
| --- | ---: | ---: | ---: | --- |
| Problem Fidelity | 9 | 0.15 | 1.35 | 원래 질문에 집중하며 과장된 대안을 제거했다. |
| Method Specificity | 7 | 0.25 | 1.75 | 개입 위치와 수식은 명확하지만 C2 optimizer·대칭·주 평가량이 미정이다. |
| Contribution Quality | 8 | 0.25 | 2.00 | VG의 KL–variance 경쟁을 실제 feature 회복과 연결하는 하나의 기여로 좁혔다. 연결이 성립할 조건을 더 선명히 해야 한다. |
| Frontier Leverage | 9 | 0.15 | 1.35 | SAE의 실제 표현 문제에 적절하다. LLM teacher나 새 모듈은 필요하지 않다. |
| Feasibility | 9 | 0.10 | 0.90 | 기존 코드와 작은 합성 실험으로 수행 가능하며 비용 상한을 추정치로 표시했다. |
| Validation Focus | 6 | 0.05 | 0.30 | 세 블록은 작지만 대칭·복제 지표·L0 matching이 C2를 혼동시킨다. |
| Venue Readiness | 7 | 0.05 | 0.35 | 제대로 성공하면 시의성 있는 diagnostic contribution이 될 수 있다. VG에 알려진 복제 대수를 적용한 수준에 머물면 top-venue 기여는 약하다. |
| **Total** |  | **1.00** | **8.00** | **REVISE** |

**COMPOSITE: 0.8000**
**CALIBRATION: none**

**GAP:** 이 검토에 인간이 선별한 known-good/known-bad proposal 3개씩이 제공되지 않아 reference anchor 점수나 근접도를 만들지 않았다. 위 점수는 명시된 rubric에 대한 보정되지 않은 판단이다. READY와의 가장 큰 간격은 문헌 수나 실험 규모가 아니라 Method Specificity와 Validation Focus에 있다. 실제로 가능한 softplus 개입이 무엇을 바꾸는지, 추가 학습에서 어떤 변화가 새로 생겼는지, 그 변화가 중복에 민감한 지표에서만 나타나는지를 사전 규칙으로 구분해야 한다. 성공적 실행을 가정해도 새로 배우는 내용은 VG의 KL–variance 경쟁과 feature 회복 사이의 조건부 연결이어야 하며 일반 dropout 현상의 재현만으로는 부족하다.

## 3. 확인한 메커니즘과 근거

### 3.1 이상적 복제식은 현재 objective와 일치한다

entropy weight가 1이고 prior `pi=sigmoid(-gamma)`가 고정이면 현재 코드의 prior minus entropy는 latent별 Bernoulli KL의 합이다. 단위 decoder atom, 동일 gate 확률, 각 amplitude `a/r`, 고정 beta에서 mean과 threshold 기반 hard reconstruction은 유지되고, independent Bernoulli의 variance 항은 `V/r`가 된다. 따라서

`ΔF(r) = (r−1) E[K_j] − beta (1−1/r) E[V_j]`

이고 `r=2`에서 `beta E[V_j] > 2 E[K_j]`라는 조건도 맞다. 이 비교에는 폭에 따라 달라지는 prior normalizer가 이미 KL 안에 포함되어 있다. hard **출력**이 보존되어도 활성 latent 개수는 보존되지 않는다. 활성인 원 feature가 두 개의 복제로 바뀌면 raw hard L0는 하나 늘어난다.

profiled beta에서 같은 선형식을 쓰지 않겠다는 제안도 맞다. B1의 주 비교를 checkpoint beta 고정으로 두면 가장 명확하다. 다른 beta 설정의 결과는 실제 전체 objective를 재계산해야 한다.

이 대수 자체의 novelty를 제한한 판단은 타당하다. Cavazza 등의 §4는 열 복제로 dropout regularizer가 감소하는 구성을 이미 제시한다. [원 논문](https://proceedings.mlr.press/v84/cavazza18a/cavazza18a.pdf)

### 3.2 가능한 softplus 복제는 작은 오차라고 가정할 수 없다

`t = w_a^T x_centered + b_a`, `a=softplus(t)`, `b=softplus(t−log 2)`라 쓰면 두 복제의 mean 기여는 `2mbd`이고 원래 기여는 `mad`다. 유한 t에서 `2b>a`이며, 복제 후 variance 비율은 `2(b/a)^2`다. 큰 양의 t에서는 variance가 감소하지 않고 증가할 수 있다.

읽기 전용 float64 점검 결과:

| t | mean 기여 비율 `2b/a` | variance 비율 `2(b/a)^2` |
| ---: | ---: | ---: |
| −8 | 1.000084 | 0.500084 |
| 0 | 1.169925 | 0.684362 |
| 4 | 1.663865 | 1.384223 |

제안은 이 차이를 이미 언급하므로 수식 오류는 아니다. 다만 ideal score가 양성인 feature를 그대로 feasible treatment라고 부르면 안 된다. `Δrecon, Δvariance, ΔKL`를 실제로 측정하고 어떤 부호 조합을 C1 성공으로 인정하는지 결정해야 한다.

### 3.3 복제의 유지가 곧 학습된 선호의 증거는 아니다

현재 loss는 Bernoulli sample을 매 step 뽑지 않는 analytic expectation이다. gate row, amplitude row, decoder column이 같고 optimizer state도 같으면 permutation symmetry에 의해 두 복제의 gradient와 AdamW update가 같다. minibatch 순서의 무작위성만으로는 이 대칭을 깨지 못한다.

현재 `VariationalGarroteSAE`를 사용한 별도 읽기 전용 CPU float64 점검에서 동일한 세 종류의 행/열을 구성하고 fresh AdamW로 10 step 학습했다. gate, amplitude, decoder의 복제 간 최대 차이는 모두 `0.0`이었다. 이는 실제 checkpoint에서 자발적 복제가 일어났다는 실험이 아니라 설계의 대칭 문제를 확인한 작은 sanity check다.

따라서 C2에서 복제 수·cosine·participation의 증가나 복제의 잔존만 관측하면, 학습이 선호한 상태와 개입으로 주입한 상태를 구별하기 어렵다.

### 3.4 기존 pilot의 제한을 올바르게 해석했다

실제 `training_seed{0,1}.json`과 `convergence_seed{0,1}.json`을 대조했다. 동일한 data seed를 쓰며, 6000-step fixed-pi의 mean–hard EV 차이는 width 32에서 약 0.0177–0.0179, width 128에서 약 0.0238이다. 마지막 기록 구간 beta 변화도 약 0.45–0.90%로 문서와 맞는다. 이것은 초기 width 차이에 학습 예산이 관여했다는 근거이며 수렴 보증은 아니다. 이 수치를 C1/C2의 확인된 결과로 올리지 않은 판단이 적절하다.

## 4. 수정할 핵심 사항

### CRITICAL — C2의 대칭과 즉시 개입 효과를 분리하라

추가 학습 전에 `t=0−` 원 checkpoint, `t=0+` feasible clone 직후, `t=T` 학습 종료를 각각 기록하라. C2의 학습 후 손상은 최소한 `Q(T)−Q(0+)`의 treatment–control 차이로 정의해야 한다. `Q(T)−Q(0−)`만 보면 softplus 출력 증가와 주입된 중복의 효과가 섞인다.

계속 학습할 모델에는 calibration에서 크기를 고정한 작은 비대칭 perturbation을 넣고 동일한 규칙·seed를 모든 clone arm에 적용하라. perturbation 자체의 mean/hard drift도 `t=0+`에서 측정하라. 새 trainable module은 필요 없다. exact clone은 B1 대수 검증이나 대칭 sanity check로 남길 수 있다. perturbation 없이 tied trajectory를 연구한다면 C2를 그 제한된 대상의 주장으로 낮춰야 한다.

optimizer 상태 이식과 reset을 혼용하지 말라. 가장 단순한 기본값은 모든 warm-start arm의 Adam state를 함께 reset하고, 원 checkpoint에서 beta를 읽어 주 C2 동안 고정하며, 같은 minibatch 순서·learning rate·step 수·decoder normalization을 쓰는 것이다. 실제 beta 재학습의 영향은 이후 필요할 때만 별도 범위로 다룰 수 있다.

### CRITICAL — 복제에 불변인 품질과 raw sparsity 비용을 분리하라

현재 `support_precision_recall`은 Hungarian matching에 포함된 latent만 평가한다. 넓은 dictionary에서는 남은 복제 열의 firing이 precision에 잡히지 않을 수 있다. 반대로 복제 후 raw L0를 원래 값으로 맞추려고 threshold를 올리면 동일 gate 두 개를 함께 제거하여 품질 손상을 만들 수 있다. 전자는 손상을 놓치고 후자는 budget 조정의 효과를 의미 손상으로 오해하게 한다.

최소 수정은 다음과 같다.

1. 주 semantic endpoint 하나를 ground-truth feature 기준으로 정의하여, 완전히 동일한 feature를 나누는 것만으로는 나빠지지 않도록 하라. 예를 들어 calibration에서 정한 dictionary-to-true-feature 할당과 cosine cutoff 아래에서 중복 latent의 gate score를 max로 합쳐 support AP/F1을 측정할 수 있다. 이는 평가용 집계이며 posterior probability라고 부르지 않는다. 정확한 duplicate가 들어간 sanity example에서 지표 불변성을 확인하라.
2. raw latent L0와 경성 reconstruction EV는 따로 보고하라. raw L0 비용 증가는 실제 비용일 수 있지만 ground-truth feature recovery 손상과 같은 주장이 아니다.
3. native threshold의 결과와 calibration에서 budget을 맞춘 결과를 구별하라. discrete ties 때문에 같은 budget을 달성할 수 없다면 결과를 임의 보간하거나 같은 L0라고 쓰지 말고 비교 불가/공통 budget 부재로 기록하라.

원래 Hungarian 지표를 없앨 필요는 없다. 그것만으로 C2를 수용하지 않으면 된다. grouped score와 hard EV가 보존되고 raw L0만 증가하면 결론은 표현 중복의 비용이며, semantic feature 회복 손상이 아니다.

### IMPORTANT — C1의 성공 판정을 정확한 식으로 고정하라

`R=E[0.5||x−mu||²]`, `V=E[sum_j V_j]`, `K=E[sum_j K_j]`라 두면 feasible intervention에서 정확히

`ΔF = beta ΔR + beta ΔV + ΔK`

다. recon drift를 제외한 기여를 `S=beta ΔV+ΔK`로 기록하라. C1의 강한 양성 기준은 held-out에서 `ΔF<−epsilon_F`와 `S<−epsilon_S`가 함께 성립하고, 사전에 정한 mean/hard drift 허용범위를 통과하는 것이다. `ΔF<0`인데 `S>=0`이면 recon 변화가 만든 개선이며 주장한 variance–KL 복제 유인의 양성 증거가 아니다. `S<0`이어도 `ΔF>=0`이면 실제 feasible clone은 objective를 낮추지 못했다.

epsilon과 drift bound의 값은 test를 읽기 전에 calibration/numerical precision/실용적 크기를 근거로 고정하라. pilot을 더 실행하여 양성 결과를 만들라는 요구가 아니다. 목적함수 합계는 width로 다시 평균내지 말고 원래 sample 평균을 유지하라. `KL/V` 비율의 0 근처 불안정성보다 `beta E[V_j]/2−E[K_j]`라는 signed ideal margin으로 rank하면 더 단순하다.

### IMPORTANT — matched control과 C2가 답하는 인과 범위를 하나로 정하라

본문의 random-eligible clone과 B2의 random expansion은 다른 treatment다. 주 비교를 amplitude/frequency 및 feasible drift가 비슷한 high-margin clone 대 low-margin clone으로 정하면 clone 조작 자체가 양쪽에 존재한다. matching caliper와 실패 규칙을 calibration에서 고정하고, KL–variance margin 자체는 treatment 차이이므로 match해 없애지 말라. random expansion은 capacity 확인용 보조 대조이며 필수 주 비교와 혼용하지 말라.

이 비교만으로는 selected feature 중요도 등 관측하지 않은 차이가 남는다. 목적함수의 variance 항이 후속 손상을 **일으킨다**고 쓰려면 같은 B2 안에서 기존 `use_variance_term`을 끈 paired continuation 같은 작은 mechanism ablation이 필요하다. beta·gamma·초기화·optimizer·데이터를 맞추고 high/low clone 차이가 variance 항 on/off에 따라 달라지는지 보라. 새 독립 benchmark 블록은 필요 없다. 이 ablation을 하지 않으면 C2는 사전에 측정한 loss preference와 개입 후 손상의 조건부 연관으로 제한하면 된다. 어느 선택을 하는지 full proposal에서 확정하라.

### IMPORTANT — 수용·기각·판단 유보를 나누어라

현재의 “negative이면 기각”은 범위가 너무 넓다. 다음 세 결과를 구분하라.

- **C1 수용:** 사전에 고른 eligible feature에 대한 feasible loss preference가 고정된 부호·drift 기준을 held-out에서 통과한다. prevalence와 효과 크기도 분모 전체와 함께 보고한다.
- **C1 비지지:** 평가 가능한 후보가 충분했으나 기준을 통과하지 않는다. 검토한 조건과 개입에 대해 비지지라고 쓴다. 일반적인 복제 선호 부재를 증명하지 않는다.
- **판단 유보:** caliper를 만족하는 대조군이 없거나, 모든 feasible clone의 drift가 너무 크거나, 신뢰구간이 실용적 효과 양쪽을 덮는다. 이 경우 C2로 진행하지 않는다.

C2는 하나의 primary semantic metric, 사전에 정한 최소 손상 크기, paired seed 집계 단위와 불확실성 계산을 정하라. `3 data × 3 train`에서 9개를 완전히 독립인 데이터 세계로 취급하지 말고 data seed를 최상위 묶음으로 다루라. B3는 같은 기준의 한 조건 반복이면 충분하다. C1 양성/C2 비지지는 유효한 결과이며, “local preference가 존재하지만 이 조건에서는 확인된 feature harm이 없다”로 끝낼 수 있다.

### MINOR — “충분히 학습됨”의 실행 정의를 붙여라

6000이라는 숫자나 beta의 작은 변화만으로 stationary checkpoint를 보장하지 말라. 고정 evaluation subset에서 마지막 두 사전 checkpoint의 objective components와 clone margin 순위·부호가 안정적인지 기록하고, 불안정하면 bounded extension 또는 미수렴 표기를 적용하라. global convergence 증명을 요구하지 않는다. source checkpoint 선정이 test support 지표에 의존하지 않도록 하면 된다.

## 5. 7 미만 차원의 구체적 조치

**Validation Focus — 6/10.**

- Weakness: C2의 duplicate 유지가 optimizer symmetry로 발생할 수 있고, support matching과 L0 matching이 품질 효과를 바꾸며, 양성·음성·inconclusive 기준이 미정이다.
- Concrete fix: `t=0−/0+/T` paired protocol, 사전 perturbation과 optimizer 규칙, duplicate-invariant primary semantic endpoint, feasible loss 분해의 sign gate, matching 실패 규칙을 B1/B2 내부에 넣는다. causal language를 유지할 경우 기존 variance-term on/off continuation을 포함한다.
- Priority: **CRITICAL**.

## 6. 단순화·현대성·드리프트

**Simplification Opportunities**

1. B2의 주 대조를 high/low-margin matched cloning으로 통일하고 random expansion은 capacity 질문이 실제로 남을 때만 보조로 둔다.
2. AP/F1/recovery/EV 전체를 동등한 성공 기준으로 쓰지 말고 duplicate-invariant semantic endpoint 하나와 hard EV·raw L0 보조 지표로 정리한다.
3. 주 B1/B2에서 checkpoint beta를 고정하여 clone mechanism을 먼저 구분한다. profiled/learned beta 변형의 전체 비교는 핵심 결과가 성립한 뒤 필요할 때 추가한다.

**Modernization Opportunities: NONE.** 현재 SAE 평가·ground-truth benchmark 활용이 질문과 맞는다. LLM teacher, auxiliary network, joint gate 모듈을 넣으면 식별할 원인이 늘어난다.

**Drift Warning: NONE.** 단, “injected clone 이후 취약성”을 “일반 학습의 자발적 복제 원인”으로 확대하지 말 것. 원인 ablation이 없으면 C2의 causal language를 조건부 연관으로 낮출 것.

**Remaining blockers:** C2 symmetry/optimizer protocol; duplicate-invariant semantic endpoint와 L0 비교 규칙; C1의 feasible acceptance gate; control 정의 및 C2 인과 범위.

**Final Verdict: REVISE.** 이 단계에서 필요한 것은 새 긍정 실험 결과가 아니라 위 네 설계 결정을 full proposal에 고정하는 일이다. 수정된 계획은 같은 agent로 다시 검토할 수 있다. READY는 연구 계획의 준비도에 대한 same-family provisional 판단이며 실험 결과나 논문 채택을 보증하지 않는다.

</details>
