# Research Refine — Round 2 Review

<details open>
<summary>Full raw reviewer response — gpt-6-astra / ultra, same reviewer</summary>

검토일: 2026-09-19
Reviewer model: `gpt-6-astra`
Reasoning effort: `ultra` — 사용자의 명시적 override
Canonical agent ID: `/root/refine_ultra`
Review provenance: **same-family provisional**. Round 1과 같은 reviewer이며 cross-family acceptance가 아니다.
검토 대상: `refine-logs/round-1-refinement.md` 전체. 제안 준비도를 검토했으며 새 실험 결과를 요구하거나 만들어 내지 않았다.

**Verdict: REVISE. OVERALL SCORE: 8.65 / 10.**

이전 네 가지 문제는 대부분 해결됐다. 동일한 복제가 analytic loss 아래에서 계속 같게 남는 문제를 비대칭 초기 perturbation과 `t=0−/0+/T` 기록으로 분리했고, Adam reset과 beta/gamma 고정도 명확하다. high/low clone × variance on/off의 네 arm은 현재의 제한된 인과 질문에 맞는다. 남은 주요 문제는 새 primary metric의 cosine cutoff가 작은 기하 변화에 큰 AP 손상을 만들 수 있다는 점이다. 또한 B1의 양성 판정은 perturbation을 포함한 실제 B2 초기조건에서 확인해야 한다. 추가 학습이나 새 모듈 없이 해결할 수 있는 마지막 평가·연결 규칙이다.

## 1. Anchor와 연구 범위

Round 0와 수정본의 Problem Anchor 문자열이 동일함을 확인했다. **Anchor status: preserved.**

dominant contribution은 이제 “고정된 VG checkpoint에서 예측한 loss 선호가, 사전에 정의한 clone 개입 이후 variance 항에 의존하는 feature 품질 변화와 연결되는가”로 더 명확하다. 자발적인 복제 발생, 범용 SAE 우위, calibrated semantic probability를 주장하지 않는다. 이 제한은 타당하며 full-scale real-model 실험을 이번 계획의 필수조건으로 추가할 이유가 없다.

Method simplicity: **tight**. 새 trainable module은 없고 기존 loss의 variance 항과 실험용 초기조건만 사용한다. Frontier leverage: **appropriate**. 현대 SAE의 feature recovery 질문을 직접 검사하며 teacher나 새로운 backbone은 필요하지 않다.

## 2. 점수

| Dimension | Score / 10 | Weight | Weighted contribution | 판단 |
| --- | ---: | ---: | ---: | --- |
| Problem Fidelity | 9 | 0.15 | 1.35 | Anchor가 보존됐다. AP의 순위 손상과 경성 support 손상은 결론에서 구분해야 한다. |
| Method Specificity | 9 | 0.25 | 2.25 | split, source 안정성, matching, perturbation, optimizer, arm 및 부호 기준이 구체화됐다. 실제 perturbation 이후 C1 gate 연결이 남았다. |
| Contribution Quality | 8 | 0.25 | 2.00 | 조건부 KL–variance 경쟁과 feature 회복의 관계라는 한 기여를 유지한다. 알려진 복제 대수만으로 novelty를 주장하지 않는다. |
| Frontier Leverage | 9 | 0.15 | 1.35 | 질문에 적합한 SAE ground truth 평가를 사용한다. 추가 foundation-model 구성요소는 불필요하다. |
| Feasibility | 9 | 0.10 | 0.90 | 네 arm과 작은 synthetic grid는 현실적이며 throughput과 범위 제한도 정직하다. |
| Validation Focus | 8 | 0.05 | 0.40 | 대조·시간 차분·개입 범위는 개선됐다. 새 hard cosine cutoff의 metric artifact가 남았다. |
| Venue Readiness | 8 | 0.05 | 0.40 | 실행이 성공하고 결론이 cutoff와 초기조건 artifact를 견디면 날카로운 진단 연구가 될 수 있다. |
| **Total** |  | **1.00** | **8.65** | **REVISE** |

**COMPOSITE: 0.8650**
**CALIBRATION: none**

**GAP:** 인간이 선별한 reference proposal 묶음이 없으므로 이번에도 calibration을 주장하지 않는다. Round 1보다 Method Specificity와 Validation Focus가 좋아졌고, 성공적으로 실행했을 때의 연구 질문도 선명해졌다. READY와의 차이는 새 empirical positive result의 부재가 아니다. 현재 성공 기준에는 평가용 cosine threshold를 넘나드는 것만으로 만들어지는 양성 결과가 들어갈 수 있고, B1이 검사한 초기조건과 B2가 학습하는 초기조건이 완전히 일치하지 않는다. 이 두 항목과 AP/경성 결정의 해석 규칙을 고정하면 제안 단계에서 요구할 핵심 식별 조건은 충족할 수 있다.

## 3. Round 1 조치의 해결 여부

| 기존 이슈 | 현재 변경 | 판정 |
| --- | --- | --- |
| 동일 복제의 optimizer 대칭 | seeded anti-symmetric perturbation, cal drift bound, 네 arm Adam reset | 해결. exact copy의 잔존을 성공 기준으로 쓰지 않는다. |
| 주입 직후와 학습 후 변화 혼동 | `t=0−`, `t=0+`, `t=T`, `Q(T)−Q(0+)` | 해결. |
| Hungarian matching과 raw L0의 혼동 | duplicate-invariant grouped max AP, raw L0/hard EV 분리, ties에서 비교 유보 | 원래 문제 해결. 새 cosine cutoff artifact는 아래에서 다룬다. |
| feasible loss 선호 정의 | `ΔF`, `S=beta ΔV+ΔK`, signed margin, drift·epsilon 기준 | 대부분 해결. perturbation 포함 초기조건에 재적용 필요. |
| random clone과 random expansion 혼용 | high/low matched clone의 단일 주 비교 | 해결. |
| 목적함수 의존성과 선택 feature 차이 혼동 | clone high/low × variance on/off interaction | 제한된 injected-intervention 주장에는 적합. mediation이나 자발 발생을 증명한다고 쓰지 않는다. |
| success·non-support·inconclusive 구분 | match/stability/coverage 실패의 판단 유보, 세 data world 조건 | 해결. 세 world의 운영상 go rule을 유의성 검정이라고 부르지 않는 점이 좋다. |

## 4. 남은 조치

### CRITICAL — cosine cutoff의 불연속성이 C2 양성을 만들 수 있다

현재 metric은 atom의 best positive cosine이 0.8 미만이면 unmatched로 처리하고 해당 true feature의 AP를 0으로 둔다. 따라서 gate가 바뀌지 않아도 단 하나의 좋은 atom이 `0.80001 → 0.79999`로 이동하면 해당 feature의 AP가 1에서 0으로 떨어질 수 있다. true feature가 32개라면 macro AP 변화는 `−1/32 = −0.03125`로, 제안의 `−0.01` 손상 cutoff를 넘는다. 이 예에서는 실제 atom 방향 변화와 hard reconstruction 변화가 임의로 작을 수 있다.

duplicate invariance sanity test는 이 문제를 잡지 못한다. 동일 열을 추가하면 threshold 양쪽 위치가 그대로이기 때문이다. high/on arm만 경계를 넘고 나머지 arm은 넘지 않으면 `Q_H` 변화, `D_on`, `I`가 모두 음수가 되는 양성 패턴도 가능하다.

최소 수정은 같은 저장 출력에서 사전에 정한 가까운 cutoff들을 재계산하는 것이다. 예를 들어 0.75/0.80/0.85를 test 전에 고정하고 semantic-harm 결론의 방향·크기가 모두에서 유지되도록 요구할 수 있다. 0.80에서만 양성이거나 경계 통과가 전부라면 threshold-sensitive/inconclusive로 남겨라. 각 true feature의 best positive cosine 변화, cutoff별 coverage와 경계 근처 atom 수를 함께 보존하면 판정 원인을 확인할 수 있다. 추가 학습은 필요 없다. 다른 연속적인 진단을 택해도 되지만 새 복잡한 metric 체계를 발명할 필요는 없다.

이 수정은 cutoff 0.8이 잘못된 보편 값이라는 주장이 아니다. 현재의 이분법적 metric만으로 실질적인 semantic harm을 확정하지 말라는 요구다.

### IMPORTANT — C1은 B2의 실제 perturbed 초기조건에 적용해야 한다

현재 B1은 bias를 `−log 2`로 옮긴 unperturbed feasible clone을 평가한다. B2는 이후 decoder·gate·amplitude에 perturbation을 추가한다. drift bound를 통과하는 것만으로 `ΔF`와 `S`의 음수 부호가 유지된다는 보장은 없다. `epsilon=1e-3`의 parameter 변화가 `1e-4 nat/sample`의 loss margin을 넘을 수도 있다.

perturbation 크기·seed를 calibration에서 고정한 뒤, 실제 `t=0+` high/on 초기조건에 C1의 `ΔF/S/drift` gate를 다시 적용하라. low/on control의 실제 margin과 차이도 기록하라. on/off 두 branch는 같은 초기 parameters에서 출발해야 하므로 이 판정 때문에 branch마다 다른 perturbation을 쓰지 말라. on에서 판정한 C1 유인이 off에서도 존재할 필요는 없다. off는 원인 항을 제거하는 대조군이다.

실제 perturbed high clone이 held-out gate를 통과하지 못하면 “B1 local result는 남지만 해당 B2 continuation은 판정 대상 아님”으로 정리하라. test failure 뒤 epsilon·atom을 다시 고르는 것은 허용하지 말라. 이것은 새 실험 블록이 아니라 기존 B1/B2가 같은 treatment를 가리키게 하는 연결 규칙이다.

### IMPORTANT — AP 순위와 경성 support의 결론을 명확히 구분하라

AP는 threshold-free metric이므로 “native threshold .5가 primary”라는 문장은 현재 AP 정의와 맞지 않는다. native .5는 hard EV, raw L0 및 경성 support 평가에 적용된다고 정확히 적어라.

또한 AP는 내려가도 모든 gate가 threshold 0.5의 같은 쪽에 남아 hard support 결정은 전혀 바뀌지 않을 수 있다. 이 경우에는 feature-ranking deterioration이 관측된 것이며 hard support deterioration은 입증되지 않았다. AP를 primary로 유지해도 된다. 다만 immutable Anchor의 경성 feature 품질에 대한 답을 내거나 hard-feature harm이라고 쓰려면 같은 grouped score에서 `.5` support F1 같은 간단한 경성 지표를 보조로 확인하고, 그 지표가 지지한 범위로 결론을 제한하라. 새 모델·학습·large benchmark를 요구하는 수정이 아니다.

내 Round 1에서도 AP를 가능한 primary endpoint로 제안했다. 이번 수정으로 평가 대상이 구체화된 만큼 AP의 threshold-free 성질과 hard decision을 구별하는 해석 규칙까지 명시하는 것이 필요하다. 순위 변화와 경성 변화가 다르면 그 차이 자체를 정직하게 보고하면 된다.

### MINOR — 불명확한 실행 문구를 짧게 고정하라

- 첫 설명의 “복제 선호가 없거나 손상이 없으면 가설을 기각”은 뒤의 더 정확한 non-support/inconclusive 규칙으로 통일하라.
- source checkpoint는 어떤 규칙으로 하나를 정하는지 명시하라. 예를 들어 10000에서 마지막 두 checkpoint 안정성을 확인하고 실패할 때만 14000으로 연장하는 방식이면 충분하다. test 결과로 6000/8000/10000 중 하나를 고르지 않는다.
- optimizer seed 하나에 valid pair가 없을 때 나머지만 평균내는지, 해당 data world 전체를 inconclusive로 두는지 사전에 정하라. 빠진 seed를 좋은 seed로 대체하지 않는다.
- SynthSAEBench의 “미관찰 feature가 많으면 보류”에서 coverage 유보 cutoff를 test 전에 숫자로 고정하라.

이 네 항목은 `/experiment-plan`으로 넘길 수 있는 작은 실행 선택이다. 추가 학습이나 새로운 연구 방향은 필요 없다.

## 5. 점수 7 미만 차원

**NONE.** 낮은 차원이 없어도 primary outcome의 식별 blocker가 남으면 READY가 아니다. 위 CRITICAL 항목은 이번 round의 blocking issue다.

## 6. Simplification / Modernization / Drift

**Simplification Opportunities**

1. C1 판정과 B2 initial-state 기록을 같은 artifact에서 계산하여 “unperturbed clone에서는 양성, 실제 treatment에서는 미확인”이라는 이중 기준을 없앤다.
2. cutoff robustness와 native hard support 평가는 기존 저장 output으로 수행한다. 추가 training arm이나 학습 가능한 평가기를 만들지 않는다.

**Modernization Opportunities: NONE.** 현재 방법이 질문에 적합하다.

**Drift Warning: NONE.** 주입된 clone과 고정 beta/gamma의 conditional intervention이라는 범위를 유지했다. 다만 AP 결과를 실제 hard support 결과와 같은 것으로 쓰지 않아야 한다.

**Remaining action items:** cosine-cutoff robustness gate; actual perturbed `t=0+`에 대한 C1 일치; AP와 hard-support claim의 구분. 이 항목을 해결한 full proposal을 같은 agent에 보내면 된다. 새로운 긍정 empirical result를 얻는 것은 다음 review의 전제조건이 아니다.

**Final Verdict: REVISE.** 이 점수와 판정은 초기 제안의 준비도에 대한 same-family provisional 평가다. 긍정 실험 결과나 submission readiness를 인증하지 않는다.

</details>
