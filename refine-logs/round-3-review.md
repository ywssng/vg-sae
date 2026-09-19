# Research Refine — Round 3 Review

<details open>
<summary>Full raw reviewer response — gpt-6-astra / ultra, same reviewer</summary>

검토일: 2026-09-19
Reviewer model: `gpt-6-astra`
Reasoning effort: `ultra` — 사용자의 명시적 override
Canonical agent ID: `/root/refine_ultra`
Review provenance: **same-family provisional**. Round 1–2와 같은 reviewer의 연구 제안 검토이며 cross-family acceptance가 아니다.
검토 대상: `refine-logs/round-2-refinement.md` 전체. 통합 확인을 위해 `refine-logs/EXPERIMENT_PLAN.md`, `refine-logs/EXPERIMENT_TRACKER.md`도 읽었다. 새 실험은 실행하지 않았다.

**Verdict: READY — research-proposal readiness only.**
**OVERALL SCORE: 9.10 / 10.**

핵심 식별 blocker가 해결됐다. 실제 parameter 개입의 loss 선호를 검사한 뒤 같은 초기조건을 추가 학습하며, 주입 직후 효과와 학습 효과를 분리한다. primary outcome은 사전에 정한 cosine cutoff 집합에서 반복 확인하고, AP 순위 손상과 경성 support 손상도 별도 결론으로 둔다. B1의 go/no-go 표본과 B2 최종 평가 표본을 나눈 것은 좋은 추가 수정이다. 이 계획은 제한된 injected-intervention 질문을 구현하고 반증할 준비가 됐다. C1/C2의 실증적 지지나 논문 제출 준비가 확인됐다는 뜻은 아니다.

## 1. Anchor, 초점과 복잡도

Round 0와 수정본의 Problem Anchor가 문자열 수준에서 동일함을 확인했다. **Anchor status: preserved.**

제안은 VG objective가 동일 정보를 나누는 표현을 국소적으로 선호하는지, 그 선호가 사전에 정의한 개입 이후 feature 품질과 어떤 관계를 갖는지 묻는다. 현재 설계는 자발적 복제 발생이나 VG 전체의 유해성을 주장하지 않는다. 경성 feature 품질이라는 Anchor의 질문에는 grouped support F1을 통해 답하고, AP만 변하면 순위 변화로 제한한다.

Dominant contribution: **sharper and focused**. 기존 duplication 대수를 새로운 일반 정리로 포장하지 않고, 실제 learned conditional gates의 KL–variance 경쟁과 제한된 후속 현상을 연결하는 하나의 진단 연구다.

Method simplicity: **tight**. 추가된 항목은 대조군, 데이터 분리와 평가 규칙이다. 새로운 학습 가능한 구성요소나 병렬 연구 주제를 넣지 않았다. Frontier leverage: **appropriate**. 연구 질문에 직접 필요한 SAE 및 ground-truth synthetic evaluation을 사용한다.

## 2. 점수와 보정 상태

| Dimension | Score / 10 | Weight | Weighted contribution | 판단 |
| --- | ---: | ---: | ---: | --- |
| Problem Fidelity | 10 | 0.15 | 1.500 | Anchor를 그대로 보존하고 AP와 실제 경성 support의 결론을 구분했다. |
| Method Specificity | 9.5 | 0.25 | 2.375 | loss, 실제 clone, perturbation, optimizer, matching, split과 판정 경로가 구현 가능하게 정의됐다. 작은 실행 기본값은 handoff에서 고정하면 된다. |
| Contribution Quality | 8.5 | 0.25 | 2.125 | 하나의 mechanism-level diagnostic contribution으로 정리됐다. 참신성은 C1/C2의 실제 연결에 달리며 대수 자체는 기여가 아니다. |
| Frontier Leverage | 9 | 0.15 | 1.350 | 현재 SAE feature 선택 문제에 맞는 접근이다. LLM teacher나 새 모듈을 강제할 이유가 없다. |
| Feasibility | 9 | 0.10 | 0.900 | 작은 synthetic grid와 네 arm의 paired continuation으로 수행 가능하다. 큰 benchmark의 throughput을 미리 가정하지 않았다. |
| Validation Focus | 9 | 0.05 | 0.450 | 주요 대안 설명, metric artifact 및 adaptive gate의 표본 재사용을 통제한다. 세 data world와 제한된 cutoff 집합의 범위를 정직하게 유지한다. |
| Venue Readiness | 8 | 0.05 | 0.400 | 실행이 성공하면 날카롭고 시의성 있는 진단 논문이 될 수 있다. 효과의 실질적 크기와 외적 반복이 아직 없으므로 제출 수준을 보증할 수는 없다. |
| **Total** |  | **1.00** | **9.100** | **READY — proposal only** |

**COMPOSITE: 0.9100**
**CALIBRATION: none**

**GAP:** 인간이 선별한 known-good/known-bad proposal reference가 제공되지 않아 anchored calibration을 주장하지 않는다. Round 2에서 READY를 막던 것은 cutoff 하나의 불연속성이 C2 양성을 만들 가능성과 B1/B2가 서로 다른 초기조건을 검사하는 문제였다. 이번에는 사전 cutoff 집합의 강건성 판정, 경성 F1 corroboration, 실제 perturbed 상태의 C1 재검증과 별도 최종 holdout으로 그 간격을 메웠다. 남은 불확실성은 empirical claim의 참·거짓, 실질적 중요도, 한정된 합성 조건 밖의 범위다. 이는 이 제안이 수행할 연구의 대상이며 긍정 결과를 미리 요구하여 계획의 준비도와 혼동하지 않는다.

## 3. Round 2 조치의 해결 여부

| Round 2 이슈 | 현재 해결 | 판정 |
| --- | --- | --- |
| cosine 0.8 경계가 작은 방향 변화를 큰 AP 손상으로 바꿈 | 0.75/0.80/0.85 모두에서 C2 조건 확인, best-positive cosine·coverage·경계 atom 수 기록 | 해결. 결론은 이 사전 cutoff 집합에서의 robustness로 표현한다. |
| B1은 unperturbed clone, B2는 perturbed clone | 실제 `t=0+`에 cal 및 B1 holdout의 `ΔF/S/drift` gate 재적용, test failure 후 재튜닝 금지 | 해결. |
| AP와 native 0.5의 해석 혼동 | AP는 ranking, hard EV/L0/F1은 native 0.5로 구분 | 해결. |
| AP 하락을 hard support 손상으로 해석할 위험 | grouped F1도 같은 change/difference/interaction 및 cutoff/world 기준을 통과할 때만 hard-support harm 표현 | 해결. |
| go/no-go에 사용한 평가 표본을 최종 확인에 재사용 | synthetic 4096/4096, B3 32k/32k로 B1/B2 holdout 사전 분할 | 해결. |

이전 round의 loss identity, softplus feasible drift, 복제 symmetry, optimizer reset, `t=0−/0+/T`, matched cloning 및 variance on/off 조건도 유지됐다. 수정 과정에서 새 module이나 관련 없는 baseline 요구가 들어오지 않았다.

## 4. 해석할 수 있는 결과와 남는 한계

**C1 양성:** 고정 beta/gamma 및 지정된 feasible clone에서 실제 objective 감소와 variance–KL 기여의 감소가 함께 관측됐다. 이는 해당 checkpoint·개입의 국소 결과다. 이상적 function-level 식의 재현만으로 C1을 지지하지 않는다.

**C2 AP 양성, F1 비지지:** 주입된 clone과 지정된 continuation 조건에서 feature-ranking 저하가 확인됐다. 경성 support 손상은 확인되지 않았다.

**C2 AP/F1 모두 양성:** 지정된 cutoff 집합과 data world에서, 이 주입 초기조건의 feature 회복 손상이 variance 항 on/off에 따라 달라졌다. 일반 학습에서 복제가 자발적으로 생기는 이유, posterior calibration 실패, 다른 beta 최적화 또는 실제 LLM semantic feature의 보편 인과를 증명한 것은 아니다.

**C1 또는 C2 비지지/유보:** 계획의 유효한 결과다. C1 local preference만 남거나 후보 matching이 불가능하면 그 범위로 기록하면 된다. 이를 새로운 module이나 유리한 test 재선택으로 양성화할 필요가 없다.

세 data world는 강한 모집단 일반화나 정교한 유의성 주장의 근거가 아니다. 현재 문서는 이를 operational go criterion으로 명시하므로 proposal blocker가 아니다. cutoff robustness도 metric의 모든 한계를 제거하는 보장은 아니다. 특히 B3의 상관된 true dictionary에서 nearest-truth 할당이 바뀌는 사례는 저장된 atom mapping과 cosine을 살펴 제한을 보고할 필요가 있다. 이는 실제 결과 해석 단계의 잔여 위험이다.

## 5. 비차단성 handoff 메모

아래는 method 재검토를 반복할 사유가 아니라 runner를 만들기 전에 정할 작은 실행 기본값이다.

- source 선택을 10000-step checkpoint의 안정성 확인, 실패 시 14000까지 한 번 연장하는 식으로 명시하고 test 성능으로 checkpoint를 고르지 않는다.
- optimizer seed 일부에 valid matched pair가 없을 때 world 평균에서 조용히 제외하지 않는다. 가장 단순한 처리는 해당 world를 inconclusive로 두고 실패 수를 공개하는 것이다.
- SynthSAEBench의 test 미관찰 feature에 대한 coverage 유보 기준을 test를 보기 전에 숫자로 고정한다.
- 도입부의 “가설을 기각”은 본문의 더 정확한 “tested protocol에서 비지지 또는 판단 유보”와 문구를 맞춘다.
- experiment plan의 “native .5는 hard EV/L0에만”은 아래의 grouped F1 규칙까지 포함하도록 정리한다. 본문 전체의 의도는 이미 분명하다.

`EXPERIMENT_TRACKER.md`는 기존 pilot과 미실행 R001–R004를 분리하고 있다. 후속 runner가 이미 구현되거나 B1/C2가 통과했다고 적지 않은 점도 확인했다. 이 review 자체가 실험 결과를 생성하거나 승인한 것은 아니다.

## 6. 점수 7 미만 차원과 최종 판정

Dimensions below 7: **NONE**.
Remaining blocking issues: **NONE at research-proposal level**.

**Simplification Opportunities**

1. B1 최종 판정과 B2 초기 상태 기록을 하나의 저장 artifact에서 계산하여 두 구현의 수치·설정 차이를 방지한다.
2. AP/F1, cutoff sensitivity와 mapping 진단은 같은 저장 outputs를 재사용한다. 새로운 training arm은 추가하지 않는다.

**Modernization Opportunities: NONE.**

**Drift Warning: NONE.** 고정된 Anchor, 하나의 dominant contribution, 새 trainable module 0개를 유지했다.

**Final Verdict: READY, 9.10/10, same-family provisional.** 연구 계획의 refine loop는 종료할 수 있다. 다음 단계는 문서에 적힌 R001부터 실제 intervention runner와 지표를 구현·검증하고, B1 gate가 통과할 때만 B2를 실행하는 것이다. 실증적 주장, 신규성의 최종 판단 및 submission readiness는 그 결과에 따라 별도로 평가해야 한다.

</details>
