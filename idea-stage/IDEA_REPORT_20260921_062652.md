# VG-SAE 개발 파이프라인을 다듬기 위한 Idea Discovery

실행: `vg-sae-development-20260921` · 날짜: 2026-09-21 · 상태: idea-discovery 완료 · 후속 확증 실험은 계획 단계.

**주 연구는 Variational Garrote 기반 새로운 SAE 개발이다.** 이 저장소는 그 연구를
지금까지 구현·실험한 작업물이며, 이번 실행은 방법·학습·추론·검증 절차를 다듬는다.
2026-09-19의 복제 진단 주제 선정은 요청을 오해한 것으로 현재 목표를 규정하지 않는다.
이전 문서와 수치는 [보존본](runs/vg-sae-development-20260921/prior_scope_snapshot/README.md)에 남겼다.

## Problem Anchor

현재 구현된 Variational Garrote 기반 SAE를 출발점으로, VG의 확률적 support
선택과 amplitude 추정이 SAE 학습에 제공하는 이점을 명료화하고 필요한 방법·학습·
추론·평가 절차를 개선한다. 합성 ground truth, 현실적인 synthetic benchmark,
실제 모델 activation의 단계적 검증으로 새로운 SAE 방법으로서의 기여를 평가한다.
주 연구 목표는 VG-SAE 개발이며, 진단 실험은 그 방법을 개선하고 검증하기 위한 도구다.

## Literature Landscape

[현재 문헌 비교](runs/vg-sae-development-20260921/evidence/literature_update.md)와
[32개 출처](runs/vg-sae-development-20260921/evidence/literature_sources.json)를 검토했다.
VAEase(ICML2025), Tonolini VSC, Fallah–Rozell thresholded VSC, Geadah SVAE를
이전 28개에 추가했다. '확률적 SAE 최초' 대신 목적함수·추론·precision 차이를 비교한다.

현재 방법은 Bernoulli support를 변분화하고 amplitude를 점 추정한다.
선형 dictionary 아래 support expectation을 정확히 계산하여 variance correction과
정규화 Bernoulli KL로 학습한다. 이 training objective에는 latent MC 또는
hard threshold의 STE가 필요하지 않다. 이것이 방법 정의이며, 성능상의 이득은 별도 검증한다.

Gated의 support/magnitude 분리, variational sparse coding의 analytic expectation,
spike-and-slab prior는 선행이다. 읽은 자료에서 현재 전체 조합과 동일한 방법을
확인하지 못했다. 아래 신규성 심사는 조건부 진행으로 판정했다.
입력별 a(x)의 prior·entropy를 포함하지 않으므로 일반 생성모델 전체의 ELBO나
보정된 semantic posterior라고 주장하지 않는다. learned global beta와 minibatch
profiled 구현은 동일한 stochastic objective라고 간주하지 않는다.

## Current Pipeline

[현재 코드·전체곡선 감사](runs/vg-sae-development-20260921/evidence/pipeline_audit.md)와
[원본 근거](runs/vg-sae-development-20260921/evidence/pipeline_evidence.json)를 보존했다.

- Stage1:12개 amplitude/frequency/beta 조건, 각273controls의 one-seed 결과가 있다.
  exponential에서 낮은 coefficient error가 보이지만 기존 test-curve 최소값 및
  서로 다른 L0로 고른 관찰이므로 새 cal/test에서 다시 검사한다.
- Stage2:13개 평가root를 확인했다. 낮은L0에서 mean/hard gap이 작은 설정도 있으며,
  중간L0에는 강한 baseline 대비 개선할 여지가 있다. calibration stream 재사용과
  one-seed 한계를 유지해서 읽는다.
- Stage3:완료된 모델 checkpoint 일부가 있으나 평가표가 없다. 새 대형 training보다
  저장한 Gemma L5 checkpoint의 평가를 먼저 계획한다.

## Ranked Ideas

여기서 순위를 매긴 대상은 **이미 선택된 VG-SAE 안의 개발 선택지**다.
VG-SAE 개발 자체를 진단 연구와 경쟁시켜 탈락시키지 않는다.
[전체12개 선택지](runs/vg-sae-development-20260921/evidence/development_options.json)와
[독립 jury의 실행 경로](runs/vg-sae-development-20260921/evidence/development_routes.json)를 보존했다.

| 순서 | 개발 선택 | 이번 처리 |
|---:|---|---|
| 1 | VG-D05: 독립 calibration과 같은 sparsity·tuning budget으로 비교한다 | P1 |
| 2 | VG-D02: beta를 같은 목적함수 안에서 충분히 학습하는 방법을 정한다 | P2 |
| 3 | VG-D08: 기존 VG encoder 뒤에 소수의 정확한 gate 좌표 업데이트를 시험한다 | P3 |
| 4 | VG-D04: 학습 posterior와 배포 hard code의 사용 규칙을 확정한다 | P3 |
| 5 | VG-D01: 현재 VG 목적함수의 variance와 entropy 기여를 분리한다 | first_required_mechanism_followup |
| 6 | VG-D06: exponential amplitude와 적은 학습표본에서 주방법의 장점을 확인한다 | P1 |
| 7 | VG-D12: 저장된 Stage3 모델부터 평가해 현실 activation 증거를 완성한다 | next_application_evaluation |
| 8 | VG-D09: support 선택 이득과 amplitude shrinkage 이득을 분리한다 | P3 |
| 9 | VG-D03: 같은 최종 VG 목적함수로 가는 sparsity warmup을 검사한다 | deferred_training_option |
| 10 | VG-D07: noise와 support correlation에서 기본 VG-SAE의 유지 범위를 검사한다 | deferred_generalization |
| 11 | VG-D11: gate 확률에 허용되는 해석의 범위를 검증한다 | claim_conditional_validation |
| 12 | VG-D10: 희귀 feature 실패가 확인될 때만 feature별 prior를 시험한다 | bottleneck_conditional_extension |

후보 탈락은0개다. D04 출력 계약과 D05 독립 선택 절차는 결과 부호와 관계없이 필요하다.
D01 variance×entropy ablation은 이후 주방법의 기여를 검증하는 첫 후속이다.
진단은 방법 개발의 도구이며 주 연구를 바꾸는 종착점이 아니다.

## Pilot Results

이번 파일럿은 같은 VG-SAE의 세 개발 경로를 검사했다. P3b는 P3 결과를 본 뒤
추가한 탐색 후속이며 네 번째 독립 연구 아이디어나 사전 확증 실험이 아니다.
재현 코드·프로토콜·작은 원본은 [실행 폴더](runs/vg-sae-development-20260921/)에 있다.

### P1: 기존 방법의 신호를 새 샘플에서 확인

기존 두 Stage1 조건의 546개 checkpoint를 새 calibration 2,048개에서 평가했다.
L0 상한 4/8/16 아래 cal hard latent error로 36개 선택을 동결한 뒤,
24개 고유 checkpoint를 독립 test 4,096개에서 평가했다. L1-GMM threshold는
원래 train에서 fit한 값이다. Dictionary는 원래 seed/config에서 복원하고
원본 test의 support/z가 bitwise 일치하는지 확인한 뒤 고정했다.

| 조건·L0 상한 | VG test error | VG actual L0 | 비교 baseline | Baseline error | Baseline actual L0 |
|---|---:|---:|---|---:|---:|
| exponential · 4 | .69878 | 3.002 | TopK 4 | .74837 | 3.992 |
| exponential · 8 | .69878 | 3.002 | TopK 6 | .73693 | 5.879 |
| exponential · 16 | .69878 | 3.002 | TopK 6 | .73693 | 5.879 |
| constant · 4 | .87248 | 3.366 | TopK 4 | .96311 | 4.000 |
| constant · 8 | .85281 | 4.158 | L1-GMM, coef 1 | .88977 | 5.193 |
| constant · 16 | .85281 | 4.158 | L1-GMM, coef .5 | .84428 | 13.101 |

표의 baseline은 각 방법 내부의 cal-selected 결과 중 **test error가 가장 낮은
비교 항목을 서술적으로 표시**한 것이다. 새 B1의 go/no-go 비교자는 test 전 cal에서
고정한다. 이 표 자체를 사전 고정한 baseline 대비 확증적 승리로 사용하지 않는다.

Exponential cap 8에서 VG의 F1은 .32709, TopK 6은 .29710이지만,
input EV는 .47784 대 .51410이다. 계수/선택과 fidelity 사이 절충이 남는다.
이는 같은 상한 아래 결과이며 exact matched-L0가 아니다. 원래 불균등 HPO와
한 training/dictionary seed의 한계가 남고, 같은 checkpoint가 반복 선택된 cap을
독립 반복으로 세지 않는다. Constant cap 16의 결과도 함께 유지한다.

[원본 표](runs/vg-sae-development-20260921/evidence/pilots/holdout/selected_test_metrics.csv) ·
[선택 동결](runs/vg-sae-development-20260921/evidence/pilots/holdout/frozen_selection.json) ·
[프로토콜](runs/vg-sae-development-20260921/pilots/holdout_protocol.md).

### P2: beta 초기화보다 checkpoint 선택이 중요했던 작은 조건

3 recipe × 2 gamma × 2 amplitude × 3 paired seeds로 36개 모델을 학습했다.
작은 d16/width64, true K=4, noise .05 조건에서 1,001 및 6,000 updates를 평가했다.
Learned beta-init 1과 train-only moment 초기화는 같은 full objective다.
Minibatch-profiled arm은 다른 stochastic objective의 reference다.

Moment 초기화의 이득은 조건/seed에 따라 바뀌므로 기본값으로 채택하지 않는다.
Default learned/gamma 2의 세 seed 평균에서는 더 긴 학습이 입력 EV를 개선하면서
coefficient recovery를 악화시켰다.

| Amplitude | Updates | Hard latent error | F1 | Input EV | Actual L0 |
|---|---:|---:|---:|---:|---:|
| exponential | 1,001 | .90063 | .28779 | .81980 | 3.14657 |
| exponential | 6,000 | 1.12136 | .21238 | .94164 | 6.05607 |
| constant | 1,001 | .99517 | .31260 | .62535 | 2.58496 |
| constant | 6,000 | 1.20330 | .18336 | .95709 | 9.34196 |

따라서 다음 비교는 모든 방법에 같은 수의 checkpoint 후보를 제공하고,
cal recovery와 L0로 선택한다. 이 작은 조건의 변화만으로 full-size에서의
보편적 실패나 순수 과적합을 확정하지 않는다. LR decay/warmup은 이번에 시험하지 않았다.

[36-model 분석](runs/vg-sae-development-20260921/evidence/pilots/training/analysis.json) ·
[프로토콜](runs/vg-sae-development-20260921/pilots/training_protocol.md).

### P3: gate 보정의 품질 신호와 비용

원래 dictionary와 5개 고정 checkpoint, train에서 얻은 beta를 사용했다.
Fresh cal/test 각 512개에서 순차 gate 0/1/3 sweeps와 native/count-matched readout,
모든 방법에 같은 support-fixed NNLS control을 비교했다. Cal-selected 1 sweep은
입력별 활성 수를 유지하며 test error .737010→.700069(약 5.01% 감소),
EV .417488→.500653을 얻었지만 현재 구현의 latency는 약 992배였다.
Source support NNLS만으로 error .651088을 얻어 generic amplitude refit의 효과가 컸다.
NNLS 이후 추가 gate error 개선은 cal에서는 악화, test에서는 약 1.01%였다.

같은 경로의 **탐색 후속 P3b**는 별도 fresh split에서 한 번의 vectorized gate update를
검사했다. Cal-selected eta .5는 다음 결과를 보였다.

| VG readout | Test error | F1 | EV | Actual L0 | ms/128 inputs |
|---|---:|---:|---:|---:|---:|
| Native | .706038 | .338201 | .444171 | 3.25 | .1702 |
| eta .5 + count-preserving | .678962 | .347255 | .502693 | 3.25 | .7436 |
| 공통 amplitude PG | .675913 | .338201 | .488605 | 3.25 | .4166 |

Error 개선은 약 3.835%지만 latency 4.3687배로 사전 3배 기준을 넘었다.
PG는 더 빠르고 coefficient error도 조금 낮다. Gate에는 EV/F1의 다른 절충이 남는다.
동일 NNLS 뒤 추가 gate error 개선은 cal .689%, test 1.422%였지만 actual L0가
달라 exact-L0 비교라고 부르지 않는다. 병렬 Jacobi에는 일반적인 단조 감소 보장이 없다.

**결정:** 기본 feed-forward VG 추론을 유지한다. 현재 solver/teacher를 채택하지 않는다.
순차 구현의 비용이 모든 보정 구현의 하한이라고도 결론내리지 않는다. 이번 run에서
test 결과를 본 추가 최적화를 반복하지 않았다.

[P3 결과](runs/vg-sae-development-20260921/evidence/pilots/refinement/findings.md) ·
[P3b 결과](runs/vg-sae-development-20260921/evidence/pilots/parallel_refinement/findings.md) ·
[P3b 사전 프로토콜](runs/vg-sae-development-20260921/pilots/parallel_followup_protocol.md).

### 실행 범위와 계산량

P1 41.27초, P2 train+test 합계 약 931.32초, P3 63.11초, P3b 2.25초의 주요
타이머 합계는 약 **0.2883 GPUh**다. 서로 다른 GPU에서 겹쳐 실행한 시간을 합산했다.
Import·개별 smoke·독립 검증 시간은 포함하지 않았다. 대규모 새 training sweep과
Stage3 평가는 다음 계획이며 이번에 완료했다고 표시하지 않는다.

## Novelty Verification

Fresh reviewer `/root/vg_method_novelty`, GPT-6 Astra **ultra**의 실제 판정은
**PROCEED WITH CAUTION, 6/10**이다. Same-family provisional이며 최초성 확증은 아니다.
[32개 검색과 22개 1차 출처](runs/vg-sae-development-20260921/evidence/novelty_search.json)를
사용해 최근 6개월과 각 핵심 주장에 대해 여러 query 형태를 확인했다.

직접 선행은 [원래 Variational Garrote](https://arxiv.org/abs/1109.0486),
[Gated SAE](https://arxiv.org/abs/2404.16014),
[entropy-based ELBO sparse coding](https://arxiv.org/abs/2311.01888),
[VAEase](https://proceedings.mlr.press/v267/lu25w.html),
[VSC](https://proceedings.mlr.press/v115/tonolini20a.html),
[thresholded VSC](https://proceedings.mlr.press/v162/fallah22a.html)와
paired mean-field/S3C 계열이다. Analytic VI, spike-and-slab, selection/amplitude 분리는
기존 개념이므로 넓은 최초성 주장은 제외한다.

검토한 자료에서 현재 **Bernoulli support-only variational objective + 입력별 point
amplitude + amortized encoder + learned linear dictionary** 전체와 동일한 조합은
확인하지 못했다. 그러나 조합의 이름만으로 충분한 기여가 되지 않는다. 구체적인
hard-code recovery 이득과 조건, full VG objective의 작동을 B1/B2로 보여야 한다.
VAEase/VSC보다 우수하다는 주장은 해당 방법을 실제로 비교하기 전까지 하지 않는다.
ProbTopK는 1차 indexed PDF에서 방법을 확인했지만 직접 PDF 접근 및 일부 구현·저자
정보의 확인 한계가 남는다. 이를 확정된 source detail로 채우지 않는다.

## External Critical Review

Fresh reviewer `/root/vg_pipeline_review`, GPT-6 Astra **ultra**의 실제 판정은
**PROCEED WITH CAUTION**이다. Same-family provisional이며 논문 제출 준비 판정이 아니다.
원래 VG-SAE 개발 anchor가 유지됐고, P1의 실제 신호와 P2/P3의 혼합 결과를 확인했다.

| 검토 지적 | 반영한 수정 | 남은 검증 |
|---|---|---|
| Reconstruction plateau가 recovery 선택 기준일 수 없음 | 모든 방법의 checkpoint 후보 수를 맞추고 cal recovery/L0로 선택 | 원래 크기의 새 training seeds |
| No-variance arm의 m↓/a↑ compensation 가능 | C2를 support–amplitude coupling으로 좁히고 m/a scale·세 risk 기록 | 실제 학습에서 발생 여부 |
| Baseline/readout/HPO 비교가 모호함 | Native L1과 train-GMM을 구분하고 cal-only comparator·trial 예산 명시 | 계획의 runner 구현과 실행 |
| Synthetic oracle 선택은 실제 activation에서 불가능 | Stage3는 L0와 사용 가능한 reconstruction/CE로 선택 | Fresh text 기반 평가 |
| Gate 보정의 비용·generic refit 대조 필요 | P3/P3b 결과 전체를 보고하고 기본 추론 유지 | 향후 병목이 확인될 때만 재검토 |

기존 복제 진단 제안의 9.10 점수는 현재 방법에 사용하지 않았다. Raw review는
project-local private trace에 보존하고, 결과·반영·한계는 이 문서에 통합했다.

## Refined Proposal

주방법은 원래 VG-SAE다. **C1**은 정해진 sparsity 예산 아래 유용한 coefficient/feature
recovery tradeoff, **C2**는 현재 profiled-beta recipe에서 variance와 entropy가 각각 기여하는
support–amplitude coupling이다. 실제 지지된 항만 결론에 쓰며 learned-beta recipe로
자동 일반화하지 않는다. 두 주장의 확증 실험은 남아 있다.

다음 실행은 B1의 공정한 원래 크기 비교, B2의 좁은 objective ablation,
B3의 기존 Stage2/3 checkpoint fresh 평가 순서로 정리한다.
최종 방법과 상세 실행 범위는 [제안서](../refine-logs/FINAL_PROPOSAL.md),
[실험 계획](../refine-logs/EXPERIMENT_PLAN.md),
[실행 상태표](../refine-logs/EXPERIMENT_TRACKER.md),
[연구 계약](docs/research_contract.md)에 있다.


## Workflow Completion

방법 제안 심사는 두 rounds 뒤 **READY, 9.05/10**이다. 첫 round도 9.05였지만
C2의 적용 범위와 항별 성공 판정이 맞지 않아 REVISE였다. 두 번째 round에서 이를
고쳐 blocker를 해소했다. 점수를 성능이나 신규성 확증으로 해석하지 않는다.
Reviewer는 `/root/vg_pipeline_review`, GPT-6 Astra ultra, same-family provisional이며
CALIBRATION:none이다. [심사·반영 기록](../refine-logs/REVIEW_SUMMARY.md)에 경위를 남겼다.

새 pilot runner 네 개의 관련 테스트 **14개 통과**, compile 검사 통과, 독립 최종
artifact audit에서 미해결 오류 없음. 전체 repository suite는 이번에 재실행하지 않았다.
검증 범위와 실제 명령은 [validation](runs/vg-sae-development-20260921/evidence/validation.json),
실제 reviewer별 판정은 [receipts](runs/vg-sae-development-20260921/evidence/review_receipts.json)에 있다.

다음 구현은 독립 split/shared batching/cal-only checkpoint 선택을 갖춘 M0 runner다.
B1/B2/B3를 이미 실행했다거나 VG-SAE의 논문 기여가 완성됐다고 보고하지 않는다.

<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:START -->
## Evidence Gate
**Status:** PASS

All required stage records, review receipts, artifacts, and report sections are present.
<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:END -->

Evidence gate도 PASS다. 이는 필요한 산출물과 실제 검토 receipt의 완결성 검사이며,
과학적 주장이나 실험 성공을 대신하는 승인이 아니다.
