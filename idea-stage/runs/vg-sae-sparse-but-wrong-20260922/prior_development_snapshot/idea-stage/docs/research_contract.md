# Research Contract: Variational Garrote 기반 SAE 개발

실행 `vg-sae-development-20260921` · 2026-09-21.
사용자가 확정한 원래 연구는 **새로운 VG-SAE 방법 개발**이다.
[RESEARCH_BRIEF](../../RESEARCH_BRIEF.md)가 범위의 기준이다.
이전 복제 진단 주제의 계약은 [역사 보존본](../runs/vg-sae-development-20260921/prior_scope_snapshot/README.md)에만 남긴다.

## Selected Method

입력에서 Bernoulli support 확률 m과 nonnegative point amplitude a를 예측하고,
learned linear dictionary D로 복원한다. Support를 표본 추출하지 않고 적분한
quadratic energy와 normalized Bernoulli prior/entropy로 학습한다.
실제 readout은 hard support × amplitude다. 확률적 support 선택이 유용한 sparse
code를 학습하도록 하는 것이 중심이며, 진단은 이 방법의 개발과 검증을 돕는다.

## Core Claims

- **C1, 후속 확증 필요:** 정해진 sparsity 예산 아래 강한 SAE baseline과 비교해
  유용한 coefficient/feature recovery tradeoff를 제공한다. Actual L0, 입력 fidelity,
  parameter·학습·추론 비용을 같이 보고한다. 범용적인 모든 지표 우위를 뜻하지 않는다.
- **C2, 미검증:** 현재 parameterization의 minibatch-profiled beta recipe에서
  variance와 entropy가 support–amplitude 결합과 hard-code 품질에 기여하는지 각각 판정한다.
  Variance만, entropy만, 둘 모두, 둘 모두 비지지를 구분하고 지지된 항만 결론에 쓴다.
  Learned-beta recipe의 C1 결과로 자동 일반화하지 않는다.
  No-variance의 scale compensation을 고려하며 deletion collapse만으로
  확률적 방법 전반이나 같은 capacity의 모든 deterministic SAE에 대한 우위를 주장하지 않는다.

현재 objective는 입력 의존 amplitude의 생성 prior/entropy를 포함하지 않는다.
따라서 full generative ELBO, unbiased amplitude, calibrated semantic posterior,
최초 probabilistic SAE/analytic VI를 수식만으로 주장하지 않는다.
Learned global beta와 minibatch-profiled beta를 다른 stochastic objective로 구분한다.

## Minimum Convincing Evidence

1. B1: 원래 Stage1 크기, 새 paired training/data worlds 100/101/102, 같은 tuning 및
   checkpoint 후보 예산. Cal hard latent error와 L0로 모든 방법의 control/checkpoint를
   고르고 동결 뒤 test 평가. Cap 8 primary, 4/16 secondary. 비교 baseline도 cal에서 고정한다.
2. B2: 같은 architecture의 variance×entropy 2×2, 같은 beta mode와 selection budget.
   Mean/sample/hard risk, m/a scale, actual L0를 함께 기록한다. 이 실험의 의미는
   현재 support–amplitude coupling의 내부 검증이다.
3. B3: 저장된 SynthSAEBench와 Gemma L5 checkpoint부터 fresh split에서 평가한다.
   Ground truth가 없는 실제 activation에서는 L0 및 reconstruction/CE로 선택한다.
   여기서 true support recovery나 semantic calibration을 주장하지 않는다.

상세 grids, job 수, split, 구현 prerequisite, resource cap과 go/no-go는
[실험 계획](../../refine-logs/EXPERIMENT_PLAN.md)에 둔다. 동일 L0 상한 아래 비교를
exact matched-L0라 쓰지 않는다. 3-seed screening은 통계적 확증과 구분한다.

## Completed Evidence and Decisions

| 확인 | 결과와 결정 |
|---|---|
| P1: 기존 546 checkpoints의 새 cal/test | Exponential cap 8에서 VG error .69878 대 TopK .73693, EV .47784 대 .51410. 한 training world의 recovery–fidelity 절충이다. |
| P2: 36개 작은 학습 | Moment beta 초기화의 일관된 이득 없음. 더 긴 학습에서 recovery가 악화된 조건이 있어 cal recovery 기반 checkpoint 선택을 채택한다. |
| P3: 순차 gate 보정 | Same-count error 약 5.01% 개선, 현재 latency 약 992배. 기본 추론에 채택하지 않는다. |
| P3b: 탐색 병렬 후속 | Same-count error 약 3.835% 개선, latency 4.3687배로 사전 3배 기준 미달. PG control과의 절충이 남는다. |
| 문헌·신규성·비판 검토 | 32개 문헌 registry, 실제 fresh ultra review 모두 조건부 진행. Same-family provisional이다. |

새 trainable module을 금지하는 사용자 제약은 없다. 현재 파일럿이 필요성을 뒷받침하지
않아 기존 architecture와 기본 추론을 후속 비교의 기준으로 유지한다.

## Current Status

- [x] 원래 VG-SAE 방법 개발을 anchor로 고정
- [x] 문헌, 방법 비교와 신규성 검토
- [x] 세 개발 경로의 파일럿 및 P3b 탐색 후속
- [x] 비판 검토와 주장 범위 수정
- [ ] B1의 공정한 새 학습 비교
- [ ] B2 objective mechanism 검증
- [ ] B3 fresh benchmark/activation 평가
- [ ] 논문 수준의 기여 확증과 작성

[통합 보고서](../IDEA_REPORT.md), [최종 제안](../../refine-logs/FINAL_PROPOSAL.md),
[실행 상태표](../../refine-logs/EXPERIMENT_TRACKER.md)가 다음 작업의 시작점이다.
