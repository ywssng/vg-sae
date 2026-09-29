# Research Brief: VG-SAE from statistical-physics principles

사용자 의도 확정:2026-09-23. 실험 설계·구현 검토·실행 승인:2026-09-24.

## Problem Statement

Variational Garrote 기반 SAE를 개발하며, 통계물리학의 명시적 모형 가정과 변분원리에서 SAE의 목적함수·선택·불확실성·근사추론을 이해한다. 원리적 유도와 설명력도 주기여이며 성능 우위나 Sparse but Wrong의 모든 문제 해결을 필수 조건으로 요구하지 않는다.

## Starting Literature

사용자가 제공한 Sparse but Wrong(2508.16560v4)와 Soh et al. Variational Garrote(2509.06383v1)에서 출발한다. VAEase, Variational Sparse Coding, Entropy-Based ELBOs 및 autoencoder/free-energy 선행은 같은 연구 흐름으로 비교한다. 알려진 VI/free-energy identity를 새 정리로 주장하지 않는다.

## Current Scope

기존 계획과 코드는 참고·재사용하되 새로운 실험을 그 계획에 자동 종속시키지 않는다. 특히 SAE 구현을 논문과 공식 GitHub에 대조한 뒤 실행한다. 현재 초기 캠페인은 exact conditional inference, frozen encoder, joint objective behavior를 연결한다. 구체적 claims/controls/판정은 아래 plan이 기준이다.

## Constraints

- 같은 조건의3 independent training worlds, 고정 seed/config, 독립 train/cal/test.
- 데이터의 실제 ground truth와 numerical exact inference reference를 구별한다.
- Full generative/semantic posterior는 input-dependent point amplitude에 주장하지 않는다.
- 로컬 .venv 및 idle GPU 사용, 최초 캠페인 상한2GPUh. 외부 메시지·유료 자원 구매는 포함하지 않는다.
- 실제 오류 수정과 수식 회귀 검증을 수행하고 기존 사용자 변경·실험 결과는 보존한다.
- LLM 대규모 검증·새 variational baseline 구현·광범위HPO는 이번 필수 범위 밖이다.

## Current Artifacts

- Method: `refine-logs/FINAL_PROPOSAL.md`
- Plan/Tracker/Results: `refine-logs/EXPERIMENT_PLAN.md`, `EXPERIMENT_TRACKER.md`, `EXPERIMENT_RESULTS.md`
- Detailed run: `refine-logs/runs/vg-sae-first-principles-20260924/`
- Config: `configs/first_principles_20260924.json`
- Evidence: `outputs/first_principles_20260924/`

이전 brief는 현재 run의 prior_scope_snapshot에 보존했다. 이전 empirical claims나 reviewer 점수는 새 캠페인의 확증으로 재사용하지 않는다.
