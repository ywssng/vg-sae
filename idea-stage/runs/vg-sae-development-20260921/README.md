# VG-SAE 방법 개발을 위한 2026-09-21 실행

이 실행은 사용자의 원래 연구인 **Variational Garrote 기반 새로운 SAE 개발**을
다듬는다. 이전 복제 진단 주제는 현재 연구 목표를 대신하지 않는다.

- 현재 통합 결과: [IDEA_REPORT](../../IDEA_REPORT.md)
- 고정 연구 범위: [RESEARCH_BRIEF](../../../RESEARCH_BRIEF.md)
- 최종 방법: [FINAL_PROPOSAL](../../../refine-logs/FINAL_PROPOSAL.md)
- 후속 실행: [EXPERIMENT_PLAN](../../../refine-logs/EXPERIMENT_PLAN.md)
- `evidence/`: 문헌 metadata, 코드 감사, 후보 경로, 작은 원본 수치
- `pilots/`: 실행 전에 기록한 protocol; `parallel_followup_protocol.md`는 P3 이후의 탐색
- `prior_scope_snapshot/`: 범위를 오해했던 09-19 문서의 보존본

재현 runner는 `scripts/run_vg_development_*.py` 네 개다. 큰 checkpoint와 NPZ는
`outputs/vg_sae_development_20260921/`에 있으며 Git에 추가하지 않는다.
공유 JSON/CSV는 계산된 수치의 작은 사본이며 경로는 저장소 기준이다.
실제 리뷰 응답은 project-local `.aris/traces/`에 두고 공개 보고서에는 판단과
수정 사항을 통합했다. 모든 이번 심사는 GPT-6 Astra ultra, same-family provisional이다.

후속 B1/B2/B3는 계획이며 이번 run에서 실행한 것으로 표시하지 않는다.
