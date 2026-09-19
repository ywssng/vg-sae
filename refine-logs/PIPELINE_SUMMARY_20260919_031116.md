# Pipeline Summary

2026-09-19 · **READY: proposal only, same-family provisional**.

선택 방향은 VG-SAE의 feature 복제에 대한 국소 loss 선호와 실제 feature 선택의
관계다. 28개 문헌과11개 후보, 두 연구 방향의 파일럿, 신규성/비판 검토를 거쳤다.
방법은3차 refinement에서9.10/10으로 구현 가능한 계획 판정을 받았다.
아직 C1/C2가 참이거나 새 방법이 우수하다는 결과는 없다.

## Deliverables

- [통합 보고서](../idea-stage/IDEA_REPORT.md)
- [선택 연구 계약](../idea-stage/docs/research_contract.md)
- [최종 제안](FINAL_PROPOSAL.md)
- [리뷰 기록](REVIEW_SUMMARY.md)
- [실험 계획](EXPERIMENT_PLAN.md)
- [실행 상태](EXPERIMENT_TRACKER.md)

## First Three Tasks

1. R001: 실제clone intervention과 중복 불변 metric을 별도 runner로 구현·검증.
2. R002: 새로운 cal/holdout에서 C1 loss-preference 검사.
3. R003: C1이 통과한 경우에만 high/low×variance on/off continuation.

기존파일럿28 tests PASS, GPU 계산 타이머 합계약0.0721h(import/추가독립검증 제외).
후속실험의7.05GPUh는 계획상상한이며 아직 소비하지 않았다.
신규성은조건부이고 단순prior/readout 처방은실제파일럿에서지지되지 않았다.
