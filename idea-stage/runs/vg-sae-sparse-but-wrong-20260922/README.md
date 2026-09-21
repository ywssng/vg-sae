# Sparse but Wrong 기반 VG-SAE 개발 — 2026-09-22

현재 연구는 원래 VG-SAE를 L1 대안으로 개발하며 실제 feature/support 복원을 검증한다.
[참조 요약](../../REF_PAPER_SUMMARY.md), [통합 보고서](../../IDEA_REPORT.md),
[최종 제안](../../../refine-logs/FINAL_PROPOSAL.md),
[실험 계획](../../../refine-logs/EXPERIMENT_PLAN.md)이 현재 진입점이다.

- `evidence/`: 문헌35개 registry,20원안/13families,실제review receipts와수치근거.
- `pilots/`: 결과 이전의 프로토콜과 실행상 변경 사항.
- `figures/`: frozen P2/P3 결과의 설명용 그림. 선택/성공 기준을 바꾸지 않았다.
- `prior_development_snapshot/`: 09-21 계획의 보존본. 현재 anchor의 승인이나 결과가 아니다.
- P1/P2/P3는225/72/54 fits로 완료. 후속계획은미실행.

재현용 source는 `scripts/run_sbw_*_pilot.py`, 공통자료/평가는 `scripts/sbw_pilot_utils.py`다.
현재 pilot runner의고정된grid/device/world/steps를후속계획CLI로오인하지않는다.
큰checkpoint는 `outputs/sbw_20260922/`에만두며Git에추가하지않는다.
Private review 원문은 `.aris/traces/`에있고공유문서에는고유한판정과반영사항을통합했다.

과학적연구계속은PROCEED WITH CAUTION,제안서는REVISE/7.85이며
same-family provisional이다. Workflow완료가C1/C2성능확증이나논문준비완료는아니다.
