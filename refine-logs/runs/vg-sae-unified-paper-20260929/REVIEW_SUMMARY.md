# Phase 1–2 통합 계획 검토 요약

2026-09-29. 실제 secondary GPT-6 Astra ultra reviewer와3차례 검토했다. 최종 **READY/9.27**, 대상은 계획의 준비도다. Same-family provisional, calibration:none. Phase2는 미실행이며 review가 가설의 성공·신규성 확정·GPU 실행승인을 뜻하지 않는다.

## Problem Anchor

- 연구 문제: Variational Garrote 기반 SAE를 명시한 통계물리적 모형과 변분원리에서 유도·이해하고, 그 모형의 밀도 응답과 독립 재학습 간 선택 불확실성으로 적절한 sparse operating density를 추정할 수 있는지 검증한다.
- 핵심 병목: 재구성 오차와 sparsity만으로 올바른 feature 분해나 밀도를 선택할 수 없다. 한 모델 내부의 posterior uncertainty, 모델 간 선택 불안정성, 실제 feature recovery 사이의 연결은 별도로 검증해야 한다.
- 유지할 목표: VG-SAE 자체의 개발과 이해를 중심에 둔다. Phase 1은 모형·추론·목적함수의 근거, Phase 2는 정답을 보지 않는 밀도 추정과 그 유용성 검증이다. Phase 1 성공이 Phase 2 성공을 함의한다고 가정하지 않는다.
- 비목표: 별도 복제 진단 주제로 전환, 모든 SAE를 VG의 특수형으로 선언, SOTA를 유일한 성공 조건으로 강제, 파라미터 없는 보편적 true-L0 발견, semantic posterior calibration 또는 thermodynamic phase transition의 무근거 주장.
- 제약: 이번 요청은 계획 작성·검토다. 새 학습·GPU 실험은 실행하지 않는다. 기존 결과와 원고는 보존한다. 후속 비용은 제안이며 9월 24일의 2 GPUh 승인 범위를 확장한 것으로 취급하지 않는다.
- 완료 기준: 원래 밀도 추정 목표를 포함한 하나의 논문 서사, 실행 가능한 추정 절차, 주장별 실험·반증 기준, Phase 1의 실제 근거와 Phase 2의 미검증 가설을 구분한 계획을 제공한다.


## 수정과 해결

| Round | 핵심 지적 | 반영 | 결과 |
|---|---|---|---|
| 1 | native-density curve 미정, template 전이 가정 부족, 보류/선택편향/예산 불명확 | density matching을 primary로 고정, pooled-support 계산, finite profile/abstention, all-world fallback, exact fit budget | REVISE7.95 |
| 2 | deployment coverage 실패가 유효한 C2a estimate를 지움; edge-case 규칙 부족 | estimate/deployment status 분리, 빈curve fallback, zero norm/tie/개별repeat drift/loop stop | REVISE8.965 |
| 3 | 수정된전체 proposal/protocol/outline 재평가 | 새component나benchmark를 추가하지 않고 같은목표·수치계약 유지 | READY9.27, remaining blocker 없음 |

## 최종 판단의 한계

- Phase1의 조건부 설명과 Phase2의 생성밀도/선택효용을 한논문으로 연결한다.
- Template transfer, achievable native coverage, stable-wrong, finite-R 변동, 실제GPU timing은 남은 경험적위험이다.
- 판정은 연구계획 준비도이며 결과의지지나 출판가능성의 보장이 아니다.
- 원래연구를 다른진단주제로 바꾸거나 SOTA를 유일한성공조건으로 강제하지 않았다.
- 숫자threshold와compute는 계획제안이다. 실행은 이번작업에 포함하지 않았다.

## 기록

`refine-logs/runs/vg-sae-unified-paper-20260929/round-1-review.md`, `round-2-review.md`, `round-3-review.md`에 full raw response가 있다. `round-1-refinement.md`와 `round-2-refinement.md`에 전체수정제안을 보존했다. 최종 논문구성은 저장소루트의 `PAPER_PLAN.md`, 실행프로토콜은 `refine-logs/EXPERIMENT_PLAN.md`다.
