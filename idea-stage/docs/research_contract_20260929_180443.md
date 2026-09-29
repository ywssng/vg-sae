# Research Contract: VG-SAE Phase 1–2

2026-09-29 사용자 요청은 현재 Phase1을 기반으로 원래 밀도추정 Phase2를 refine하고 하나의 논문계획을 작성하는 것이다. 이번run은 planning_only다.

## 고정 연구 문제

명시한 VG 선택모형에서 SAE를 유도·이해하고, 그 관측량을 활용한 density inference가 feature recovery에 유용한지 검증한다. 원래추정 목표를 없애거나 다른진단 연구로 대체하지 않는다. 성능 우위가 유일한 기여기준은 아니지만 추정효용은 독립실험으로 검증한다.

## Claims와 evidence

- C1/Phase1: 조건부 free-energy 항과 exact/MF/encoder/readout 역할. 기존9월24일 작은 캠페인의 제한적근거를 재사용.
- C2a/Phase2: specified synthetic의 generating expected density 추정. 아직미검증.
- C2b/Phase2: selected operating density의 recovery utility. C2a와 구분하고 아직미검증.
- Known identity, 원문kernel, Hungarian matching 자체를 새로운정리/최초방법으로 주장하지 않는다. Full generative/semantic posterior는 point-amplitude SAE에 주장하지 않는다.

## 실행·평가 계약

- World와 optimization repeat를 분리. 같은world의 공통입력에서 정렬된 masks만 비교.
- Truth-free selector, independent train/align/select/test, prediction hash 후test evaluation.
- Native density matching, signed correspondence, single-template profile/abstention.
- Estimate status와 deployment status 분리. All-world recovery policy 및 이유별fallback 공개.
- Generating count, empirical sample count, finite-bank recovery optimum을 혼동하지 않음.
- Numerical grids/thresholds·exact counts는 `refine-logs/EXPERIMENT_PLAN.md`가 기준. 계획작성은 새GPU 실행승인이 아님.
- Phase1 seed를 Phase2 신규world에 합치지 않음. Mechanism subset 재사용을 추가독립증거로 세지 않음.

## 파일과 상태

- `PAPER_PLAN.md`: 하나의 논문서사·본문/그림/claim map.
- `refine-logs/FINAL_PROPOSAL.md`: 방법제안.
- `refine-logs/EXPERIMENT_PLAN.md`, `EXPERIMENT_TRACKER.md`: 실행프로토콜/TODO.
- `refine-logs/runs/vg-sae-unified-paper-20260929/`: 문헌근거·실제review·이력.
- `refine-logs/runs/vg-sae-first-principles-20260924/`: Phase1원본 보존.

Current evidence: Phase1 초기결과만 있다. Phase2의 성공/실패는 아직없다. 검토의 READY가 나오더라도 계획의구체성에 대한 same-family provisional 판단이지 과학적가설 지지나 paper acceptance가 아니다.
