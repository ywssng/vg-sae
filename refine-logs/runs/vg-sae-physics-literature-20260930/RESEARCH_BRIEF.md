# Research Brief: VG-SAE의 물리 모형에서 ensemble 기반 밀도 추정까지

2026-09-30. 사용자는 새 세 논문과 기존 문헌을 바탕으로 전체 논문 및 Phase 1·2를 재검토하고, ARIS·scientific 두 agent의 토론을 PI가 감독하도록 요청했다. 문헌·수식·설계 검토만 수행했으며 새 학습은 없다.

## 연구 목표

VG-SAE를 명시한 통계물리적 선택모형에서 유도·이해하고, 불확실성 곡선으로 밀도를 추정해 feature recovery와 대조한다. 모형 설명(C1), generating density 추정(C2a), operating-point 선택효용(C2b)을 구분한다. SOTA를 유일한 성공조건으로 두거나 원래 추정 연구를 별도 진단 주제로 바꾸지 않는다.

## 이번 개정

- Phase1: conditional Gram interaction, exact covariance 대 MF response, 좌표 sparsity 대 occupancy를 구분한다. 기존exact response 검증은 재사용한다. Rotation은0-fit 대조이며 새 correlated-prior oracle는 후속기전으로 둔다.
- Phase2: 고정데이터 optimizer variation(E_opt)과 data+optimizer challenger(E_joint), hard-between과soft-total을 각native density축에서 구분한다. 현재 operational primary는 E_opt+hard이며 이론적 우월성의 선언이 아니다.
- 같은 올바른 hardcode로 수렴해도 U_hard=0일 수 있다. 이를 stable-wrong와 함께 검사하며 soft-total에는 within term이 남을 수 있음을 명시한다.
- 처음부터7k–10kfits를 실행하지 않는다. 공통reference의두 teacher pilot238–350fits를모두8k까지학습하는계획으로전체2k/4k/8k곡선을먼저비교한다. 추가 controls는4k hardcoverage에서만고른다.
- 본검증36freshworlds와smallL1은bridge·개발·lock이후의조건부단계다. Same-bank/L1 비교는결과 무관보고하며광범위ablation은사전조건부로둔다.

## 문헌 해석의 범위

Nature Perspective의모형·복원·해석가능성구분을서사에반영한다. PRL2017의RBM weight sparsity/compositional phase나PRL2020의weight-posterior/Nishimori결과를VG의hardL0/learnedbeta에그대로옮기지않는다. Nature의세부가정은공개저자preprint로읽었고최종출판본전체와동일하다고단정하지않는다.

두scientist의독립입장·실제이견·교차수정·공동 권고와PI판정은 `refine-logs/runs/vg-sae-physics-literature-20260930/`에있다. Operational 합의는완료했지만어떤ensemble이유용한지는실험 전 미확인이다. 이전READY9.27점수를이번설계의실증으로재사용하지않는다.

## 현재 문서

- `PAPER_PLAN.md`: 통합서사·주장·그림·본문 구성.
- `refine-logs/FINAL_PROPOSAL.md`: 방법과ensemble/observable의정의.
- `refine-logs/EXPERIMENT_PLAN.md`, `EXPERIMENT_TRACKER.md`: 현재단계별protocol/미실행 상태.
- `refine-logs/runs/vg-sae-physics-literature-20260930/PI_SYNTHESIS.md`: 세논문의반영과PI판정.
- `refine-logs/runs/vg-sae-first-principles-20260924/`: Phase1원본.
- `refine-logs/runs/vg-sae-unified-paper-20260929/`: 직전통합계획및3라운드검토원본.

기존LaTeX 원고와Phase1결과는보존한다. 새계획의비용은제안이며문헌검토요청을GPU 실행승인으로확장하지않는다.
