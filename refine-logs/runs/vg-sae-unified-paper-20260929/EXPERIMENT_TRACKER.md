# Phase 1–2 실행 추적

2026-09-29. 이번 작업은 통합 계획 작성이다. **새 학습·GPU 작업은 시작하지 않았다.** Phase1의 완료 기록과 아직 실행하지 않은 Phase2를 구분한다. Fit 수는 EXPERIMENT_PLAN의 unique training runs 기준이며 iteration/reference/retry를 독립world로 세지 않는다.

| ID | 단계 | 목적 | 크기/예산 | 우선순위 | 상태 | 근거 또는 다음 산출물 |
|---|---|---|---|---|---|---|
| P1-B1 | Phase1 | 구현·수식 감사 | 과거378 tests/compileall 기록 | 기존근거 | EXISTING_DONE | 9월24일 IMPLEMENTATION_AUDIT/EXPERIMENT_RESULTS |
| P1-B2 | Phase1 | exact/MF response | 135 conditional cells | 기존근거 | EXISTING_DONE | outputs/first_principles_20260924/exact_summary.csv |
| P1-B3 | Phase1 | frozen inference error | 9 fits/3worlds | 기존근거 | EXISTING_DONE | frozen_summary.csv |
| P1-B4 | Phase1 | joint objective/readout | 54 fits/3worlds | 기존근거 | EXISTING_DONE | joint_summary.csv/profile_summary.csv |
| P2-M0 | Phase2-A | tensor axes/순열/퇴화/heterogeneity/kernel 계약 | 작은CPU 수치검사, 학습0 | MUST | TODO | tests와 typed estimator/schema; 아직미구현 |
| P2-M1 | Phase2-A | 개발·timing·native coverage | 8worlds×4runs×17–25controls=544–800fits | MUST | TODO | dev manifest, estimator, timing, lock receipt |
| P2-M1b | Phase2-A | optimization budget 민감도 | 24개 추가4000-update 구간 | MUST | TODO | 전체recipe 고정 또는 버전개정 |
| P2-M2 | Phase2-B | blind density/recovery 본검증 | 36worlds×6runs×17–25=3672–5400fits | MUST | TODO | sealed predictions → evaluator test |
| P2-M3 | Phase2-C | beta/variance/entropy 연결 | main6world subset, primary bank 재사용;1836–2700newfits | MUST | TODO | paired objective/estimator differences |
| P2-M4 | Phase2-C | L1 generic selector/TopK recovery comparator | L1 612–900 +TopK576fits | MUST | TODO | comparator/coverage 및 recipe 기록 |
| P2-M5 | Phase2-C | amplitude/skew/overlap 단일축 transfer | 9freshworlds,918–1350fits | SHOULD | TODO_CONDITIONAL | primary 결과와 비용 검토 후 |
| P2-S2 | Stage2 data | pinned SynthSAEBench transfer | staged20M sample pilot36fits 제안, timing미정 | CONDITIONAL | NOT_SCHEDULED | fresh streams·capacity target·matcher scaling |
| P2-S3 | Stage3 data | real LM external utility | 한 layer/작은width pilot36fits 제안, timing미정 | OPTIONAL | NOT_SCHEDULED | native coverage·CE/KL, trueL0 지표없음 |

Core M0–M4의 새 fit budget: 기본7240/최대10376. M1b 포함4000-step equivalents:7264/10400. Measured cost가 없으므로 GPUh는 제안 범위25–145, 제안cap160이다. 이 표는 실행 승인이나 비용 사용 기록이 아니다.

Phase1의3worlds와 Phase2의36worlds를 합쳐 새 확증표본이라고 쓰지 않는다. P2-M3/M4는 P2-M2의 사전 subset을 사용하므로 추가 independent world도 아니다. Stage2/3의 conditional studies가 끝나지 않아도 완료된 core와 미실행 확장을 구분해 기록한다.
