# Experiment Tracker — VG-SAE 후속 wave

작성일: 2026-09-21. 기준: [EXPERIMENT_PLAN](EXPERIMENT_PLAN.md).
**모든 아래 신규 실험은 TODO이며 이번 계획 작성에서 실행하지 않았다.**
P1/P2/P3/P3b는 완료된 탐색 근거로 별도 보존하고 이 표의 완료 수에 넣지 않는다.

| Run ID / 묶음 | 단계 | 목적 / 시스템 | 데이터·선택 | 최대 수 | 우선순위 | 상태 | 선행 조건·종료 gate |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| PRE-01 | M0 | 독립 train/cal/test sampler와 paired data order | d128/true1024/SAE1024; worlds100/101/102 | 구현 1식 | MUST | TODO | split RNG, data hash, constant RMS 일치 |
| PRE-02 | M0 | trial manifest·네 checkpoints·selection | cal hard latent error, cap8 primary; 4/16 secondary | 구현 1식 | MUST | TODO | native L1와 GMM secondary 분리; test leakage 없음 |
| PRE-03 | M0 | 공식 baseline/native readout 및 deletion 검증 | metric/energy/gradient fixture와 관련 기존 tests | 검증 1식 | MUST | TODO | completed updates, auxiliary/schedule, m/a 보상 검증 |
| TIME-01 | M0 | full-size throughput, memory, checkpoint I/O | 6 B1 methods + 3 deletion, 300 updates씩; 본 trial에 포함 | 9 partial trials | MUST | TODO | 총 2 GPUh; quality 기반 grid 수정 금지 |
| B1-E-W100 | M1 | VG/L1/TopK/BatchTopK/JumpReLU/Gated | exponential, world100, train8196/cal4096/test8192 | 72 trainings / 288 candidates | MUST | TODO | 12 trials/method, 4 checkpoint/trial |
| B1-E-W101 | M1 | 위와 같음 | exponential, world101 | 72 / 288 | MUST | TODO | selection SHA 동결 후 test |
| B1-E-W102 | M1 | 위와 같음 | exponential, world102 | 72 / 288 | MUST | TODO | M1 전체 cap 12 GPUh |
| B1-C-W100 | M2 | 위와 같음 | constant, world100 | 72 / 288 | MUST | TODO | exponential과 독립 scope 판단 |
| B1-C-W101 | M2 | 위와 같음 | constant, world101 | 72 / 288 | MUST | TODO | 동일 max opportunity |
| B1-C-W102 | M2 | 위와 같음 | constant, world102 | 72 / 288 | MUST | TODO | M2 전체 cap 12 GPUh |
| SELECT-B1 | M1/M2 | cap별 control+checkpoint 및 non-VG reference 동결 | cal only; 최대 108 primary/secondary test rows | 36 primary method/world/condition 선택 + secondary caps | MUST | TODO | 평균 5% error 감소, F1와 seed reversal guardrails |
| B2-W100 | M3 | no-variance/no-entropy/neither; full B1 재사용 | exponential100; profiled, LR.01, gamma6 | 신규18 / 72 | MUST | TODO | full6 trainings는 B1에서 재사용 |
| B2-W101 | M3 | 위와 같음 | exponential101 | 신규18 / 72 | MUST | TODO | profiled 범위; variance/entropy 각각 판정; m/a·세 risk |
| B2-W102 | M3 | 위와 같음 | exponential102 | 신규18 / 72 | MUST | TODO | M3 cap 6 GPUh; collapse를 넓은 우월성으로 해석하지 않음 |
| PRE-B3A | M4 | saved Stage2 4methods × 3controls manifest | 동일 102.4M training budget, seed0 | 12 saved checkpoints | MUST | TODO | checkpoint SHA와 source revision 확인 |
| B3A-CAL | M4 | Stage2 operational selection | RNG620100; 262144 samples; hard caps32/64/128, cal MSE | 12 evaluations | MUST | TODO | test labels로 control 선택 금지 |
| B3A-TEST | M4 | MCC/classifier F1/coverage와 fidelity | RNG630100; 1048576 samples | 최대12 evaluations | MUST | TODO | B3a 총 2 GPUh; 서로 다른 두 cap points에서 go 판정 |
| PRE-B3B | M4 | GemmaL5 seed0 4methods × 3controls 및 document split | train/과거eval/cal/test 문서 provenance, hashes | 12 saved checkpoints | MUST | TODO | packed row offset만으로 문서 독립을 가정하지 않음 |
| B3B-CAL | M4 | real activation operational selection | 131072 tokens; CE prefix16384; hard caps128/256/512 | 12 evaluations | MUST | TODO | cal CE 최소 및 comparator 동결 |
| B3B-TEST | M4 | hard L0, EV/MSE, CE/KL, latency | 262144 tokens; CE prefix16384 | 최대12 evaluations | MUST | TODO | B3b 총 6 GPUh; ground-truth recovery 주장 금지 |
| REPORT-01 | 최종 | C1/C2/범위·실패·비용 표 | 세 paired differences와 모든 baseline 비교 | 보고서 1식 | MUST | TODO | 데이터 미존재·미완료·negative 결과를 구분 |
| EXT-S2 | M5 | Stage2 추가 training seeds100/101 | 4 methods × 1 frozen operating control × 2seeds | 최대8 trainings / 8 final checkpoints | CONDITIONAL | NOT STARTED | B3a go 후 별도 16 GPUh cap/throughput 검토 |
| EXT-S3 | 이후 | Gemma 신규 training 검토 | 아직 실행 grid/budget 없음 | 현 계획 0 trainings | DEFERRED | NOT PLANNED | B3b 결과와 실제 throughput 이후 별도 계획 |

기본 wave의 정확한 신규 학습 수는 **B1 432 + B2 54 = 486**,
후보 checkpoint 수는 **1,728 + 216 = 1,944**다. M0의 partial runs는 해당 trial로
재개하므로 추가하지 않는다. B2 full의 18 trainings/72 checkpoints도 재사용이다.
기본 wave cap은 **40 GPUh**이며 완료 시간의 예측값이 아니다. B3는 저장 모델 24개를
cal/test에서 최대 48번 평가하며 신규 학습은 0개다. 조건부 EXT-S2는 기본 합계에 포함하지 않는다.

실행 기록은 각 묶음에 config/SHA, train/cal/test RNG, 선택 checkpoint,
actual hard L0, candidate completion count, elapsed GPUh, walltime, peak memory,
성패와 이유를 연결한다. 기존 산출물을 삭제하지 않고 새 run 폴더에 기록한다.


## 이번 idea-discovery에서 완료한 근거

| Pilot | 상태 | 범위 |
| --- | --- | --- |
| P1 fresh holdout | DONE | 기존 546 controls, 새 cal/test, 한 training world |
| P2 beta recipe | DONE | 작은 조건의 36 models, 3 paired seeds |
| P3 conditional refinement | DONE | 23 arms, cal-selected 순차 보정과 generic NNLS |
| P3b parallel follow-up | DONE: exploratory | 22 arms, 새 split, 사전 latency 기준 미달 |
| 관련 수치·선택 tests | PASS | 새 tests 14개; 전체 repository suite 재실행은 아님 |

이 완료표는 위 B1/B2/B3의 결과를 대신하지 않는다.
