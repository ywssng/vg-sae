# VG-SAE 후속 실험 추적표

2026-09-22. 아래 SBW-N 실행은 모두 **계획이며 미실행**이다. 상세 계약은
`experiment-plan-v2.md`를 따른다. 기존 pilot 결과를 새 main 결과로 재사용하지 않는다.

| Run ID | 목적 | System / split | Fits 상한 | 후보 수 상한 | 상태 | 완료/진행 gate |
|---|---|---|---:|---:|---|---|
| SBW-P1 | 완료한 직접 비교 pilot | 5방법, worlds 210–212 | 225 | 675 cal, 254 test 고유 | COMPLETED | target1.8 VG 0/9; target2 7/9; Jump 전 target 0; robust superiority 미확인 |
| SBW-P2 | 완료한 scalar prior pilot | fixed2 / learned init−2/2/6 | 72 | final 72 | COMPLETED | joint 0/18; primary residual>2가 14/18; wrong-EB fixed point 증거 없음 |
| SBW-P3 | 완료한 posterior pilot | amortized / optimized product / exact32 | 54 | cal162, final test54 | COMPLETED | matching 4/18 target entries, 3 unique pairs; 한 same-gamma total-effect rescue |
| SBW-N00 | normalization/readout/metric와 처리량 | 5방법 × 500 updates; scientific selection 제외 | 5 probes | 0 | TODO | 독립 RNG, native readout, train/eval/I/O 계측 |
| SBW-N01 | baseline range/convergence scout | world310, p=.4, rho±.4, 5방법×3controls | 30 | 90 | TODO | Jump {.0003,.003,.03}와 공식 input normalization 우선 |
| SBW-N02 | 한 번의 range 보충 | 방법별 control 1개 × 2 signs | 10 | 30 | CONDITIONAL | test 미생성; source-cal 규칙만 사용; 부족하면 unresolved |
| SBW-N03 | precision 정책 개발 | world310, 3policies×gamma{0,2,4}×2signs | 18 | 54 | TODO | gamma 고정; new module 없음 |
| SBW-N04 | 보충 gamma의 precision 적용 | 나머지 2policies×2signs | 4 | 12 | CONDITIONAL | N02가 VG gamma를 추가했을 때만 |
| SBW-FREEZE | recipe와 selection 동결 | 3 controls/method; T; 3 checkpoints; selectors | 0 | 0 | TODO | source truth와 target truth 권한 분리; manifest/source/cal hash |
| SBW-N05 | 새 직접 비교 | worlds311/312/313, p=.4, rho±.4, 5방법×3controls | 90 | 270 | TODO | source/blind/oracle를 구분; rescaled L1 추가 fit 없음 |
| SBW-N06 | precision 확인의 추가 arms | 같은 main cells, 나머지 2policies×3gamma | 36 | 108 | TODO | N05 VG18을 공유; gamma2 primary total effect |
| SBW-N07 | unknown-count recipe transfer | p=.2/.6, rho±.4, 3worlds, 5방법×source recipe1 | 60 | 180 | TODO | target true count/recovery로 control·checkpoint 변경 금지 |
| SBW-N08 | 50-feature range scout | g=h50,d100; 5방법×3controls + 최대5 보충 | 20 | 60 | CONDITIONAL | method freeze; exact source settings; 새 처리량 계측 |
| SBW-N09 | 50-feature 단일 setting 확장 | 5방법×3controls×3init/train seeds | 45 | 135 | CONDITIONAL | 15M examples를 실제 채웠을 때만 그 budget으로 표기 |
| SBW-N10 | 기존 realistic/source inventory | Stage2 SynthSAEBench, Stage3 Gemma | 0 | 평가 가능 checkpoint만 | TODO_READ_ONLY | recipe/data offsets/source hashes/eval labels 확인 |
| SBW-N11 | 새 realistic source 한 개 | final VG / rescaled L1 / strong comparator ×3seeds | 9 | 9 final | CONDITIONAL | N10 뒤 exact width/sample manifest; 새 method로 실제 학습 |
| SBW-N12 | Gemma 한 layer 후속 | 같은 3systems×3seeds | 9 | 9 final | OPTIONAL_UNBUDGETED | sparse probing/task utility, CE/KL, disjoint text; c_dec 단독 금지 |

N01–N07 scientific fits 상한은 248개, 최종 eligible selection candidates 상한은 744개다. Superseded 개발 checkpoints를 포함한 실제 저장 상한은 864개다.
Development 62 fits/186 후보에는 test가 없고, main/transfer 186 fits/558 후보는
selection 동결 뒤 fresh test에서 평가한다. N00 5 probes와 N08 이후 조건부 확장은
별도다. 16k→32k의 공통 development 연장은 fit 수를 늘리지 않지만 updates를 늘린다.

M0–M5 제안 hard cap 8 GPUh(또는 별도 CPU process-hours 8), 50-feature 별도 8 GPUh,
첫 realistic training 별도 12 GPUh는 **미래 계획값**이다. 사용자가 확정한 장기 계산
승인으로 기록하지 않는다. Throughput 예측이 cap을 넘으면 main 시작 전에 incomplete
scope를 기록한다. 실행 중 cap에 닿으면 완료하지 않은 항목을 TODO/INCOMPLETE로 남긴다.

각 실행의 필수 기록: 실제 seed, config/source/data hashes, 초기 state hash, normalization,
parameter count, training updates/examples, wall/GPU/CPU time, selection rule/IDs,
cal/test boundary, signed/absolute geometry, support, native/expected L0, fidelity, failure.
반복 checkpoint/selector/target entry를 독립 world나 추가 fit으로 세지 않는다.

완료 판정은 C1/C2의 evidence와 coverage를 각각 기록한다. 유용한 oracle point,
source-supervised transfer, input-only selector, same-gamma total effect와 matched-L0
보조 분석을 한 종류의 성공으로 합치지 않는다.


## 선택과 회계의 고정 사항

- C1 primary는 source-supervised `S_source`; blind/oracle/matched 결과는 별도다.
- Main 3-control grid는 S_source 우승 coefficient를 포함한다. VG는 gamma2도 포함한다.
- C2는 gamma2와 모든 policy의 공통 final T에서 비교한다.
- B1/B2 main은 같은 `main_confirmation` RNG와 실제 checkpoint를 공유한다.
- 744는 eligible candidates다. B1 개발 연장 시 superseded 최대120개를 별도로
  보존해 실제 저장 checkpoint는 최대864개다. 개발과정에 test는 없다.
- Runner 인자화와 normalization/batch-stream 검증은 아직 구현하지 않은 M0다.
