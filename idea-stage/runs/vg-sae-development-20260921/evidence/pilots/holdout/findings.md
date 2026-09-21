# P1 결과: 기존 Stage 1 VG-SAE의 새 calibration/test 평가

2026-09-21. 기존 profiled Stage 1의 exponential/constant amplitude, frequency skew .5, d128/true1024/SAE1024, training seed0 checkpoint를 재평가했다. 기존 source·checkpoint를 바꾸거나 새 모델을 학습하지 않았다.

각 조건의 전체 273-control grid, 총 546개 checkpoint를 새 calibration 2048개에서 평가했다. 방법/조건/L0 cap 4·8·16별로 calibration hard latent relative error를 최소화하고, MSE·actual L0·fixed control order로 동률을 처리했다. 36개 선택 행, 고유 checkpoint 24개를 기록·동결한 후에만 test 4096개를 생성하고 평가했다. GPU2 전체 본 실행은 41.27초였다.

| 조건 | L0 cap | 선택된 VG gamma | VG test L0 | VG hard latent error | 비교상 가장 낮은 baseline error | 해당 baseline test L0 |
|---|---:|---:|---:|---:|---|---:|
| exponential | 4 | 3.0 | 3.002 | .69878 | TopK k4: .74837 | 3.992 |
| exponential | 8 | 3.0 | 3.002 | .69878 | TopK k6: .73694 | 5.879 |
| exponential | 16 | 3.0 | 3.002 | .69878 | TopK k6: .73694 | 5.879 |
| constant | 4 | 3.5 | 3.366 | .87248 | TopK k4: .96311 | 4.000 |
| constant | 8 | 3.0 | 4.158 | .85281 | L1 coeff1: .88977 | 5.193 |
| constant | 16 | 3.0 | 4.158 | .85281 | L1 coeff.5: .84428 | 13.101 |

모든 방법의 checkpoint는 calibration에서 고정했다. 표의 baseline 최소값은 그 고정된 방법별 결과를 읽기 위한 기술적 요약이며, test로 checkpoint를 다시 고르지 않았다. 이 표는 **같은 L0 cap 아래의 선택된 frontier**이며 정확히 같은 L0끼리의 비교가 아니다. 이번 선택 결과에서는 test 평균 L0가 cap을 넘은 경우가 없었다. 더 큰 cap에서 같은 checkpoint가 반복되는 것은 오류가 아니라 calibration error 목적에 따른 선택이다.

Exponential 조건의 VG 양성 신호가 새 예시에서도 유지됐다. VG gamma3의 test hard EV는 .47784, F1은 .32709다. Cap8의 TopK k6은 EV .51410, F1 .29710으로, VG가 모든 평가 지표에서 우세하다는 뜻은 아니다. Constant 조건에서도 낮은 L0 cap의 VG는 개선 가치가 있는 기준선이며, cap16에서는 L1이 더 낮은 latent error를 냈다. 후속 방법 개발은 exponential gamma3와 constant gamma3/3.5를 우선 anchor로 사용할 수 있다.

원본 artifact에 dictionary tensor가 직접 없어서 original saved config와 seed0로 원래 데이터를 복원했다. 저장된 old-test support와 latent가 bitwise 일치하고, 복원된 dictionary/probabilities가 원래 생성기의 값과 일치함을 확인한 후 동결했다. 새 calibration/test seed로 dictionary를 생성하지 않았다. 32개의 L1 control 모두 기존 original-training GMM threshold를 그대로 사용했으며 fresh split으로 fit한 경우는 없다.

검증은 세 단위 테스트와 기존 evaluator 동등성 확인으로 구성했다. Fixed dictionary·독립 RNG, train-only L1 threshold, cap/tie selection의 test-field 무관성 테스트 3개가 통과했다. 여섯 방법 각각 한 checkpoint를 original test에서 평가한 핵심 metric은 저장된 Stage 1 값과 최대 절대오차 약 5.31e-8 이내였다. 상세값은 `validation.json`에 있다.

주요 산출물:

- `protocol.json`: 결과 관측 전 규칙, seed, 예산, source 및 script hash.
- `source_manifest.json`, `*_frozen_source.npz`: source config와 동결 dictionary/probabilities.
- `calibration_metrics.csv`: 모든 546개의 raw calibration metrics 및 checkpoint hash.
- `frozen_selection.json`: test 생성 이전에 고정한 36개 선택. SHA-256 `c9af239490599a9d73cfe72ecb06e4111dc5c0672e461efd657f6e6a208be576`.
- `selected_test_metrics.csv` / `.json`: 모든 방법의 선택 결과, 실제 calibration/test L0, cap 초과 및 hard metrics.
- `*_calibration.npz`, `*_test.npz`, `*_matching.npz`: P3가 재사용 가능한 동일 dictionary 데이터와 선택된 checkpoint의 weight-only Hungarian mapping.
- `environment.json`, `validation.json`, `completion.json`: 실행 환경·검증·시간 기록.

한 dictionary/training seed의 fresh-sample 확인이며, training-seed 강건성이나 SOTA를 입증하지 않는다. 원래 조건을 고른 판단은 과거 결과를 봤고, 방법별 control-grid·optimizer·parameter 예산이 다르다. 이번 결과는 VG-SAE 자체의 후속 최적화와 조건부 추론 개선에 쓸 근거다.
