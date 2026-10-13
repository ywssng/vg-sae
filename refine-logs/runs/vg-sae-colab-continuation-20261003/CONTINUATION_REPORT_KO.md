# VG-SAE 연구 재개 보고서 — 2026-10-03

이관본의 최신 연구를 현재 프로젝트 계획과 연결하고, Colab CPU에서 기존
수치표의 미완료 검증과 두 가지 이론 후속 검사를 진행했다. **원래 native VG의
밀도 추정 문제는 아직 해결되지 않았다.** BR5는 `measurement_unresolved`,
W2는 gated, C2a/C2b는 unentered이며 실제 LLM activation은 실행하지 않았다.

## 1. 이관 기록에서 확정한 현재 상태

W1의 350 fits/1,050 snapshots, 별도 21-fit·9-fit probe, fresh precision-reset,
G1/G2와 후속 수리 검토를 구분했다. 이미 실패로 종료한 학습 분기는 반복하지
않았다. 상세 완료조건과 증거 경로는 [연구 상태 감사](RESEARCH_STATE_AUDIT.md)에
있다. 이관 요약의 G1 `known-dictionary` 표현은 세부 protocol/result와 달랐다.
G1은 orthogonality·unit-amplitude 구조를 주고 D/q/v를 학습했으며, 알려진 D를
쓰는 별도의 exact-mixture control과 혼동하지 않는다.

## 2. 저장된 population 수치표: 독립 검증 20/20 완료

새 verifier는 기존 optimizer를 다시 실행하지 않았다. 원본 manifest·seal 및
실제 사용한 결과·trace를 해시로 연결하고, 전체 threshold 구간의 정확한
분할을 복원했다. 과거에 제외한 leaf도 포함해 모든 최종 구간의 목적함수
상한과 파라미터 이미지를 독립적인 512-bit Arb 계산으로 검사했다.
최종 leaf의 모든 certificate 필드가 생성/precision-update 기록과 일치하는지도
검사했다. 기존 replayer의 ID·domain 대조만으로는 이 마지막 연결이 충분하지
않다는 문제를 보완했다.

| 검증 항목 | 과거 기록 | 이번 별도 결과 |
|---|---|---|
| 독립 numerical replay | 3/20 완료, 17 audit-unresolved | 새 terminal-partition 검증 20/20 완료 |
| 양의 crossover 16개 사례 | 4개 image-resolved, 12개 image-unresolved | 엄격히 열등한 저장 leaf를 제거한 뒤, 모든 남은 연결성분의 q/v 이미지가 지정 폭 기준을 충족 |
| 정확한 lambda=0 anchor | 4개 | 동일한 해석적 값 검증 |
| 신규 optimizer·subdivision·sample | 없음 | 없음 |

새 이미지 판정은 각 연결성분에 대한 것이다. 일부 사례에는 여러 성분이
남으므로 유일한 optimizer나 전체 합집합의 직경을 보장한다고 표현하지 않는다.
이전 결과와 상태는 수정하지 않았다. 과거의 12개 primary 미해결과 17개 replay
미해결, 이번 별도 검증의 완료는 서로 다른 기록이다.

Colab CPU 354.994초, 최대 메모리 182,460,416 bytes를 사용했고, 사전 상한
600 CPU초·512 MiB 안에서 완료했다. [전체 audit와 구간](results/population-terminal-v1/TERMINAL_AUDIT.json),
[물리·수리 검토](PHYSICS_AUDIT.md)에 정확한 범위와 근거가 있다.

## 3. “충분히 작은 잡음”을 명시적 충분조건으로 정량화

알려진 two-atom normalized population 모형에서, 고정 crossover 경로
`h=(lambda*sigma/2)^2`와 joint scalar variance를 사용했다. 모든 prelimit 전역
최적점이 들어가는 variance 범위를 먼저 정하고, 전체 제외 activity 영역의
목적함수 상한과 entropy·Gaussian-tail 오차를 비교했다. 선택한 stationary
root의 위치만 검사한 결과가 아니다.

`r=q0(1-q0)`로 쓰면 아래 값은 `0<sigma<=sigma0` 전체에서 해당 부등식을
보장하는 **충분한** 잡음 상한이다. 최대로 허용되는 실제 잡음값은 아니다.
`—`는 고정된 bound/grid로 인증하지 못했거나 lambda=0에서 하한 주장이
해당하지 않음을 뜻한다.

| q0 | d | lambda | q*<q0의 sigma0 | r<q*<q0의 sigma0 |
|---:|---:|---:|---:|---:|
| .1 | 2 | 0 | 1e-4 | — |
| .1 | 2 | .5 | 1e-4 | — |
| .1 | 2 | 2 | 1e-4 | 1e-5 |
| .1 | 2 | 8 | 1e-8 | 1e-8 |
| .1 | 2 | 16 | — | — |
| .1 | 10 | 0 | 1e-4 | — |
| .1 | 10 | .5 | 1e-4 | — |
| .1 | 10 | 2 | 1e-4 | 1e-5 |
| .1 | 10 | 8 | 1e-8 | 1e-8 |
| .1 | 10 | 16 | — | — |
| .3 | 2 | 0 | 1e-3 | — |
| .3 | 2 | .5 | 1e-3 | 1e-10 |
| .3 | 2 | 2 | 1e-4 | 1e-4 |
| .3 | 2 | 8 | 1e-8 | 1e-8 |
| .3 | 2 | 16 | — | — |
| .3 | 10 | 0 | 1e-3 | — |
| .3 | 10 | .5 | 1e-3 | 1e-10 |
| .3 | 10 | 2 | 1e-3 | 1e-3 |
| .3 | 10 | 8 | 1e-8 | 1e-8 |
| .3 | 10 | 16 | — | — |

전체 20개 중 16개에서 위쪽 activity 제외를 인증했고, 양의 crossover 16개 중
10개에서 strict bracket 양쪽을 인증했다. 원래 잡음 `sigma=.05`에서는 20개
모두 이 충분조건으로 판단할 수 없었다. 이는 이론의 정량적 예시이며, 실용적인
잡음 범위나 현재 native VG의 추정 성능을 입증한 결과가 아니다. 원래 analytic
정리를 부정하는 결과도 아니다. 계산은 Colab CPU 0.153초였다.

[유도와 가정](FINITE_NOISE_DERIVATION.md), [독립 검토](FINITE_NOISE_INDEPENDENT_REVIEW.md),
[모든 오차항·한쪽 하한·고정 grid 결과](results/finite-noise-v1/FINITE_NOISE_CERTIFICATES.json)를
함께 보관한다.

## 4. 원래 native 코드의 유한 beta·bias 검사

Noiseless iid Bernoulli 두 좌표, teacher decoder D=I, 고정 beta/gamma에서
비음수 amplitude와 sigmoid gate를 가진 실제 native 모델을 구성했다.
해석적으로 정한 한 candidate와 공통의 양의 amplitude floor만 사용했으며
학습이나 candidate 탐색을 하지 않았다. 모든 파라미터는 유한하고 JSON으로
저장했다. 코드의 `free_energy`와 저장된 float64 파라미터를 독립 평가한
240자리 Decimal 값을 비교했다.

- 고정된 q 2개 × beta 4개 × gamma 2개, 총 16개 사례를 모두 보존했다.
- 10개에서 목적함수 감소를 float64로 구분했다.
- 6개는 차이가 해상도보다 작아 `unresolved_float64_difference`로 남겼다.
- implementation failure는 없었다. 전체 process CPU는 3.880초였다.

수식은 finite beta에서 정확한 zero-bias 점을 복원 목표로 두는 데 한계가 있음을
보여 준다. Decoder는 계속 teacher이므로 dictionary gap의 반증이 아니다.
현재 beta=400 사례의 효과는 극도로 작으며, 이전의 큰 support 실패를 설명한다고
쓰지 않는다. 여기서 사용한 float64 해상도 기준과 Decimal 비교는 formal
outward-rounded certificate가 아니다.

[유도·고정 사양](NATIVE_FINITE_BETA_DERIVATION.md),
[독립 검토](NATIVE_FINITE_BETA_INDEPENDENT_REVIEW.md),
[16개 실제 모델과 결과](results/native-bias-v1/native_bias_witness_results.json)에 상세가 있다.

## 5. 유한 precision에서의 원래 native 목적함수 한계

기존 평균위험 bridge의 열린 가정인 **목적함수 개선이 작은 clean stochastic
risk를 보장하는가**를 직접 다뤘다. sigma=.05의 동일한 noisy input만 쓰는
유한 sigmoid/softplus 모델을 구성했다. 관측 잡음까지 복사하는 모델의 목적함수
상한과, clean stochastic risk가 `epsilon=1/20000` 이하인 **모든 native 모델**의
하한을 비교했다. 이는 원래 잡음 에너지의 1%를 복원 오차 목표로 두는 엄격한
조건이다. 같은 정보로 그 조건을 만족하는 유한 native 모델도 구성했으므로
비어 있는 비교집합을 이용한 결과가 아니다.

고정된 q=.1/.3, beta=400/1600/25600, gamma=1/16의 12개 조건을 모두 검사했다.
512-bit outward arithmetic에서 6개는 엄격한 분리를 인증했고 6개는 충분조건이
미분리였다. 별도 optimizer·sample·quadrature는 없었고 전체 process CPU는
0.151초였다.

| beta | gamma | beta × sigma² | 두 q 사례의 결과 | 인증한 objective gap의 보수적 하한 |
|---:|---:|---:|---|---:|
| 400 | 1 | 1 | 둘 다 판단 보류 | — |
| 400 | 16 | 1 | 둘 다 판단 보류 | — |
| 1600 | 1 | 4 | 둘 다 복원 장애 인증 | > .61347 |
| 1600 | 16 | 4 | 둘 다 판단 보류 | — |
| 25600 | 1 | 64 | 둘 다 복원 장애 인증 | > 49.21341 |
| 25600 | 16 | 64 | 둘 다 복원 장애 인증 | > 19.83993 |

따라서 분리된 조건에서는 전역 최적점의 존재를 가정하지 않아도, 충분히 낮은
목적함수값이 위의 작은 clean-risk 조건과 양립할 수 없다. 과거의 무한 precision
발산 예시와 달리 beta와 모든 모델 파라미터가 유한하며, 한 복원 checkpoint만
상대로 비교한 결과도 아니다.

**실제 잡음에 맞춘 beta=400은 이 결과로 반박하지 못한다.** Decoder identity,
hard support 또는 q 추정 실패를 직접 증명한 것도 아니다. 고정된 큰 bias가
허용되는 모형과 해당 복원 오차 목표에서의 조건부 결과다. 새 연구 계획에는
precision과 실제 noise의 관계를 명시해야 한다는 근거를 추가한다.

[수리·source 검토](NATIVE_FINITE_BETA_NOISE_OBSTRUCTION_REVIEW.md),
[사전 고정 protocol](NATIVE_NOISE_OBSTRUCTION_PROTOCOL.md),
[12개 전체 bound와 유한 모델](results/native-noise-v1/native_noise_obstruction_results.json)에
정확한 부등식·구간 끝점·제약을 보관했다. 인증은 실수값 native architecture에
대한 것으로, float64로 반올림한 파라미터에 자동으로 이전하지 않는다.

## 6. 남은 연구와 실제 LLM 실행 경계

Calibrated precision에서의 복원 보장, 관측 가능한 noise/representation 조건,
hard-between 관측량의 식별 조건은 남는다. 새 toy나 bound를 원래 q 추정기의
성공으로 대체하지 않으며, 그 근거가 없으면 W2와 확인 world를 열지 않는다.
다음 순서와 금지된 재시작 분기는 [후속 연구 계획](NEXT_RESEARCH_PLAN.md)에
명시한다. 전체 pre-LLM 연구 목표는 아직 미완료다.

## 재현성과 실행 위치

모든 테스트·수치 계산·모델 forward는 Colab Standard CPU에서 실행했다.
로컬은 코드·문서 편집, 파일 해시·압축·전송·결과 정리와 Git에 사용했다.
Controller 47개, 수리/증거 계약 51개, native witness 10개 테스트가 각각
원격에서 통과했다. 51개 통과는 저장된 당시 source/test snapshot에 대한
판정이다. 이후 optional flint가 없는 기본 환경에서 collection을 실패시키지
않도록 test import만 수정했으며, scientific source는 바꾸지 않았다. 이 변경과
새 noise-obstruction·선택적 import 경로는 후속 28개 Colab 테스트에서 검증했다.
테스트 묶음에는 중복 항목이 있으므로 합계를 독립 검증 수로 표현하지 않는다.

Drive 마운트 실패 뒤 짧은 작업은 runtime 저장소를 명시하고 종료 전 모든
결과를 fetch했다. 첫 세션의 관측 잔액은 399.91→399.84 units, 관측 CPU 요율은
0.08 units/hour였다. GPU 할당은 0이었다. 개별 연구의 CPU 시간과 Colab 계정의
잔액 차이는 서로 다른 측정치다. [결과 archive·자원 기록](COLAB_RESOURCE_RECEIPT_FIRST_SESSION.json)에
해시와 관측 범위가 있다.

두 번째 CPU 세션의 검증과 결과 회수까지 마친 뒤 잔액은 399.83 units였고,
세션을 종료한 후 active assignments와 사용 요율은 모두 0이었다.
두 세션 동안 표시된 잔액 차이는 약 0.08 CU다.
[두 번째 세션 기록](COLAB_RESOURCE_RECEIPT_SECOND_SESSION.json)도 별도로 보관한다.
기존 이관본의 lazy optional-import 두 파일은 동일한 bytes를 검토·검증해
통합했다. 새 검증 경로가 Git 밖의 기존 미커밋 코드에 의존하지 않도록 한
호환성 변경이며, 모델의 수식은 바꾸지 않았다.

완료된 연구 job의 archive·source·plan·출력 해시와 주장 범위는
[사후 독립 검토](POST_RUN_INDEPENDENT_REVIEW.md)에서도 확인했다.
검토는 요청한 ARIS, Scientific Agent Skills, physics-guide-solver 및
physicist-analyst를 활용했다. 독립 reviewer는 같은 모델 계열이므로
cross-family 검증이나 사람의 최종 승인을 뜻하지 않는다. 사용한 scientific
workflow 출처는 연구 상태 감사에 검증된 서지와 함께 기록했다.
