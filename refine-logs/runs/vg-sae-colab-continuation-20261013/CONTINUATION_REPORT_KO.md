# VG-SAE 연구 재개 결과 — 2026-10-13

정밀도와 prior를 실제 생성분포에 맞춰도, native VG 목적함수의 감소가
hard 계수 오차의 감소를 보장하지 않는 **유한 모델 두 개의 명시적 비교**를
추가했다. 사전과 hard support는 동일하다. 이 결과는 전체 복원 모형 집합에
대한 최적값 비교, 실제 학습 실패, Phase 2 밀도 추정 실패를 뜻하지 않는다.

## 이전 중단 지점 정리

10월 3일 완료된 네 검증을 다시 실행하지 않고 보존했다. 다운로드된 네 결과
묶음의 25개 파일이 작업 파일 및 Git index와 바이트 단위로 일치함을 확인했다.
두 CSV에는 원본 CRLF를 보존하는 한정된 Git attribute를 적용했다. 해당 결과,
기존 검토, 실행 코드와 scoped test는 `4dd54d6`으로 커밋하고 origin/main에
푸시했다. 다른 기존 미커밋 변경은 보존했다.

## 새 분석의 조건과 결과

입력은 X=S+sigma Z, 두 독립 Bernoulli 단위 진폭 원천, D=I, bias=0이다.
sigma=.05, beta=400으로 beta*sigma^2=1이며, q=.1/.3에 각각 정확한
prior와 Bayes sigmoid gate를 공급했다. 이는 알려진 생성모형의 분석 대조군이다.
q를 추정하거나 teacher 정보를 사용하지 않는 실제 알고리즘이라고 주장하지 않는다.

기준 진폭 1을 `softplus(log(exp(1)-1)+(X-1)/2)`로 바꿨다. 두 모델 모두
원래 affine sigmoid/softplus native 구조의 유한 파라미터를 사용한다.
Gate, 사전, bias, beta, 표준 KL, entropy 및 full stochastic variance는 같다.

| 생성 q | 목적함수 감소량의 인증 하한 | hard 계수 NMSE 증가량의 인증 하한 | 변경 후 hard 계수 NMSE 인증 상한 |
|---|---:|---:|---:|
| .1 | > .010498 | > .00004872 | < .000626 |
| .3 | > .031495 | > .00004872 | < .000626 |

표는 정확한 outward endpoint에서 보수적으로 반올림했다. 실제 계수 오차나
목적함수 값을 추정한 표가 아니라, population 차이 및 오차의 **부등식**이다.
상세 상한은 .000625보다 극히 조금 크므로 `NMSE <= .000625`라고 쓰면 안 된다.
원본 구간과 파라미터는
[결과 JSON](results/matched-amplitude-v1/matched_amplitude_results.json)에 있다.

Softplus의 기울기가 0과 1 사이이므로 변경 진폭은 1과 관측 X 사이에 있다.
따라서 관측 입력에 대한 stochastic reconstruction loss는 감소한다.
반면 true-active 좌표의 원래 hard 계수는 1이므로 입력 잡음을 따라가는 진폭은
계수 오차를 만든다. 고정 Gaussian 구간에서의 양의 오차 증가가 inactive
꼬리에서 가능한 감소보다 큼을 인증했다. Gate가 같으므로 모든 입력의 hard
support와 그 FP/FN, support F1은 정확히 같다. Dictionary도 정확히 같다.

수식과 실행 전 계약은 [프로토콜](MATCHED_AMPLITUDE_PROTOCOL.md),
독립 유도와 원래 C2 endpoint 대조는 [Astra 검토](ASTRA_INITIAL_REVIEW.md)에 있다.
정확한 real-parameter 수학적 인증과 float64 Torch 구현 검증은 구분한다.

## 검증과 리소스

- 실행 ID: `matched-amplitude-cpu-20261013-v1`. 고정 두 사례를 모두 보존했고
  두 사례 모두 `pair_certified`다. 검색, seed 선택, quadrature, 학습은 없었다.
- Colab CPU에서 관련 테스트 **3 passed, 0 failed, 0 skipped**,
  JUnit 시간 7.488초. 파라미터 매핑, 동일 gate/support 및 full stochastic
  risk와 native loss의 일치를 확인했다. 전체 저장소 테스트를 실행한 것은 아니다.
- Arb 512-bit, 한 thread의 인증 계산은 process CPU **0.105746705초**,
  wall **0.155477822초**였다. import와 주 결과 직렬화를 포함하며 60/120초
  한도 이내다. 이는 Colab 전체 세션 비용이나 설치 시간을 나타내지 않는다.
- Python 3.13.16, torch 2.11.0+cpu, NumPy 2.1.3, python-flint 0.8.0을
  기록했다. 전체 Python 3.14/SAELens 환경을 재현했다고 주장하지 않는다.
- 결과·worker state·환경·JUnit·로그를 다운로드하고 CPU 세션을 종료했다.
  GPU와 서버의 수치 계산 자원은 사용하지 않았다. 서버에서는 파일·전송·Git만
  처리했다. 종료 후 표시 잔액 399.83 units, rate 0, active assignments 0이다.
  시작 잔액도 399.83이었지만 표시 정밀도 때문에 실제 과금이 정확히 0이라는
  뜻은 아니다.
- [소스 freeze](MATCHED_SOURCE_FREEZE_v1.json),
  [아카이브 receipt](results/matched-amplitude-v1/ARCHIVE_RECEIPT.json),
  [원격 환경](results/matched-amplitude-v1/remote-environment.json),
  [런타임 receipt](results/matched-amplitude-v1/runtime_receipt.json)를 보존했다.

## 요청한 세 모델 검토 상태

| 요청 모델 | 상태 |
|---|---|
| GPT-6 Astra extra high | xhigh로 생성한 독립 subagent가 수학·실행 전 코드·실행 후 결과 검토 완료 |
| Opus 5.5 extra high | 미실행. 협업 도구 roster에 없고, 별도 Claude CLI에도 사용 가능한 인증 토큰이 없음 |
| Gemini 3.1 Pro high | 미실행. 협업 도구 roster에 없고, 저장된 Google OAuth grant가 `invalid_grant`로 거절됨 |

[실행 후 Astra 검토](POST_RUN_ASTRA_REVIEW.md)는 소스 9개와 회수 파일 8개의
아카이브 연결, 해시, 두 사례의 주장 및 테스트 범위를 확인했다. 수치 계산을
다시 실행하지 않았다.

연결된 OpenCodex의 모델 관리 API도 이 서버의 client role에서는 접근할 수
없었다. 다른 모델을 대신 호출하거나 세 모델 합의로 기록하지 않았다. 연결이
복구되면 동일한 frozen evidence를 두 모델에 제공해야 한다. 검토 요청 때문에
프로젝트 밖의 설정·인증 파일을 변경하지 않았다.

ARIS formula-derivation, scientific-critical-thinking, physicist-analyst,
physics-guide-solver를 적용했다. Scientific Agent Skills의 절차 기여는
[arXiv 논문](https://doi.org/10.48550/arXiv.2609.00065)으로 표기했으며,
현재 metadata 확인 내용은 독립 검토 문서에 기록했다.

## 남은 연구

정밀도 보정만으로 원래 생성 계수의 정확성이 목적함수에 의해 보존된다는
단순한 주장은 배제했다. 그러나 이 두 모델의 계수 오차는 모두 작다. C2b의
.02는 selector 간 개선량 기준이므로 이번 결과가 C2b를 반증하지 않는다.
10월 3일의 matched beta400 all-class bound도 여전히 미분리다.

BR5 `measurement_unresolved`, W2 gated, C2a/C2b unentered를 유지한다.
Confirmation worlds 32000–32035 및 실제 LLM activation은 사용하지 않았다.
다음 단계는 [갱신 계획](NEXT_RESEARCH_PLAN.md)에 있다. 이번 두 사례의 성공을
이유로 실패한 gamma/seed sweep이나 큰 fit 캠페인을 재개하지 않는다.
