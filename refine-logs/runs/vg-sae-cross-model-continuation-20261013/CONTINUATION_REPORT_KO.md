# 연구 재개 기록 — 2026-10-13

## 이번에 진행한 연구

원래 matched-precision amplitude 결과와 C2 계획을 유지하면서, 고정 finite bank의 full clean stochastic risk를 label-free 신뢰상한으로 연결하는 분석을 진행했다. Astra xhigh는 기존 zero-pattern 조건을 대체하는 one-/two-bit local-flip 기하 정리를 제안했다. 이 조건은 K=16, p=.125/.25에서도 구조적으로 배제되지 않는다. 정확한 가정과 증명은 ASTRA_INITIAL_REVIEW.md에 있다. square/orthogonal binary generator 및 알려진 noise/origin 계약에 한정되며 overcomplete family R에는 적용되지 않는다.

고정 gamma12 비교집합의 population 비공허성은 보였으나, 일반적인 global moment envelope/MOM 신뢰상한은 현재 표본 예산에 사용할 수 없다. 가장 유리한 기준 모델에서도 n<=8192이면 모든 가능한 관측 배열에 대해 C_U>19/25인 반면 기하 조건은 C_U<67/(16*10^12)을 요구한다. FINITE_SAMPLE_BUDGET_OBSTRUCTION.md의 짧은 정확한 부등식을 Astra가 독립 검토하여 승인했다. ASTRA_AUDIT_DESIGN.md는 정확한 유리수 envelope와 필요/충분 표본수의 구분을 보충한다.

이 결과는 해당 신뢰구간 구성의 한계다. VG 복원 실패, 실제 SURE 분산의 하한, C2b의 .02 paired contrast, q 식별 또는 다른 인증 방식의 불가능성을 의미하지 않는다. 실제 체크포인트를 인증한 것도 아니다.

## 후속 기하 후보

THIRD_CUMULANT_GEOMETRY_CANDIDATE.md에 더 관대한 대안을 기록했다. 동질적인 square orthogonal binary 계약의 third cumulant와 decoder의 polar factor를 이용하면 clean-risk가 극도로 작다는 조건 없이 양의 permutation 기하 오차를 상한할 수 있다. 주 에이전트의 row-max 제안을 Astra가 doubly stochastic/permutation 분해로 강화하여 작은 오차 전제까지 제거했다. Astra 및 Opus xhigh의 해당 기하식 검토는 완료했으나 Gemini 검토와 유용한 유한표본 tensor 신뢰반경은 없으므로 다음 연구 후보이며 인증된 estimator가 아니다. 같은 정보의 classical moment/tensor 비교가 필요하다.

## 자원 결정

표본을 생성하지 않아도 해당 방식의 실패가 결정되므로 Colab 세션·학습·수치 검증을 시작하지 않았다. Mac/deepstat에서는 문서·코드 읽기 및 작성, 모델 API 검토 요청과 Git 관리만 수행했다. confirmation worlds와 실제 LLM activation은 접근하지 않았다.

## 지정 모델 검토와 연결

Astra xhigh의 이론 유도 및 primary-agent obstruction 독립 검토를 완료했다. Opus/Gemini native subagent는 encrypted task 오류로 시작되지 않아 collective-inference에서 실제 사용한 OpenCodex plaintext API 패턴을 적용했다. 토큰 제한으로 끝난 응답과 실패 요청은 완료 검토로 집계하지 않는다. Opus xhigh의 focused 최종 검토는 예산 모순과 cumulant 기하식 모두 blocking defect 없이 통과했다. 전체 local-flip 정리나 기존 matched-amplitude 증거의 완전 검토로 확대하지 않는다. REVIEW_RESPONSE.md 및 각 REVIEW_RECEIPT.json, REVIEW_CONNECTION_STATUS.json에 적용 의견과 미완료 상태를 기록한다.

사용자는 Gemini 3.1 Pro high를 고정하고 Flash 대체를 거부했다. 기존 daily-cloudcode 주소에서의 404 원인을 확인하여 같은 Google 제공자의 cloudcode-pa 주소로 연결을 수정했다. 새 주소의 모델 조회에는 Gemini 3.1 Pro (High)/gemini-pro-agent가 존재한다. 실제 생성 요청은 현재 RESOURCE_EXHAUSTED/429로 거절된다. 모델 선택창의 존재와 제공자 생성 성공을 구분하며, Gemini 검토 완료나 세 모델 합의를 주장하지 않는다.

## 다음 연구 순서

1. 지정 Opus/Gemini의 실제 완료 검토를 받아 local-flip 정리와 예산 obstruction을 반영한다. 기존 matched-amplitude 고정 증거의 누락된 두 모델 검토도 별도로 완료한다.
2. 표본을 더 모으기 전에 localized derivative와 Stein scalar의 cancellation을 이용한 더 날카로운 신뢰상한 또는 더 관대한 기하 조건을 유도한다. 이번에 실패한 global-envelope recipe로 4096/8192-row audit을 실행하지 않는다.
3. 새 경로의 finite-sample 예산과 정보 계약이 유효할 때만 Colab CPU 소규모 검증을 사전 고정한다. 수렴 존재나 유한 비교집합의 비공허성을 optimizer가 그 집합에 도달한다는 주장으로 확대하지 않는다.
4. hard-between q 식별은 독립 과제로 유지한다. BR5 measurement_unresolved, W2 gated, C2a/C2b unentered이며 confirmation과 LLM activation은 보류한다.
