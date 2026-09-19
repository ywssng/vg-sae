# Review Summary

2026-09-19 ·3/5 rounds ·최종9.10/10 ·**READY: 연구 계획만**.
같은 GPT-6 Astra ultra reviewer를 사용한 same-family provisional 검토다.

## Problem Anchor

VG-SAE의 Bernoulli gate와 입력별 amplitude를 함께 학습할 때, 목적함수가
feature를 선택하는 표현과 동일 정보를 여러 latent에 나누는 표현 중 무엇을
선호하는지 밝히고, 그 선호가 실제 경성 feature 품질에 영향을 주는지 검증한다.
기존 VG-SAE 코드와 synthetic/실제 activation 평가 구조를 사용한다.
일반적인 dropout 복제 정리, 확률적 SAE 최초성, 보편적 calibration 또는 SOTA는
주장하지 않는다. 새 학습 모듈 없이 원인 하나를 판별하는 것이 목표다.

## Round-by-Round Resolution Log

| Round | 우려 | 변경 | 결과 | 남은 불확실성 |
|---|---|---|---|---|
|1|clone 대칭·지표 중복·L0 비교·C1 기준|비대칭 초기개입, Adam reset, t0+/T, grouped AP, actual loss 분해, high/low×variance4arms|8.00 REVISE → full proposal 수정|효과의 실제 존재|
|2|cosine cutoff artifact, 다른 초기조건, AP와 hard support 혼동|.75/.8/.85 robustness, 실제 t0+ 재검증, grouped F1 확인, B1/B2 holdout 분리|8.65 REVISE → full proposal 수정|지정 cutoff 밖의 범위|
|3|수정의 통합 확인|source10000/14000 cap, missing pair world 유보, B3 missing5% 기준과 문구 정리|9.10 READY, 핵심 blocker 없음|C1/C2 empirical evidence·외적 일반화|

## Scope and Pushback

Reviewer는 추가 trainable module이나 새 benchmark 묶음을 요구하지 않았다.
점수 향상을 위해 empirical positive 결과를 만들거나 연구 문제를 바꾸지 않았다.
과거 mean-hard gap을 objective 실패로 부르던 잠정 해석은 철회했고,
calibration도 Brier 하나로 주장하지 않는다. clone의 즉시 출력 보존과 추가 학습의
효과를 구분했으며, 자연 학습의 자발적 복제라고 확대하지 않는다.

## Raw Reviewer Responses

- [Round1](round-1-review.md), [그에 대한 full revision](round-1-refinement.md)
- [Round2](round-2-review.md), [그에 대한 full revision](round-2-refinement.md)
- [Round3](round-3-review.md), [clean final proposal](FINAL_PROPOSAL.md)

초기 아이디어의 신규성 판정5/10 및 비판 리뷰의 현재 논문 준비도2/10과
이번9.10이라는 **계획의 준비도**는 평가 대상이 다르다. 서로 바꿔 인용하지 않는다.
