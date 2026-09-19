# Refinement Report

2026-09-19 ·Rounds3/5 ·최종점수9.10/10 ·**READY: 연구 계획 단계만**.

## Problem Anchor

VG-SAE의 Bernoulli gate와 입력별 amplitude를 함께 학습할 때, 목적함수가
feature를 선택하는 표현과 동일 정보를 여러 latent에 나누는 표현 중 무엇을
선호하는지 밝히고, 그 선호가 실제 경성 feature 품질에 영향을 주는지 검증한다.
기존 VG-SAE 코드와 synthetic/실제 activation 평가 구조를 사용한다.
일반적인 dropout 복제 정리, 확률적 SAE 최초성, 보편적 calibration 또는 SOTA는
주장하지 않는다. 새 학습 모듈 없이 원인 하나를 판별하는 것이 목표다.

## Final Method

현재 checkpoint의 gate KL와 variance를 이용해 clone의 loss 선호를 예측하고,
실제 softplus parameter 개입에서 검증한다. 유효한 조건에서만 high/low margin과
variance on/off의 paired continuation으로 feature-ranking 및 hard support의
변화를 따로 검사한다. 새 trainable module은0개다.

## What Changed

1. 복제의 수식적 가능성을 실제 optimization 결과와 분리했다.
2. 대칭이 유지되는 학습을 피하고 주입 직후와 추가 학습을 구분했다.
3. 복제 불변 지표, cutoff robustness, native hard 지표를 구분했다.
4. actual t0+에 C1을 다시 검사하고 go/no-go와 최종 holdout을 나눴다.

## Review History

[점수표](score-history.md), [라운드별 해결 기록](REVIEW_SUMMARY.md),
[Round1 원문](round-1-review.md), [Round2 원문](round-2-review.md),
[Round3 원문](round-3-review.md)에 full reviewer responses를 보존했다.
실제 reviewer identity는 `/root/refine_ultra`, model은gpt-6-astra, effort는ultra다.
Route는same-family provisional이고 CALIBRATION:none이다.

## Remaining Weaknesses

- C1/C2는 아직 실험하지 않은 가설이다.
- 알려진 dropout/SoftSAE 관찰을 VG에 대입한 수준에 머물 위험이 크다.
- 세 data world 및 제한된 cosine cutoff 집합은 강한 모집단 보증이 아니다.
- SynthSAEBench의 상관된 true dictionary는 mapping 해석을 더 어렵게 할 수 있다.
- 실제 모델 activation에는 generating support 정답이 없어 calibration을 바로 주장할 수 없다.

## Planning Gate

- Method thesis: frozen loss-preference → feasible clone → conditional paired continuation.
- Dominant contribution: 실제 VG의 conditional KL–variance 경쟁과 feature 선택의 관계.
- Rejected complexity: 새 prior/encoder/calibration net, 무관한 LLM teacher.
- Remaining validation: 실제효과·실질적크기·외적반복.
- Frontier primitive: 새 primitive 없음; 현대SAE 평가가 연구 대상.
- Next: [EXPERIMENT_PLAN](EXPERIMENT_PLAN.md)의R001 구현, B1 통과시에만B2.
