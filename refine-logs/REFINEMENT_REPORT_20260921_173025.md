# Refinement Report — VG-SAE 방법 개발

2026-09-22. 현재 판정 REVISE, advisory7.85/10, same-family provisional.

## Immutable Problem Anchor

Sparse but Wrong가 제기한 잘못된 희소성 수준과 feature 혼합 문제를 출발점으로,
L1 진폭 벌점에 의존하지 않는 Variational Garrote 기반 새로운 SAE를 개발한다.
확률적 support 선택과 amplitude 추정이 실제 feature 복원을 개선하고, 올바른 활성 수를
모르는 상황에서 희소성 설정에 따른 오류를 줄이는지 검증한다. 주 연구는 VG-SAE 방법
개발이며, 재구성–희소성 곡선·decoder 진단은 성공의 대리 목표가 아니라 이를 검증하는 도구다.

## 방법 정교화

L1 대안이라는 큰 목표를 실제 feature 방향과 firing support의 복원으로 구체화했다.
Sparse but Wrong의 직접 L0/feature 문제와 Gated 등에서 확인하는 L1 shrinkage를
구분해 인용했다. 기존 VG의 Bernoulli support, point amplitude, analytic risk를
유지하며 실제 prior/precision/inference 약점을 분리했다.

새 모듈의 묶음을 만드는 대신 precision 정책 한 가지를 다음 대조로 골랐다.
Scalar gamma 학습은 현재 recipe에서 비지지이고, joint32는 한 total-effect 신호가
있어도 일반적/scalable 이득을 확보하지 못했다. 이를 기본 구성에 자동으로 넣지 않는다.

## Full revision 기록

- [초기 제안](runs/vg-sae-sparse-but-wrong-20260922/round-0-initial-proposal.md): P1 진행 중,
  완성된 P2/P3와 핵심 수식을 정리했다.
- [Revision1](runs/vg-sae-sparse-but-wrong-20260922/round-1-refinement.md): P1 완료,
  nonstationary prior와 signed leakage 해석, 최소 precision 대조를 통합했다.
- [Revision2](runs/vg-sae-sparse-but-wrong-20260922/round-2-refinement.md): profiled/learned
  epsilon branch 구분, bounded coefficient 신호, finite-beta 주장 범위를 정리했다.
- [상세 계획 v2](runs/vg-sae-sparse-but-wrong-20260922/experiment-plan-v2.md): C1 selector와
  C2 checkpoint, source winner, 공유 RNG/state, candidate/checkpoint 회계를 명시했다.

## 바꾸지 않은 핵심과 아직 바꾸지 않은 가설

원래 VG-SAE 방법 개발이라는 목표는 모든 full proposal에 그대로 있다.
진단 결과나 lower-bound 분석은 주방법을 개선하는 근거이며 새 진단 논문을
중심 기여로 바꾸지 않았다. 필요 없는 teacher/RL/새 backbone을 붙이지 않았다.

C1의 가능성과 비교 우위, source-supervised와 target-blind 선택, total effect와
matched-L0 질문을 각각 구분한다. C2의 empirical benefit은 아직 가설이다.
점수 상승으로 연구 근거의 부족을 덮지 않는다.

## 미해결 사항

조율된 JumpReLU 범위·convergence, VG의 distinct useful effect, non-oracle 선택,
precision의 실제 역할,50-feature와실제source전이가 남아 있다. 이는 다음 실험의
질문이며 이번 파일럿이 완료되지 않았다는 뜻은 아니다.

[심사·반영](REVIEW_SUMMARY.md) · [최종 제안](FINAL_PROPOSAL.md) ·
[실험 계획](EXPERIMENT_PLAN.md) · [점수 이력](score-history.md).
