# VG-SAE 개발 제안 정교화 기록

실행 `vg-sae-development-20260921` · reviewer GPT-6 Astra ultra,
same-family provisional, CALIBRATION:none. 과거 09-19의 9.10은 이번 평가에 사용하지 않았다.
**최종 READY, 9.05/10, readiness review 2 rounds.** Proposal 준비도에 한정된다.

## Problem Anchor

현재 구현된 Variational Garrote 기반 SAE를 출발점으로, VG의 확률적 support
선택과 amplitude 추정이 SAE 학습에 제공하는 이점을 명료화하고 필요한 방법·학습·
추론·평가 절차를 개선한다. 합성 ground truth, 현실적인 synthetic benchmark,
실제 모델 activation의 단계적 검증으로 새로운 SAE 방법으로서의 기여를 평가한다.
주 연구 목표는 VG-SAE 개발이며, 진단 실험은 그 방법을 개선하고 검증하기 위한 도구다.

## 방법의 정리

핵심은 Bernoulli support를 적분하고 amplitude를 입력별 점 추정으로 남기는
amortized VG-SAE다. 이미 구현된 정확한 quadratic support risk와 normalized prior/entropy를
명확히 기술했다. Full generative ELBO·unbiased amplitude·semantic posterior 해석은 하지 않는다.

P1의 fresh-sample 신호는 후속 연구를 정당화하지만 한 training world와 불균등 과거
HPO의 한계가 남는다. P2의 beta 초기화 변경은 채택하지 않았다. P3/P3b의 비용·공통
amplitude control 결과를 반영해 기본 inference를 보존했다. 이는 모든 효율적 보정이
불가능하다는 결론이 아니다.

## Full revision 기록

- [Initial proposal](runs/vg-sae-development-20260921/round-0-initial-proposal.md): 원래 목표,
  수식, 실제 파일럿, C1/C2와 세 블록을 정리했다.
- [Revision 1](runs/vg-sae-development-20260921/round-1-refinement.md): P2에 따른 checkpoint
  선택, native L1, cal-only comparator, scale compensation과 좁은 C2를 반영했다.
- [Revision 2](runs/vg-sae-development-20260921/round-2-refinement.md): C2의 profiled 범위와
  variance/entropy별 판정을 명시하고 GMM fit 단위·Stage2 metric 정의를 정리했다.
- 상세 [변경 기록](runs/vg-sae-development-20260921/round-2-change-log.md),
  [최종 clean 제안](FINAL_PROPOSAL.md), [실험 계획](EXPERIMENT_PLAN.md).

## 주장–증거 경계

C1은 sparsity 상한 아래 recovery–fidelity tradeoff다. Exact-L0가 다르면 같은 L0라고
쓰지 않는다. 모든 후보의 실제 비용·실패·세 paired world 차이를 보고한다.
C2는 profiled recipe의 내부 coupling에 관한 두 항의 개별 판정이다. 한 항만 지지되면
그 항만 결론에 쓰고 collapse/비유한/feasibility 부재를 수치 우위로 바꾸지 않는다.

## Simplicity와 방향 유지

새 teacher, RL, head, 복제 정리를 억지로 붙이지 않았다. LLM은 자연스러운 application
activation source다. 동일한 full model을 B1과 B2에서 재사용하고 저장된 Stage2/3 모델의
평가를 우선한다. 새 trainable module 수를 0으로 제한하는 사용자 규칙은 만들지 않았다.

## 남은 구현과 경험적 불확실성

Cal checkpoint runner, shared batch order, 네 completed-update checkpoints,
Stage3 문서 provenance, throughput/공간 측정이 M0/B3의 실제 작업이다.
아직 본 실험이 실행되지 않았으므로 novelty의 실질적 강도와 C1/C2 성공은 미확정이다.
Stage2/3 전이가 약하면 주장의 적용 범위를 제한하고 같은 VG-SAE 안에서 개선한다.

## 심사 기록

[REVIEW_SUMMARY](REVIEW_SUMMARY.md)와 [점수 이력](score-history.md)에 판정·반영을 통합했다.
Full prompt/response는 project-local private `.aris/traces/`에 보존한다.
공유 산출물에는 raw reviewer transcript를 중복 게시하지 않는다.
