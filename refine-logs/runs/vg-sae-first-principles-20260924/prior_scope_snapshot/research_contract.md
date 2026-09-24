# Research Contract — Sparse but Wrong에 기반한 VG-SAE 개발

2026-09-22 · run `vg-sae-sparse-but-wrong-20260922`.
[RESEARCH_BRIEF](../../RESEARCH_BRIEF.md)가 사용자의 고정 연구 목표다.

## Selected Method and Purpose

**L1 amplitude sparsification의 대안으로 Variational Garrote 기반 SAE를 개발한다.**
Bernoulli support 확률과 point amplitude를 예측하고, analytic expected risk와
normalized prior/entropy로 learned dictionary를 학습한다. 성공은 실제 feature와
firing support의 복원으로 판단한다. 낮은 MSE, 낮은 L0, 낮은 c_dec만으로 대신하지 않는다.

Sparse but Wrong는 wrong-L0/feature 문제를 제공한다. L1 전반의 실패나 VG의 성공을
그 논문의 결론으로 쓰지 않는다. L1 shrinkage와 selection/magnitude 선행은 Gated 등에서
별도로 인용한다. 원래 VG 연구를 별도 진단 논문으로 바꾸지 않는다.

## Claims and Evidence

- **C1:** 적절히 조율한 비교와 사용 가능한 선택 규칙 아래 VG가 유용한 feature/support
  recovery를 제공하는가? 작은 toy의 가능성과 일부 L1 대비 coefficient 신호는 있다.
  Strong-baseline 우위 및 unknown-L0 선택은 아직 미검증이다.
- **C2:** 같은 VG 구조에서 precision 정책을 통제하면 잘못된 dense support 영역을
  줄이고 유용한 복원 구간을 더 안정적으로 확보하는가? 다음 실험의 가설이다.
  고정 finite beta의 수학적 하한을 경험적 성능이나 semantic recovery 보장으로 쓰지 않는다.

C1 primary는 development/source truth로 고정한 `S_source` recipe의 새 main/target
결과다. 완전히 truth-free 학습이라고 부르지 않는다. Input-only `S_blind`, oracle 및
matched-L0는 별도 비교다. C2는 gamma2와 공통 final T에서 paired precision policies를
비교한다. L0가 바뀌며 recovery가 좋아지는 total effect도 유효하되, 밀도와 분리한
좁은 주장과 혼동하지 않는다.

## Completed Pilots

| 경로 | 완료 근거 | 결정 |
|---|---|---|
| P1 baseline comparison |225 fits; target1.8 VG0/9 coverage, target2 7/9 거의 완벽. L1 rescale 후4/5 coefficient NMSE 우위. Jump coverage없음 | 방법 feasibility와 제한된 L1-alternative 신호; broad superiority 미확인 |
| P2 scalar prior |72 fits; joint0/18, init-stable1/18; train gamma residual>2 in14/18 | 현재 recipe 채택하지 않음; converged wrong-EB point라고 해석하지 않음 |
| P3 exact32 posterior |54 fits;4/18 matching entries(3unique), joint0; 한 same-gamma rescue | total effect 보존, intrinsic covariance/scalable method 주장 보류 |
| Dense-offset witness |수식과2 deterministic tests; trueD와densecode가공존 | precision 대조의 근거; SGD경로 원인이나 새 training pilot 아님 |

Signed-assignment leakage는 pure sign/mapping 오류도 포함한다. Absolute geometry와
실제 다중성분 혼합을 따로 보고한다. Point-amplitude conditional objective를
full generative ELBO나 calibrated semantic posterior라고 쓰지 않는다.

## Minimum Convincing Next Evidence

1. 개발 전용 calibration에서 baseline 범위와 충분한 학습 길이를 확보한다. Source
   우승 coefficient와 checkpoint를 새 main에 보존하고 선택을 test 전에 동결한다.
2. 같은 VG의 fixed10/global-learned-init10/profiled beta를 gamma·data·init·budget·readout을
   맞춰 비교한다. Gamma 학습이나 새로운 joint module을 동시에 추가하지 않는다.
3. 새 recipe를 고정한 뒤50-feature와 한 realistic source로 조건부 확장한다. 기존
   checkpoint는 평가 경로의 근거이며 새 method 효과를 대신하지 않는다.

정확한 grid/조건/추가시도 상한/중단 기준은
[EXPERIMENT_PLAN](../../refine-logs/EXPERIMENT_PLAN.md), 실제 상태는
[EXPERIMENT_TRACKER](../../refine-logs/EXPERIMENT_TRACKER.md)가 기준이다.
후속 main/transfer 및 큰 source 실험은 아직 실행하지 않았다.

## Decision and Next Pointer

새 novelty와 scientific review는 PROCEED WITH CAUTION이다. Proposal readiness는
advisory7.85/10, REVISE이며 same-family provisional이다. 이 경험적 gap을 해결하려고
같은 method의 다음 최소 대조로 진행한다. 높은 점수를 얻기 위한 목표 변경이나
불필요한 module 추가는 하지 않는다.

[통합 보고서](../IDEA_REPORT.md) · [최종 제안](../../refine-logs/FINAL_PROPOSAL.md) ·
[심사 정리](../../refine-logs/REVIEW_SUMMARY.md).
