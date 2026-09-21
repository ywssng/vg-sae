# Pipeline Summary — Sparse but Wrong 기반 VG-SAE

2026-09-22 · run `vg-sae-sparse-but-wrong-20260922`.

주 연구는 L1 대안인 Variational Garrote SAE의 실제 feature/support 복원이다.
문헌35개 registry,20개 원안을13개 development families로 통합하고 세 pilot을
실행했다. 총351 fits이며 독립 training seeds351개라는 뜻은 아니다.

- P1: 적절한 target2 구간의7/9 covered worlds에서 VG 복원은 거의 완벽했다.
  L1 rescale 이후에도4/5 matched pairs의 coefficient NMSE가 낮았다. 그러나
  target1.8과Jump coverage가 부족하고 strong-baseline joint 우위는 미확인이다.
- P2: scalar-prior recipe는 joint0/18, 대부분 gamma gradient도 아직 크다.
  잘못된 EB 정상점으로 수렴한 증거로 해석하지 않는다.
- P3: 한 same-gamma world의 유효한 total-effect rescue가 있지만 intrinsic
  covariance/scalable-method 근거는 부족하다. MAP와 marginal readout도 분리한다.
- 구성적 dense-offset 점검은 precision 대조의 근거다. 실제 SGD 원인으로
  확정하지 않고, profiled floor와 learned branch를 정확히 구분한다.

과학적 계속 여부는 PROCEED WITH CAUTION, 방법 제안은 REVISE/7.85다.
모든 review는 GPT-6 Astra ultra의 same-family provisional이다.

[최종 제안](FINAL_PROPOSAL.md) · [실험 계획](EXPERIMENT_PLAN.md) ·
[추적표](EXPERIMENT_TRACKER.md) · [심사 정리](REVIEW_SUMMARY.md).

다음은 M0 구현·처리량 확인, source calibration, same-VG precision 대조다.
개발후 선택한 정책을 고정해 새 main/transfer를 검증하고, 유용한 경우에만
50-feature/현실적 source로 확장한다. 제안 core 상한248 fits와8 GPUh는
미래 계획이며 현재 실행한 작업이나 사용자 확정 장기 예산이 아니다.
