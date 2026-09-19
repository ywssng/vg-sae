# Research Contract: VG-SAE 복제 유인과 feature 선택

날짜: 2026-09-19 · source: [IDEA_REPORT의 B04](../IDEA_REPORT.md#ranked-ideas).
이 문서는 현재 선택한 연구만 유지한다. B05는 별도 대안이며 C1/C2와 합산하지 않는다.

## Selected Idea

학습된 VG-SAE에서 한 feature를 둘로 나누면 어떤 조건에서 loss가 낮아지는지
검사하고, 그 초기조건에서 추가 학습한 결과가 실제 feature 선택을 손상하는지
대조한다. 기존 gate KL와 variance 항을 직접 조작할 수 있어 현재 코드에 잘 맞고,
짧은 실험으로 가설을 지지하지 않는 경우를 구분할 수 있다는 이유로 선택했다.

기본 replication 대수와 width-dependent prior는 알려진 선행연구다.
현재 선택은 새 정리·새 architecture·SOTA를 확보했다는 뜻이 아니다.

## Core Claims

1. **C1, 미검증:** 실제 학습 checkpoint의 일부 atom에서 parameter로 구현 가능한
   복제가 normalized gate KL 비용까지 포함해 loss상 유리하고, 출력 drift만으로
   그 이득을 설명할 수 없다.
2. **C2, 미검증:** 사전에 정한 clone 개입 뒤 high/low-margin 간 feature-ranking
   학습 변화의 차이가 variance 항 on/off에 의존하는 조건이 있다. hard support
   손상은 별도 grouped F1 확인까지 있을 때만 주장한다.

이 조건부 주장들은 injected initialization에 대한 것이다. 자연 학습에서의
자발적 복제, 일반적인 calibration, real-model semantic feature 보장은 포함하지 않는다.

## Method Summary

gate·amplitude·decoder가 학습된 모델에서 atom별 KL와 Bernoulli variance를 잰다.
고정 beta에서 ideal r-copy loss 변화는 `(r−1)K−beta(1−1/r)V`지만
linear-softplus amplitude는 정확히 a/r로 바뀌지 않으므로 feasible 개입의
reconstruction/variance/KL를 실제로 다시 계산한다. 새 trainable module은 없다.

calibration에서 high/low-margin atom pair를 고정하고, 작은 반대 방향 perturbation을
넣어 동일 clone의 optimizer 대칭을 깬다. 네 arm(high/low × variance on/off)은
같은 Adam reset·beta·gamma·batch 순서로 추가 학습한다. 주입 직후와 종료 시점의
변화만 비교하며, duplicate-invariant grouped AP를 primary로 사용한다.
cutoff 민감도와 grouped hard support F1, hard EV/raw L0는 별도로 확인한다.

## Minimum Convincing Evidence

- C1: 새 holdout에서 ΔF와 `S=beta ΔV+ΔK`가 모두 사전 음수 기준을 통과하고
  mean/hard drift가 작아야 한다. 실제 perturbed 초기조건에도 같은 검사를 적용한다.
- C2: source3 data seeds×3optimizer seeds, high/low×variance on/off에서 주입 직후를
  뺀 paired interaction이 사전 방향·크기 기준을 통과해야 한다. raw L0 증가만으로
  semantic 손상이라고 부르지 않는다.
- cal matching·stability·metric coverage가 성립하지 않으면 유보한다.
  유효한 실험에서 효과가 없으면 검증한 범위의 비지지로 남긴다.
- B1 gate holdout과 B2 final holdout은 서로 다르다. 조건/threshold 선택에 final test를
  쓰지 않는다. 상세 수치 규칙은 [FINAL_PROPOSAL](../../refine-logs/FINAL_PROPOSAL.md).

## Experiment Design

- 먼저 independent spike-and-slab synthetic d16/truth32/width128, density2/32,
  noise.05, train8192/cal2048/B1holdout4096/B2holdout4096.
- 주 대조는 high/low matched cloning 및 기존 variance 항 삭제다. random expansion이나
  더 큰 encoder를 동시에 새로운 기여로 넣지 않는다.
- B1/B2가 성립하면 기존 pinned SynthSAEBench 한 조건으로 외적 반복한다.
- 새 학습법의 baseline 우위를 주장할 때만 Gated/JumpReLU/BatchTopK를 동일
  sparsity·계산 예산으로 추가한다. 현재 계약은 superiority 연구가 아니다.
- 후속 compute ceiling7.05GPUh는 계획상 상한이다. 기존 파일럿에 기록된 GPU 계산은
  약0.0721GPUh이며 시작/import·독립 검증 시간은 이 수치에 포함하지 않았다.

## Current Results

| 관찰 | 실제 근거 | 허용되는 해석 |
|---|---|---|
| Constructed replication 수식 일치 | [16구성 JSON](../evidence/pilots/replication_results.json) | feasibility와 구현 일치; 학습 원인 미입증 |
| 큰 폭의 mean-hard gap, 학습 연장 시 감소 | [risk 표](../evidence/pilots/readout_risk_comparison.json) | 최적화 예산의 영향; sampled risk와 구분 |
| fixed-count prior / 단순 ma readout의 부정 결과 | [파일럿 집계](../evidence/pilots/pilot_summary.json) | 이 작은 조건의 간단한 처방 비지지 |
| B05 Brier 개선/NLL 악화 | [posterior 결과](../evidence/pilots/posterior_report.md) | oracle 진단의 가치; calibration 전반 개선 아님 |

C1/C2의 actual clone intervention 결과는 없다. 과거 Stage2 결과도 one-seed와
calibration stream 재사용 조건 때문에 확증적 baseline 승리로 기록하지 않는다.

## Key Decisions and Next Pointer

- 본 실행의 생성·심사는 GPT-6 Astra ultra; 리뷰는 same-family provisional.
- generic duplication/width-scaling novelty 주장은 철회한다.
- mean, sampled, hard risk를 분리하고 probability count와 raw firing count를 혼동하지 않는다.
- 구현 시작점과 예산은 [EXPERIMENT_PLAN](../../refine-logs/EXPERIMENT_PLAN.md),
  실행 여부는 [EXPERIMENT_TRACKER](../../refine-logs/EXPERIMENT_TRACKER.md)가 기준이다.

## Status

- [x] Idea selected conditionally
- [x] Literature and novelty review
- [x] Exploratory pilots and deterministic checks
- [ ] Clone intervention and duplicate-invariant evaluation implemented
- [ ] C1 confirmed on new held-out data
- [ ] C2 confirmed with paired controls
- [ ] SynthSAEBench external replication
- [ ] Paper draft
