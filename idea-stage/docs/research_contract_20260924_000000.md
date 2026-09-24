# Research Contract: VG-SAE first-principles campaign

## Selected Idea

VG의 이산 선택 모형과 변분 자유에너지로 SAE의 목적함수·선택 불확실성·추론 및 공동 학습을 이해한다. 사용자가09-23에 연구 의도를 명확히 했고09-24에 구현 감사·실험 설계·실행을 승인했다. 기존 결과나 이전 주제 선택을 이 캠페인의 확증으로 사용하지 않는다.

## Core Claims

1. C1: 명시한 Bernoulli 모형의 energy/prior/entropy와 dictionary overlap으로 conditional support response 및 factorized inference의 한계를 설명한다.
2. C2: 고정 모형에서 encoder 추론의 추가 오차와 joint dictionary/amplitude 학습의 효과를 구분한다.

Known identities의 수치 재현은 구현 검증이다. 새로운 설명력은 결과에 따라 평가한다. SOTA, 자동 true-L0, calibrated semantic posterior, thermodynamic phase transition은 주장하지 않는다.

## Method Summary

Amplitude1·고정 dictionary의 작은 생성모형에서는 exact64-state posterior와 actual support labels가 모두 있다. 동일 조건에서 mean-field와 gate encoder를 비교한다. Joint VG-SAE의 amplitude는 입력별 point estimate이므로 support conditional risk로 해석한다. Entropy/variance 삭제와 precision 정책은 각각 다른 objective/optimization을 만든다.

## Experiment Design

- Plan: `refine-logs/runs/vg-sae-first-principles-20260924/EXPERIMENT_PLAN.md`
- Config: `configs/first_principles_20260924.json`
- Exact response135 cells; frozen9 fits; joint54 fits; paired worlds2401/2402/2403.
- Baselines: pinned SAELens ReLU(RI-L1), norm-weighted TopK. Recipe 차이를 명시한다.
- Budget: 최대2 GPUh, 로컬 A6000, 기존 .venv.
- Results/Tracker: 같은 refine-logs run 폴더. 완료 상태는 tracker와 raw result를 따른다.

## Key Decisions

현재 구현에서 검증된 numerical/export 문제를 먼저 수정했다. Baseline library를 사용한다는 사실만으로 paper-exact라고 하지 않는다. Training/calibration/test streams는 분리하고 모든 final grid를 보존한다. Test를 보고 controls를 고르거나 확대하지 않는다. 본 캠페인 뒤 연구 주제를 자동 변경하지 않는다.

## Current Results — initial campaign complete

135 exact cells,9 frozen fits,54 joint fits completed with3 worlds.378 tests passed. C1 is supported within the known-model conditions; C2 is partially supported. Overlap creates conditional dependence and a factorization gap. Frozen refinement reduces reverse-KL but worsens Brier/marginal scores at high overlap. Removing variance yields excellent mean-code reconstruction with poor full stochastic/hard risk. No SOTA or semantic calibration claim follows.

Details: `refine-logs/runs/vg-sae-first-principles-20260924/EXPERIMENT_RESULTS.md`. Main GPU wall allocation was approximately0.15GPUh. Optional LLM/stronger tuning studies remain outside this completed initial campaign.
