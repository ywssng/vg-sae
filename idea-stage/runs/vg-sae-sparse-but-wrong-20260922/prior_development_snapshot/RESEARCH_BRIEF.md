# Research Brief: Variational Garrote 기반 SAE 개발

확정일: 2026-09-21. 사용자가 기존 요청의 의미를 직접 정정한 내용이 기준이다.

## Problem Statement / 고정 연구 목표

**Variational Garrote 기반의 새로운 Sparse Autoencoder를 개발한다.**
VG-SAE 개발 자체가 사용자의 원래 연구 아이디어이며, 이 저장소는 그 연구를
지금까지 구현·실험해 온 작업물이다. 이번 작업은 진행 중인 방법과 연구
파이프라인을 더 명료하고 설득력 있게 다듬는 것이다.

## Immutable Problem Anchor

현재 구현된 Variational Garrote 기반 SAE를 출발점으로, VG의 확률적 support
선택과 amplitude 추정이 SAE 학습에 제공하는 이점을 명료화하고 필요한 방법·학습·
추론·평가 절차를 개선한다. 합성 ground truth, 현실적인 synthetic benchmark,
실제 모델 activation의 단계적 검증으로 새로운 SAE 방법으로서의 기여를 평가한다.
주 연구 목표는 VG-SAE 개발이며, 진단 실험은 그 방법을 개선하고 검증하기 위한 도구다.

## Context

- 기존 코드: `src/sae_model.py`, `src/sae_loss.py`, `src/sae_train.py`,
  `src/saelens_vg.py`. Bernoulli gate, deterministic amplitude, analytic
  expected reconstruction variance, normalized prior/entropy, learned/profiled beta.
- 기존 실험: Stage1 independent synthetic sparse coding, Stage2 pinned
  SynthSAEBench, Stage3 real-model activations. 관련 코드·runner·평가 결과를 재사용한다.
- 사용자의 연구에서 이미 구축한 방법을 보존하고, 변경은 실제 병목과 검증 가능한
  VG-specific 가설에 연결한다. 새 module 수를0으로 제한하지도, 신규성만을 위해
  불필요한 module을 추가하지도 않는다.
- 참고 논문의 방법과 실제 구현 차이를 확인한다. '확률적 SAE 선행연구가 있다'는
  사실만으로 이 연구를 탈락시키지 않는다. 기존 연구와 같은 내용인 주장은 정확히 좁힌다.

## Constraints

- 검토/후속 reviewer: GPT-6 Astra `ultra` (기존 사용자 지정 유지).
- 사용 가능한 자원은 실행 직전 확인한다. 최초 확인은 A6000 48GB4대 idle.
- 이 skill run의 pilot 상한: 아이디어 최대3개, 각 GPU당2시간 추정 이내,
  총8 GPU-hours 이내. 실제 elapsed/조건/seed/config를 남긴다.
- 기존 public API/config 의미를 보존한다. 실험적 변형은 먼저 별도 runner로 검증한다.
- 데이터/계수/threshold 선택용 calibration과 최종 test를 분리한다.
- 프로젝트 밖 파일 생성·수정·삭제 금지. 기존 변경과 결과·논문을 보존한다.
- 외부 메시지 발송과 외부 GPU 구매는 이번 작업에 포함하지 않는다.
- 목표 venue 및 장기 총 compute budget은 미지정이다. 이를 확정 사실로 만들지 않는다.

## What We Already Tried

기존 Stage1/2/3의 구체적 사실은 각 run config와 raw summary를 우선한다.
2026-09-19 agent의 파일럿도 계산 결과로는 남지만, 그때 선택한 '복제 진단 연구'
문제 설정은 사용자의 원래 목적을 오해했으므로 현재 목표를 규정하지 않는다.

- 기존 Stage2에는 one-seed와 calibration/evaluation stream 재사용 한계가 있다.
- 기존 파일럿의 fixed-count prior 및 단순 ma readout 부정 결과는 제한된 조건이다.
  새 VG-SAE 방법 전체의 실패 근거로 일반화하지 않는다.
- posterior-mean, Bernoulli-sampled, hard reconstruction을 구분한다.
  큰 expected L0나 mean-hard gap 자체만으로 문제를 단정하지 않는다.
- known-dictionary posterior 파일럿은 Brier/NLL tradeoff와 undertraining 영향을
  보여주며, 실제 learned SAE의 확률 해석에 그대로 전이하지 않는다.

## Requested Outputs

`idea-discovery` 단계는 생략하지 않되, 생성·비교하는8–12개 후보는 동일한
VG-SAE 개발 목표 안의 구체적 개선안/검증 경로다. 이 중2–3개 cheap pilot으로
다음 개발 선택을 좁힌다. 최종 산출물은 현재 방법의 기여, 필요한 수정,
우선 실행 순서, baseline/ablation 및 성패 판단 기준을 포함한 연구 파이프라인이다.

Canonical outputs: `idea-stage/IDEA_REPORT.md`, `refine-logs/FINAL_PROPOSAL.md`,
`refine-logs/EXPERIMENT_PLAN.md`, `refine-logs/EXPERIMENT_TRACKER.md`,
`idea-stage/docs/research_contract.md`. 이전 결과는 날짜별 이력으로 보존한다.

## Non-Goals

- VG-SAE를 연구 대상으로 삼아 별개의 비판·실패분석 논문으로 주제를 교체하지 않는다.
- 원래 연구를 새로운 주제로 재선정해야 하는 후보 중 하나로 낮추지 않는다.
- 근거 없이 성공, 신규성 확증, baseline 우위, calibration을 선언하지 않는다.
- 방법의 실제 문제를 숨기지도 않는다. 부정 결과는 원래 목표 안에서 다음 설계·
  검증 결정을 바꾸는 근거로 사용한다.
