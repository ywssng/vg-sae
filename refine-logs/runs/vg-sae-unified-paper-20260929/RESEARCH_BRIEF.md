# Research Brief: VG-SAE Phase 1–2 통합 논문

사용자 요청 확정:2026-09-29. 최근 Phase1 모형·추론 계획에 원래 Phase2 밀도 추정 목표를 연결하여 하나의 논문 계획을 작성한다. 이번 작업은 계획·검토이며 새 학습 실험을 시작하지 않았다.

## Problem Anchor

VG-SAE를 통계물리학의 명시적 선택모형과 변분원리에서 유도·이해하고, 독립 재학습의 선택 불확실성 곡선으로 적절한 sparse operating density를 추정할 수 있는지 검증한다. Phase1의 성공이 Phase2의 성공을 함의하지 않는다. VG-SAE 개발을 별도 clone 진단 주제로 바꾸거나 SOTA를 유일한 성공조건으로 삼지 않는다.

## 두 단계와 한 논문

- **Phase1/C1:** conditional free energy, exact/MF/encoder, variance·entropy·precision와 readout 역할. 9월24일의 작은 실제 결과를 재사용한다.
- **Phase2/C2a:** truth-free feature correspondence와 native achieved density를 사용해 generating expected density를 추정한다. 아직미실행이다.
- **Phase2/C2b:** 그 추정으로 고른 SAE의 held-out feature recovery와 same-bank oracle regret를 평가한다. C2a와 별도의 endpoint다.
- 논문은 모형 가정→불확실성 해석→밀도 추정→recovery 검증의 한 흐름으로 구성한다. Existing Stage1/2/3는 synthetic/benchmark/LM 데이터규모이며 Phase1/2와 다른 구분이다.

## 현재 제안한 방법

Global learned-beta VG와 고정 native gate를 사용한다. 같은 training world에서 initialization/batch order가 다른5ensemble+1reference를 학습하고, native density로 checkpoint를 맞춘 뒤 signed Hungarian으로 대응한다. Whole uncertainty curve에 단일template를 fit하고 식별성·coverage 부족이면 추정을 보류한다. NNLS normalized weights를 Bayesian posterior로 부르지 않는다.

추정 성공(estimate_status)과 그 밀도에서 checkpoint를 확보했는지(deployment_status)는 별도로 기록한다. Test truth는 selector에 들어가지 않는다. Recovery 평가의 all-world policy는 필요 시 reconstruction selector로 fallback하되 이를 밀도추정 성공으로 세지 않는다.

## 근거와 상태

- Phase1:135 exact cells,9 frozen fits,54 joint fits,3worlds의 기존 초기결과. 조건부 지지이며 semantic calibration이나 자동true-L0 결과가 아니다.
- Phase2:설계만 존재한다. 새 confirmatory worlds와 구현이 필요하다. 모든미실행항목은 tracker에서TODO다.
- 계획된 numerical thresholds, worlds, budget는 제안이다. 이전2GPUh 승인이나 과거 실험 결과를 새 실행승인·확증표본으로 확대하지 않는다.
- 새학습·새teacher/encoder·대형LM benchmark는 이번 작업에서 수행하지 않았다. 큰Stage2/3는 core synthetic 이후 조건부 확장이다.

## 현재 문서

- 통합 논문: `PAPER_PLAN.md`
- 방법: `refine-logs/FINAL_PROPOSAL.md`
- 상세 실행계획/상태: `refine-logs/EXPERIMENT_PLAN.md`, `refine-logs/EXPERIMENT_TRACKER.md`
- 이번 run·review·문헌: `refine-logs/runs/vg-sae-unified-paper-20260929/`
- Phase1 원본: `refine-logs/runs/vg-sae-first-principles-20260924/`
- 기존 결과 요약: `refine-logs/EXPERIMENT_RESULTS.md`는 Phase1만의 결과이며 통합계획 전체의 완료기록이 아니다.

원래원고 `tex/aps/apssamp.tex`는 역사적 density-estimation 구상을 보존한다. 통합 outline에 맞춘 LaTeX 개정은 이후 별도 작업이다.
