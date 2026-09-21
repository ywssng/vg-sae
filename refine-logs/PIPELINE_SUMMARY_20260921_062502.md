# VG-SAE 방법 개발 파이프라인

실행 `vg-sae-development-20260921` · 2026-09-21.
원래 Variational Garrote 기반 새로운 SAE가 주방법이다. 이전 진단 주제 선택은
현재 목표를 대신하지 않는다. 고정 범위는 [RESEARCH_BRIEF](../RESEARCH_BRIEF.md)다.

문헌 registry 32개, 같은 방법 안의 개발 선택지 12개, 세 pilot 경로와 P3b 탐색 후속을
완료했다. 신규성 검토와 과학적 비판 검토는 모두 **PROCEED WITH CAUTION**이며,
GPT-6 Astra ultra의 same-family provisional 판단이다. 최초성이나 성능 우위의 확증이 아니다.

- P1은 같은 L0 상한 아래 새 샘플의 recovery–fidelity 절충을 확인했다.
- P2는 beta 초기값 변경을 지지하지 않았고, 더 긴 학습의 EV 개선과 recovery 악화가
  함께 나타나 cal recovery 기반 checkpoint 선택이 필요함을 보여줬다.
- P3/P3b는 조건부 보정의 일부 품질 신호가 있지만 사전 비용 기준을 넘었다.
  새 solver/teacher를 기본 방법에 넣지 않았다.
- 새 runner의 관련 테스트 14개와 compile 검사를 통과했다. 전체 repository suite를
  이번에 재실행한 것은 아니다. 파일럿 주요 타이머 합계는 약 0.2883 GPUh다.

현재 clean method는 [FINAL_PROPOSAL](FINAL_PROPOSAL.md), 구체적인 grids·예산·선택은
[EXPERIMENT_PLAN](EXPERIMENT_PLAN.md), 완료/미완료는 [EXPERIMENT_TRACKER](EXPERIMENT_TRACKER.md)에 있다.
최종 방법 제안 심사 결과는 [REVIEW_SUMMARY](REVIEW_SUMMARY.md)에 기록한다.

다음은 M0의 독립 split/shared-batch/checkpoint-selection runner 구현과 full-size
throughput 측정이다. 기본 계획의 B1 432 + B2 신규 54 = 최대 486 trainings와
1,944 checkpoint 후보는 **향후 계획**이다. 40 GPUh는 제안한 상한이며 완료 시간의
예측이나 이미 실행한 작업이 아니다. 저장된 Stage2/3 모델의 24개 checkpoint를 먼저
평가하도록 했다. Gemma 문서 provenance를 확인하기 전 B3b split 완료를 주장하지 않는다.

원래 core model과 기본 inference의 의미는 보존했다. 새 trainable module을 금지하는
사용자 제약은 없고, 필요성이 확인될 때 같은 anchor 안에서 검토한다.
