# Research Brief: Sparse but Wrong에서 출발한 VG-SAE 개발

확정일: 2026-09-22. 사용자가 Sparse but Wrong (Chanin & Garriga-Alonso, ICML 2026)을
핵심 연구 동기로 지정했다. VG 기반 새로운 SAE 개발이 원래 연구 목표라는 결정은 유지한다.

## Problem Statement

**L1 sparsification의 대안으로 Variational Garrote 기반 SAE를 개발하고, 희소성 수치나
재구성 성능만으로는 드러나지 않는 실제 feature 복원 문제를 해결하는 것을 주목적으로 삼는다.**
Sparse but Wrong의 직접 근거는 잘못된 L0가 상관·반상관 feature의 혼합을 유도하고,
희소성–재구성 곡선만으로 SAE 품질을 판단하면 잘못된 dictionary를 선호할 수 있다는 것이다.
이 문제에서 VG의 확률적 support 선택과 amplitude 분리가 유용한 대안인지 검증한다.
논문 자체가 VG의 성공이나 L1 전반의 실패를 입증했다고 기술하지 않는다.

## Immutable Problem Anchor

Sparse but Wrong가 제기한 잘못된 희소성 수준과 feature 혼합 문제를 출발점으로,
L1 진폭 벌점에 의존하지 않는 Variational Garrote 기반 새로운 SAE를 개발한다.
확률적 support 선택과 amplitude 추정이 실제 feature 복원을 개선하고, 올바른 활성 수를
모르는 상황에서 희소성 설정에 따른 오류를 줄이는지 검증한다. 주 연구는 VG-SAE 방법
개발이며, 재구성–희소성 곡선·decoder 진단은 성공의 대리 목표가 아니라 이를 검증하는 도구다.

## Reference Paper and Source Scope

- David Chanin, Adrià Garriga-Alonso, Sparse but Wrong: Incorrect L0 Leads to
  Incorrect Features in Sparse Autoencoders. 기준 local PDF는 arXiv 2508.16560v4,
  2026-07-07, 표지 ICML 2026 / PMLR 306.
- Local library: `refs/`. 기준 PDF:
  `refs/Sparse but Wrong Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders.pdf`.
- 본문은 BatchTopK/JumpReLU를 중심으로 low/high L0의 혼합, decoder pairwise
  cosine `c_dec`, 실제 activation sparse probing의 연결을 검사한다.
- L1 대체는 사용자의 VG-SAE 개발 방향이다. 참조 논문의 직접 결론과 이 새 방법의
  가설을 구분하고, hard TopK를 soft gate로 바꾸기만 하면 문제가 해결된다고 가정하지 않는다.

## Method Context

기존 `src/sae_model.py`, `src/sae_loss.py`, `src/sae_train.py`, `src/saelens_vg.py`를
출발점으로 쓴다. Bernoulli support, 입력별 point amplitude, analytic expected
quadratic risk, normalized prior/entropy, learned/profiled beta가 현재 방법이다.
고정 gamma는 여전히 sparsity를 조절하므로 VG가 true L0를 자동 발견한다고 주장할 수 없다.
필요한 개선은 파일럿으로 판단하며 새 module을 금지하거나 장식적인 component를 강제하지 않는다.

## Existing Evidence and Required Change

- 09-21의 P1은 새 cal/test의 유용한 recovery 신호와 fidelity 절충을 보였지만,
  한 training world와 과거 불균등 HPO이며 support correlation을 직접 다루지 않았다.
- P2에서 학습 연장 중 EV 개선과 recovery 악화가 함께 나왔다. Loss/MSE만으로
  checkpoint를 고르지 않는다. Oracle recovery 선택과 실제 사용 가능한 선택을 구분한다.
- P3/P3b gate correction은 일부 품질 개선을 보였으나 사전 latency 기준 미달이었다.
  기본 solver/teacher로 채택한 사실은 없다.
- Generic Stage1 dictionary coherence는 support correlation이 아니다. 참조 논문의
  correlated/anticorrelated firing을 재현하는 데이터와 signed mixing 평가가 필요하다.
- Mean, Bernoulli-sampled expected, hard readout risk와 actual L0를 구분한다.
  Point amplitude의 prior/entropy가 없어 full generative ELBO나 semantic posterior를 주장하지 않는다.

## Constraints and Workflow

- GPT-6 Astra ultra를 idea generation, novelty, critique와 refinement 리뷰에 사용한다.
  사용자 effort 지정이 일반 skill의 낮은 effort/다른 model 기본값보다 우선한다.
- AUTO_PROCEED=true. 세 pilot 경로 이내, 각 pilot GPU당 예상 2시간 이내,
  총 8 GPUh 이내. 실제 GPU 사용은 실행 직전 재확인한다. 최초 확인 GPU0는 busy,
  GPU1/2/3은 idle이므로 GPU0 작업을 건드리지 않는다.
- 새 cal/test split, 재현 가능한 config/seed, 결과를 보기 전 판정 규칙을 기록한다.
- 기존 API/config 의미는 유지하고 변형은 별도 runner에서 먼저 검증한다.
- 파일 생성·수정·삭제는 프로젝트 안에서만 한다. 기존 변경·논문·결과를 보존한다.
- 외부 메시지, 유료 GPU 구매는 포함하지 않는다. 장기 확증 실험 budget/venue는 미지정이다.
- 이전 09-21 계획과 결과는 보존한다. 이전 review 점수나 receipt를 이번 anchor의
  승인으로 재사용하지 않는다.

## Requested Outputs

Reference paper summary → 연결 문헌 → 같은 VG-SAE 목표 안의 8–12개 개발 경로 →
2–3개 판별력 있는 pilot → 새 novelty/비판 리뷰 → focused method proposal와 실험 계획.
주 목적은 ground-truth feature 복원과 잘못된 sparsity 설정에 대한 견고성이다.
Matched-L0/MSE는 보조 비교이며 좋은 점수 자체를 성공으로 삼지 않는다.

Canonical outputs: `idea-stage/REF_PAPER_SUMMARY.md`, `idea-stage/IDEA_REPORT.md`,
`refine-logs/FINAL_PROPOSAL.md`, `refine-logs/EXPERIMENT_PLAN.md`,
`refine-logs/EXPERIMENT_TRACKER.md`, `idea-stage/docs/research_contract.md`.
HTML과 대화 안에서 읽는 report도 제공한다. 이번 본 연구를 별도의 진단 논문으로 바꾸지 않는다.
