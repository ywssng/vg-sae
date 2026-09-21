# VG-SAE 프로젝트 메모리

사용자와 확정한 결정 및 다음 작업에 필요한 맥락을 보관한다. 행동 지침은
[AGENTS.md](../AGENTS.md)에 두고, 여기에는 사실과 결정의 근거를 간결하게
남긴다. 현재 코드와 사용자 정정으로 확인하면서 갱신한다.

## 확정된 결정 — 2026-09-07

- 사용자는 이 프로젝트에서 작업하는 Codex의 모델로 GPT-6 Astra를 선택했다.
  [`.codex/config.toml`](../.codex/config.toml)에 `model = "gpt-6-astra"`를
  기록했으며, reasoning effort는 프로젝트에서 지정하지 않는다.
- 모델 선택의 대상은 프로젝트의 Codex이다. SAE 실험은 로컬 모델의 내부
  activation을 사용한다. 관련 구현은
  [`src/gpt2_activations.py`](../src/gpt2_activations.py)와
  [`src/real_activations.py`](../src/real_activations.py)에서 확인한다.
- 사용자는 Astra의 프롬프팅 권장사항을 프로젝트 지침과 메모리에 반영해 달라고
  요청했다. 승인된 작업의 자율적 완료, 필요한 경우에만 질문, 명확한 지침
  우선순위, 간결한 설명, 도움이 되는 병렬 위임, 변경 영향에 맞는 검증이
  확정된 협업 선호다. 구체적인 적용 규칙은 `AGENTS.md`가 기준이다.

## 연구 맥락의 확인 위치

- 실행 환경과 실험 흐름: [`README.md`](../README.md),
  [`pyproject.toml`](../pyproject.toml), [`configs/`](../configs/).
- 수식·정규화·gradient·재현성 선택:
  [`REPRODUCTION_NOTES.md`](../REPRODUCTION_NOTES.md), [`tests/`](../tests/).
- Stage 3 모델과 activation 실험:
  [`docs/stage3_real_activations.md`](../docs/stage3_real_activations.md).
- 실험 결과를 해석할 때는 해당 run에 저장된 seed, config, checkpoint 및
  평가 방식을 함께 확인한다. 메모리에 숫자를 옮길 때도 결과 경로와 조건을
  근거로 남긴다.

## VG-SAE idea-discovery — 2026-09-19 (문제 설정은 09-21 정정으로 대체)

- 사용자는 현재 프로젝트의 구현을 출발점으로 `idea-discovery`를 실행하고,
  이 작업의 아이디어 생성·신규성·비판·방법 리뷰를 GPT-6 Astra의 `ultra`
  effort로 수행하도록 지정했다. 프로젝트의 전역 모델 설정을 바꾼 요청은 아니다.
- 통합 보고서는 [`idea-stage/IDEA_REPORT.md`](../idea-stage/IDEA_REPORT.md),
  후속 방법과 실험 계획은 [`refine-logs/`](../refine-logs/)에서 확인한다.
  28개 문헌과 11개 후보를 검토했으며, 리뷰는 same-family provisional이다.
- 당시 에이전트의 선택은 learned VG checkpoint에서 복제의 loss 선호와 feature 품질의
  관계를 검사하는 조건부 진단 연구다. 정확한 support posterior를 이용한
  gate 대상 검증을 대안으로 남겼다. 일반 dropout 복제 원리와 폭별 prior는
  기존 연구이므로 새 정리나 해결책으로 주장하지 않는다.
- 파일럿 코드와 사전 계획은 `scripts/idea_discovery_*`, `idea-stage/pilots/`,
  작은 원본 결과는 `idea-stage/evidence/pilots/`에 보존한다. 기존 Stage-2
  결과는 한 seed와 calibration stream 재사용 조건을 함께 읽어야 한다.
- `expected_ev`라는 파일럿 필드는 posterior-mean reconstruction EV다.
  stochastic reconstruction risk에는 Bernoulli variance가 추가되며 hard
  inference와도 구분한다. 큰 expected L0나 mean-hard 차이만으로 실제 복제,
  calibration 실패, objective 실패를 확정하지 않는다.
- 후속 clone intervention과 duplicate-invariant 평가 구현·확증 실험은 아직
  계획 단계다. 파일럿의 성공과 후속 C1/C2의 검증 완료를 혼동하지 않는다.

## 사용자 정정과 원래 연구 목표 — 2026-09-21

- **Variational Garrote 기반 새로운 SAE를 개발하는 것 자체가 사용자의 원래
  연구 아이디어**다. 이 저장소는 그 연구를 지금까지 구현하고 실험한 작업물이다.
- 요청의 목적은 VG-SAE에서 별개의 새 연구 주제를 찾는 것이 아니라, 진행 중인
  VG-SAE 방법 개발 및 연구 파이프라인을 다듬는 것이다. 09-19의 복제 진단 주제
  선택은 에이전트의 범위 오해였으며 현재 연구 목표로 사용하지 않는다.
- 새 실행의 고정 목표·제약은 [`RESEARCH_BRIEF.md`](../RESEARCH_BRIEF.md)에 있다.
  후보 생성은 같은 방법의 개선·검증 경로 안에서 수행한다. 진단 실험은 방법
  개발을 돕는 도구이며 주 연구를 대체하지 않는다.
- 기존 수치·문헌·코드는 조건을 재확인해 재사용한다. 기존 판단이나 리뷰 점수를
  변경된 목표의 검증으로 가져오지 않는다. `ultra` reviewer 지정은 유지한다.

## 원래 연구 목표로 재실행한 결과 — 2026-09-21

- 현재 실행 ID는 `vg-sae-development-20260921`이다. 이전 목표의 문서 사본은
  `idea-stage/runs/vg-sae-development-20260921/prior_scope_snapshot/`에 보존했다.
  최신 proposal/plan/contract는 VG-SAE 자체의 방법 개발을 기준으로 한다.
- 새 P1은 기존 546 checkpoints를 독립 calibration에서 선택한 뒤 fresh test로
  평가했다. Exponential의 coefficient/F1 신호와 input fidelity 손실이 함께 있었다.
  같은 L0 상한 아래 한 training world의 결과이며 exact-L0 또는 multi-seed 우위가 아니다.
- P2의 36개 작은 학습에서 moment beta 초기화의 일관된 이득은 없었다.
  일부 조건에서는 학습 연장으로 EV가 개선돼도 latent recovery/F1는 악화했다.
  다음 본 비교는 모든 방법에 같은 checkpoint 후보 기회를 주고 calibration recovery와
  L0로 선택한다. 기존 core beta 기본값과 기본 추론은 변경하지 않았다.
- P3 순차 보정과 P3b 탐색 병렬 후속은 일부 품질 개선을 보였으나 사전 latency 기준을
  넘었다. Generic amplitude control 뒤의 추가 이득과 비용을 더 검증해야 하므로
  solver/teacher를 기본 방법에 채택하지 않았다.
- Objective는 input-dependent point amplitude를 조건으로 한 support variational
  risk다. Full generative ELBO나 calibrated semantic posterior라고 부르지 않는다.
  Global learned beta와 minibatch profiling의 stochastic objective를 구별한다.
- 다음 B1/B2/B3는 아직 계획이다. 독립 paired training worlds와 동일 HPO/checkpoint
  기회, profiled-beta 조건에서 variance/entropy 각각의 objective coupling 검증,
  기존 Stage2/3 saved checkpoint의 fresh 평가를
  순서대로 진행한다. 구체적인 grids·실행 상한·미구현 prerequisites는
  `refine-logs/EXPERIMENT_PLAN.md`와 `EXPERIMENT_TRACKER.md`가 기준이다.
- 이번 새 runner 네 개의 관련 수치/선택 테스트 14개가 통과했다. 전체 repository
  test suite를 이 실행에서 다시 돌렸다고 기록하지 않는다. 작은 evidence는
  `idea-stage/runs/vg-sae-development-20260921/evidence/`에 있다.

## 갱신 원칙

- 다음 작업에 도움이 되는 확정된 결정·사용자 정정·재현성 맥락을 갱신한다.
  일시적인 진행 로그는 필요한 결정, 검증 결과, 남은 작업으로 압축한다.
- 변경 가능한 환경 상태, 브랜치 상태, 모델 목록은 필요할 때 다시 확인한다.
  개인 설정값이나 미완료 작업을 영구적인 프로젝트 정책으로 기록하지 않는다.
- 이 파일은 저장소에서 관리하는 프로젝트 메모리다. `AGENTS.md`의 시작 지침을
  통해 읽으며, Codex의 자동 생성 메모리 기능에 필수 규칙의 보존을 의존하지
  않는다.
