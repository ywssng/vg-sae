# VG-SAE 프로젝트 메모리

사용자와 확정한 결정 및 다음 작업에 필요한 맥락을 보관한다. 행동 지침은
[AGENTS.md](../AGENTS.md)에 두고, 여기에는 사실과 결정의 근거를 간결하게
남긴다. 현재 코드와 사용자 정정으로 확인하면서 갱신한다.

## 사용자 범위 정정 — 2026-09-23

- 이번 요청은 **Variational Garrote 기반 SAE 개발 자체의 연구 주제 적합성과
  계속할 가치**를 판단하는 것이다. 기존 코드 분석을 출발점으로 추가 구현이나
  실험을 수행하는 요청이 아니다. 09-22 brief의 파일럿·개발 단계는 이번 요청에
  적용하지 않는다.
- 문헌 검토는 사용자가 제공한 Sparse but Wrong와 Soh et al.의 Variational
  Garrote 두 논문에서 출발한다. 주제를 다른 진단 연구로 바꾸지 않는다.
- 이번 문헌 평가와 별도 에이전트 비판은
  `idea-stage/runs/vg-sae-topic-suitability-20260923/IDEA_REPORT.md`에 기록한다.
  에이전트의 판단은 연구 지속 가치 YES이며, 현재 구현의 우위나 최종 기여의
  신규성이 확인됐다는 뜻이 아니다. 사용자에게 확정받은 과학적 결론으로
  취급하거나, 이 판단만으로 과거 실험계획을 자동 실행하지 않는다.

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

## 새 핵심 연구 동기 — 2026-09-22

- 사용자는 ICML 2026의 Sparse but Wrong (Chanin & Garriga-Alonso)을 기준으로
  idea-discovery를 다시 실행하도록 요청했다. L1 sparsification 외의 방법론으로
  VG-SAE를 개발하는 것을 주목적으로 삼는다.
- 논문의 직접 L0/feature-mixing 결론과 사용자의 VG 대안 가설을 구분한다.
  주 연구는 여전히 VG-SAE 개발이며 별도 진단 주제로 변경하지 않는다.
- 현재 run ID는 `vg-sae-sparse-but-wrong-20260922`다. 이전 09-21 계획은
  해당 run의 `prior_development_snapshot/`에 보존했다. 기존 결과·review를
  새 문제의 확증으로 재사용하지 않는다.
- 사용자는 보고서를 별도 브라우저 없이 대화 안에서 읽는 방식을 선호한다.

## Sparse but Wrong 기반 재실행 결과 — 2026-09-22

- 참조 논문 v4/ICML2026의 직접 문제는 wrong L0와 feature identity이며, L1 대체는
  사용자의 VG 개발 가설이다. 논문이 L1 전반을 기각하거나 VG를 제안한 것으로 쓰지 않는다.
- 현재 보고서와 contract는 세 새 pilot을 통합했다: P1 225 fits, P2 72, P3 54.
  합계351은 독립 seed 수가 아니다. 각 pilot은3 paired worlds의 짧은 toy 검증이다.
- P1은 target2의7/9 covered cases에서 거의 완벽한 VG 복원을 보였고, simple L1
  rescale 후에도4/5 matched pairs의 coefficient NMSE가 낮았다. Strong-baseline
  joint 우위는 미확인이고 target1.8 및 Jump의 coverage가 부족했다.
- P2는 finite-budget scalar-prior recipe 비지지다. 대부분 gamma gradient도 아직
  크므로 잘못된 EB 정상점으로 수렴했다고 표현하지 않는다. P3의 한 total-effect
  rescue는 보존하되 density/readout/부호 효과와 고유 covariance 효과를 구분한다.
- Frozen mixing_energy는 signed-assignment leakage로 pure sign/mapping 오류도
  포함한다. 실제 다중성분 혼합은 absolute cosine/전체 projection과 함께 해석한다.
- Dense-offset 구성은 true decoder와 dense code로 profiled loss를 낮출 수 있는
  parameter family다. Profiled branch만 epsilon floor를 사용한다. Global learned
  branch에는 같은 energy clamp가 없다. 실제 SGD 원인이나 recovery 보장은 미입증이다.
- 다음 개발은 같은 VG의 precision 정책 대조와 baseline/selector calibration이다.
  Gamma 학습이나 exact32를 새 기본 구성으로 채택하지 않았다. 후속248-fit core
  roadmap은 미실행이며 큰 source 확장은 조건부다.
- Scientific continuation은 PROCEED WITH CAUTION, proposal은REVISE/7.85,
  same-family provisional이다. 점수의 변화만으로 연구 목표를 바꾸지 않는다.
  최종 실행 규칙과 남은 작업은 최신 refine-logs/EXPERIMENT_PLAN.md 및 tracker를 읽는다.

## 갱신 원칙

- 다음 작업에 도움이 되는 확정된 결정·사용자 정정·재현성 맥락을 갱신한다.
  일시적인 진행 로그는 필요한 결정, 검증 결과, 남은 작업으로 압축한다.
- 변경 가능한 환경 상태, 브랜치 상태, 모델 목록은 필요할 때 다시 확인한다.
  개인 설정값이나 미완료 작업을 영구적인 프로젝트 정책으로 기록하지 않는다.
- 이 파일은 저장소에서 관리하는 프로젝트 메모리다. `AGENTS.md`의 시작 지침을
  통해 읽으며, Codex의 자동 생성 메모리 기능에 필수 규칙의 보존을 의존하지
  않는다.
