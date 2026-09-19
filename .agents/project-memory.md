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

## VG-SAE idea-discovery — 2026-09-19

- 사용자는 현재 프로젝트의 구현을 출발점으로 `idea-discovery`를 실행하고,
  이 작업의 아이디어 생성·신규성·비판·방법 리뷰를 GPT-6 Astra의 `ultra`
  effort로 수행하도록 지정했다. 프로젝트의 전역 모델 설정을 바꾼 요청은 아니다.
- 통합 보고서는 [`idea-stage/IDEA_REPORT.md`](../idea-stage/IDEA_REPORT.md),
  후속 방법과 실험 계획은 [`refine-logs/`](../refine-logs/)에서 확인한다.
  28개 문헌과 11개 후보를 검토했으며, 리뷰는 same-family provisional이다.
- 현재 선택은 learned VG checkpoint에서 복제의 loss 선호와 feature 품질의
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

## 갱신 원칙

- 다음 작업에 도움이 되는 확정된 결정·사용자 정정·재현성 맥락을 갱신한다.
  일시적인 진행 로그는 필요한 결정, 검증 결과, 남은 작업으로 압축한다.
- 변경 가능한 환경 상태, 브랜치 상태, 모델 목록은 필요할 때 다시 확인한다.
  개인 설정값이나 미완료 작업을 영구적인 프로젝트 정책으로 기록하지 않는다.
- 이 파일은 저장소에서 관리하는 프로젝트 메모리다. `AGENTS.md`의 시작 지침을
  통해 읽으며, Codex의 자동 생성 메모리 기능에 필수 규칙의 보존을 의존하지
  않는다.
