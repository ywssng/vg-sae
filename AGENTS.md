# 프로젝트 작업 지침

이 문서는 `vg-sae` 저장소 루트와 그 하위 디렉터리에서 작업하는 Codex 및
기타 자동화 에이전트를 위한 기본 지침이다.

## 작업 시작과 프로젝트 메모리

- 작업 시작 시 [프로젝트 메모리](.agents/project-memory.md)를 읽고, 대상 경로에
  적용되는 하위 `AGENTS.md` 또는 `AGENTS.override.md`가 있는지 확인한다.
- 행동 규칙은 이 문서에, 사용자와 확정한 결정·선호·재현성 맥락은 프로젝트
  메모리에 기록한다. 같은 규칙을 여러 파일에 복사해 중복 관리하지 않는다.
- 메모리의 사실은 현재 코드, config, 사용자 정정과 대조한다. 오래된 기록은
  수정하고, 추정이나 검증하지 않은 실험 결과를 확정된 사실로 저장하지 않는다.
- 장기 작업에서는 확정한 결정, 관련 파일, 검증 결과, 남은 작업을 저장소 안에
  간결하게 기록해 다음 작업에서도 이어갈 수 있게 한다. 비밀정보와 개인 경로는
  기록하지 않는다.

## GPT-6 Astra 협업 지침

프로젝트 Codex 기본 모델은 [`.codex/config.toml`](.codex/config.toml)의
`gpt-6-astra`이다. 아래 지침은 사용자가 요청한
[OpenAI의 Astra 프롬프팅 권장사항](https://developers.openai.com/api/docs/guides/latest-model#prompting-best-practices)을
이 저장소의 연구·개발 흐름에 적용한 것이다.

### 자율적인 실행과 완료

- 사용자 요청과 이전 대화에서 의도, 범위, 이미 받은 승인을 파악한다.
  “할 수 있어?”, “하고 싶어”, “도와줘” 같은 표현도 작업 요청으로 받아들이고
  구현, 필요한 검증, 결과 전달까지 수행한다. 가능하다는 답변이나 계획만으로
  멈추거나 시간·노력 절약을 위해 임의로 일부 작업만 완료하지 않는다.
- 통상적인 구현 선택은 문맥에 맞는 합리적인 가정으로 진행한다. 이미 승인된
  작업, 읽기 전용 확인, 되돌릴 수 있는 편집·수정·검토는 반복해서 묻지 않는다.
  필요한 경우 저장소 안의 격리된 체크아웃과 충돌 해결도 직접 수행한다.
- 답에 따라 연구의 의미, 작업 범위, 되돌리기 어려운 결과가 달라지고 문맥으로
  해결할 수 없을 때만 짧게 질문한다. 이미 허용된 독립 작업은 계속 진행한다.
- 최종 승인이 실제로 필요한 작업은 승인과 무관하게 진행할 수 있는 준비와
  검증을 먼저 끝내고, 검토 가능한 결과를 제시한 뒤 마지막 단계에서 묻는다.
  필요한 답변이나 승인은 시간 경과로 대신하지 않는다.
- 실제 근거 없는 경고, 면책 문구, 승인 단계, 안전 체크리스트를 덧붙이지 않는다.
- 진행 중 추가 요청은 기존 목표에 반영한다. 상태 질문이나 곁가지 질문에
  답한 뒤 진행하던 작업을 이어가며, 취소나 목표 변경은 사용자가 명확히
  요청한 경우에 적용한다.

### 지침의 우선순위와 충돌 처리

- 시스템·개발자 지침을 준수하며, 현재 사용자 요청과 대화에서 확정된 제약을
  작업 기준으로 삼는다. 사용자 지시와 skill의 가이드라인이 충돌하면 사용자
  지시를 우선한다. 오래된 메모리나 일반적인 문구로 현재 요청을 뒤집지 않는다.
- 관련 지침과 skill만 읽고, 불필요한 중단이나 반복 승인을 유발하는 모호한
  문구가 있는지 확인한다. 이미 주어진 승인과 해당 규칙의 적용 범위를 먼저
  판단하고, 예외가 있다는 이유만으로 사용자에게 다시 묻지 않는다.
- skill 때문에 질문·승인 요청·중단·미완료 또는 사용자 의도와 다른 행동이
  필요해지면 정확한 `SKILL.md` 이름과 링크, 해당 지시의 인용, 적용 이유를
  짧게 제시한다. 명시된 요구사항과 에이전트의 해석을 구분한다.
- 자동 승인 검토가 작업을 거부해 허용되는 방법으로 완료할 수 없으면,
  거부된 작업과 제시된 이유를 진행 보고와 최종 답변에 분명히 알린다.

### 설명과 문체

- 별도 요청이 없으면 한국어로, 핵심 결과나 다음 행동부터 설명한다. 익숙한
  단어와 능동형 문장으로 짧고 연결된 문단을 쓰고, 문단마다 한 가지 요점을
  다룬다. 기술 용어와 구현 세부사항은 이해와 검증에 필요한 만큼만 포함한다.
- 비교·순서·병렬 항목을 읽기 쉽게 만들 때 목록과 표를 사용한다.
  불필요한 제목, 중첩 목록, 반복 결론을 줄인다.
- 과장, 상투어, 막연한 수식어, 임의로 만든 합성 용어를 피한다.
  “Bottom Line”, “In short”, “delve”, “leverage” 같은 관용구와
  “X가 아니라 Y” 식의 불필요한 대비, 요청하지 않은 미변경 사항의 나열을
  덧붙이지 않고 수행할 행동을 직접 말한다.
- 진행 보고는 확인한 사실, 남은 불확실성, 다음 단계의 목적에 집중한다.
  최종 답변은 변경 내용과 이유, 수행한 검증과 실제 한계를 담아 독립적으로
  이해할 수 있게 쓴다. 실행하지 않은 검증을 통과했다고 표현하지 않는다.

### 하위 에이전트 위임

- 독립적으로 진행할 구체적인 작업을 나누면 시간이나 품질에 도움이 되는
  경우 협업 도구로 하위 에이전트에 위임한다. 조사와 구현, 변경 검토와 관련
  검증처럼 병렬 진행의 이점이 있는 작업에 적용하고, 단순한 수정에 불필요한
  에이전트를 늘리지 않는다. 하위 에이전트도 같은 기준을 적용한다.
- 위임할 때 목표, 읽거나 수정할 파일 범위, 기대 결과를 명시한다. 같은 파일의
  동시 수정을 피하고, 주 에이전트는 독립 작업을 진행하며 결과 통합과 필요한
  검증을 책임진다. 같은 조사를 중복시키거나 필요한 결과를 받기 전에 완료를
  선언하지 않는다.
- 에이전트 간 메시지도 사람이 읽을 수 있게 구체적으로 쓰고, 단어와 숫자의
  띄어쓰기를 지킨다. 도구가 없거나 이점이 없으면 직접 작업을 이어간다.

## 프로젝트 개요

이 저장소는 Variational Garrote sparse regression과 VG-SAE(Variational
Garrote Sparse Autoencoder)를 연구·구현한다. 주요 관심사는 다음과 같다.

- Variational Garrote 자유에너지 목적함수와 sparsity/selection 지표
- synthetic spike-and-slab regression 실험
- VG-SAE, L1/ReLU SAE, Top-K SAE, gated SAE 비교
- synthetic sparse-coding 실험과 GPT-2 residual-stream 실험
- 논문 수식 검증 및 재현성 기록

## 저장소 구조

- `src/`: 모델, loss, 데이터 생성, 학습, 평가 구현
  - `model.py`, `loss.py`, `train.py`: 기본 Variational Garrote
  - `sae_model.py`, `sae_loss.py`, `sae_train.py`: VG-SAE와 baseline SAE
  - `data.py`, `sae_data.py`, `evaluate.py`, `sae_evaluate.py`: 데이터·평가
  - `gpt2_activations.py`: GPT-2 activation 수집 및 캐시
- `scripts/`: sweep, notebook 생성, activation 캐시 실행 스크립트
- `configs/`: YAML 기반 실험 설정
- `tests/`: 논문 수식·모델 동작·작은 smoke test
- `notebooks/`: 제안 실험별 분석 notebook
- `outputs/`: 추적할 가치가 있는 현재 실험 결과. 대형 캐시와 checkpoint는 제외
- `docs/`: 방법론 문서
- `refs/`: 참고 논문
- `.github/workflows/`: CI가 있는 경우 실제 workflow에서 필수 검사를 확인
- `README.md`: 사용자용 실행 안내
- `REPRODUCTION_NOTES.md`: 논문과 구현 사이의 선택사항 및 한계
- `.agents/project-memory.md`: 확정된 프로젝트 결정과 작업 맥락
- `.codex/config.toml`: 프로젝트 Codex 설정

## 실행 환경

- Python 버전은 `.python-version`과 `pyproject.toml`을 따른다. 현재 기준은
  Python 3.14 이상이다.
- 명령은 항상 프로젝트 루트에서 실행한다.
- 일반 의존성·테스트 의존성 설치:

  ```bash
  python -m pip install -r requirements.txt
  ```

- `pyproject.toml`과 `uv.lock`을 사용하는 환경에서는 lockfile을 임의로
  삭제하거나 수동으로 깨뜨리지 않는다.

## 주요 명령

기본 학습:

```bash
python -m src.train --config configs/base.yaml
```

synthetic VG-SAE sweep 예시:

```bash
python -B scripts/run_synthetic_sweep.py \
  --output-dir outputs/synthetic_first_pass \
  --input-dim 8 \
  --widths 16 \
  --n-samples 256 \
  --support-density 0.125 \
  --lambdas 0.0,0.5,1.0,2.0 \
  --steps 80 \
  --include-no-variance \
  --include-baselines
```

전체 Python 검증이 필요한 경우:

```bash
python -m compileall -q src tests
python -m pytest tests -q
```

## 검증 범위

- 변경의 실제 영향에 맞는 검증과 프로젝트의 필수 검사를 수행한다. 문서·지침·
  모델 선택 설정처럼 되돌릴 수 있고 영향이 작은 변경은 내용, 링크, 형식,
  diff를 확인한다. 구현을 그대로 반복하는 테스트를 새로 만들지 않는다.
- 동작을 바꾸거나 오류를 고칠 때는 기존 관련 테스트를 우선 사용하고, 기존
  검증에 빈틈이 있으면 의미 있는 동작·회귀 테스트를 추가하거나 수정한다.
- 수식, 정규화, sparsity penalty, mask parameterization, gradient처럼 연구
  결과에 영향을 주는 변경은 작은 deterministic test를 먼저 추가하고,
  `REPRODUCTION_NOTES.md`와 관련 테스트에 반영한다.
- 필요한 검사가 통과하면 마무리한다. 새로운 변경, 실패, 남은 의문이 있을
  때만 검증을 확대하거나 반복한다. 이미 실행한 검사와 실행하지 못한 검사를
  구분하고, 환경 문제로 막히면 실패한 명령과 원인을 보고한다.

## 구현 및 실험 규칙

- 기존 public API와 config 필드의 의미를 유지한다. 변경이 필요하면
  `README.md`, 테스트, 관련 notebook 또는 reproduction notes를 함께 갱신한다.
- 실험은 재현 가능한 seed와 config를 사용한다. 생성 결과는 적절한
  `outputs/` 하위에 저장한다.
- `outputs/gpt2/`, `outputs/checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`,
  `*.safetensors` 같은 대형 산출물은 Git에 추가하지 않는다.
- 참고 논문, notebook, 현재 추적 중인 CSV/PNG 결과는 사용자가 명시적으로
  정리하라고 하지 않는 한 삭제하지 않는다.
- 선택적 의존성은 가능한 한 lazy import를 유지해 핵심 VG 코드의 import
  가능성을 보존한다.
- 비밀키, token, `.env` 파일, 개인별 절대경로를 소스·문서·커밋에 넣지 않는다.

## Git 버전관리 규칙

사용자가 요청한 기능·소스·프로젝트 설정·공유 지침·메모리 변경은 구현과
필요한 검증이 끝나면 자동으로 commit하고 push한다. 기존 승인 범위의 일반
commit과 push에 별도 확인을 요구하지 않는다. 사용자가 commit이나 push를
하지 말라고 한 경우 그 요청을 따른다. 기본 순서는 다음과 같다.

```bash
git status --short
git diff
git add -- <이번 작업의 파일 경로>
git diff --cached
git commit -m "변경 내용을 간결하게 작성"
git push
```

- 작업 전후의 상태와 staged diff를 확인하고 이번 작업의 변경만 커밋한다.
  기존 변경은 보존하며, 같은 파일에 섞인 경우 작업 부분만 선별해 stage한다.
  구분할 수 없는 변경이 충돌할 때에만 사용자에게 필요한 범위를 확인한다.
- commit message의 `변경 내용을 간결하게 작성`은 실제 변경 내용을 요약한
  짧은 문장으로 바꾼다.
- push 전에 가능한 검증 명령을 실행하고 결과를 사용자에게 보고한다.
- 현재 기본 브랜치는 `main`이며 원격은 `origin`이다.
- `git reset --hard`, 무관한 파일 삭제, force push, 공개 저장소 설정 변경은
  사용자가 명시적으로 요청하지 않는 한 실행하지 않는다.
- 루트 `AGENTS.md`, 프로젝트 메모리, `.codex/config.toml`은 함께 버전관리한다.

## 파일 범위 및 안전 규칙

- 파일 생성·수정·삭제는 이 저장소 루트와 그 하위 경로에서만 수행한다.
- 프로젝트 scope 밖의 파일이나 디렉터리는 절대 생성·삭제·수정하지 않는다.
  특히 부모 디렉터리, home 디렉터리, 다른 저장소, 시스템 경로를 건드리지
  않는다.
- 삭제나 덮어쓰기 전 대상 경로가 저장소 안에 있는지와 기존 변경사항을
  에이전트가 직접 확인한다.
- 작업과 무관한 기존 변경사항은 보존한다. 실제 충돌이나 사용자 결정이
  필요한 범위 문제에만 앞의 질문 기준을 적용한다.
