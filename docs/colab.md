# Colab CLI에서 작은 실험 실행하기

공식 `google-colab-cli==0.7.4`를 프로젝트 안에 설치한다. 이 서버에서는
CLI와 파일 전송만 실행하며, 아래 smoke 실험의 학습은 Colab에서 실행한다.
`scripts/colab`은 인증 정보·세션·로그·캐시를 Git에서 제외한 `.colab/`에 둔다.
HOME이나 시스템 Python은 변경하지 않는다. 처음 로그인할 때 Google이 보여주는
권한을 확인하고 **Colab Pro 구독 계정**을 선택한다. 인증 코드는 터미널에만 입력한다.

## 설치와 로그인

프로젝트 루트에서 실행한다. `uv`가 필요하다.

```bash
bash scripts/colab setup
bash scripts/colab login
bash scripts/colab usage
```

`login`은 공식 CLI의 OAuth 인증을 거쳐 사용량을 조회한다. 브라우저에서 최초
동의는 필요하지만 실험 중 Colab 탭을 계속 열어 둘 필요는 없다.
이 어댑터는 upstream의 하드코딩된 저장 경로를 import 전에 한정적으로 바꾸므로
CLI 버전을 고정한다. CLI를 직접 실행하면 이 격리가 적용되지 않는다.

## 런타임과 영구 저장소

```bash
# 첫 연결 검증은 CPU로 충분하다.
bash scripts/colab new -s vg-sae
# GPU가 필요한 다음 실험은 위 명령 대신 다음을 사용한다.
# bash scripts/colab new -s vg-sae --gpu T4
bash scripts/colab drivemount -s vg-sae
```

Drive 연결에서 추가 동의가 나오면 표시된 절차를 따른다. 이 단계는 실제 로그인
후 확인해야 한다. Drive 마운트를 확인하지 못하면 실행기는 학습을 시작하지 않는다.
자원 할당은 가용량·구독·컴퓨팅 유닛에 따르며 CLI가 이를 보장하지 않는다.

## 작은 실험 실행·확인

```bash
python scripts/colab_experiment.py run --session vg-sae --run-id colab-smoke-v1
python scripts/colab_experiment.py status --session vg-sae --run-id colab-smoke-v1
python scripts/colab_experiment.py fetch --session vg-sae --run-id colab-smoke-v1
```

예제는 VG sparse regression을 seed 0·1에서 각각 100 step 실행하는 인프라
검증이다. 논문 결과나 VG-SAE baseline 비교가 아니다. Colab에 GPU가 있으면 사용한다.
Colab 기본 Python과 torch/numpy/PyYAML을 이용하는 제한된 실행 경로이며, 프로젝트의
전체 Python 3.14/SAELens 환경과 동일하다고 가정하지 않는다. 전체 SAE 실험은
별도 의존성 준비·검증이 필요하다. 환경 버전은 `environment.json`에 기록한다.

결과는 Drive의 `MyDrive/vg-sae-colab/<run-id>/`에 저장한다.

- `state.json`: 작업별 실행·완료·실패와 plan/source SHA-256.
- `tasks/<id>/stdout.log`, `tasks/<id>/output/`: 작업 로그·설정·결과·모델.
- 완료 후 `source.zip`, `plan.json`, `environment.json`, `results.zip`.

`fetch`는 완료된 결과 묶음을 `.colab/runs/<run-id>/results.zip`에 받는다.
업로드는 `src/*.py`, 기본 config, worker와 smoke 스크립트만 포함한다.
현재 체크아웃의 미커밋 코드도 포함되며 실제 파일 내용의 해시로 식별한다.
소스나 계획이 달라지면 새 run-id를 써야 한다. 기존 실행 파일은 보존한다.

CLI 기본 30초 제한 대신 기본 6시간(`--timeout 21600`)을 사용한다. 이는 CLI가
결과를 기다리는 시간이며 런타임 수명이나 학습 시간 제한을 늘리는 옵션이 아니다.
CLI가 성공 코드를 반환해도 완료 상태와 plan/source 해시를 추가 확인한다.

## 연결이 끊겼을 때

**바로 재실행하지 않는다.** 통신이 끊겨도 원격 계산은 진행 중일 수 있다.
`status` 명령은 파일 API로 상태를 읽으므로 바쁜 커널에 실행을 추가하지 않는다.
로그도 공식 `download`로 가져올 수 있다.

```bash
bash scripts/colab sessions
python scripts/colab_experiment.py status --run-id colab-smoke-v1
bash scripts/colab download -s vg-sae \
  /content/drive/MyDrive/vg-sae-colab/colab-smoke-v1/tasks/seed-0/stdout.log \
  .colab/seed-0.log
```

상태의 시간은 작업 경계에서만 바뀐다. 갱신이 없다는 이유로 런타임 종료를
판단하지 않는다. 공식 `colab status`의 BUSY/IDLE도 로컬 기록이라 확정 근거가 아니다.

기존 런타임이 종료된 것을 확인하면 새 런타임을 만들고 Drive를 다시 연결한 뒤,
**같은 체크아웃·계획·run-id**로 `run` 명령을 실행한다. 완료한 작업은 건너뛴다.
중간에 끊긴 작업은 처음부터 다시 실행한다. 현재 예제는 optimizer/RNG 상태에서
step 단위로 이어가는 기능이 없다. 긴 학습에는 해당 학습 코드의 checkpoint/resume
기능을 따로 연결하고 결과를 `COLAB_TASK_OUTPUT` 경로에 저장해야 한다.

다른 런타임 두 개에서 같은 run-id를 동시에 실행하지 않는다. 중복 실행 잠금은
같은 런타임의 같은 작업 디렉터리에만 적용한다. 자동 런타임 재할당·무한 재시도는
구현하지 않았으며, 자원 제한이나 불확실한 실행 상태에서는 멈추고 확인한다.

Drive는 영구 저장 위치이지만 FUSE flush가 Google 서버까지 즉시 동기화됐다는
보장은 없다. 중요한 완료 결과는 `fetch`로 한 번 더 보관한다.
작업을 마치고 결과를 확보한 뒤 자원을 해제한다.

```bash
bash scripts/colab stop -s vg-sae
```

## 세션 유지의 실제 범위

실제 커널 작업이 실행되는 동안의 세션 관리는 Colab에 맡긴다. 더미 계산,
브라우저 클릭, keep-alive 루프로 유휴 회수를 우회하지 않는다. Pro의 런타임
상한·유휴 제한·가용 자원·컴퓨팅 유닛 제한은 남는다. 끊김 없는 실행을 보장하지 않는다.

공식 근거(2026-10-02 확인):
[CLI](https://github.com/googlecolab/google-colab-cli),
[세션 관리](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md),
[Colab FAQ](https://research.google.com/colaboratory/faq.html).
