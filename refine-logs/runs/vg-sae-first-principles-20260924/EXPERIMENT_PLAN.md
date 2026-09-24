# VG-SAE first-principles 실험 계획

2026-09-24. 사용자 요청에 따라 새로 설계한 초기 핵심 캠페인. 기존 코드·계획은 검토·재사용 자료이며 이전 실험의 결론을 전제로 삼지 않는다. 설정 원본은 `configs/first_principles_20260924.json`이다.

## Problem Anchor

이산적인 feature 선택을 Variational Garrote의 변분 자유에너지로 정식화하여 SAE의 목적함수, 선택 불확실성, 근사추론과 공동 dictionary 학습의 역할을 이해한다. 수식의 수치 재현은 구현 검증이며 그 자체가 새 연구 기여는 아니다. SOTA, 정답 L0 자동 발견, 모든 feature 혼합 해결을 성공 조건으로 강제하지 않는다.

## Claim Map

| 주장 후보 | 최소 근거 | 반증·범위 축소 | 블록 |
|---|---|---|---|
| C1: 조건부 Bernoulli 선택모형의 에너지·선택 비용·entropy가 support 분포를 결정하며, dictionary overlap이 만드는 posterior 의존성은 factorized 근사의 한계를 설명한다 | exact golden control, field response, 동일 target의 exact/MF KL와 수렴 잔차, 직접 support covariance | 높은 overlap의 지정 조건에서 gap이 없으면 그 기전 관찰은 비지지. 미수렴이면 최적화와 근사오차를 분리했다고 주장하지 않음 | B1/B2 |
| C2: SAE encoder를 동일 조건부 모형의 amortized 추론으로 검사하면 encoder의 추가 오차와 dictionary/amplitude 학습의 영향을 구별할 수 있다 | frozen 비교에서 같은 x,a,D,β,γ 사용; encoder/refined gate/exact 차이; joint 학습에서 objective 각 항과 precision 정책의 효과 | frozen에서조차 posterior를 근사하지 못하면 gate 해석을 제한. joint의 결과를 생성 posterior나 semantic calibration으로 확대하지 않음 | B3/B4 |

Anti-claims: 알려진 free-energy identity가 새 정리라는 주장; VI를 썼으므로 semantic feature가 옳다는 주장; 낮은 MSE/높은 sparsity만으로 방법 우위를 선언; mf solver가 전역 최적이라는 주장. Full generative ELBO는 amplitude=1의 정확히 명시한 생성모형에 한정하고 input-dependent point amplitude의 VG-SAE에는 조건부 support risk라고 명시한다.

## B1 — 논문·공식 GitHub와 구현 대조 / MUST

범위: scalar VG, core VG-SAE, SAELens VG adapter, ReLU/L1·TopK·BatchTopK·JumpReLU·Gated baseline, 학습 wrapper와 추론 변환. 고정 SAELens commit `8be14080485952f729ed58d674bcddf9778e0aa4`와 설치본을 대조한다. 각 모델이 따르는 논문 변형과 upstream recipe 차이를 기록한다. 버그, 수치 정책, 의도적 SAE 확장을 구분한다.

Float64 tiny independent enumeration으로 기대 SSE·정규화된 prior·entropy·Gaussian normalization·gradient를 검사한다. Profiled β의 정상점, 위험 floor, batch duplication invariance, 저정밀 entropy, baseline의 학습/추론 동치도 해당 영향에 맞게 검사한다. 통상 영역 tolerance는 절대/상대 1e-8, 저정밀은 명시한 dtype tolerance다. Blocker가 남으면 관련 학습을 시작하지 않는다.

## B2 — 정확 support 분포와 물리적 응답 / MUST

- d=K=6, amplitude=1. 첫 두 dictionary atom의 cosine c=0/.7/.95; 다른 atom은 orthogonal. 이는 firing correlation을 바꾸는 실험과 다르다.
- 3 seeds, seed마다 독립 Bernoulli(.25) support와 Gaussian noise precision8으로 생성한 128개 입력을 모든 inference controls에 재사용한다.
- β=[2,8,32], γ=[−2,0,log3,2,4], p(s)∝exp(−γN). 총135 conditional cells, 각 cell에서64 support를 모두 열거한다.
- gamma log3, beta8만 data-generating model과 일치한다. 다른 control은 고정 입력에서 misspecified conditional response를 검사하며 calibration 성공을 요구하지 않는다.
- Exact negative log evidence, entropy, mean N, Var(N), pair covariance를 기록한다. `−d E[N]/dγ=Var(N)`을 δ=1e−4 중앙차분으로 검사한다. 알려진 항등식의 수치 검증이며 새 상전이 정리가 아니다.
- MF는 prior, .5, .01, .99, seeded random의5개 시작점에서 최대100 coordinate sweeps. sample별 가장 낮은 F의 best-found 해를 사용하고 잔차를 저장한다.
- Golden control: orthogonal marginal error<1e−8, F+logZ−KL consistency<1e−8, response residual<1e−5. Correlated 설명 신호는 mean reverse-KL≥.05 nats와 ≥95% samples residual≤1e−6을 함께 만족할 때 명료한 관찰로 표시한다. 더 작은 값·음성 결과도 모두 보존한다.
- 산출물: exact/MF response CSV, sample-level NPZ, covariance/KL/occupancy 그림. 유한 계의 response/crossover라고 부른다.

## B3 — 고정 모형에서 encoder를 연결 / MUST

3 seeds ×3 coherences=9 fits. B2의 생성모형(beta8, pi.25, amplitude1)과 동일한 dictionary를 고정한다. Decoder, amplitude encoder(상수1), beta는 frozen이며 gate encoder만 학습한다. train8192/cal2048/test2048은 별도 seed streams. Adam lr.003, batch256,2000 updates. 모든 hyperparameter와 최종 checkpoint를 사전에 고정한다.

Test에서 exact, best-found MF, encoder를 같은 target으로 평가한다. Brier/support NLL는 실제 생성 support에 대해 계산한다. Exact와의 차이는 numerical inference reference metric으로 따로 명시한다. MF 시작점에 encoder를 추가하되 raw F 차이를 clamp하지 않고 solver 잔차를 함께 기록한다.

Primary: F_encoder−F_MF, reverse-KL to exact, marginal error, fixed-point residual. Mean F drop≥.01 nats이고3 seeds에서 같은 방향이면 접근 가능한 gate refinement 이득으로 판단한다. 작은 gap도 실패로 강제하지 않는다. Finite training·모형 용량·최적화 차이가 섞인 amortization gap이며 전역 최적 encoder gap은 아니다.

## B4 — 공동 dictionary 학습과 목적함수 항 / MUST, 초기 evidence

- 하나의 overcomplete synthetic family: d8, true/learned K16, random unit dictionary, independent support pi=.125, exponential amplitude scale1/sqrt2(active second moment1), observation noise std.1.
- 3 independent training worlds. world별 같은 dictionary와 train8192/cal2048/test4096을 모든 방법에 재사용. Dictionary/init/train/cal/test/batch seeds를 역할별로 명시한다.
- VG global learned β, VG minibatch profiled β, profiled−entropy, profiled−variance, SAELens ReLU(RI-L1), SAELens norm-weighted TopK의6개 방법. 각3 controls: VG gamma0/2/4; L1 coefficient .001/.01/.1; TopK k1/2/4. 총54 fits. TopK는 pinned recipe(aux coefficient1, unconstrained decoder)이며 원 논문의 unit decoder/aux1/32 recipe와 구분한다.
- 2000 updates, batch256, lr.003, β initial100, no input rescaling, no weight decay, clip1. VG variants는 같은 초기 parameter state와 batch stream; baseline은 native 공식 구조/optimizer/hooks를 사용한다. 동일 parameter count나 원 논문의 전체 training recipe 재현을 주장하지 않는다.
- Checkpoints500/1000/2000의 calibration metrics를 모두 남긴다. Test는 최종2000 checkpoint의 전체 grid를 평가한다. Test-best 선택, test 기반 grid 확대, matched-L0 보간은 하지 않는다. Calibration도 이번 캠페인에서는 hyperparameter 선택에 쓰지 않는다.
- 지표: posterior-mean MSE, **full Bernoulli stochastic risk**(variance 삭제 모델도 동일식으로 평가), native hard-code MSE/L0, expected mask count, hard mask count, effective mean-code count(threshold1e−6), entropy, β, 각 objective 항. 생성 truth와의 signed Hungarian dictionary cosine, support F1, coefficient NMSE는 보조다. Nonorthogonal truth에 orthogonal-only mixing metric을 쓰지 않는다.
- 동일 final checkpoint의 calibration에서 full-data profile vs batch16/256 평균 profile을 비교하고 floor가 작동하지 않은 경우 Jensen gap≥−1e−6 및 β dispersion을 기록한다. Log-floor가 작동하면 부등식 보장 대상으로 세지 않는다. 이는 objective의 batch dependence이며 minibatch 학습 실패의 인과 증명은 아니다.
- 효과는 same-world/same-gamma paired differences로 보고한다. Entropy/variance 삭제가 개선·악화할 방향을 미리 가정하지 않는다. 삭제 목적은 다른 확률적/결정론적 objective이므로 동등 ELBO라고 하지 않는다. 3 seeds는 초기 방향성 evidence다.

## 실행 순서와 예산

| 단계 | 작업 | 종료 조건 | 예상 비용 |
|---|---|---|---|
| M0 | B1 audit, regression tests, CUDA witness, tiny script sanity | 유효성 검사·독립 code review 통과 | CPU/짧은 GPU, <.05 GPUh |
| M1 | B2 exact/MF controls | golden controls 통과; 모든135 cells 저장 | CPU, 수분 예상 |
| M2 | B3 frozen9 fits |9 fit 결과·control checks 저장 | 약.2–.4 GPUh |
| M3 | B4 joint54 fits |54 fit 결과와 paired 분석 | 약.5–1.2 GPUh |
| M4 | 결과 무결성 검토·그림·tracker | raw records에서 표/주장 추적 가능 | CPU |

최초 캠페인 상한은2 GPUh, 1 fit2000steps의 smoke timing으로 확인한다. Run time이 estimate의2배를 넘으면 해당 worker 상태·로그를 확인하며, 비용 상한 도달 시 미실행을 명시한다. 학습 성공 여부를 보고 budget를 몰래 늘리지 않는다. GPU 점유는 실행 직전 재확인하고 기존 다른 프로세스를 종료하지 않는다.

예산의10%는 smoke,30%는 frozen,60%는 joint에 배분한다. 따라서3 seeds의 frozen worker는 각각720초, joint worker는 각각1440초를 넘기지 않는다. Runner가 fit마다·각 training update마다 deadline을 검사하고 elapsed wall time과 failure도 기록한다. 외부 `timeout`도 동일 상한으로 적용하여 평가 중에도 종료 한계를 둔다. Smoke GPU commands는 각각240초 이하이며 총720초를 넘지 않게 운영한다.

Local `.venv` Python3.14 / Torch2.13+cu126 / SAELens6.47 환경을 재사용한다. Shell의 기본 Python3.11을 사용하지 않는다. Deterministic algorithms, TF32 off, CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP/OPENBLAS2. Cache는 outputs/.cache 아래다. 유료 GPU·외부 메시지는 포함하지 않는다.

배치 실행은 world별 local worker3개로 나누고 각fit의 manifest/status/result를 저장한다. `experiment-queue`의 공식 helper는 SSH+conda+screen 전용이며 local venv와 저장소 밖 쓰기 금지에 맞지 않으므로, 기존 local subprocess 실행 방식과 per-fit status를 사용한다. 실행 backend를 SSH로 바꾸거나 home 아래 queue를 만들지 않는다. 실패한 fit은 success marker를 쓰지 않고 worker를 종료해 로그부터 검토한다.

## 이번 범위 밖

LLM 대규모 benchmark, VAEase 신규 구현, 여러 새 amplitude 분포, posterior teacher/새 encoder, 거대한 HPO는 NICE-TO-HAVE 후속이다. 이번54fits는 튜닝을 마친 baseline 대비 우위나 실제 LLM 의미적 feature 검증이 아니다. 현대 frontier primitive는 핵심 가정이 아니므로 억지로 추가하지 않는다.

기존 이론의 수치 확인(B1/B2 golden controls), 이번 새로운 관찰(B2 dependence/B3 inference gap/B4 objective behavior), 아직 검증하지 않은 일반화를 구분하여 결과를 작성한다.
