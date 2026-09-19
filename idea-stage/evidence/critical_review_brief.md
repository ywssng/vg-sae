# VG-SAE idea-discovery: 비판 리뷰용 근거

요청: 이 프로젝트의 VG-SAE 뼈대를 이용한 연구 idea-discovery. 사용자 지정
GPT-6 Astra `ultra`; 코드 변경·실험 산출물은 프로젝트 안에서만 관리.

## 현재 선택 방향과 고정 문제

**Problem Anchor:** VG-SAE가 학습한 확률적 표현의 좋은 기대 복원이 실제로
사용하는 희소한 경성 표현의 품질로 이어지지 않는 조건과 원인을 밝힌다.
기존 코드의 Bernoulli gate, deterministic amplitude, normalized decoder와
공정한 hard-L0 평가를 출발점으로 삼는다. 새 확률적 SAE의 최초성이나
일반적 SOTA, 확률 calibration을 미리 주장하지 않는다.

dominant contribution 후보: 알려진 dropout duplication 원리와 VG의
prior/KL 및 경성 추론을 연결하여, 실제 learned checkpoint에서 검증할 수 있는
원인분해 실험을 만든다. 복원값 차이가 관찰됐다는 사실과 원인 입증을 분리한다.
보조 후보 B05는 작은 known model의 exact posterior로 meanfield/encoder 오차를
분리한다. 별도 일반 uncertainty 방법 논문으로 합치지 않는다.

## 코드 및 기존 근거

`src/sae_model.py`: m=sigmoid(gate(x)), a=softplus(amplitude(x)), h=m*a.
E=.5||x-b-Dh||²+.5Σm(1-m)a²||D_j||².
F=beta E + normalized Bernoulli KL - Gaussian normalizer (learned beta),
혹은 minibatch profiled-beta objective. 배포는 1[m>.5]*a.
a=0이면 optimum m=pi=sigmoid(-gamma); expected L0만으로 정보 누출을
판정할 수 없다. decoder_bias와 input normalization도 구분해야 한다.

`idea-stage/evidence/code_audit.md`에 실제 Stage2 raw CSV의 경로/조건이 있다.
1seed, calibration-stream reused이므로 우열 확증에 쓰지 않는다.
gamma1.99 width4096의 hard/expected L0=30.2536/543.1684,
prior expected count492.5721, hard/expected EV=.77504/.86096.
근처 hardL0 JumpReLU EV=.80413. readout만 바꾸면 dictionary recovery는 바뀌지 않는다.

## 새 파일럿: 관측 범위

사전 계획은 `idea-stage/pilots/replication_protocol.md`,
`idea-stage/pilots/training_protocol.md`. 원본 결과는
`outputs/idea_discovery_20260919/replication/results.json`,
`outputs/idea_discovery_20260919/training/seed0/results.json`, `seed1/results.json`.

1. r replicated columns, m=pi, a=A/(r*pi), bias disabled, constant input에서는
   mean recon exact, KL0, hard0; variance=A²(1-pi)/(2r*pi).
   Actual implementation r4→256 variance .923632→.014432 at gamma2;
   fixed prior count1에서는 .375→.498047. 20 deterministic tests pass.
   이 구조는 Cavazza2018의 알려진 mechanism이며 actual training prevalence를
   증명하지 않는다. 자유 bias의 한 상수 입력은 trivial bias-only solution이 있다.
2. 기존 model/trainer 작은 synthetic d16, truth32, true mean count2,
   train8192/cal2048(미사용)/test2048, noise .05, exponential amplitude,
   learned beta, 1200steps, optimizer seeds0/1, 동일 data seed, widths32/64/128.
   fixedpi=2/32에서 EV gap은 width32 약 .025→width128 약 .051.
   그러나 variance_energy는 width 증가 때 **증가**한다. replication path를
   optimizer가 선택했다고 단정 불가.
3. pi=2/width control은 width128 hard EV .805/.814로 fixedpi .851/.852보다
   낮고 gap도 해결하지 않았다. stronger prior/lower actual L0 confound가 있다.
   matched per-sample count top-(ma), same-support ma는 EV를 .011–.014가량
   낮췄다. 단순 readout 개선 가설은 이 pilot에서 지지되지 않았다.

## 직접적인 선행연구와 인정하는 중복

- Cavazza et al. AISTATS2018, https://proceedings.mlr.press/v84/cavazza18a.html
  duplication variance dilution와 width-dependent retention 이미 존재.
- SoftSAE, https://arxiv.org/abs/2605.06610, tiny soft weights 정보 누출 및
  hard-topK stabilization 이미 존재(문헌 reviewer가 원문 확인).
- A Dominant Diffuse Phase in the Sparse Autoencoder Phase Diagram,
  https://arxiv.org/abs/2609.10299 : ReLU/L1 synthetic diffuse representations.
- Probabilistic TopK, https://openreview.net/pdf?id=zMIIHeKivz : stochastic gate
  +feature confidence, 최종 출판/저자 상태 미확인.
- Analysis of Variational Sparse Autoencoders, https://arxiv.org/abs/2509.22994 :
  naive Gaussian variational SAE의 부정 결과.
- SynthSAEBench, https://arxiv.org/abs/2602.14687 : recovery와 reconstruction
  분리, matching pursuit의 실패; 저장소 Stage2와 직접 연결.
- Sparse but Wrong, https://arxiv.org/abs/2508.16560 : L0 선택과 feature 오류.
- Compute Optimal Inference and Provable Amortisation Gap,
  https://arxiv.org/abs/2411.13117 : amortization refinement 일반 개념 중복.

문헌 shard의 정식 목록/판정은 진행 중이다. 검색하지 않은 논문이 없다고
주장하거나 최신 연구 전체의 부재를 증명할 수는 없다.

## 검토 요청

강한 연구자로서 위 숫자와 실제 코드/파일을 검증하고, candidate JSON 전체를
참고해 선택 가설에 가장 강한 반론을 제시하라. 어떤 부분이 알려졌고,
어떤 조건부 VG-specific 결과가 나와야 기여가 되는지 한 문장으로 구분하라.
선정 방향을 더 거대한 architecture로 대체하지 말고 다음 판별 실험을 정하라.
기본 연구 계속 여부를 PROCEED / PROCEED WITH CAUTION / REVISE / ABANDON 중
하나로 명시하고 이유, 연구가치 score, 구현 가능성, 현재 논문준비 수준,
allowed/unsupported claims matrix, 최대3 core blocks와 반증/중단 조건을 써라.
긍정 판정은 이 연구 계획을 계속할 가치이지 empirical acceptance가 아니다.
정확한 model/effort/agent identity 및 same-family provisional 성격을 명시하라.
