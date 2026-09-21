# Sparse but Wrong: reference analysis for VG-SAE development

검토일: 2026-09-22. 기준은 사용자가 제공한 27쪽 PDF의 arXiv v4 (2026-07-07)다. 아래 쪽수는 PDF에 인쇄된 쪽수와 같으며, 원문 전체를 복제하지 않고 분석과 재현 메모만 기록한다. 이 문서는 논문과 공식 코드의 읽기 전용 조사 결과이며 새로운 실험 결과가 아니다.

## 1. 연구 동기를 정확히 연결하기

**고정 연구 목표는 Variational Garrote 기반의 새로운 SAE 개발이다.** 이 논문은 sparsity를 확보하고 MSE를 낮추는 것만으로 실제 feature를 잘 찾았다고 판단할 수 없다는 문제를 제공한다. 이를 계기로 L1 amplitude penalty 이외의 support 선택 원리를 연구하는 것은 타당한 우리의 연구 동기다. 그러나 논문이 VG를 제안하거나 VG의 우위를 입증한 것은 아니다.

현재 v4의 주 실험은 **BatchTopK와 JumpReLU**다. L1 SAE를 주 비교 대상으로 삼아 실패를 입증하지 않았다. L1 shrinkage, snapping 또는 rescaling 실험도 없다. 도입 각주 2 (p.1)는 L1과 JumpReLU가 coefficient를 통해 sparsity를 조정한다는 설명이다. §3.7 (p.6)의 sparsity coefficient에 따른 L0의 plateau는 **JumpReLU** 결과다. 이를 “L1 snapping”이라고 쓰면 잘못된 귀속이다. Appendix I (pp.23–24)는 c_dec를 사용한 meta-level L0 controller이며 L1 대체 방법을 권고하는 장이 아니다.

L1 shrinkage의 직접 근거는 별도 일차문헌인 [Gated SAE 논문](https://arxiv.org/abs/2404.16014v2)에 있다. 이 논문은 feature 존재 판단과 크기 추정을 분리해 shrinkage를 줄이는 기존 해법을 제시한다. 따라서 “support와 amplitude를 분리했다”만으로 VG-SAE의 신규성을 주장할 수 없다. VG의 변분 expected reconstruction risk, Bernoulli variance, entropy/prior 구조가 실제 feature 복원에 추가 이득을 주는지 검증해야 한다. **L1 대비 개선**, **Gated/JumpReLU 대비 추가 가치**, **잘못된 L0에 대한 강건성**은 서로 다른 주장이다.

## 2. 출판 및 버전 확인

- 저자: David Chanin, Adrià Garriga-Alonso.
- [arXiv 공식 메타데이터](https://arxiv.org/abs/2508.16560): v1 2025-08-22; v2 2025-09-26; v3 2025-12-05; v4 2026-07-07.
- [ICML 2026 공식 poster 페이지](https://icml.cc/virtual/2026/poster/63049)는 제목·저자·학회를 확인하고 [OpenReview aYqF2Cv6Cr](https://openreview.net/forum?id=aYqF2Cv6Cr)로 연결한다. 해당 ICML 페이지는 HTTP로 실제 읽었으며 web renderer는 restricted URL 오류였다. OpenReview 본문/API는 browser challenge/403으로 직접 읽지 못했다.
- PDF p.1에는 ICML 43회, Seoul, PMLR 306, 2026이라고 적혀 있다. `https://proceedings.mlr.press/v306/`은 이번 조회에서 HTTP 404였다. 따라서 **ICML 2026은 공식 학회 페이지로 확인**, **PMLR volume 306은 PDF 자체의 서지 표기**, proceedings 개별 페이지·page range는 미확인으로 구분한다.
- [공식 구현](https://github.com/chanind/sparse-but-wrong-paper)의 읽기 기준 commit은 `d5886b540dc5b9cac4f76e6db2b0cce1b0b7c585`다. GitHub API의 마지막 push 시각은 2025-11-16이었다. 이 구현이 v4의 모든 추가 실험을 그대로 재현한다고 가정하지 않는다.

버전 차이는 연구설계에 중요하다. [v1](https://arxiv.org/html/2508.16560v1)은 BatchTopK와 n-th decoder-input projection에 중심이 있었고, L1/JumpReLU에 대한 일반화는 예상으로 썼다. [v3](https://arxiv.org/html/2508.16560v3)는 c_dec를 주 지표로, n-th projection을 별도 Appendix A.9–A.10으로 다뤘다. v4는 c_dec를 주 지표로 유지하고 n-th projection 장을 제거했으며, decoder-input histogram은 §4.2에 남아 있다. v1/v3 공식 HTML의 본문에서도 shrinkage/snapping/rescaling 논의는 확인되지 않았다.

## 3. 논문이 실제로 보인 것

| 근거 위치 | 논문에서 보인 결과 | VG-SAE 계획에 적용할 범위 |
|---|---|---|
| §3.1, pp.3–4, Figs.2–3 | 5개의 orthogonal feature, true L0≈2에서 BatchTopK L0=1.8은 상관 부호에 따라 다른 feature 성분을 섞음. 낮은 L0 조건은 ground-truth 초기화에서도 dictionary가 이동함. | amplitude bias만 보지 말고 decoder 방향의 오염을 측정한다. VG가 이를 막는지는 새 가설이다. |
| §3.2, pp.3–4, Fig.1 | 50-feature toy의 낮은 L0와 높은 L0 모두에서 혼합 발생. 이 실험에서 낮은 L0는 많은 latent를 손상하고 높은 L0에서는 일부 정상 latent가 남음. | low/correct/high realized L0 모두 평가하며 “더 sparse면 더 좋다”를 채택하지 않는다. |
| §3.3–3.4, pp.4–5, Figs.4,6 | 같은 낮은 L0에서 학습한 혼합 dictionary가 true dictionary보다 reconstruction이 좋음. p.4의 2.73 대 4.88은 저자 보고 MSE 수치다. | MSE 우위만으로 semantic recovery 우위를 주장하지 않는다. 원문 수치는 재현 전 이 프로젝트의 결과로 사용하지 않는다. |
| §3.5, p.5, Fig.5 | 높은 L0에서 낮추는 schedule은 toy feature를 회복하지만, 너무 낮은 L0에서 시작한 schedule은 최종 L0를 바꿔도 완전히 회복하지 못함. | prior/sparsity warmup 후보의 동기다. VG만의 특성이나 일반적 불가역성 정리가 아니다. |
| §3.6–3.7, p.6, Figs.7–8 | c_dec가 toy의 적정 L0 부근에서 작아짐. JumpReLU의 L0는 넓은 coefficient 범위에서 적정 L0 근처에 머묾. | JumpReLU가 이미 강한 경쟁 baseline이다. adaptive support 자체가 VG의 독점적 장점은 아니다. |
| §4, pp.6–8, Figs.9–10 | Gemma-2-2b 및 Llama-3.2-1b의 일부 layer에서 c_dec elbow가 sparse probing 성능 peak와 관련됨. | 실제 LLM에서 true feature는 알려져 있지 않다. proxy/probing 상관을 semantic ground truth로 바꾸지 않는다. |
| §4.1–4.2, pp.7–8; Appendix L, pp.24–25 | 높은 L0에서는 JumpReLU가 BatchTopK보다 probing이 더 좋음. per-latent threshold 조정이 원인일 수 있다는 저자의 해석. | per-latent 또는 per-sample adaptive support의 동기지만 인과 검증 완료 사실은 아니다. |
| Appendix D, pp.14–15 | 작은 toy에서 mixing의 부호/크기는 feature coactivation 상관 및 L0 deficit과 연동됨. 독립 조건에서는 해당 mixing이 관찰되지 않음. | 독립·양의 상관·음의 상관을 분리하고, feature-vector cosine과 coactivation correlation을 혼동하지 않는다. |
| Appendix E, pp.15–17 | 50 features / 40 dimensions의 superposition과 width=25의 좁은 SAE에서도 관련 현상. 좁은 SAE의 적정 L0는 표현할 수 있는 feature subset에 따라 달라짐. | 모든 width에서 데이터 전체의 true mean L0를 정답으로 강제하지 않는다. |

§6 (p.9)와 Appendix A (p.13)의 제한: LRH/linear features에 한정하며, nonlinear concept geometry와 복잡하게 불균형한 상관 구조를 충분히 다루지 않았다. LLM layer/model 범위가 작다. LLM에서 per-sample TopK를 별도 실험하지 않았다. c_dec의 넓은 minimum과 특히 high-L0 구간의 불완전한 probing 상관 때문에 이 지표 하나로 precise optimum을 보장하지 않는다.

## 4. 정리의 정확한 범위

**Theorem F.1, pp.17–20, Eqs.(5)–(46).** 두 orthonormal feature, 두 unit decoder latent, tied encoder/decoder, bias 없음, Top-1 projection 및 양의 feature 크기에 대한 구성이다. 혼합하는 latent 한 개의 구체적인 경로로 MSE가 감소하는 예를 제시한다. 이는 L1 penalty, unconstrained learned encoder, BatchTopK의 batch selection, VG의 변분 목적함수 전반에 대한 정리가 아니다.

**Theorem F.2, pp.21–22, Eqs.(47)–(52).** g개의 orthonormal feature, g개의 tied latent, bias 없음, Top-K에서 선택된 i와 활성화됐지만 누락된 j가 동시에 존재할 확률이 양수라는 event A를 둔다. unit-norm perturbation `d_i(ε)=(f_i+ε f_j)/sqrt(1+ε²)`의 expected MSE 방향미분이 `-2 P(A) E[m_i m_j | A] < 0`가 된다. proof는 active magnitude가 atomless라 tie가 거의 없다는 가정을 사용한다. 이를 VG-SAE의 목적함수에 옮기려면 variance·entropy·prior·amplitude 변화까지 포함하는 별도 유도가 필요하다.

평균 true L0와 per-sample K는 다르다. 이 정리의 핵심은 **P(number of active features > K)>0**로 발생하는 누락 event다. 평균값이 K와 같거나 더 작아도 이런 event는 있을 수 있다. 논문의 평균-L0 toy 결과와 per-sample Top-K 정리를 동일한 명제로 취급하지 않는다.

인용 시 주의할 원문 내부 표현 두 가지도 있다. F.1의 `P12>0`만으로 평균 active count가 1보다 크다는 문장은 일반적으로 성립하지 않는다. `P0=1-P1-P2-P12`라면 필요한 조건은 `P12>P0`다. F.3의 anti-correlation에 따라 Eq.(52) 기댓값 부호가 뒤집힌다는 해석도 `m_i,m_j>0 on A` 조건에서는 나오지 않는다. 음의 상관 mixing은 §3/Appendix D의 **empirical result**로 인용하고 이 remark로 증명되었다고 쓰지 않는다. 이는 제한된 독해상 점검이며 전체 proof audit를 수행했다는 뜻은 아니다.

**Theorem G.1, pp.22–23, Eqs.(53)–(63).** orthonormal 고유 feature f_i와 공통 feature g에 대해 `d_i=sqrt(1-γ_i²)f_i+γ_i g`로 혼합한 특정 family에서 c_dec가 증가한다. 적어도 두 γ_i가 0이 아니어야 한다. 모든 잘못된 dictionary를 검출하거나 최소 c_dec가 true features라는 역명제를 증명하지 않는다. 예를 들어 다른 orthonormal rotation도 c_dec=0일 수 있다. 따라서 VG의 c_dec 개선을 단독 성공 기준으로 삼지 않는다.

## 5. 지표를 구분하는 평가 정의

1. **c_dec (paper metric)** — §3.6 Eq.(4), p.6; Appendix M/Fig.23, pp.25,27. Unit decoder atoms d_j에 대해 `mean_{i<j} |d_i^T d_j|`. 데이터가 필요 없고 대각선을 제외한다. 원 논문 버전은 모든 decoder pair를 포함한다. dead latent를 제외한 보조값을 추가할 수 있지만 원 정의와 다른 이름으로 보고한다. 폭·차원·실제 dictionary coherence가 달라지면 raw c_dec를 직접 순위화하지 않는다.
2. **Decoder-input projection histogram (paper exploratory diagnostic)** — §4.2/Fig.10, pp.7–8. decoder 방향으로 input을 투영한 값의 분포이며 c_dec와 다른 물건이다. 코드의 related n-th projection helper는 `P=(X-b_dec) D^T`, flatten 후 내림차순 정렬한 index `n*batch_size` 값을 반환한다. 큰 절댓값 tail이 mixing과 관련될 수 있으나 scale, mean, frequency도 영향을 준다. v3의 n-th quantile을 v4의 주 metric이라고 하지 않는다.
3. **Dictionary recovery (project operational metric)** — `C_{ji}=d_j^T f_i`를 signed cosine으로 계산하고 Hungarian one-to-one matching으로 평균 matched cosine을 최대화한다. nonnegative true coefficients에서 부호 반전을 무조건 동등 취급하지 않는다. `mean matched cosine`, `fraction matched cosine ≥0.95`, unmatched true features를 함께 기록한다. 논문 heatmap을 수치화하기 위한 우리 정의이며 원 논문의 별도 scalar claim이 아니다.
4. **Mixing / leakage (project operational metric)** — orthonormal toy에서 매칭 π 후 `mean_j sum_{i≠π(j)} (d_j^T f_i)^2`와 outside-true-span energy를 나눠 기록한다. small star toy는 Appendix D의 직접 metric `mean_{i=1..4} cos(d_{π^{-1}(i)}, f_0)`도 사용해 부호를 보존한다. 절댓값 또는 squared leakage만 보고 음·양 상관 구분을 잃지 않는다.
5. **True / realized / expected L0** — 데이터 true L0는 `mean_n sum_i 1[m_ni>0]`; 모델 hard L0는 실제 배포 추론에서 `mean_n sum_j 1[z_nj>0]`; VG expected support count는 `mean_n sum_j q_nj`. soft amplitude의 양수 개수, expected support, threshold-hard support를 같은 값으로 쓰지 않는다. 비교 budget은 calibration에서 정한 동일 hard inference 규칙의 realized L0다.
6. **Support recovery and amplitude** — 별도 calibration에서 정한 dictionary alignment를 고정해 test support micro/macro F1와 active true-feature별 recall을 평가한다. VG hard gate 및 L1의 support threshold는 사전에 명시한다. true-positive support에서 coefficient bias/MAE, 전체 coefficient NMSE를 함께 보고한다. zero-input sample의 norm ratio를 계산할 때는 제외 수를 명시해 0/0을 피한다.
7. **Reconstruction** — sample SSE `mean_n ||x_n-xhat_n||²`와 per-coordinate MSE `SSE/d`를 분리한다. centered EV는 `1-SSE/mean_n||x_n-mean(x)||²`; 공식 notebook의 함수 이름은 variance explained이나 실제 denominator는 `mean_n||x_n||²`이므로 uncentered explained energy다. 같은 이름으로 합치지 않는다.

c_dec를 학습 objective나 HPO selection criterion으로 사용한다면 **해당 c_dec 자체는 독립 성능 증거가 아니다**. Dictionary/support recovery와 held-out task metric으로 판정해야 한다. amplitude rescaling control도 support와 dictionary 방향을 고정한 채 calibration으로만 적합하고 fresh test에서 평가한다.

## 6. 공식 generator와 최소 재현 설정

모든 코드 링크의 commit을 고정한다: [tree at d5886b5](https://github.com/chanind/sparse-but-wrong-paper/tree/d5886b540dc5b9cac4f76e6db2b0cce1b0b7c585). 실행 entrypoint는 scripts가 아니라 notebooks다. 저장소 README에 있는 디렉터리 설명 일부는 실제 파일명과 다르므로 아래 실제 경로를 따른다.

| 목적 | 공식 파일 / 읽은 cell (0-based) |
|---|---|
| 5-feature positive/negative experiment | `notebooks/small_toy_model_experiments.ipynb`: cells 2,6,10,14,18,22 |
| 상관 강도 / L0 sweep | 위 notebook: cells 25,29 |
| 50-feature main experiment | `notebooks/main_toy_model_experiments.ipynb`: cells 1,5,9,16,21,24,28,32,35 |
| JumpReLU sweep | `notebooks/toy_model_experiments_jumprelu.ipynb`: cells 1,5 |
| 데이터·학습·decoder | `sparse_but_wrong/toy_models/get_training_batch.py`, `train_toy_sae.py`, `toy_model.py`, `orthogonalize.py`, `initialization.py` |
| K schedule / decoder norm | `sparse_but_wrong/enchanced_batch_topk_sae.py` (upstream filename has this spelling) |
| LLM example | `notebooks/train_and_eval_llm_sae.ipynb` (entrypoint identified; this task did not execute it) |

**공통 sampling:** Gaussian vector `u~N(0,C)`를 뽑아 `a_i=1[u_i>Φ^{-1}(1-p_i)]`로 support를 만들고, 독립 magnitude noise ε로 `m_i=a_i max(0,1+0.15 ε_i)`를 생성한다. `x=Σ_i m_i f_i`. 따라서 C는 **latent Gaussian correlation matrix**이며 binary support의 Pearson correlation이 아니다. 가변량 분포를 다시 sampling하므로 batch마다 fresh examples다. mean magnitude=1은 함수의 기본값이다. rectification이 active probability를 미세하게 바꾸므로 true L0는 실제 m>0에서 측정한다.

**Small exact setting:** g=h=5, d=20; p_i=.4; C_ii=1; C_0j=C_j0=+.4 또는 -.4 (j=1..4); 나머지 offdiagonal 0. 비교 K=2와 1.8. 낮은 K는 ground-truth encoder/decoder 초기화도 수행. Appendix D reproduction은 latent Gaussian correlation {-0.5,-0.4,…,0.5}, K=1.8; L0 sweep은 {1.7,1.8,1.9,2.0}와 correlation ±.4. 끝점 ±.5는 PSD 경계여서 일부 Cholesky sampler가 실패할 수 있다. pilot은 ±.4,0으로 시작하거나 jitter를 명시해 사용한다.

**Large exact code setting:** g=h=50, d=100; `p_i=0.345*(49-i)/50+0.05`, i=0,…,49. 따라서 p_0=.3881, p_49=.05, Σp=10.9525 (코드/plot에서 약11). `generate_random_correlation_matrix(num_features=50, positive_ratio=.5, correlation_strength_range=(.3,.9), sparsity=.3, seed=42)`; strength를 부여한 뒤 eigenvalue를 1e-6으로 clip하고 diagonal 1로 다시 normalize한다. 재구성 후 actual pair strengths/sign count는 요청 범위와 정확히 같다고 보장하지 않는다. 공식 sweep K=1,…,25, 폴더 seed label=0,…,4; 대표 K=5,11,18.

**Feature directions:** 공식 ToyModel은 iid normal directions를 normalize한 뒤 Adam lr=.01, 1000 steps로 Gram matrix offdiagonal과 norm error를 줄인다. exact QR를 사용한 새 pilot은 타당하지만 **원 generator의 문자 그대로의 재현은 아니다**. Small superposition variation은 orthogonalization을 4 steps로 줄인다. Large superposition Appendix E는 g=50,d=40. 좁은 SAE는 g=50,h=25,d=100이며 앞25개 feature 확률 합은 7.6325로 paper의 약7.6과 일치한다.

**Training defaults in code:** Adam β=(.9,.999), lr=3e-4, 15,000,000 samples, batch=1024, constant LR, no LR warmup/decay, no autocast; `dead_feature_window=1000`, `feature_sampling_window=2000`; online fresh generator. 실제 step 수는 trainer batching에서 정해지므로 로그에 기록한다. W&B와 final checkpoint save는 helper에서 꺼져 있다. Code lockfile versions: Python≥3.11; sae-lens=6.11.1; torch=2.8.0; transformer-lens=2.16.1; sae-probes=0.1.4. 현재 vg-sae 환경에서 무작정 latest를 설치하면 같은 구현이라고 볼 수 없다.

**JumpReLU toy code:** `l0_coefficient` grid {.05,.07,.09,.1,.3,.5,.75,1,1.2,1.3,1.4,1.5}; bandwidth=2, init_threshold=.1, `jumprelu_sparsity_loss_mode='tanh'`, `l0_warm_up_steps=10000`, `normalize_activations='expected_average_only_in'`, h=50,d=100. notebook loop의 변수/폴더 이름 `l1`/`jr_saes_by_l1`는 **실제 L1 SAE라는 뜻이 아니다**. Appendix B (p.13)의 penalty는 `Σ_i tanh(c |a_i| ||d_i||)`이며 decoder norm을 포함한다. 이를 plain L1 comparator로 등록하면 안 된다.

**최소 local pilot 제안(새 실험설계, 미실행):** 우선 small toy ±.4/0, 동일 world에서 BatchTopK K={1.8,2}, L1, Gated, VG를 비교하고 JumpReLU를 강한 필수 후속 baseline으로 둔다. 3개의 명시적 seed와 독립 calibration/test stream; 2,000–5,000 updates의 짧은 run은 구현/방향 확인만 한다. 논문 fidelity claim은 15M-sample budget과 baseline 수렴이 확인된 뒤에만 한다. full main toy로 넓히기 전에 small toy의 true-dictionary initialization에서 low-L0 mixing이 재현되는지 확인한다. normalization·decoder bias·nonnegative amplitudes·realized L0를 맞추고, regularizer의 숫자 자체를 방법 간 동등 budget이라고 간주하지 않는다.

## 7. 재현 시 이미 확인한 불일치와 함정

- §3 본문 (p.3)은 batch500, Appendix C (p.13)와 helper는 batch1024. 15M/500=30k steps지만 15M/1024≈14.65k steps다. 어느 조건인지 함께 명시한다.
- §3.2 prose의 p_0=.345→p_49=.05를 단순 linspace하면 Σp=9.875다. 공식 notebook 수식의 합10.9525와 다르다. 코드를 기준으로 실행하고 차이를 보고한다.
- §3.5 prose는 첫25k steps 전환 후5k steps 고정. 코드 down-schedule은 start3000,duration8000; up-schedule은 start0,duration25000인데 helper 기본 총step≈14649라 그대로는 전환을 끝내지 못한다. Schedule experiment는 해당 차이를 해결한 명시 config 없이는 faithful replication이라고 하지 않는다.
- notebook의 `for seed in [0,...,4]`는 directory label로 쓰이며 그 loop 안에 `torch.manual_seed(seed)`가 없다. random-correlation 생성은 global Python/Torch RNG를 seed42로 바꾼다. 재현 시 world/init/train/calibration/test RNG를 명시적으로 분리·기록하는 것은 필요한 개선이다.
- notebook에서 좋은 수렴을 얻을 때까지 재실행하라는 주석이 있다. 새 연구에서는 seed를 사전 고정하고 실패한 seed도 포함해 보고한다.
- `eval_sae`는 100,000 examples에서 L0/dead counts를 평가하고 decoder norm을 fold하는 mutation을 수행한다. all-zero inputs의 output/input norm ratio는 0/0이 가능하다. 새 평가에서 zero-input 처리 규칙을 명시한다.
- code의 reconstruction plot은 uncentered explained energy, v4 Fig.4는 MSE이므로 raw 수치와 축을 섞지 않는다. 저자 reported MSE scalar도 sum/mean convention을 확인한 뒤 비교한다.

## 8. VG-SAE에 대한 반증 가능한 가설 3개

아래는 논문의 결론이 아니라 **우리가 시험할 가설**이다. 각 가설이 실패해도 주 연구는 VG-SAE 개발이며 실패 원인과 개선 방향을 기록한다.

**H1 — Support 선택과 amplitude 추정의 분리가 L1 shrinkage 이외의 feature 복원 이득을 준다.** Same world, 동일 dictionary width/encoder 용량/compute와 calibration-matched hard L0에서 VG가 L1/ReLU보다 coefficient bias와 matched dictionary recovery를 개선한다. 그 차이가 calibration-only rescaling한 L1 및 Gated baseline에도 남는지 확인한다. rescaling/Gated control 후 이득이 사라지면 “VG objective가 mixing을 줄였다”는 해석은 기각한다. Sparse but Wrong가 제공하는 anchor는 MSE와 dictionary quality의 분리이며 shrinkage 근거는 별도 Gated 논문이다.

**H2 — VG의 data-dependent support는 저차 toy의 coactivation 구조 변화에서 더 넓은 유효 sparsity 영역을 갖는다.** positive/negative/independent copula를 paired world로 사용해 prior/coefficient sweep 전체의 realized L0–dictionary/support recovery를 그린다. VG의 recovery가 좋은 구간이 BatchTopK/L1보다 넓고 JumpReLU/Gated와 비교할 가치가 있는지 본다. 좋은 구간을 test를 보고 고르지 않고 calibration으로 선택한다. optimum 한 점의 개선만 있거나 JumpReLU가 같거나 더 넓으면 broad robustness 주장은 기각한다. VG도 너무 낮은 hard L0에서 support를 누락할 수 있으므로 theorem의 구조적 한계가 자동 해소된다고 주장하지 않는다.

**H3 — VG Bernoulli variance/entropy coupling이 generic gating 이상의 사전 지정 조건 이득을 만든다.** full VG, no-variance, no-entropy, deterministic gated control을 동일 데이터·capacity·같은 HPO 기회·realized L0에서 비교한다. full objective의 유리한 c_dec만 보지 말고 fresh-test support F1, signed dictionary alignment, leakage를 판정한다. beta/profiling을 ablation마다 다시 최적화하면 objective coupling이 달라지므로 fixed-beta와 profiled-beta를 구분한다. generic amplitude rescaling/gating과 동일하고 추가 term의 이득이 없으면 VG-specific mechanism 주장을 기각한다. prior만 강화해 낮은 MSE를 얻었으나 mixing이 늘었다면 방법 성공으로 판정하지 않는다.

## 9. 인용 및 논문 서술에 사용할 짧은 결론

권장 동기: “L1 기반 희소화의 amplitude bias는 알려져 있고, 최근 Sparse but Wrong는 hard sparsity 계열에서도 부적절한 L0와 reconstruction 중심 평가가 feature 혼합을 숨길 수 있음을 보였다. 우리는 Variational Garrote의 확률적 support 선택을 SAE로 확장하고, amplitude·support·dictionary recovery를 분리해 기존 gate 및 threshold 방법을 넘는 이득이 있는지 검증한다.”

피할 주장: “이 논문은 L1을 버리라고 결론 냈다”, “non-L1이므로 feature mixing을 해결한다”, “VG의 q가 semantic posterior다”, “c_dec가 낮으면 monosemantic이다”, “L0/MSE Pareto 우위만으로 해석 가능성이 좋다”.
