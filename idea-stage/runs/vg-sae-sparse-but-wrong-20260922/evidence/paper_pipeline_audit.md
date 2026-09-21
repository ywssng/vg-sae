# Sparse but Wrong를 기준으로 한 VG-SAE 구현·파일럿 감사

2026-09-22. 범위는 현재 코드, 저장된 이전 근거, 로컬 참고 PDF의 정합성 확인이다. 새 실험·학습·GPU 점유·source 변경은 수행하지 않았다. 주 연구는 **Variational Garrote 기반 새 SAE 개발**이며 아래 진단은 방법 선택을 위한 도구다. 당시 `RESEARCH_BRIEF.md`는 09-21 목표였으므로 새 논문 anchor는 이번 사용자 정정을 우선했다.

## 1. 결론과 새로 구분해야 하는 질문

현재 VG는 amplitude에 직접 L1 비용을 부과하지 않고 Bernoulli support의 variational risk를 최적화한다. 이는 유효한 방법상의 출발점이지만, **맞는 L0의 자동 발견** 또는 **correlated feature mixing의 방지**가 이미 구현되었다는 뜻은 아니다. 고정 `lambda_sparsity = gamma`와 inference threshold가 밀도를 결정하며, 정규화된 decoder에서 variance 항은 decoder 방향을 직접 분리하지 않는다.

새 논문 anchor에 맞는 질문은 다음 두 가지다.

1. 동일한 achieved hard L0 또는 L0 오지정 구간에서 VG가 L1/Gated/JumpReLU/BatchTopK보다 ground-truth feature를 덜 섞는가?
2. true support/L0를 제공하지 않은 선택·학습 규칙으로 좋은 operating point를 찾는가?

첫 질문의 oracle calibration 성공을 둘째 질문의 해결로 쓰면 안 된다. Reconstruction EV가 높아져도 feature recovery가 낮아질 수 있으므로 두 축을 동시에 기록한다. 더 낮은 L0 자체도 성공 지표가 아니다.

## 2. 참고 논문과 기존 데이터의 차이

로컬 `refs/Sparse but Wrong Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders.pdf`는 Chanin/Garriga-Alonso, ICML 2026 / PMLR 306, arXiv:2508.16560v4, 2026-07-07이다. 본 감사는 초록, §2–3, §4 주변 metric 설명, Appendix C/D/F를 읽었다. 전체 문헌·신규성 확인은 별도 조사 담당 범위다.

- 논문 소형 예시는 orthogonal feature 5개, input dimension 20, firing probability .4, true mean L0 2다. Feature 0과 나머지의 positive/negative firing correlation을 바꾸며, BatchTopK 1.8과 2를 비교한다. Low-L0 arm의 ground-truth initialization은 최적화 실패만으로 mixing을 설명하기 어렵게 한다.
- 큰 예시는 feature 50개, dimension 100, 서로 다른 frequency, true L0 약 11에서 low/correct/high L0를 비교한다. JumpReLU도 확인하므로 L1 amplitude shrinkage를 제거하는 것만으로 이 논문의 실패가 해결된다는 해석은 맞지 않는다.
- 부록의 수학적 논의에는 truncation에 의해 한 feature가 빠지는 co-occurrence 사건만으로 descent pressure가 생기는 예시가 있다. 따라서 correlation을 모든 failure의 필요조건이라고 정리하지 않는다. 또한 positive magnitude product와 negative correlation에 관한 문구를 그대로 새 정리로 옮기지 않고, 새 실험의 실제 support correlation과 signed mixing을 직접 확인한다.
- 본문은 toy batch size 500, Appendix C는 1024라고 써 있어 정확한 재현에서는 저자 코드/config 확인이 필요하다. 새 cheap pilot의 batch/steps는 독립 protocol로 명시한다.

현재 `src/sae_data.py:make_synthetic_sparse_coding`는 각 feature support를 독립 `rng.binomial`로 만든다. `coherence`는 `make_unit_dictionary`의 공통 방향을 바꾸는 parameter이며 support correlation이 아니다. `frequency_skew`는 marginal probability만 바꾼다. Stage 1 설정은 이를 더 엄격히 제한하고 README/REPRODUCTION_NOTES도 correlated firing을 포함하지 않는다고 설명한다. **기존 Stage 1/P1을 논문 failure 재현으로 재명명할 수 없다.**

Stage 2는 pinned SynthSAEBench의 hierarchy를 포함한 별도 realistic regime다. `src/synthsaebench_sweep.py`는 `decoderesearch/synth-sae-bench-16k-v1`, revision `b2efd8b919ae46d6d487c73d46db5ee52813621d`, d768/true16384/SAE4096, `scale_children_by_parent=False`를 고정한다. Support dependency가 있는 현실성 단계로 재사용할 수 있으나, Stage 1 `coherence` knob나 이번 orthogonal controlled-correlation experiment와 동치가 아니다. Generator와 scale flag의 정확한 의미는 manifest/REPRODUCTION_NOTES를 유지한다.

## 3. 현재 VG의 정확한 목적함수와 한계

`src/sae_model.py:VGSAEConfig`와 `VariationalGarroteSAE.free_energy`를 기준으로, 입력별 `m = sigmoid(gate_encoder(x-b))`, `a = softplus(amplitude_encoder(x-b))`, `h = m*a`다. 기본 nonnegative amplitude이며 gate/amplitude는 서로 다른 선형 head다. Decoder column은 unit norm으로 유지한다.

고정 precision beta, unit entropy weight에서 입력 하나의 목적함수는 상수까지 다음과 같다.

\[
F=\frac{\beta}{2}\left(\|x-b-D(m\odot a)\|^2+
\sum_j m_j(1-m_j)a_j^2\|d_j\|^2\right)
-\frac{d}{2}\log\frac{\beta}{2\pi}
+\gamma\sum_jm_j+J\operatorname{softplus}(-\gamma)-\sum_jH(m_j).
\]

이는 **input-dependent point amplitude를 조건으로 한 mean-field support variational risk**다. Full generative ELBO, true semantic posterior, calibrated support probability라고 단정하지 않는다. `entropy_weight != 1` 또는 variance/entropy 제거는 서로 다른 objective다.

- Learned beta는 scalar `log_beta`를 optimizer가 학습한다. Profiled beta는 minibatch energy로 `beta_eff = B*d/(2*sum energy)`를 만들고 `d/2*log(mean per-coordinate expected squared risk)`를 최적화한다. `E_batch log(E_batch)`와 global learned-beta objective는 같지 않다. 작은 noiseless/undercomplete toy가 거의 정확히 fit되면 profiled precision과 epsilon floor가 결과를 지배할 수 있어 beta/risk/floor 접근 여부를 기록한다.
- `gamma`는 signed finite scalar field다. Prior probability는 `pi = sigmoid(-gamma)`. 고정 gamma가 expected support size `sum m`를 통제한다. 고정 폭에서 normalizer는 model parameter에 대한 gradient를 바꾸지 않지만 gamma를 학습한다면 필수다.
- Current core `encode()`는 `(m,a,m*a)`, `encode_inference()`는 `1[m>tau]*a`를 반환한다. SAELens `VGTrainingSAE.training_forward_pass()`는 mean reconstruction을 학습하되 public `feature_acts`와 inference는 hard code다. Expected L0와 actual hard L0를 반드시 분리한다.
- 표준 Bernoulli gate threshold .5는 symmetric support decision의 MAP 의미를 가질 수 있으나 reconstruction, latent error, correlated feature disentanglement를 자동 최적화하지 않는다.
- `m`이 0/1로 포화되면 variance와 entropy가 사라지고 reconstruction + gamma L0 형태가 된다. 따라서 hard selection을 쓴다는 사실만으로 잘못된 sparsity pressure에서 자유롭지 않다.

### 3.1 Variance 항이 decoder 방향에 주는 직접 gradient

고정 `m,a`에서

\[
V=\tfrac12\sum_jm_j(1-m_j)a_j^2\|d_j\|^2,
\quad \nabla_{d_j}V=m_j(1-m_j)a_j^2d_j.
\]

이는 radial gradient다. Unit-norm constraint의 tangent projection을 거치면 0이며, 현재 core training과 SAELens adapter는 이 projection을 실제로 수행한다. Variance에 decoder pair cosine을 벌주는 직접 항은 없다. Profiled mode에서도 fixed `m,a`의 variance-gradient 성분은 scalar beta로 곱해진 radial 항이므로 동일하다.

그러므로 VG variance가 mixing을 개선한다면 그 경로는 **m/a의 변화, precision coupling, 최종 학습 trajectory**에 대한 경험적·조건부 근거여야 한다. Variance를 줄이는 더 확신 있는 mixed representation을 objective가 허용할 수도 있다. 단지 variance 항의 부호가 양수라는 이유로 disentanglement를 보장할 수 없다.

별도 유도: orthonormal true dictionary에서 `x=D_* z`, fitted bias가 평균 residual을 흡수했다고 하자. D를 true dictionary에 놓고 code `h`를 고정하면, atom j가 true direction i로 움직이는 reconstruction gradient 성분은 `Cov(h_i-z_i, h_j)`다. Bias를 0으로 고정하면 covariance 대신 raw product expectation이다. 누락된 coefficient와 남은 code의 상관이 signed mixing pressure를 만들 수 있다. 이는 이 코드에 적용한 audit derivation이며 새 정리/논문 novelty 주장이 아니다.

### 3.2 Gamma 학습은 즉시 가능하지만 correct L0의 정답은 아니다

Normalized scalar gamma를 학습하는 candidate에서는

\[
\partial_\gamma\overline F=\sum_j\overline m_j-J\sigma(-\gamma).
\]

Stationarity는 `pi = mean_{sample,latent}(m)`라는 자기 일관성이다. 원래 support와 일치한다는 조건은 아니다. `gamma* = log((1-rho)/rho)`의 detached alternating update도 같은 한계를 가진다. Beta prior/pseudocount로 endpoint를 막을 수 있으나 hyperparameter와 width dependence가 새로 생긴다. Amplitude가 x의 함수인 현재 목적에서 이 절차를 검증된 marginal-likelihood model selection이라고 부르지 않는다.

현재 `free_energy(lambda_sparsity=...)`는 `float()` 변환을 하므로 trainable gamma tensor를 넘기면 gradient가 끊긴다. Candidate는 우선 local runner의 별도 loss에서 normalized prior를 differentiable하게 계산해야 한다. Core public config 의미를 바꾸지 않는다. Gamma-only adaptation이 실패하면 correlated support prior나 structured posterior의 필요성을 검토할 수 있으나, 그 결과를 보기 전에 복잡한 module을 기본 방법으로 채택하지 않는다.

## 4. 재사용할 코드와 공정성 경계

| 목적 | 바로 재사용할 entrypoint | 주의점 |
|---|---|---|
| 작은 synthetic tensors | `src.sae_data.SparseCodingTensors` | correlated-support sampler는 신규 run-local helper 필요 |
| VG core | `src.sae_train.build_sae('vg', d, J, lambda_sparsity=..., beta_mode=...)` | core는 `input_dim`, `n_latents`; 일반 baseline은 `d_in`, `d_sae` |
| 같은 trainer 경로 VG | `src.saelens_vg.VGTrainingSAE(VGTrainingSAEConfig(d_in=d,d_sae=J,...))` | common official trainer를 원하면 이 adapter가 적절 |
| L1 / Gated | `build_sae('l1',d,J,l1_coefficient=...)`, `build_sae('gated',...)` | 두 모델은 공식 SAELens aliases; gated도 비교에 포함해야 amplitude-head 분리만의 효과를 판별 가능 |
| JumpReLU / BatchTopK | `build_sae('jumprelu',d,J,l0_coefficient=...)`, `build_sae('batchtopk',d,J,k=float_value)` | BatchTopK는 noninteger mean k 허용; export 후 native inference threshold 사용 |
| 학습 | `src.sae_train.fit_sae` | baseline은 official SAETrainer, core VG는 AdamW route. 동일 optimizer/provider를 원하면 VG adapter 사용 |
| 공식 inference | `src.sae_baselines.to_inference_sae(model,fold_decoder_norm=True)` | BatchTopK를 training batch TopK 상태로 평가하지 말 것 |
| 공통 metrics | `src.sae_sweep_eval`의 matching/rectangular union/hard-code helper | wrapper 구조만 재사용하고 아래 sign/threshold를 새 pilot에서 명확히 고정 |
| decoder metric | `src.sae_evaluate.decoder_pairwise_cosine_similarity`, `decoder_atoms_from_model` | cdec 단독 minimum은 monosemanticity 증명이 아님 |
| prior runner | `scripts/run_vg_development_holdout.py`, `run_vg_development_training.py` | split freeze, hashed protocol, timing 기록의 패턴 재사용; 기존 generator/condition 그대로 사용하지 않음 |

Pinned baseline dependency 검증은 `tests/test_sae_baselines_saelens.py`에 SAELens 6.47.0 / revision `8be14080485952f729ed58d674bcddf9778e0aa4`로 들어 있다. 실제 새 실행에서 installed version과 commit을 확인한다. 이미 통과했다고 여기서 주장하지 않는다.

`fit_sae`의 best checkpoint는 training objective 최소값이며 feature quality oracle가 아니다. Cross-method loss 수치는 objective scale이 달라 비교 불가다. Final step을 공통으로 쓰거나 모든 방법에 동일 checkpoint candidate 수를 주고 calibration rule을 고정한다. Ground-truth calibration은 방법 개발용 oracle로 표기하고 truth-free selector를 별도 보고한다.

평가에서 필요한 추가사항:

- **Signed cross-feature matrix:** learned atoms를 정규화하고 true atoms와 signed cosine 행렬을 보존한다. Hungarian matching 후 matched diagonal뿐 아니라 off-diagonal leakage RMS, max-absolute contamination, positive/negative correlation edge별 contamination을 기록한다. cdec, absolute-MCC 하나만으로는 누가 누구를 섞는지 안 보인다.
- **Sign:** nonnegative code의 feature recovery에서 decoder sign flip은 자동 동치가 아니다. 기존 Stage1 helper는 absolute matching과 signed coefficient alignment를 사용한다. 새 paper-like positive-coefficient toy에서는 sign-preserving primary matching/diagonal accuracy를 정의하거나, 최소 signed matched cosine와 negative match 비율을 함께 보고하여 absolute-MCC가 반대 방향 atom을 정답처럼 세지 않게 한다.
- **Unmatched atoms:** rectangular union helper는 unmatched learned code를 FP, unmatched truth를 FN으로 남긴다. 이 규칙을 유지하고 dead atom·coverage를 함께 기록한다.
- **Native L1:** 기존 evaluator는 train-fit GMM threshold L1 code와 raw ReLU code를 구별한다. 새 비교의 primary는 native raw nonzero L0로 하고, GMM/threshold가 필요한 경우 모든 threshold 출처·추가 inference 변형을 별도 표기한다. VG는 native .5 hard support와 mean/stochastic risk를 나눈다.
- **Threshold boundary:** core/adapter는 `m > tau`, Stage1 `_hard_latents`는 `m >= threshold`다. 기존 extreme equality는 보통 드물지만 이번 tiny oracle initialization은 정확히 .5가 생길 수 있어 native `>`로 통일하거나 boundary 건수를 명시한다.
- **Risk:** mean SSE, hard SSE, Bernoulli expected SSE `mean SSE + 2V`를 별도 기록한다. `expected_ev` 같은 기존 이름을 sampled-risk EV로 읽지 않는다.

## 5. 가장 싼 새 pilot 패키지

아래는 **미실행 제안**이다. Root가 2–3개를 사전등록하고 시간·GPU를 배정한다. GPU0는 다른 작업이 사용 중이라는 현재 scheduling 정보를 존중하며, 이 감사는 어떤 GPU도 사용하지 않았다. 1/2/3의 availability도 실행 직전에 재확인한다.

통합 단계의 선택: root는 paper-like 5-feature orthogonal dictionary + Gaussian-copula correlation ±.4 protocol을 우선한다. 아래 exact Bernoulli pair는 realized support correlation을 검증하는 secondary option으로 보존한다. 구체적인 gradient tests와 method pilot 구현은 후보 선정 후 진행하며, 이 감사의 grids와 실행 수는 최종 preregistration이 아니다. Copula의 latent correlation과 sampled binary-support correlation을 각각 기록한다.

### P1. Correlated firing에서 VG recovery와 achieved-L0 곡선

목적은 새로운 VG 방법의 출발점이 논문의 failure에 실제로 얼마나 민감한지 구분하는 것이다. 소형 직교 toy에서는 “큰 모델이 어려워 실패했다”라는 설명을 줄인다.

- d20, true/SAE width5, p=.4, independent amplitude `Uniform[1,2]`, noise std .02, train8192/cal2048/test4096, paired worlds3. Dictionary는 QR로 orthonormal 생성하고 모든 split에 고정한다. Positive/negative/zero firing correlation만 바꾼다. 이 조건은 저자 full reproduction이 아닌 제한된 mechanistic pilot이다.
- 검증하기 쉬운 exact pair sampler: features0/1에 `P11=.16+.24*rho`, `P10=P01=.24*(1-rho)`, `P00=.36+.24*rho`; rho in `{-.5,0,+.5}`. Features2–4는 독립 Bernoulli .4. Marginals와 mean L0=2를 보존하며 부호가 다른 support correlation을 만든다. 저자의 hub correlation generator를 재사용할 수 있으면 별도 paper-reproduction arm으로 둔다. Gaussian latent correlation을 realized Bernoulli correlation이라고 적지 않는다.
- 첫 screening은 VG, L1, Gated, JumpReLU, BatchTopK 각 **동일 5-control** grid, 동일 samples/steps/checkpoint count. VG gamma 후보 `[-2,0,2,4,6]`; L1/Gated coefficients와 JumpReLU coefficients는 각 architecture의 scale가 다르므로 작은 사전 scout를 통해 low/correct/high L0를 bracket하고 cal만으로 확정한다. BatchTopK는 `[1,1.5,2,2.5,3]`를 사용할 수 있다. Grid를 그대로 정량 fairness로 포장하지 말고 scout 비용까지 기록한다.
- 3000 updates, batch256, LR3e-4를 initial common protocol로 두되 500/1500/3000 동일 checkpoints를 저장한다. Undertraining 판단은 저장 curve를 사용한다. 모든 방법이 ground truth 근처를 회복하지 못하면 full run 확장 전에 positive control/학습 길이를 수정하고 새로운 protocol version으로 분리한다.
- Oracle-cal recovery optimum, truth-free cdec minimum, achieved-L0 envelope를 **서로 다른 selection**으로 동결한다. Test support/L0는 selection에 사용하지 않는다. Threshold .5를 tuning해 true L0에 맞췄다면 correct-L0 discovery라 부르지 않는다.
- Primary: signed mixing, matched direction recovery/coverage, hard coefficient relative error, support F1. Secondary: achieved hard/expected L0, mean/hard/stochastic reconstruction, cdec/dead fraction, parameter count/latency. Corr0은 sanity control이지 무조건 no mixing이라는 이론적 보장이 아니다.
- 이어갈 기준: 세 world 중 적어도 두 world와 양 correlation 부호에서 low/high L0의 recovery degradation을 관찰하고, VG가 강한 baseline과 다른 회복 가능 operating region을 보이는지 확인한다. 유사하면 VG가 이미 해결했다고 결론 내리지 않고 P2/P3 설계의 기준선으로 삼는다. Matching metric과 L0 차이가 동시에 유리한 결과만 사전 robustness claim에 사용한다.

### P2. VG objective coupling이 mixing을 바꾸는가

P1의 VG grid 중 cal로 선택한 low/correct/high achieved-L0 checkpoint/control을 이용한다. Full VG, no variance, no entropy의 원소별 ablation은 같은 initialization/batches와 같은 checkpoint 수로 비교한다. Profiled beta에서는 variance 제거가 beta scale도 바꾸므로 term 제거를 독립적인 coefficient 변화로 해석하지 않는다.

최소 추가 실행은 positive/negative pair condition × 3 paired worlds × 3 controls × 3 variants다. P1 full-VG run은 재사용 가능하다. 원래 gamma를 유지한 total-effect 비교와 cal로 achieved-L0를 가까이 맞춘 비교를 나눠야 한다. No-variance/no-entropy가 proper variational objective라는 설명은 피한다.

그 전에 CPU float64의 작은 deterministic checks만으로 (a) Bernoulli enumeration = mean risk + variance, (b) fixed-code unit-norm decoder rotation의 variance first derivative 0, (c) gamma derivative의 normalized-prior formula, (d) coordinate gate optimum을 검증한다. 이들은 성능 pilot을 대체하지 않고 수식/구현의 의미를 고정한다.

진행 기준: full VG의 recovery 이득이 단순 density 이동을 넘어 남고, decoder mixing·support recovery 지표에서 같은 방향인지 확인한다. Full variance가 유리하면 현재 VG 핵심을 유지하며 학습/decision을 개선한다. 불리하면 variance를 무조건 삭제하지 말고 amplitude/gate/precision coupling을 후속 설계 대상으로 삼는다.

### P3. Unknown true L0를 위한 normalized-prior adaptation

방법 candidate는 현재 VG의 Bernoulli prior를 scalar learned gamma 또는 detached alternating gamma update로 추정하는 최소 확장이다. 모든 amplitude/gate/decoder 구조는 유지한다. 이것의 novelty와 선행연구 차이는 별도 문헌 검토 대상이며 현재 확증되지 않았다.

- Local runner에서 normalized prior와 gradient를 구현하고 core objective와 gamma 고정 시 수치 일치를 확인한다. Full core API/config 수정 없이 시작한다.
- Fixed-gamma VG grid, learned-prior VG, P1의 cal-cdec selector를 비교한다. True support/dictionary/L0는 학습·prior update·truth-free selection에 넣지 않는다. Oracle recovery selector는 benchmark ceiling으로 따로 둔다.
- 작은 paired-density transfer: p in `{.2,.4,.6}`에서 feasible pair rho±.3/0를 사용하고, source setting에서 정한 recipe를 그대로 적용한다. Correlation joint probability는 일반식 `p²+rho*p*(1-p)`이며 4개 joint probability가 nonnegative인지 검사한다.
- 3 initial gamma values, 3 worlds로 self-confirming density/initialization dependence를 점검한다. Prior probability, entropy, mean/hard L0, beta, dead fraction과 endpoint/clamp 빈도를 모두 저장한다. Training loss만 낮아지고 code가 dense 또는 collapse면 실패로 기록한다.
- 성공 범위는 “이 조건에서 true-L0 없이 선택한 VG가 고정 gamma transfer보다 recovery를 보존한다”까지다. `pi=mean(m)`라는 정지조건만으로 자동 correct-L0 발견이라 쓰지 않는다. Exact support identification은 별도의 identifiability와 observed recovery 근거가 필요하다.

세 pilot의 큰 조합을 전부 실행할 필요는 없다. P1이 먼저 충분히 학습된 corr-sensitive baseline을 만들고, P2/P3 중 root가 선택한 1–2개를 좁은 조건에서 수행하는 순서가 가장 싸다. Run별 최대시간과 전체 8 GPU-hour 예산은 preregistration에서 고정한다. 실제 runtime를 측정하지 않은 이 감사는 비용 보장을 하지 않는다.

## 6. 기존 결과를 활용하는 범위

09-21 P1은 existing 546 checkpoints를 fresh cal/test로 평가했다. Exponential에서 VG의 coefficient/F1와 fidelity 사이 tradeoff 신호가 있었지만 independent-support 한 training world, 불균등 control/parameter budget, L0 cap 비교였다. 이번 correlation failure나 correct-L0 선택의 증거가 아니다.

P2의 36개 beta 초기화 비교는 일관된 moment initialization 이득을 보이지 않았다. 일부 장기 학습에서 EV가 개선되면서 latent recovery가 나빠졌다. 이는 새 anchor와 양립하는 **평가 설계상 주의점**이지 correlation mixing을 측정한 결과가 아니다. Beta 초기값 개선을 새 주요 기여로 미리 채택할 근거가 없다.

P3/P3b readout/conditional correction은 fixed dictionary와 기존 한 조건에서 일부 hard-quality를 개선했다. Sequential update는 native latency 약992배, parallel도 사전3배 기준을 넘었다. Generic amplitude refit/control 뒤의 VG gate 이득이 작거나 split별로 바뀌었다. Sparse-but-wrong의 학습된 decoder mixing 자체를 고친 결과로 쓰지 않는다. 이 경로는 저비용 teacher/학습 개선의 보조 후보로 남기되 새 주제를 대체하지 않는다.

## 7. 재사용 가능한 검증과 필요한 새 검사

이번 감사에서는 tests를 실행하지 않았다. 새 구현에서 필요한 subset만 선택한다.

- `tests/test_sae_components.py`: analytic Bernoulli variance의 enumeration 일치, signed prior, dtype/endpoints, vector beta normalization.
- `tests/test_vg_sae_beta.py`: learned Gaussian/prior normalization, stationary beta, profiled/learned gradient 관계.
- `tests/test_saelens_vg.py`: official trainer dispatch, hard encode vs mean training reconstruction, decoder normalization, normalization folding/export.
- `tests/test_sae_baselines_saelens.py`: pinned official identity, trainer equivalence, BatchTopK export/EMA read-only semantics.
- `tests/test_sae_sweep_eval_rectangular.py`: unmatched FP/FN 및 hard-code reconstruction.
- `tests/test_vg_development_holdout.py`: independent RNG/fixed dictionary, train-only L1 threshold, cal-only selection.
- `tests/test_vg_development_refinement.py`: exact enumerated coordinate objective/conditional stationarity. Iterative correction을 다시 쓰는 경우에만 필요하다.

새 helper의 의미 있는 checks는 joint-support exact marginals/correlation/valid probabilities, orthonormal dictionary, split RNG 분리, signed mixing의 constructed ground-truth/known mixed atom 사례, selector의 test-field 불변성, learned gamma gradient와 fixed-gamma objective equivalence다. Pairwise cosine 하나를 objective로 넣어 metric을 직접 최소화한 결과를 feature recovery의 독립 증거로 삼지 않는다.

## 8. 방법 개발로 연결할 최종 판단

VG의 장점 후보는 probabilistic support와 amplitude의 분리, analytic stochastic risk, normalized sparsity field, residual-evidence 기반 support update다. 이를 실제 contribution으로 만들려면 **correlated firing에서 feature recovery를 보존하는 학습 및 true-L0를 쓰지 않는 operating-point 선택**을 함께 보여야 한다. 이번 감사에서 그런 성능 결론은 아직 없다. 위의 작은 controlled pilot로 어떤 VG-specific 경로가 효과가 있는지 좁힌 뒤, 기존 SynthSAEBench와 saved real-activation checkpoints에서 전이를 확인하는 순서가 현재 코드와 자원을 가장 잘 재사용한다.
