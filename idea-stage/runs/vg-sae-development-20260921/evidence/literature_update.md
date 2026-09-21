# VG-SAE 개발을 위한 문헌 갱신

검토일: 2026-09-21. 담당: `/root/vg_method_literature`, GPT-6 Astra `ultra`.
이 문서는 문헌·개발 선택지 shard의 결과이며, 독립적인 최종 신규성 판정이나 후보 순위가 아니다.
현재 연구 목표는 저장소 루트 `RESEARCH_BRIEF.md`의 **Variational Garrote 기반 새 SAE 개발**이다.
2026-09-19 보고서의 복제 진단 중심 문제 설정은 현재 목표를 규정하지 않는다.

## 판단

읽은 1차 자료 중 현재 구현의 **Bernoulli mean-field support + 입력별 point amplitude + 선형 dictionary의 정확한 기대 재구성 오차 + Bernoulli prior/entropy + Gaussian precision + 한 번의 encoder 추론**을 그대로 제시한 단일 논문은 확인하지 못했다. 이는 검색 범위 안의 미발견이며 신규성 확증은 아니다. 각각의 요소와 상당한 부분 조합은 선행연구에 존재한다. 따라서 이 조합을 VG에서 SAE로 옮기는 구체적인 방법과 그로 인해 달라지는 학습·추론 성능을 검증할 여지는 있지만, 조합의 나열만으로 충분한 기여라고 결론내리지 않는다.

특히 기존 28개 목록에 없던 **VAEase, spike-and-slab VSC, learned-threshold VSC, sparse-coding VAE**를 추가했다. 이들은 변분 SAE라는 넓은 영역의 직접 선행이다. Gaussian vSAE의 특정 부정 결과를 모든 Bernoulli VG-SAE의 실패로 일반화할 수도 없다. 확률화를 도입했다는 사실보다, **어떤 불확실성을 어떻게 적분하고 어떤 비용으로 희소 표현을 얻는가**를 비교해야 한다. [VAEase](https://proceedings.mlr.press/v267/lu25w.html), [VSC](https://proceedings.mlr.press/v115/tonolini20a.html), [learned-threshold VSC](https://proceedings.mlr.press/v162/fallah22a.html), [SVAE](https://pmc.ncbi.nlm.nih.gov/articles/PMC13377536/).

## 현재 구현의 비교 기준

`src/sae_model.py`를 확인했다. 중심화된 입력을 $x$, dictionary 열을 $D_j$, latent 수를 $L$로 쓰면 기본 설정은 다음과 같다.

\[
m(x)=\sigma(W_gx+b_g),\qquad a(x)=\operatorname{softplus}(W_ax+b_a),\qquad q(s\mid x)=\prod_j\operatorname{Bern}(s_j;m_j(x)).
\]

\[
E(x)=\frac12\|x-D(m\odot a)\|^2+
\frac12\sum_jm_j(1-m_j)a_j^2\|D_j\|^2.
\]

정규화된 Bernoulli prior $\pi=\sigma(-\gamma)$와 entropy coefficient가 1일 때 per-example 목적함수는

\[
\mathcal F(x)=\beta E(x)-\frac d2\log\frac\beta{2\pi}
+\sum_j\operatorname{KL}(\operatorname{Bern}(m_j)\|\operatorname{Bern}(\pi)).
\]

- 별도 gate/amplitude affine encoder, 양의 softplus amplitude가 기본이다. signed amplitude 옵션도 있다.
- decoder 열은 기본적으로 unit norm이다. variance 식은 실제 열 norm도 사용한다.
- learned precision은 `exp(log_beta)`이며, profiled 모드는 **현재 minibatch의 기대 energy**로 $\beta^*=Bd/(2\sum_iE_i)$를 계산한다. 따라서 minibatch profiling과 전체 데이터 precision 학습은 같은 실행 절차가 아니다.
- `forward`는 posterior-mean coefficient $m a$를 사용한다. `encode_inference`는 $1[m>\tau]a$를 반환한다. posterior mean은 일반적으로 dense이고, expected L0와 hard L0도 다르다.
- entropy coefficient가 1이 아니거나 variance를 끈 ablation은 위의 원래 Bernoulli KL 목적함수와 달라진다.
- amplitude 자체의 prior/entropy는 없다. $a(x)$가 입력의 결정론적 함수이므로 이를 일반 spike-and-slab VAE의 Gaussian slab을 단순히 zero-variance로 보낸 극한과 동일시하면 안 된다. Gaussian slab KL은 그 극한에서 발산한다. 입력 의존 amplitude까지 포함해 정규화된 생성모델 ELBO라고 주장하려면 별도 모델 정의가 필요하다.

## 목적함수·추론·precision 비교

아래 표에서 ‘없음’은 확인한 논문 목적함수에 별도 학습 observation precision이 없다는 뜻이다. $\beta$라는 기호는 논문별로 KL 계수, temperature, precision 중 서로 다른 뜻을 가질 수 있다.

| 방법 / 확인 출처 | 목적함수·잠재변수 | 추론 / 학습 | observation precision | 현재 VG-SAE와 실제 차이 |
| --- | --- | --- | --- | --- |
| 현재 VG-SAE | 위의 Bernoulli KL + exact expected quadratic risk; amplitude는 point estimate | 두 affine head의 한 pass; 학습 중 latent sampling·STE 불필요; hard threshold 별도 | learned 또는 minibatch-profiled | 비교 기준 |
| [Kappen–Gómez, The Variational Garrote, Machine Learning 2014](https://arxiv.org/abs/1109.0486) / [Soh 외, 2025 preprint](https://arxiv.org/html/2509.06383v1) | binary support를 변분 적분하고 regression coefficients는 점 추정; variance·entropy를 포함 | 원래 fixed-design regression의 최적화; 2025 자동미분 | Gaussian precision 추정 | 가장 직접적인 수식 선행. SAE의 learned dictionary와 입력별 amortized support/amplitude가 차이 |
| [Gated SAE, NeurIPS 2024](https://arxiv.org/html/2404.16014v2) | hard gate와 ReLU magnitude 분리; reconstruction + gate surrogate L1 + auxiliary reconstruction | tied/scaled gate·magnitude weight, 한 pass; gate 보조 경로로 학습 | 없음 | support/amplitude 분리와 L1 shrinkage 개선은 선행. Bernoulli posterior/KL·기대 variance 없음 |
| [JumpReLU, 2024 preprint](https://arxiv.org/html/2407.14435v1) | $z\,1[z>\theta]$, MSE + 실제 L0 penalty | 한 pass; learned threshold를 STE로 학습 | 없음 | 직접 hard sparsity·작은 magnitude bias의 강한 baseline. probabilistic support 불필요 |
| [TopK, Gao 외, 2024 preprint](https://arxiv.org/html/2406.04093v1) | MSE + AuxK; 입력당 k개 선택 | 한 pass + TopK; decoder·encoder 공동 학습 | 없음 | 고정 per-input budget. sparsity control/낮은 shrinkage만으로 차별화 불가 |
| [BatchTopK, 2024 preprint](https://arxiv.org/html/2412.06410v1) | batch 전체 Bk개 선택, MSE + AuxK | 학습은 batch selection; 추론은 학습 중 추정한 global threshold | 없음 | 입력마다 다른 L0와 평균 budget 제어는 이미 구현. VG의 이득은 같은 실제 L0에서 검증 |
| [Probabilistic TopK, anonymous ICLR 2026 submission](https://openreview.net/pdf?id=zMIIHeKivz) | Binary Concrete score + 작은 magnitude score로 exact TopK mask; ordinary MSE | 훈련은 stochastic score; 추론은 gate noise 제거; TopK 유지 | 없음; 논문 β는 gate temperature | 확률적 input gate와 confidence 동기 자체는 선행. Bernoulli KL/free energy 또는 analytic expected risk는 제시된 §3 식 6–8과 다름 |
| [Baker–Li, Analysis of Variational Sparse Autoencoders, 2025 preprint](https://arxiv.org/html/2509.22994v1) | $q(z\mid x)=N(\mu(x),I)$, Gaussian KL = $\|\mu\|^2/2$, sampled code 뒤 TopK | Gaussian reparameterization, KL annealing | 없음; β는 KL weight | support uncertainty와 amplitude uncertainty를 분리하지 않음. 이 설정의 dead feature·fidelity 부정 결과는 VG에 대한 동일방법 반증이 아님 |
| [Solomon–Leburu–Chung, vsPAIR, 2026 preprint](https://arxiv.org/html/2602.02948v1) | Bernoulli spike + Gaussian slab posterior; Bernoulli KL + active-probability-weighted Gaussian KL; Beta hyperprior | observation VAE와 sparse QoI VAE를 연결; hard-concrete sampling | 기본 유도는 unit decoder covariance; task loss weights 별도 | spike와 slab을 모두 확률화하고 nonlinear paired inverse model 학습. VG에는 slab KL·slab sampling·paired decoder가 없음 |
| [Velychko–Damm–Fischer–Lücke, AISTATS 2024](https://arxiv.org/html/2311.01888v2) | Laplace prior, Gaussian variational posterior, 선형 decoder; analytic entropy ELBO | full/diagonal Gaussian, 비amortized 및 deep amortized 학습; analytic variance | stationary noise variance를 analytic objective에 반영 | analytic variational sparse coding·noise 추정·annealing 자체는 선행. Bernoulli discrete support+point amplitude와 다름 |
| [Lu–Zhu–He–Wipf, VAEase, ICML 2025](https://arxiv.org/html/2506.04859v2) | Gaussian VAE objective를 변형; $\tilde z=(1-\sigma_z(x))\odot z$, Gaussian KL | stochastic Gaussian latent; uncertainty head가 decoder 입력을 gate | trainable scalar decoder variance γ | 입력별 uncertainty gate+learned noise+LLM activation 적용까지 선행. gate가 Bernoulli probability가 아니고 slab shrink/KL·sampling이 다름 |
| [Tonolini–Jensen–Murray-Smith, VSC, UAI 2019 / PMLR 2020](https://proceedings.mlr.press/v115/tonolini20a/tonolini20a.pdf) | spike-and-slab prior/posterior, analytic KL; pseudo-input mixture prior | nonlinear VAE; Bernoulli relaxation과 spike warm-up, MC likelihood | 현재 읽은 범위에서 별도 최적 precision 절차 미확인 | Bernoulli inclusion과 amortized sparse coding은 오래된 직접 선행. point amplitude·linear exact expected risk는 다름 |
| [Fallah–Rozell, ICML 2022](https://proceedings.mlr.press/v162/fallah22a/fallah22a.pdf) | shifted threshold로 Laplace/Gaussian samples를 hard sparse 분포로 변환; ELBO | sampled latent, threshold STE, 학습 가능한 threshold distribution | 현재 읽은 범위에서 별도 precision 최적화 미확인 | hard sparse variational coding·sampling/gradient 효율 개선 선행. 정확한 Bernoulli 기대 risk와 다른 추정 경로 |
| [Geadah 외, Neural Computation 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC13377536/) | overcomplete linear SVAE, heavy-tailed prior와 Gaussian recognition posterior | feedforward deep recognition, VAE fitting | 관측 noise parameter를 모델에 포함; 이번 검토에서 학습 세부 미확인 | feedforward variational linear dictionary 자체는 선행; exact zero support를 만드는 Bernoulli VG와 다름 |

Probabilistic TopK는 direct PDF가 browser challenge로 막혔다. 검색 인덱스가 추출한 **원 PDF의 method와 training table**을 읽었다. 저자·최종 채택 여부·저자 코드의 gradient 처리는 미확정이며, 위 표는 확인한 수식 범위의 비교다.

## 데이터·학습 예산 비교

‘미확인’ 칸은 숫자를 추정하지 않았다. 서로 다른 논문의 보고 점수나 학습량으로 현재 코드의 우위를 비교하지 않는다.

| 방법 | 확인한 실험 데이터·model | 확인한 예산 / 조건 | VG-SAE 실험 설계에 주는 기준 |
| --- | --- | --- | --- |
| Gated | GELU-1L, Pythia-2.8B, Gemma-7B의 여러 activation site | batch 4096; GELU/Gemma 300k, Pythia 400k steps; resampling | 같은 width 외에 encoder/training FLOPs도 통제 |
| JumpReLU | Gemma 2 9B 여러 layer·site | 8B training tokens; TopK에는 AuxK | 짧은 pilot로 논문의 최종 경쟁력까지 결론내리지 않음 |
| TopK | GPT-2 small 및 GPT-4 activations | 최대 16M latent/40B tokens 시연; run별 budget 다름 | 현대 강한 baseline의 AuxK·init 포함; 동일 data exposure와 FLOPs 병기 |
| BatchTopK | GPT-2 Small, Gemma 2 2B residual stream | 1B tokens, batch 4096; GPT-2 width 3072–24576, Gemma 16384 | 평균 L0에 맞춘 가변 support의 필수 baseline |
| Probabilistic TopK | GPT-2 layer 8/OpenWebText; Qwen3-0.6 layer 26/FineWeb | 원 PDF Table 1: 각각 307M/215M tokens; 500k samples; context 1024 | latent sampling 없이 VG가 비슷한 feature utilization을 달성하는지 별도 검증 |
| Analysis of vSAEs | Pythia-70M layer 0 residual/TinyStories | width 2048, 20k steps, k=64/128/256/512, RTX 3080 10GB | 작은 모델에서도 비교 가능. 논문의 Gaussian KL와 VG KL를 혼동하지 않음 |
| vsPAIR | MNIST blind inpainting, CT inverse problems | CT 예시 100 또는 200 epochs; task별 network·loss 다름 | generic uncertainty 결과를 LM SAE support calibration으로 전이하지 않음 |
| Entropy ELBO | synthetic bars, whitened natural images | image N=204800, d=256, latent 100/400; total time 미확인 | analytic training의 직접 비교 참조; LLM 결과는 이 논문에 없음 |
| VAEase | synthetic subspace/manifold, MNIST/FashionMNIST, Pythia/Pile-10k, Yelp | LLM N=2098176, d=512, latent 300, 35 epochs, batch 2048 | 해당 LM 설정은 undercomplete. 현재 overcomplete VG-SAE와 동일 setting으로 재검증 필요 |
| VSC / learned-threshold VSC / SVAE | VSC: synthetic, FashionMNIST, UCI HAR; thresholded VSC: image patches, FashionMNIST, CelebA; SVAE: natural image patches | 이번 읽기에서 전체 compute budget 미확인 | 넓은 방법 선행으로 반드시 인용; 모든 이미지 방법을 대규모 LM baseline으로 이식할 필요는 없음 |

각 행의 근거는 위 방법 표의 해당 본문·부록이다. Gated Appendix D, JumpReLU §5, TopK Appendix B, BatchTopK Appendix, Probabilistic TopK Table 1, Analysis vSAE §3, vsPAIR Appendix B, Entropy ELBO §4, VAEase Appendix D.3을 확인했다.

## VG 메커니즘으로 기대할 수 있는 이점과 입증 의무

다음은 논문에서 가져온 성공 결과가 아니라 **현재 목적함수의 수식적 해석과 검증 가설**이다. dictionary와 나머지 latent를 고정하고 $c_j=D_j^Tr_{-j}$, $\|D_j\|=1$로 놓으면 coordinate optimum은

\[
m_j=\sigma\{\beta(a_jc_j-a_j^2/2)-\gamma\},\qquad
a_j^*=[c_j]_+\quad(m_j>0).
\]

이는 entropy coefficient 1, fixed precision, 자유로운 per-example nonnegative amplitude의 조건이다. 실제 affine+softplus encoder가 이를 달성한다는 뜻은 아니다. 식의 VG 원리는 기존 연구에서 왔고 새로운 정리로 내세우지 않는다.

| 가능한 이점 | VG에서 오는 이유 | 반드시 넘어야 할 비교·반증 |
| --- | --- | --- |
| 약한 feature를 살리면서 큰 amplitude bias를 줄임 | 선택 log-odds는 residual 설명 이득에 의존하고, 선택 후 amplitude에 L1을 직접 가하지 않음 | Gated/JumpReLU/TopK도 shrinkage를 줄인다. 같은 hard L0의 support F1·amplitude bias·recovery에서 남는 이득 필요 |
| noise 증가에 따라 불확실한 support 선택 조절 | β가 residual gain의 evidence scale을 정하고 variance 항이 불확실한 큰 amplitude의 위험을 반영 | β는 모델 오차도 noise로 흡수할 수 있다. known noise, underfit noise, learned β를 분리하고 oracle 설정과 비교 |
| 입력별 다른 활성 수를 유연하게 배분 | 개별 Bernoulli gate라 per-example k가 고정되지 않음 | BatchTopK·JumpReLU·SoftSAE·VAEase가 이미 adaptive sparsity 제공. 동일 평균 hard L0에서 tail recovery와 fidelity 개선 필요 |
| 작은 데이터에서 dictionary 과적합 감소 | 정확한 support-averaged risk가 일부 불안정한 설명을 낮출 가능성 | 이는 아직 가설. 여러 N·noise·seed의 held-out recovery와 train-test gap 필요. prior만의 효과와 분리 |
| sampled variational SAE보다 안정적·저렴한 gradient | 선형 decoder 아래 support expectation을 analytic하게 계산 | entropy-ELBO도 analytic이다. ‘모든 VI보다 빠름’ 불가. 동일 family의 MC/relaxed estimator와 비용·gradient/수렴 비교 |
| support confidence를 제공 | m은 명시적인 variational inclusion parameter | learned dictionary의 식별성, input amplitude, factorization 오차가 남는다. confidence 순위와 calibrated probability를 구별; tiny known-D 검증은 보조 도구 |

## 동일한 VG-SAE 안의 개발 결정 seed 10개

최종 후보 순위·새 연구 주제 선정이 아니다. 최소한의 변경으로 무엇을 결정할지 명시했다. 병목 증거가 없는 module은 추가하지 않는다.

| ID | 개발 선택 | 최소 구현/절차 | VG-specific 예상 효과 | 결정용 증거와 중단 기준 |
| --- | --- | --- | --- | --- |
| L01 | **현 analytic VG를 완성된 기준 방법으로 고정** | 원 KL(τ=1), unit D, 학습·mean·sampled·hard 출력을 명시하는 canonical recipe | 부가 module 없이 support 위험을 적분하는 기존 핵심이 충분한지 판단 | 충분히 수렴한 matched-hard-L0 Gated/JumpReLU/BatchTopK frontier. 개선이 없으면 무엇을 바꿀지 L02–L10에서 선택 |
| L02 | **precision 추정의 학습 절차 정리** | learned global β, minibatch profile, 누적/EMA energy를 이용한 global update의 작은 비교; inference는 같은 encoder | noisy minibatch의 prior 대 reconstruction 균형 요동 감소 | β trajectory·batch-size 민감도·held-out recovery. 별도 smoothing 이득이 없으면 learned/profiled 중 단순한 절차 유지 |
| L03 | **VG evidence에 맞춘 gate 초기화 또는 학습 target** | 기존 random gate 초기화와, 초기 decoder projection의 squared residual gain으로 유도한 초기 bias/target을 비교 | gate가 magnitude 크기보다 noise-scaled 설명 이득을 학습하도록 시작 | 동일 FLOPs에서 early support 회복과 final hard quality. 최종 이득이 사라지면 initialization 팁으로만 기록 |
| L04 | **gate–amplitude 교대 학습** | 같은 free energy에서 amplitude/dictionary와 gate update를 짧게 교대; 신규 latent 없음 | 학습 초기에 a와 m이 서로 보상해 gate evidence가 흐려지는 문제를 완화할 가능성 | joint Adam과 equal-update/equal-time 비교. 이득이 단순 update 수면 채택하지 않음 |
| L05 | **원 목적함수로 돌아오는 continuation** | β/γ 또는 entropy warm-up 후 마지막 충분한 구간은 원 KL τ=1; schedule만 변경 | 초기 포화와 과도한 잡음 적합 사이에서 더 나은 optimum 탐색 | gate utilization·hard Pareto·seed 안정성. τ→0 hardening 결과는 다른 목적함수임을 별도 표시 |
| L06 | **hard sparse readout을 목적에 맞게 결정** | m>τ, budget-matched m ranking, 예측 설명 이득 score를 calibration split에서 비교; 필요하면 한정된 train-time readout alignment | confidence threshold와 reconstruction 목적의 차이를 줄여 실제 sparse inference 활용 | calibration/test 분리, 실제 hard L0·latency·recovery. dense ma 결과만 개선되면 배포 기여 주장 보류 |
| L07 | **prior의 sparsity budget 의미 정리** | γ sweep과 $\pi=\rho/L$ 같은 explicit prior-count parameterization을 비교, learned γ는 이후 선택 | width와 prior 비용의 관계를 명확히 해 해석 가능한 tuning 축 제공 | width grid에서 hard L0·recovery·calibration; fixed-count 자체를 신규 이론으로 주장하지 않으며 자동 sparsity 정답이라 부르지 않음 |
| L08 | **두 encoder head의 계산량·결합 방식 선택** | 현 independent head와 Gated식 tied/scaled head를 VG loss 아래 비교 | support/amplitude 분리를 유지하며 추가 parameter/compute 필요성 확인 | 동일 width 및 동일 FLOPs 양쪽 비교. 성능 차이가 encoder parameter 수에서만 나오면 기여를 좁힘 |
| L09 | **소량 데이터·잡음에서 쓰는 VG 학습 recipe** | architecture는 유지, N×noise×amplitude heterogeneity의 사전 고정 grid에서 β/prior tuning 절차 확정 | exact uncertainty penalty가 유용한 operating regime을 찾고 그 범위에서 방법을 완성 | clean synthetic recovery + noisy held-out test + multi-seed. pilot 승리 조건만 사후 선택하지 않음 |
| L10 | **한 pass encoder의 VG inference 근사 개선** | 기존 output을 teacher initialization으로 소수 coordinate update를 훈련 때만 계산하고 encoder에 distill; 초기 pilot에서 gap이 확인될 때만 사용 | VG stationary evidence rule을 배우되 사용 시 한 pass 유지 | amortization reference 대비 gap, 추가 train FLOPs, hard recovery. 개선이 generic depth/compute와 같으면 별도 VG 기여로 과장하지 않음 |

L03/L10은 비선형 gate 또는 teacher 연산을 도입할 수 있으므로, 현재 gate가 이미 충분하면 불필요하다. L07은 기존 prior-count 파일럿의 한정된 부정 결과를 포함해 평가한다. L06은 sparse action을 만드는 개발 경로이며, 그 자체를 별개의 posterior 진단 논문으로 전환하지 않는다.

## 선행연구 때문에 좁혀야 하는 표현

- “첫 variational/probabilistic SAE”, “최초 Bernoulli support SAE”, “최초 support/magnitude 분리”, “최초 analytic sparse coding”, “최초 adaptive sparsity”, “최초 learned observation noise”는 지지하지 않는다.
- 새 method 주장은 **구체적인 VG식 amortized sparse autoencoder와 재현 가능한 학습·희소 추론 절차**로 시작한다. 경험적 기여는 같은 sparsity·data·compute에서 확인한 범위만 쓴다.
- confidence, calibration, uncertainty, noise robustness, feature recovery는 서로 다른 평가 대상이다. 하나가 개선됐다고 다른 것을 얻었다고 쓰지 않는다.
- dropout column replication의 variance dilution과 width correction은 [Cavazza 외 AISTATS 2018](https://proceedings.mlr.press/v84/cavazza18a.html) 선행이 있다. 이는 개발상 주의점과 보조 stress test이며 이번 연구 목표를 바꾸는 이유가 아니다.
- [Sparse but Wrong](https://arxiv.org/abs/2508.16560), [SynthSAEBench](https://arxiv.org/abs/2602.14687), [Feature Hedging](https://arxiv.org/abs/2505.11756)는 reconstruction과 feature recovery를 분리할 근거다. 이를 위해 독립 calibration/test와 여러 seed를 사용한다.

## 검색 범위·한계

- 이전 `idea-stage/evidence/literature_sources.json`의 28개 canonical source를 재사용하고, 위 4개를 추가해 32개로 정리했다. inherited 항목 중 새로 본문을 읽은 source는 read level을 올렸다. 이전 overlap 메모가 현재 연구 목표를 정하지 않도록 별도 표시했다.
- 최신 검색 기준일은 2026-09-21이다. 최근 6개월(2026-03-21–2026-09-21)의 arXiv API에서 sparse autoencoder/sparse coding 및 variational/Bernoulli/probabilistic/spike-and-slab query를 실행했다. 17개 hit의 metadata는 `arxiv_targeted_search.json`에 저장했다. 이 query는 제목뿐 아니라 broad all 필드를 검색하므로 관련 없는 적용 논문도 포함한다. 신규 핵심 exact-match 논문은 확인하지 못했다.
- 2026-09-19의 broad 30-hit sweep을 함께 재사용했다. 최신 post가 두 query의 키워드 밖이거나 아직 index되지 않았을 수 있어 completeness는 보장하지 않는다.
- 최신 추가 hit 중 PADL(2606.22352), autointerpretability variance(2607.19386)는 abstract 수준의 인접 lead다. 해당 abstract만으로 세부 알고리즘 동일성·실험 공정성을 확정하지 않았다. 현재 핵심 source 32개에 편입하지 않았다.
- 기본 `papers/`·`literature/`에는 PDF가 없다. `refs/`의 기존 세 PDF와 앞선 읽기 기록을 사용했고, Zotero/Obsidian은 도구가 없어 검색하지 않았다. 외부 저장소/라이브러리에 파일을 쓰지 않았다.
- direct OpenReview 접근이 막혀 Probabilistic TopK의 최종 publication status와 코드 수준 구현은 남은 확인 사항이다. 새로 추가한 4개는 proceedings 또는 저자 원고의 본문으로 확인했다.
- 사용자가 지정한 ultra 검토는 same-family provisional이며, 최종 독립 신규성·비판 리뷰를 대체하지 않는다.
