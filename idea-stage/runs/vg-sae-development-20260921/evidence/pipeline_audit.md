# VG-SAE 개발을 위한 현재 파이프라인 감사

작성: 2026-09-21. 범위: 저장소의 현재 구현, 저장된 Stage 1/2 전체 곡선, Stage 3 artifact 상태. 이 감사는 새 실험을 실행하거나 소스·기존 결과를 수정하지 않았다. 수치는 `pipeline_evidence.json`에 원본 경로와 함께 추출했다.

**고정 연구 목표는 Variational Garrote에 기반한 새로운 SAE 개발이다.** 기존 구현은 그 연구의 출발점이다. 2026-09-19 메모리의 복제 진단 논문 선택, B04 우선순위, 새 모듈을 만들지 않는다는 제한은 이번 사용자 정정으로 대체됐다. 복제·calibration·posterior 진단은 VG-SAE의 설계와 검증에 쓰는 도구다. 이전 소규모 음성 결과로 VG-SAE 개발의 가치나 가능성을 부정하지 않는다.

## 1. 이미 구현된 VG 기여와 강점

`src/sae_model.py:VariationalGarroteSAE`와 `docs/methods_synthetic_sparse_coding_vg_sae.tex`에는 이미 다음 방법이 구현·정식화되어 있다.

- 원래 scalar regression의 전역 selector를 **샘플별 support selector와 vector decoder dictionary**로 옮겼다. Gate와 amplitude를 각각 입력에서 amortize하며 `m=sigmoid(g(x-b))`, `a=softplus(r(x-b))`, `h=m*a`를 사용한다.
- Mean-field Bernoulli 기대 재구성 오차를 Monte Carlo 없이 계산한다. 에너지는 `0.5||x-b-D(ma)||² + 0.5 sum_j m_j(1-m_j)a_j²||d_j||²`다. Decoder norm을 포함하므로 단위 norm 가정이 꺼져도 식의 의미가 유지된다.
- 정규화된 support prior `pi=sigmoid(-gamma)`와 entropy를 결합한다. Unit entropy에서 `gamma*sum(m)+L*softplus(-gamma)-sum(H(m))`는 Bernoulli KL이다. 음의 gamma도 합법적인 dense prior다.
- Gaussian precision은 minibatch profiling 또는 learned `log_beta` 중 선택한다. Profiled mode는 재구성과 선택 비용의 상대 scale을 에너지로 정하며, learned mode는 full Gaussian NLL을 최적화한다.
- Unit-norm decoder, tangent gradient, amplitude/decoder transpose 초기화, stable-logit entropy를 갖췄다. 학습 중 encoder와 decoder는 서로 다른 parameter다.
- `src/saelens_vg.py`는 공식 SAELens 등록·훈련·저장·추론 인터페이스를 지원한다. 학습 mean code와 공개 hard code를 로그에서 분리한다. 일반 inference hook과 activation normalization folding도 구현되어 있다.

이 목록은 **현재 구현된 방법적 기여**를 설명하며, 각 구성요소의 최초 발명이나 문헌상 novelty 판정을 뜻하지 않는다. 조합만으로 독창성을 확정할 수도, 확률적 SAE 문헌이 있다는 이유만으로 이 개발을 폐기할 수도 없다. 문헌 검토와 다음 실험이 검증할 것은 이 VG 구성에서 어떤 설계가 필요한지, 어떤 이익을 재현할 수 있는지다.

실험 기반도 유용하다. Stage 1은 true dictionary/support/amplitude와 직사각형 Hungarian matching을 갖고, Stage 2는 고정 pretrained 생성기와 streaming evaluation, Stage 3는 pinned real activations와 CE/KL intervention 평가를 갖췄다. 공식 Standard/L1, TopK, BatchTopK, JumpReLU, Gated 구현을 직접 사용하는 점은 baseline 오류를 줄이는 강점이다. 관련 검증 파일은 `tests/test_sae_components.py`, `tests/test_vg_sae_beta.py`, `tests/test_saelens_vg.py`, `tests/test_sae_baselines_saelens.py` 등에 존재한다. 이번 감사에서 이 테스트를 새로 실행하지는 않았다.

## 2. Stage 1: 전체 곡선이 주는 긍정·부정 근거

기준 root:

- `outputs/runs/stage1_beta_profiled_din128_gt1024_sae1024_sd001_seed0/`
- `outputs/runs/stage1_beta_learned_din128_gt1024_sae1024_sd001_seed0/`
- amplitude와 frequency factorial 결과: `outputs/runs/stage1_ablation23_factorial_analysis/`

각 precision mode에 exponential/constant/uniform amplitude × frequency skew 0.5/0.0의 6개 조건이 있다. 12개 root 각각 273개의 completed control row를 갖고 있다. 한 조건의 controls는 VG 33, L1 16, TopK 128, BatchTopK 41, JumpReLU 32, Gated 23개다. **3,276개 결과 행은 3,276개의 독립 반복이 아니다. 전부 seed 0이며 beta와 무관한 baseline 결과도 양쪽 표에 반복된다.**

공통 기본값은 `d=128`, ground truth width=SAE width=1024, support density .01, train 8196/test 1024, batch 128, 1000 steps, LR .01, no noise/coherence다. amplitude 법칙은 active second moment를 맞춘다. Equal second moment가 mean/cross-term까지 같다는 뜻은 아니다.

`summary/last/final_metrics.csv`의 `hard_generalization_error`를 control별로 최소화한 **탐색적** 요약은 다음과 같다. 이것은 같은 held-out 곡선에서 고른 값이므로 별도 validation으로 고정된 승리 판정이 아니다.

| amplitude / frequency | profiled VG 최저 hard latent error | learned VG | 최선 baseline |
|---|---:|---:|---:|
| exponential / skew .5 | .6950 | .7122 | .7332 TopK |
| exponential / uniform | .7314 | .7637 | .7943 L1 |
| constant / skew .5 | .8541 | .8631 | .8448 L1 |
| uniform / skew .5 | .8374 | .8351 | .8193 L1 |
| constant / uniform | .9750 | .9753 | .9773 L1 |
| uniform / uniform | .9504 | .9478 | .9478 L1 |

Exponential amplitude 두 조건은 후속 검증 가치가 있는 양성 신호다. Constant/uniform에서 거의 모든 방법의 latent error가 높은 조건도 있어, 그 조건의 실패를 곧바로 VG만의 실패로 해석하면 안 된다. 꼬리가 긴 amplitude에서의 강점은 후보 설명이지 확인된 causal mechanism이 아니다.

기본 exponential/skew 조건의 profiled VG:

| gamma | hard L0 | expected L0 | hard EV | mean-code EV | decoder cosine | hard F1 | hard latent error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3.0 | 3.388 | 51.732 | .4780 | .6176 | .7513 | .3260 | .6950 |
| 2.5 | 3.914 | 82.338 | .4673 | .6354 | .7603 | .3332 | .6961 |
| 1.25 | 10.467 | 245.361 | .2845 | .6641 | .5447 | .2493 | .8585 |

True empirical L0는 10.447이다. 따라서 현재 결과에서 true L0와 동일하게 맞추는 것 자체는 최적 모델 선택 기준이 아니다. 동시에 gamma 2.5–3의 양성 신호는 버릴 이유가 없다. 이 구간을 새 seed와 validation에서 확인하는 것이 넓은 control sweep을 다시 하는 것보다 저렴하다.

학습 곡선 `summary/training_curves.csv`의 profiled gamma 3은 step 500→999에서 train mean MSE .05375→.04238, loss -155.39→-167.94로 계속 개선된다. Test mean MSE는 .06411, hard MSE는 .08752다. 1,000 steps가 최적화 수렴을 보장하지 않고, train/test gap과 mean/hard gap도 동시에 존재한다. 더 긴 훈련, LR 변경, sparse 추론 변경을 한 번에 넣으면 원인을 분리할 수 없다.

Stage 1 비교시 `explained_variance`는 VG mean code이고, 실제 sparse 비교에는 `hard_explained_variance`가 필요하다. L1은 train-fitted GMM mask를 쓰며 `hard_*`에서는 그 mask로 ReLU code를 자른다. Stage 2의 L1 raw native firing과 의미가 다르므로 두 stage의 F1/dead fraction을 동일 정의처럼 합치면 안 된다.

## 3. Stage 2: 강한 feature 신호와 hard-code 병목이 공존

정식 200M 결과 원본:

- learned VG: `outputs/runs/stage2_synthsaebench16k_l0calibrated_sae4096_train200m_test25m_beta_learned_seed0/summary/last/final_metrics.csv`
- profiled VG: 위 root의 `beta_profiled` 버전
- 공식 baseline 35개: `outputs/runs/stage2_synthsaebench16k_l0calibrated_sae4096_train200m_test25m_seed0/summary/last/final_metrics.csv`
- 넓은 104개 곡선: `outputs/runs/stage2_synthsaebench16k_rho1e-3to1e-1_logspaced_4methods_sae4096_iter100k_beta_learned_seed0/summary/last/final_metrics.csv`

데이터는 pinned `decoderesearch/synth-sae-bench-16k-v1`, revision `b2efd8b919ae46d6d487c73d46db5ee52813621d`, `d=768`, true width=16384, SAE width=4096다. 200M root의 정확한 sample 수는 train=199,999,488/eval=24,999,936이고 seed 0이다. 고정 artifact의 `scale_children_by_parent=false`는 논문 생성 스크립트의 설명과 다르며 이미 reproduction notes에 기록되어 있다.

다음은 보간하지 않은 가까운 hard L0 값이다.

| 방법/설정 | hard L0 | hard EV | MCC | macro F1 | expected L0 | mean EV |
|---|---:|---:|---:|---:|---:|---:|
| VG learned gamma 6 | 15.273 | .7747 | .6204 | .7249 | 18.317 | .7798 |
| VG learned gamma 2.8 | 19.220 | .7735 | .7173 | .7798 | 253.883 | .8337 |
| VG profiled gamma 2.84 | 19.344 | .7789 | .7240 | .7903 | 242.090 | .8337 |
| L1 coeff 2.42 | 19.937 | .7664 | .5821 | .5621 | 19.937 | — |
| TopK k 20 | 19.981 | .7904 | .6755 | .6253 | 19.981 | — |
| BatchTopK k 20 | 20.015 | .7869 | .7085 | .7534 | 20.015 | — |
| JumpReLU coeff 1.16 | 19.939 | .7932 | .7357 | .8251 | 19.939 | — |
| Gated coeff 2.17 | 20.177 | .7913 | .6265 | .6063 | 20.177 | — |

VG는 이 구간에서 L1·Gated보다 높은 MCC와 F1을 내며, TopK보다 높은 MCC/F1을 보인다. JumpReLU는 현재 가장 중요한 강한 대조군이다. 이 한 표로 우위를 주장하지 않되, VG의 좋은 feature 구조와 충분하지 않은 sparse reconstruction을 모두 인정할 수 있다.

learned gamma 1.99에서는 hard L0 30.254/EV .7750/MCC .7015/F1 .5546, expected L0 543.17/mean EV .8610이다. gamma 1.63에서는 hard L0 45.663/EV .7831, expected L0 744.68/mean EV .8778이다. gamma 6에서 EV gap은 .0052지만 gamma 1.63–2.8에서는 .060–.099다. 문제의 크기가 gamma에 강하게 의존한다.

104-point root의 더 넓은 곡선도 확인했다. VG gamma 3.81은 hard L0 16.02, EV .7703, MCC .6752, F1 .8044다. gamma 2.58은 L0 20.97, MCC .7134다. 매우 sparse한 gamma 18.8–21에서는 dead fraction 약 .916–.929와 낮은 MCC가 나타난다. Dense gamma .834는 L0 614/EV .9612이지만 MCC .2381/F1 .0463이다. 따라서 현재 VG의 제어 범위는 충분히 넓고, 무조건 sparse/dense 끝점으로 이동하는 것은 해결책이 아니다. 강한 support 신호가 있는 중간 영역을 개선해야 한다.

200M train history에서 learned gamma 1.99의 step 49,000→195,311은 expected L0 545.61→543.85, hard L0 29.66→30.16, mean MSE .14496→.14287이다. Mean/hard 의미 차이는 장기 훈련에도 남는다. 다만 gamma 6의 loss 640.68→627.23과 variance energy 2.036→1.411처럼 추가 훈련 이익도 있어 모든 run이 수렴했다고 단정할 수 없다. History는 해당 minibatch 관측값이며 endpoint validation 곡선은 아니다.

## 4. 해석과 평가의 정확한 경계

1. **세 코드/위험을 분리한다.** Posterior-mean reconstruction은 `D(ma)`다. Bernoulli stochastic reconstruction의 기대 squared risk는 mean residual에 variance correction을 더한다. 공개 sparse inference는 `D(a*1[m>tau])`다. `vg_expected_explained_variance`는 첫 번째의 EV이고 stochastic EV가 아니다. Expected L0와 hard L0가 크다고 복제나 calibration 실패가 증명되는 것은 아니다.
2. **현재 확률은 모델 조건부 변분 확률이다.** Amplitude는 입력에 의존하는 deterministic encoder 출력이고, mean-field·dictionary 오차·amortization 제한이 있다. 전체 생성 문제의 정확한 support posterior 혹은 실제 LM feature의 calibrated probability라는 주장은 별도 근거가 필요하다. Tiny exact-posterior 실험은 특정 조건부 문제에만 적용한다.
3. **Stage 2의 기존 eval은 calibration이다.** `README.md`와 `REPRODUCTION_NOTES.md`가 명시하듯 stream seed 40000을 coefficient 선택과 최종 표에 재사용했다. `calibration_seed=20000` 필드는 activation scaler 추정용이며 별도 control-selection holdout을 뜻하지 않는다. 기존 결과를 폐기하지 말고 exploratory/calibration으로 유지하고, 고정한 설정을 새 stream에서 평가한다.
4. **Stage 2 macro classifier와 MCC는 다른 matching이다.** `src/synthsaebench_eval.py`는 MCC에 Hungarian assignment를 쓰고, classification은 latent별 best absolute-cosine true feature를 쓴다. Uniqueness도 함께 보고한다. Width4096이 true width16384를 모두 복구할 수 없고, classification은 unmatched true feature를 Stage 1 union 방식으로 벌하지 않는다.
5. **현재 Stage 2 AP/AUC는 probability ranking을 평가하지 않는다.** `support_average_precision`는 binary firing의 `precision*recall+prevalence*(1-recall)`이고 AUC는 `.5*(recall+specificity)`다. `m`의 calibration, Brier, log score, continuous AUPRC와 다르다. 추가한다면 명확한 새 필드가 필요하다. Probability quantile도 전체 25M이 아닌 첫 80-row preview에서 계산된다.
6. **Stage 1 dead fraction과 Stage 2 dead fraction은 다르다.** 전자는 train mean activity threshold, 후자는 evaluation stream에서 hard firing 한 번도 없음이다. 전 stage의 단일 dead-neuron 지표로 합치지 않는다.
7. **Scale·loss 값은 방법 사이의 성능지표가 아니다.** VG는 Gaussian precision, prior와 entropy를 포함한다. Baseline objective, profiled/learned objective 절대값을 직접 비교하거나 다른 gamma의 raw loss만으로 모델을 선택하지 않는다.

## 5. 현재 최적화 선택과 개선 가능한 부분

| 부분 | 현재 근거 | 저비용 개선 후보 |
|---|---|---|
| Stage 1 | `src/sae_train.py`: VG AdamW(weight decay 0), LR .01 constant, clip 1, tangent projection+renorm. Baseline은 공식 Adam/SAETrainer. Full-train objective로 best 저장 | Original objective 그대로 LR .003/.001, 더 긴 budget, 마지막 구간 decay를 분리해서 확인. Train/validation 모두 기록 |
| Stage 2 | `runs/run_SynthSAEBench_sweep.py:_trainer`: Adam beta1=.9/beta2=.999, LR 3e-4 constant, LR warmup 0, BF16 autocast. Final-third decay 옵션은 존재 | Decay와 gate/prior continuation을 작은 좁은 grid에서 확인. Whole-model clipping 때문에 beta/gate/amplitude 중 어떤 gradient가 지배하는지 먼저 로그 |
| Gate 시작 | `gate_bias_init=-2`, random gate weights; amplitude만 decoder transpose 초기화 | Prior-aware gate bias `-gamma`, residual-based gate 초기화, 또는 gamma warmup을 별도 ablation. 모두 유리하다고 미리 가정하지 않음 |
| VG coefficient schedule | Adapter에 `lambda_warm_up_steps`가 있지만 Stage 2 VG builder는 기본 0. L1/Gated는 전체 steps의 1/3 warmup | VG continuation과 baseline의 정식 권장 schedule을 모두 인정하되, 동일 budget에서 결과 비교 |
| Precision | Stage 2 learned와 profiled 둘 다 비슷한 gap. Stage 1에서는 profiled가 exponential 조건에서 더 좋음 | Beta-mode 교체만을 새 방법의 핵심으로 삼지 않음. Learned beta의 group LR/EMA 또는 profiling 안정성은 optimizer ablation |
| Mixed precision | Dense GEMM뿐 아니라 gate/entropy 계산도 autocast context 안. BF16 posterior 평균 로그의 반올림 흔적은 실제 history에 있음 | 동일 minibatch/weights에서 FP32 loss/gradient와 비교하는 smoke audit. 차이가 중요할 때만 probability·entropy reduction을 FP32로 유지. 오류가 입증된 상태는 아님 |
| 추론 rule | VG는 항상 probability threshold .5와 conditional amplitude `a`를 사용 | Budget와 decision loss에 맞는 sparse code를 비교. Threshold=.5가 모든 reconstruction budget의 최적 rule이라는 근거는 없음 |
| Compute fairness | VG는 gate+amplitude 두 encoder, baseline 다수는 한 encoder | Equal width/token 결과에 train seconds·parameter 수·inference FLOPs를 함께 기록. Equal walltime 부가 비교로 비용을 숨기지 않음 |

현재 Stage 2 200M VG 7개 run의 기록된 train time 중앙값은 learned 4,899.6초, profiled 4,938.9초다. 장비 동시 실행 조건과 worker scheduling이 섞인 값이므로 미래 실행시간 보장은 아니다. Stage 1에서 빠른 판별 후 좁은 20M/50M Stage 2로 넘어가는 순서가 타당하다. A6000 4개는 사용자 제공 현재 자원이며 이번 감사는 GPU를 점유하지 않았다.

## 6. Stage 3는 새 대형 sweep보다 기존 checkpoint 평가가 우선

저장된 `train_status.json` 기준:

| root target | complete | queued | running 라벨 | 완료 내용 |
|---|---:|---:|---:|---|
| Gemma-2-2B L5 | 54 | 108 | 0 | VG 15, L1 15, BatchTopK 15, JumpReLU 9; 완료 seed는 0 |
| Gemma-2-2B L12 | 16 | 64 | 4 | BatchTopK 16 |
| Llama-3.2-1B L7 | 0 | 144 | 0 | 없음 |

세 root는 `outputs/runs/stage3_real_activation_*`이며 정확한 경로는 JSON inventory에 있다. L5에서 54개, L12에서 20개의 checkpoint 파일이 실제 존재한다. L12의 20개에는 interrupted/running artifact도 포함되므로 모두 완료 checkpoint로 세지 않는다. **오래된 running 라벨은 현재 실행 중임을 뜻하지 않는다.**

세 root 모두 최종 evaluation metric 파일/CSV가 없다. 따라서 현재 real-activation 성능 결론을 낼 수 없다. 그러나 L5에는 VG와 강한 baseline의 완료 checkpoint가 충분하다. 일부 checkpoint의 config/fingerprint를 확인하고 소규모 새 validation activation 범위에서 먼저 hard/mean reconstruction, expected stochastic risk, achieved L0, CE/KL을 평가하면, 새 390-run 학습 없이 실제 feature 환경의 방향성을 얻을 수 있다. 이후 한 model/layer의 2–3 operating point와 추가 seed로 확장한다.

## 7. 가장 판별력 높은 저비용 파일럿 세 개

다음은 **제안이며 실행 결과가 아니다.** 각 파일럿이 VG-SAE의 어떤 설계 결정을 바꾸는지 명시한다. 새 데이터 seed/분할/선택 규칙을 실행 전에 고정하며, 기존 40000 stream을 새 test라 부르지 않는다.

### P1. 기존 VG의 최적화 상한을 먼저 확인

**질문:** 현재 encoder/objective가 충분한데 1000-step/high-LR 학습이 이익을 가리는가, 또는 hard-code gap이 잘 최적화한 뒤에도 남는가?

- Stage 1 exponential/skew .5, d128/L1024, train8192/validation2048/test4096으로 새 split을 만든다. Seeds 1,2,3. Gamma는 양성 anchor 3과 true-L0 근처 1.25만 사용한다.
- 4 arms: 기존 LR .01; LR .003; LR .003 + 마지막 1/3 decay; LR .003 + gamma를 첫 10% 동안 0에서 target으로 warmup. 총 24 runs, 각각 4,000 steps. 동일 initialization와 batch sequence를 공유하고 step 1000도 저장한다. Warmup 최종 objective는 원래 VG와 동일하다.
- Beta는 첫 판별에서 profiled로 고정한다. Learned beta, entropy weight, architecture를 동시에 바꾸지 않는다. Same-batch FP32/BF16 gradient smoke는 별도 작은 수치 확인이며, 틀어진다면 개선 run 전에 원인을 기록한다.
- 매 250 steps에 validation mean MSE, stochastic risk, hard MSE/L0/F1, beta, gate/amp/decoder gradient norm, clipping frequency를 기록한다. Test는 arm 선택 후 한 번 사용한다. Threshold .5는 일단 고정하여 optimization과 decision 변경을 분리한다.
- **진행 기준:** 세 seed 중 적어도 두 seed에서 검증 hard error가 기존 동일-step anchor 대비 5% 이상 개선되고 MCC/F1을 크게 희생하지 않으면 training protocol을 갱신한다. 기준 수치는 탐색 gate이며 statistical significance가 아니다. Mean risk만 개선되면 P3로, optimizer를 바꿔도 inference residual이 남으면 P2로 간다.

### P2. VG 조건부 최적화로 encoder를 개선

**질문:** 선형 amortized gate/amplitude가 VG 자체의 좋은 해를 충분히 찾지 못하는가? 그렇다면 짧은 VG refinement 또는 teacher distillation이 새 SAE의 핵심 구조가 될 수 있는가?

고정된 a,D,b,beta와 unit entropy에서 `r_-j=x-b-sum_(k!=j)m_k*a_k*d_k`라 두면 coordinate optimum은

```
logit(m_j) = beta * (a_j * d_j^T r_-j - 0.5 * a_j^2 * ||d_j||^2) - gamma.
```

고정 m과 다른 amplitude에 대해서는 활성 branch의 최적 amplitude가

```
a_j = max(0, d_j^T r_-j / ||d_j||^2)
```

다. 첫 식은 residual evidence를 gate에 직접 공급하는 구체적인 VG 기반 설계 근거다. 이 식 자체의 문헌상 novelty를 주장하지 않는다. `m_j=0`에서 amplitude는 식별되지 않는다. Sequential coordinate update의 조건부 objective 감소와 parallel simultaneous update의 동작은 다르며, 후자에 감소 보장을 자동으로 붙이지 않는다. Profiled mode에서는 fixed-beta 조건부 update 뒤 전체 batch energy로 beta를 재최적화하는 alternation의 목적함수를 따로 확인한다.

- P1에서 선택한 3-seed VG checkpoint를 frozen dictionary로 사용한다. 먼저 작은 held-out minibatch에서 기존 encoder, 1/3/10 sweep의 gate-only coordinate refinement, gate+amplitude refinement를 비교한다. D나 true support를 inference에 제공하지 않는다.
- 동일 VG free energy, mean/stochastic/hard errors, support F1, L0, walltime를 기록한다. 작은 exact-enumeration 조건부 문제로 식과 update 방향을 확인하되 이를 large learned-model posterior calibration으로 확대하지 않는다.
- Gate refinement만 효과가 크면 residual-conditioned amortized correction 1 step 또는 refined probability teacher를 기존 gate에 distill하는 2k-step branch를 구현한다. Amplitude update에서만 이익이 나오면 amplitude head 개선을 우선한다. 고정 D에서조차 이익이 없으면 dictionary learning/decision 문제로 초점을 옮긴다.
- **진행 기준:** validation stochastic risk 5% 이상 감소와 hard matched-L0 error/F1의 일관된 개선이 함께 나타날 때 short-refinement VG를 다음 방법 후보로 올린다. Objective만 낮고 feature metric이 나빠지면 채택하지 않는다. 추가 inference cost와 baseline에 동일 refinement를 허용한 경우를 별도 보고한다.

### P3. 확률을 실제 sparse code로 바꾸는 규칙을 분리 검증

**질문:** 현재 gate/dictionary의 좋은 정보가 `a*1[m>.5]`라는 deployment rule에서 손실되는가? 단순한 VG-consistent decision으로 hard-code 품질을 회복할 수 있는가?

- 기존 Stage 2 learned gamma 6/2.8/1.99의 3 checkpoint를 사용한다. 별도 새 calibration stream 131,072 samples와 untouched test stream 524,288 samples를 고정한다. Train stream/기존 40000 stream과 분리한다. 추가 seed checkpoint는 P1/P2의 선정 후에만 훈련한다.
- Fixed dictionary, fixed encoder 아래에서 (a) 현재 hard amplitude, (b) `m*a`를 같은 mask에서 유지한 sparse mean code, (c) 같은 support에서 nonnegative amplitude만 짧게 refit하는 code를 비교한다. Mask는 .5 기본값과 calibration으로 선택한 probability threshold를 분리한다. 별도 후보로 mean-contribution magnitude `m*a*||d||` ranking을 비교할 수 있으나 이를 support posterior threshold라고 부르지 않는다.
- Target achieved L0를 15/20/30으로 고정하고 calibration에서 선택한 rule/threshold를 test에 그대로 적용한다. Test의 achieved L0가 달라지면 사실대로 기록한다. Amplitude refit은 baseline 동일 support에도 적용한 별도 공정성 대조를 둔다. 같은 threshold의 code 변경과 mask 변경을 합치지 않는다.
- Primary metrics는 hard reconstruction/CE 또는 synthetic latent error와 F1이며, mean risk·Bernoulli stochastic risk·probability ranking/Brier는 보조다. Brier는 식별 가능한 matching/coverage 조건을 명시하고 arbitrary duplicate와 unmatched feature를 명확히 처리한다.
- **진행 기준:** matched-L0 hard EV가 최소 .02 개선되고 support F1 하락이 .02 이내인 안정된 rule을 후속 학습에 연결한다. 예컨대 sparse mean retention이 이기면 amplitude/gate 학습과 deployment decision을 통일하는 VG 목적함수/decoder 구조를 개발한다. 단순 refit만 모든 방법을 똑같이 개선하면 VG 고유의 방법 이득으로 주장하지 않는다.
- 추가 training 없이 이미 완료된 Gemma L5 VG와 JumpReLU/BatchTopK checkpoint 각각 1–2개에 같은 구분을 적용해 transfer 방향을 본다. 기존 Stage 3 전체 sweep 재개는 이 파일럿의 전제가 아니다.

## 8. 다음 방법 개발로 이어지는 판단

현재 근거로 가장 생산적인 출발점은 **좋은 feature 구조가 관측된 gamma 영역을 보존하면서, VG free energy의 조건부 추론과 amortized sparse deployment를 가깝게 만드는 것**이다. 최적화 개선은 기준선을 세우고, residual-conditioned VG 업데이트는 구조적 개선 후보를 주며, probability-to-code 파일럿은 사용자에게 실제로 쓰이는 SAE code의 목적을 정한다.

새 결과의 최종 비교에는 최소 JumpReLU와 BatchTopK를 포함한다. 단순 variance 제거·entropy 제거는 full probabilistic objective와 다른 ablation으로 분명히 표기한다. P1–P3 결과가 나온 후 필요한 항목만 확장한다. 기존 Stage 1/2의 폭넓은 결과를 다시 전부 실행하거나 복제 진단을 독립 논문 목표로 전환할 필요는 없다.

## 감사 재현성

- 사용한 원본: root `AGENTS.md`, `.agents/project-memory.md`, `README.md`, `REPRODUCTION_NOTES.md`, 위 methods TeX, `src/sae_*`, `src/saelens_vg.py`, `src/synthsaebench_*`, 관련 `runs/` runner, 저장된 configs/status/CSV.
- 원본 수치는 표에 명시한 CSV column을 그대로 사용했다. 일부 표는 출력 precision만 반올림했다. Optimum은 재선택한 독립 test 결과가 아니다.
- `pipeline_evidence.json`에는 모든 Stage root 상태 inventory, Stage 1 12조건의 VG 전체 curve/방법별 탐색 optimum, Stage 2 13개 evaluated non-smoke root의 metric row를 보존했다.
- 변경은 이 보고서와 companion JSON만이다. 학습·평가 실행, 소스 변경, 기존 artifact 삭제, commit/push는 수행하지 않았다.
