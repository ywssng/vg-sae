# Experiment Plan — VG-SAE 방법 개발 후속 계획

작성일: 2026-09-21. **이 문서는 향후 실행 계획이며, 아래 B1–B3는 아직 실행하지 않았다.**
P1/P2/P3/P3b의 완료된 탐색 결과를 새 확증 실험의 결과로 세지 않는다.

## 고정 연구 목표와 주장

Variational Garrote 기반 새로운 SAE를 개발한다. 현재의 Bernoulli support,
입력별 point amplitude, linear dictionary와 analytic support-averaged quadratic
risk를 주방법으로 유지하고, 실제 sparse code의 회복 성능과 목적함수의 역할을
검증한다. 진단은 방법 개발의 도구다.

| 주장 | 검증할 범위 | 최소 근거 | 블록 |
| --- | --- | --- | --- |
| C1 | 사전 지정 활성 상한에서 VG-SAE가 강한 baseline 대비 유용한 coefficient/feature recovery와 input fidelity의 절충을 제공한다. | 원래 크기의 새 paired worlds, 같은 최대 탐색 기회, 독립 calibration/test, 실제 L0와 EV·비용을 함께 공개 | B1; B3는 적용 범위 확인 |
| C2 | 현재 VG parameterization에서 full objective의 variance/entropy 항이 support–amplitude 결합과 hard-code 성능을 유지하는 데 기여한다. | 같은 beta mode·architecture·탐색 기회의 2×2 deletion, m/a scale과 mean/stochastic/hard 지표 | B2 |

C2는 일반 deterministic SAE보다 확률적 불확실성 처리가 우수하다는 주장이나,
추가 encoder capacity의 영향을 배제했다는 주장이 아니다. 이 계획에는 별도
capacity-matched deterministic control이 없으므로 그 문구를 사용하지 않는다.
Analytic support expectation은 구현 속성으로 기술하고 MC 방법보다 빠르다고
주장하지 않는다. LLM은 B3의 activation 제공 모델이며 teacher/RL 구성요소는 없다.

## 현재 코드에서 출발하는 선택

- 기존 Stage 1 저장 설정은 d=128, true/SAE width=1,024, n_train=8,196,
  p=0.01, skew=0.5, noise=0, coherence=0, amplitude scale=1,
  batch=128, LR=0.01, 1,000 updates다. Constant amplitude는 현 generator의
  RMS matching 규칙을 그대로 사용하며 임의로 값 1로 바꾸지 않는다.
- 기존 Stage 1은 방법별 grid 수가 다르며 `best.pt`는 full-training objective로
  선택한다. 따라서 그 runner를 그대로 실행하는 것으로 아래 공정 비교가 구현되지는 않는다.
- P2의 작은 조건에서는 EV가 좋아지는 동안 latent error/F1가 악화했다.
  최종 loss plateau나 final checkpoint를 primary recovery 선택 기준으로 삼지 않는다.
- 기존 공식 SAELens baseline의 optimizer, decoder 처리, auxiliary loss, threshold
  상태, sparsity schedule을 유지하고 resolved config로 기록한다. Stage 1의
  constant LR를 유지한다. 특정 baseline의 공식 세부 설정을 VG에 맞춰 삭제하지 않는다.
- P2의 moment beta 초기화와 P3/P3b의 보정 solver는 기본 방법에 추가하지 않는다.

## B1 — 원래 크기의 공정한 recovery 확인

**우선순위: MUST.** Main table과 cap-constrained tradeoff figure를 만든다.

### 데이터, seed와 분리

조건은 exponential primary, constant scope control 두 개다. 각 조건에서
world `100, 101, 102`를 쓴다. World 간에는 dictionary, train/cal/test sample,
model initialization과 minibatch order가 함께 달라진다. 이는 세 개의 joint
world 반복이며 data seed와 optimizer seed를 독립 요인으로 분해한 실험이 아니다.

| 항목 | 사전 지정 |
| --- | --- |
| 공통 구조 | d=128, true width=1,024, SAE width=1,024, p=0.01, skew=0.5, coherence=0, noise=0 |
| Dictionary RNG | `world`; 두 amplitude 조건에서 같은 dictionary 재사용 |
| Train / calibration / test sample RNG | `1_000_000 + world`, `2_000_000 + world`, `3_000_000 + world` |
| 표본 수 | train 8,196; calibration 4,096; test 8,192 |
| 초기화 RNG | 기존 방법 순서의 `method_index`를 써서 `100_000 + 1_000 * world + method_index` |
| Minibatch RNG | `4_000_000 + world`; 같은 데이터 순서를 모든 방법·trial에 제공 |
| Trial 사이 | 같은 방법/world의 초기 weights와 batch sequence를 재사용; beta scalar만 해당 recipe대로 설정 |

Sample RNG는 dictionary를 다시 뽑지 않는 별도 sampler에 전달한다. 두 amplitude의
support draws를 같게 만들고 amplitude draws만 바꾸면 범위 비교가 더 직접적이다.
현 generator는 dictionary와 sample 생성을 한 RNG에 묶으므로 이 분리는 후속 runner
구현 사항이다. Train 통계만 normalization, bias initialization, L1 GMM fitting에 쓴다.
Cal/test의 분포 파라미터는 train과 같고 표본은 독립이다. Test data는 selection JSON을
동결한 뒤 생성한다.

### 정확한 후보 grid와 최대 기회

모든 방법은 **condition/world당 최대 12 training trials × 4 checkpoints = 48개
primary selection candidates**를 가진다. Trial 하나는 한 control과 한 training
recipe의 조합이다. VG의 beta mode 탐색도 이 12개 안에서 지불한다.

| 방법 | Control grid, 6개 | Recipe grid | Training trials |
| --- | --- | --- | ---: |
| VG-SAE | gamma `[1, 2, 3, 4, 6, 8]` | `(profiled beta, LR 0.01)`; `(learned beta, beta-init 1, LR 0.003)` | 12 |
| L1/ReLU | coefficient `[0.05, 0.2, 0.5, 1, 2, 4.5]` | LR `[0.003, 0.01]` | 12 |
| TopK | K `[2, 4, 6, 8, 12, 16]` | LR `[0.003, 0.01]` | 12 |
| BatchTopK | K `[2, 4, 6, 8, 12, 16]` | LR `[0.003, 0.01]` | 12 |
| JumpReLU | L0 coefficient `[0.05, 0.2, 0.5, 1, 2, 4]` | LR `[0.003, 0.01]` | 12 |
| Gated | L1 coefficient `[0.05, 0.2, 0.5, 1, 2, 5]` | LR `[0.003, 0.01]` | 12 |

Control 값은 기존 Stage 1 grids에서 가져온 좁은 범위다. LR=0.003은 P2에서
사용한 안정적인 대안 후보이며 더 좋은 기본값으로 확증된 것은 아니다. 두 VG recipe는
beta와 LR를 함께 바꾸므로 B1만으로 beta mode의 인과 효과를 말하지 않는다.

공통 최대 training budget은 **6,000 optimizer updates × batch 128 = 768,000
sample presentations/trial**이다. 고유 train sample은 8,196개다. 저장 시점은
**1,000 / 2,000 / 4,000 / 6,000 완료 updates**로 고정한다. 로그의 zero-based
step와 혼동하지 않도록 checkpoint에 `completed_updates`를 저장한다.

이 선택은 original 1,000-update 후보를 보존하면서 더 늦은 회복 성능 악화를 직접
확인한다. 특정 방법의 곡선을 본 뒤 추가 checkpoint나 longer training을 주지 않는다.
실패 trial도 12개의 기회를 소비한다. Infrastructure 장애는 같은 config/RNG의
재시작만 허용하며 변경된 LR나 control의 무료 재시도로 바꾸지 않는다. OOM 등으로
완료 기회가 달라지면 같은 최대 기회와 실제 완료 수를 모두 보고한다.

**B1 합계: 6 methods × 2 amplitudes × 3 worlds × 12 = 432 trainings,
1,728 saved candidate checkpoints, 최대 331,776,000 sample presentations.**

### Readout, matching과 calibration 선택

- VG primary hard code는 `a * 1[m >= 0.5]`다. Threshold grid를 추가하지 않는다.
  Posterior mean `m*a`와 analytic stochastic reconstruction risk를 별도 기록한다.
- L1 primary는 **native ReLU code**, support는 `h > 0`이다. P1의 train-GMM
  thresholded code는 같은 선택된 L1 checkpoint에서만 secondary로 평가한다.
  GMM은 해당 trial의 train activations로 한 번 fit하고 cal/test에서 refit하지 않는다.
  GMM 결과는 primary control/checkpoint selection이나 non-VG reference 선택에 넣지 않는다.
- TopK/BatchTopK/JumpReLU/Gated는 공식 native inference code와 nonzero support다.
  BatchTopK inference threshold는 training state에서 가져오고 cal/test에 맞춰 바꾸지 않는다.
- Learned/true unit dictionary 사이 absolute cosine Hungarian matching과 sign alignment를
  checkpoint별 weights만으로 계산한다. Test coefficient/support로 alignment를 다시 fit하지 않는다.
  d_true=d_SAE=1,024이므로 full bijection이다. 향후 unequal-width 실험에 이 점수를
  그대로 외삽하지 않는다.
- Primary metric은 `E_z = sqrt(sum ||z_hat_aligned-z||² / sum ||z||²)`다.
  Support F1는 전체 sample-feature TP/FP/FN으로 계산하는 micro F1다. Input EV,
  per-element MSE, decoder matched cosine, actual hard L0, dead fraction, 시간과
  parameter 수를 함께 낸다. `expected_ev` 등 legacy 이름은 실제 정의를 함께 표기한다.

각 method/condition/world/cap에서 **calibration actual mean hard L0 ≤ cap**인
후보 중 calibration `E_z`가 가장 작은 control+checkpoint를 선택한다. Primary cap은
8, secondary는 4와 16이다. 순서는 `E_z`, 더 낮은 hard reconstruction MSE, 더 낮은 hard L0,
사전 고정 candidate index(control/recipe/checkpoint 순서)로 고정한다. Feasible 후보가 없으면 `no feasible candidate`다.
새 threshold나 grid extension으로 채우지 않는다. Test L0가 cap을 넘더라도 threshold를
조정하지 않고 일반화된 L0와 초과량을 보고하며 해당 cap의 성공으로 세지 않는다.

이것은 true latent를 아는 synthetic benchmark의 **oracle-assisted model selection**이다.
실제 activation에서 그대로 사용할 tuning rule이라고 부르지 않는다.

각 world의 primary cap 8에서 다섯 non-VG 방법의 calibration-selected 결과 중
calibration `E_z` 최소 방법을 **reference R_world**로 동결한다. Test에서 비교 상대를
교체하지 않는다. 각 baseline과의 개별 차이도 전부 보고하므로 reference가 약한
방법으로 선택되는 경우를 숨기지 않는다. Cal로 선택한 각 checkpoint를 test에서 한 번
평가해 cap별 재사용하며, 최대 primary/secondary test rows는 108개다. 별도로 같은
L1 checkpoint의 GMM secondary row는 최대 18개다.

### B1 판단 규칙

Exponential cap 8의 세 paired 차이 `r_world = 1 - E_z(VG)/E_z(R_world)`를 전부
공개한다. 다음은 후속 개발 자원을 배정하는 **screening go 기준**이며 유의성 검정이 아니다.

1. 세 world 모두 양쪽에 feasible candidate가 있고 test mean L0도 cap 이내다.
2. 평균 상대 error 감소 ≥ 5%, 최소 2/3 worlds에서 error가 낮고, 어떤 world도
   상대 error가 5% 넘게 악화하지 않는다.
3. 평균 F1 차이 ≥ −0.02이고 어떤 world도 −0.05보다 나쁘지 않다.
4. Input EV 하락의 평균/최대값, actual L0 차이, parameter 수, walltime과 latency를
   함께 표시한다. EV dominance를 C1의 조건으로 가정하지 않으며 fidelity 손실이 있으면
   주장을 해당 recovery–fidelity tradeoff로 제한한다.

Constant는 같은 규칙으로 별도 결과를 내고 exponential 실패를 대체하는 성공으로
합산하지 않는다. 동일 cap 비교를 exact-L0 comparison이라고 쓰지 않는다. 세 joint
world는 screening이며 population uncertainty의 확증 근거로 과장하지 않는다.
실패하면 같은 VG-SAE 안에서 optimization, operating band, readout을 재검토한다.

## B2 — full objective의 support–amplitude 결합

**우선순위: MUST.** Exponential, worlds 100/101/102, B1과 동일 데이터·width·batch·
checkpoint 네 시점으로 제한한다. Full/variance deletion/entropy deletion/both deletion
4 arms를 비교한다. 모든 arm의 beta는 **profiled**, LR는 **0.01**, gamma grid는
B1의 `[1,2,3,4,6,8]`이다. Arm마다 6 trials × 4 checkpoints = 24 candidates다.
C1에서 learned recipe가 선택되더라도 B2의 beta를 사후 교체하지 않는다.

Full의 18 trainings/72 candidates는 B1의 profiled subset을 재사용한다.
나머지 **3 deletions × 6 gamma × 3 worlds = 54 신규 trainings / 216 checkpoints**다.
비교 전체는 72 training arms/288 candidates지만 신규 계산을 72개로 중복 집계하지 않는다.
B2의 selection 결과를 B1 후보에 추가하거나 B1 reference를 다시 고르지 않는다.

Variance on/off와 entropy coefficient 1/0 외에는 architecture, normalization,
parameter count, RNG, optimizer를 같다. Profiled beta는 각 arm의 정의된 reconstruction
energy로 계산되므로 값이 달라질 수 있다. 이는 **같은 profiling rule**이며 동일한
수치 beta를 고정한 ablation은 아니다. Beta도 함께 기록한다. Normalized prior의
normalizer를 임의로 삭제하지 않는다. 손실의 절대값은 서로 다른 objective이므로
arm 간 성능 순위로 쓰지 않는다.

No-variance에서 `h=m*a`를 고정하고 자유로운 amplitude를 `a=h/m`으로 놓으면,
entropy가 있을 때 gate KL은 `m=pi`에서 최소가 된다. Entropy까지 없고 gamma>0이면
`m→0, a→∞`로 같은 h를 유지하면서 gamma*m 비용을 줄이는 방향이 있다. Full variance는
고정 h에서 `0.5 * ((1-m)/m) * h² * ||D_j||²`이므로 이 compensation을 억제한다.
현재 affine/softplus encoder가 모든 입력에서 이 경로를 정확히 실현한다는 주장도,
이미 이 현상이 관측됐다는 주장도 아니다. Decoder unit normalization만으로 m/a
재배율 자유도가 제거되지는 않는다.

각 저장 시점에 train/cal `m`의 q01/q10/q50/q90/q99, mean gate entropy,
`a`와 `m*a`의 RMS/q50/q90/q99/max, beta, expected L0, actual hard L0를 기록한다.
Primary selection은 B1과 같은 cal cap 8 아래 hard `E_z`이며 cap 4/16은 추가 학습 없이
secondary로 낸다. 같은 checkpoint의 mean reconstruction MSE, **analytic** expected
stochastic MSE(variance 포함), hard MSE/EV/F1를 함께 낸다. Deletion-trained model도
진단용 analytic risk에는 원래 Bernoulli variance를 포함하며 training energy와 구분한다.

C2 지지 기준은 full이 variance deletion 대비 세 world 중 2개 이상에서 낮은 hard
`E_z`, 평균 3% 이상 감소, F1 평균 저하 0.02 이내를 보이며 m/a 및 mean-hard 분석이
그 해석과 일치하는 것이다. Entropy 및 interaction 효과는 모든 paired 차이를 공개하고
유리한 arm만 골라 결론을 쓰지 않는다. Collapse만 관측되면 결론은 **이 parameterization의
support–amplitude 결합을 해당 항이 유지한다**는 좁은 해석이다. 일반 deterministic
baseline에 대한 승리나 calibrated uncertainty의 증거로 세지 않는다. 비유한 수치,
0 firing, 급격한 amplitude 증가도 결과이며 test 확인 후 clamp/penalty를 추가하지 않는다.

## B3 — 저장된 현실적인 benchmark와 모델의 범위 확인

**우선순위: MUST인 것은 저장 모델 평가까지다. 추가 training은 조건부 후속이다.**
B1/B2의 결론을 기다리는 동안 split·manifest 준비는 병행할 수 있다. Existing grid를
모두 재학습하지 않으며 B3의 single-seed evaluation을 multi-training-seed 근거로 바꾸지 않는다.

### B3a: pinned SynthSAEBench, 12개 저장 checkpoint

Source는 `outputs/runs/stage2_synthsaebench16k_rho1e-3to1e-1_logspaced_4methods_sae4096_iter100k_beta_learned_seed0/`다.
현재 VG/L1/BatchTopK/JumpReLU 각각 26개 `last.pt`가 있다. 명목상 config.methods에는
다른 방법도 남아 있으나 실제 비교는 존재 확인한 네 방법이다. Generator revision은
`b2efd8b919ae46d6d487c73d46db5ee52813621d`, d=768, true width=16,384,
SAE width=4,096, scale_children_by_parent=false다. 선택된 모델은 모두 seed 0,
100,000 updates × batch 1,024 = **102,400,000 training samples** 조건이다.
200M VG와 100k baseline을 섞어 같은 예산이라고 부르지 않는다.

| 방법 | 저장된 control 3개, 전체 grid의 indices 8/12/16 |
| --- | --- |
| VG learned beta | gamma `[3.81, 1.79, 1.38]` |
| L1 native ReLU | coefficient `[3.0, 1.11, 0.774]` |
| BatchTopK | K `[17.8797, 37.356, 78.0477]` |
| JumpReLU | coefficient `[1.42, 0.484, 0.316]` |

Fresh calibration은 seed `620100`, 262,144 samples; test는 seed `630100`,
1,048,576 samples다. 기존 train/cal/eval seed 30000/20000/40000과 분리하고,
모든 방법에 같은 stream을 재생한다. Data generation state와 artifact SHA를 저장한다.

각 방법은 cal의 native hard L0 caps `[32,64,128]` 아래 **cal reconstruction MSE**가
가장 낮은 저장 모델을 고른다. 이는 실사용 가능한 selection 신호를 쓰는 범위 확인이다.
Test true labels로 control을 선택하지 않는다. Feasible point가 없으면 공백으로 남긴다.
비교 reference도 cap마다 같은 cal MSE 최소 non-VG 방법으로 동결한다.

Test에서 official MCC와 per-latent best-cosine classifier F1/precision/recall,
uniqueness, true-feature coverage `unique best matches / 16384`, hard L0, EV/MSE,
clean-latent generalization 정의를 모두 보존한다. Stage 1의 bijective micro F1와
Stage 2 classifier macro F1를 같은 수치로 합치지 않는다. 이 runner의 generalization
metric은 4,096/16,384의 불완전 coverage 영향을 별도로 설명한다.

최대 **12 checkpoint-cal evaluations + 12 checkpoint-test evaluations**다. 같은
checkpoint가 여러 cap에 선택되면 test를 재사용한다. Calibration에서는 관측 가능한
reconstruction 지표로 선택하고 dictionary/feature metrics는 사전 정의한 test readout이다.

Stage 2 추가 seeds 검토 조건: 서로 다른 selected checkpoints를 사용하는 최소 두 caps에서
VG test EV가 reference보다 0.02 이상 낮지 않고, MCC 또는 classifier F1가 평균 0.02 이상
높으며, 다른 한 metric의 평균 하락은 0.02 이내일 것. Actual L0 및 coverage 차이를 함께
기록한다. 이를 실패하면 Stage 1의 범위로 주장을 제한하고 Stage 2 재학습을 자동 시작하지 않는다.

### B3b: Gemma-2-2B layer 5, 12개 저장 checkpoint

Source는 `outputs/runs/stage3_real_activation_gemma-2-2b-layer5_sae32768_train500m_eval1048576_ctx1024_beta_learned_seeds0-1-2/`다.
실제 완료된 seed 0의 final 모델만 쓴다. 현재 저장된 last는 VG/L1/BatchTopK 각 15개,
JumpReLU 9개다. d=2,304, SAE width=32,768, hook=`model.layers.5`,
500,002,816 training tokens와 pinned model/tokenizer revision을 그대로 유지한다.

| 방법 | 평가할 저장 control 3개 |
| --- | --- |
| VG learned beta | gamma `[10.624448634154069, 3.061698016670821, 1.7868662754501037]` |
| L1 native ReLU | coefficient `[7.178574121956187, 2.4646294753417255, 1.1073258883840886]` |
| BatchTopK | K `[80,160,300]` |
| JumpReLU | coefficient `[0.4375,0.5625,0.6875]` |

VG/L1의 세 좌표는 기존 target K=80/160/300의 사전 interpolated control이며
achieved L0가 아니다. JumpReLU는 기존 9점 grid 안의 세 값이다. 범위 밖 결과를 보고
유리한 저장 모델을 추가하지 않는다. 다음 protocol revision에서는 독립 cal/test를 써야 한다.

Fresh 문서는 **문서 단위로 cal/test를 분리**하고 document hash와 token spans를 저장한다.
기존 train 0:500,002,816 및 과거 heldout 범위를 재사용하지 않는다. 목표는 calibration
131,072 tokens와 test 262,144 tokens이고 각 split의 사전 고정 prefix 16,384 tokens에서
CE/KL intervention을 한다. Tokenizer/model revision, BOS 처리, context 1,024,
padding/BOS 제외 mask, truncation 정책을 두 split에 동일하게 고정한다.

**선행 구현 조건:** 현재 pretokenized row offset은 서로 다른 token range만 보장하며
원문 document provenance를 보장하지 않는다. Original raw source document와 packed row의
대응 또는 별도 heldout-document manifest를 먼저 확보하고 train/과거 eval/cal/test의
문서 중복을 점검해야 한다. 이 확인이 안 되면 단순 fresh token range를 document-disjoint라
표기하지 않으며 B3b의 go 판정을 보류한다. 단순 offset 증가만으로 이 요건을 충족했다고
취급하지 않는다. 이 계획은 provenance가 이미 구현됐다고 가정하지 않는다.

각 방법은 cal native hard L0 caps `[128,256,512]` 아래 **cal CE increase**가 가장 작은
모델을 선택한다. 동률은 cal hard MSE, L0, 사전 grid 순서로 푼다. CE original/replacement/
zero ablation, KL original→replacement 및 original→ablation, normalized CE/KL scores,
hard reconstruction EV/MSE, L0, dead fraction, latency를 test에서 낸다. Reference는 cap별
cal CE increase 최소 non-VG 방법이다. Zero-ablation denominator가 0에 가까우면 normalized
score를 결측 처리하고 raw CE/KL를 유지한다. VG mean readout은 secondary다.

최대 **12 checkpoint-cal + 12 checkpoint-test evaluations**이며 token budget은
12 × (131,072 + 262,144) = 4,718,592 checkpoint-token evaluations다.
이는 모델 forward/intervention 원본·ablation cache 재사용 전의 개수이며 GPU 시간과 같지 않다.

추가 real-data training의 screening go는 서로 다른 selected checkpoint를 쓰는 두 caps
이상에서 VG가 reference 대비 normalized CE score −0.02 이내, normalized KL score
−0.02 이내, hard EV −0.02 이내이며 실제 L0 cap을 지키는 것이다. 회복 ground truth가
없으므로 이 통과는 **배포 fidelity가 현저히 나쁘지 않은 범위가 존재함**을 뜻한다.
True feature recovery, semantic calibration, SOTA를 뜻하지 않는다.

### B3 이후 조건부 확장

B3a go 이후에만 같은 4 methods × 사전 선택된 하나의 control/cap band × new training
seeds 100/101 = **최대 8 Stage 2 training trials**를 별도 계획으로 고려한다. Seed 0은
기존 모델이고 추가 두 개와 총 세 training seeds를 이룬다. 동일 102.4M sample budget,
1 final checkpoint/trial이며 새 training/cal/test seed mapping을 기록한다. 새 HPO는 없다.

B3b 이후 **Stage 3 신규 training은 이 계획의 실행 예산에서 0개**다. 500M-token seed
확장은 실제 throughput, fidelity 결과, 기존 checkpoint storage를 확인한 뒤 별도 예산과
실행 계획을 고정한다. Stage 2의 짧은 시간이나 P2 pilot으로 이를 추정하지 않는다.

## 실행 순서, 계산량과 중단 조건

장기 총 budget은 사용자와 확정되지 않았다. 아래는 **제안한 후속 wave의 hard caps**이며
현재 승인되거나 이미 소비된 예산으로 기술하지 않는다. 이 문서 작성 중 새 큰 job을 돌리지 않는다.

| 단계 | 실행 내용 | 수/후보 | 제안 cap | 통과 조건 |
| --- | --- | ---: | ---: | --- |
| M0 | runner/split/metric/checkpoint 규칙 구현·검증; full-size throughput 측정 | 6 B1 방법 + 3 B2 deletion의 원래 첫 trial을 300 updates까지 측정 후 같은 trial로 재개 | 2 GPUh | 아래 accounting 검증 및 projected budget 이내 |
| M1 | B1 exponential, 모든 방법/world | 216 trainings / 864 checkpoints | 12 GPUh | 공정한 기회와 frozen selections 확보 |
| M2 | B1 constant scope | 216 / 864 | 12 GPUh | Primary와 독립 scope 결과 확보 |
| M3 | B2 deletions; full은 B1 재사용 | 신규 54 / 216 | 6 GPUh | 수학·진단 규칙대로 해석 가능 |
| M4 | B3a/b saved-checkpoint cal/test | 합계 최대 48 checkpoint-split evaluations | 8 GPUh: B3a 2, B3b 6 | Provenance, fresh split, frozen selection |
| M5 | 조건부 Stage 2 추가 seeds | 최대 8 trainings, 8 final checkpoints | 별도 16 GPUh | B3a go + throughput 재계산; 기본 wave에 포함하지 않음 |

M0의 300 updates는 추가 model-selection checkpoint가 아니다. 같은 B1/B2 trial로
재개하면 training 수에 중복 합산하지 않는다. M0에서는 quality에 따라 grid를 바꾸지 않고
시간·memory·I/O만 측정한다. Model 1개/GPU를 기본으로 하고 최대 병렬 수는 실행 직전
확인한 idle GPU 수와 4 중 작은 값이다. GPU 번호, device, peak memory, 실제 walltime,
checkpoint bytes, versions와 모든 data/config hash를 기록한다.

각 방법에서 warmup 뒤 200 updates의 median update time, checkpoint save/load time,
cal 4,096 samples 평가 시간을 잰다. **Estimated hours = 각 trial의 남은 updates ×
해당 method update time + checkpoint/evaluation/I/O time**을 합하고 25% 여유를 붙인다.
GPUh와 wall-clock 시간은 따로 보고한다. 이 추정이 cap을 넘으면 full wave를 시작하지 않고
동일 checkpoint 수·동일 최대 기회를 보존한 예산 수정안을 기록한다. 빠른 방법만 늘리거나
VG만 계속 돌리지 않는다. 실행 중 cap에 도달하면 완료된 결과와 미완료를 구분하고 공정 비교
완료를 선언하지 않는다. 현재 runtime 숫자는 미측정이므로 확정 예상 소요시간을 적지 않는다.

**기본 wave 신규 합계: 486 training trials, 1,944 candidate checkpoints,
최대 373,248,000 training sample presentations + saved B3 24 checkpoints 평가.
제안 총 cap 40 GPUh(M0 2 + B1 24 + B2 6 + B3 8)**다. P1–P3b의 pilot 예산과 별개다.
Full-size training이 40 GPUh 안에 끝난다는 예측이 아니라 그 안에서 시작 가능한지를 M0에서
판정하는 상한이다. Final checkpoint 파일 크기를 실측해 신규 저장소 예산 20 GiB와
남은 여유 공간을 확인한다. 초과가 예상되면 오래된 결과를 삭제하지 않고 출발 전에 기록 정책을
수정한다. Optimizer resume states는 active trial만 유지하며 candidate는 weights/config/metadata다.

## 구현 전제와 검증

1. **새 runner 또는 기존 runner의 opt-in 경로:** 개별 trial recipe, 네 completed-update
   checkpoint, native L1 primary, train-only GMM secondary, 별도 cal/test 분리와
   frozen selection manifest. 기존 public config 의미나 과거 결과를 덮어쓰지 않는다.
2. **공통 data order:** 현재 VG와 공식 baseline batch provider가 완전히 같은 순서를
   쓰는지 확인하고 shared index schedule을 제공한다. Normalization 통계를 train에 제한한다.
3. **Metric consistency:** Native code의 reconstruction과 L0/support가 같은 code를
   가리키는지, dictionary sign/permutation과 RMS-matched constant amplitude를 확인한다.
   Small deterministic identity/relabel/permutation fixture로 latent error/F1와 selection tie-break를 검증한다.
4. **Checkpoint isolation:** Cal로 뽑힌 SHA와 test-loaded SHA가 같음을 확인한다.
   Test를 읽지 않고 selection 완료가 가능한지, malformed/no-feasible/NaN cases를 명시적으로 처리한다.
5. **B2 수식:** 현재 variance/entropy opt-in 경로를 확인하고 four arms가 정확한 항만
   바꾸는 deterministic gradient/energy test를 수행한다. m/a rescaling 사고실험을 수치
   fixture로 검증하되 실제 training의 관측 결과라고 기록하지 않는다.
6. **B3 runner:** saved-only 평가를 전체 target의 training 완료에 묶지 않는 별도 manifest,
   Stage 2 두 fresh RNG streams, Stage 3 document-level provenance 및 CE/KL selection 경로가 필요하다.
7. **필수 검증:** 수정한 코드에 관련 기존 tests와 위 회귀 tests를 적용하고, 수식 변경이
   있으면 `REPRODUCTION_NOTES.md`에 실험 opt-in 의미를 반영한다. 이 문서 작성 단계에서는
   구현 및 test 실행을 완료했다고 주장하지 않는다.

## 논문 산출물과 명시적 제외

Main은 B1의 세 world별 cap8 recovery/F1/EV/L0 및 각 baseline 차이, B2의 four-arm
표와 m/a·mean-hard 분석, B3의 현실적인 적용 범위 표다. Cap4/16, 모든 후보 곡선,
training horizon, GMM secondary, 비용·parameter·실패 trials는 appendix에 둔다.
같은 selected checkpoint가 여러 cap에 나타난 결과를 독립 성공으로 세지 않는다.

현재 제외한 것은 폭별 대규모 sweep, 새 teacher/module, conditional inference solver
채택, 여러 확률적 선행의 일괄 재구현, 전체 Stage 3 grid 재학습이다. VAEase/VSC보다
우수하다는 주장을 추가하면 그때 동일 조건의 직접 baseline이 필요하고, capacity-independent
주장을 추가하면 명시적인 deterministic capacity control이 필요하다. 지금의 좁은 C1/C2
검증을 시작하기 위한 필수 행렬로 늘리지 않는다.

근거 파일: `RESEARCH_BRIEF.md`, 본 run의 `round-0-initial-proposal.md`,
`.aris/traces/research-review/2026-09-21_vg-development/001-pipeline-review.response.md`,
`src/sae_sweep.py`, `src/sae_train.py`, `src/sae_data.py`, `src/sae_sweep_eval.py`,
`src/synthsaebench_eval.py`, `src/real_activation_sweep.py`, `src/real_activations.py`,
각 위 source directory의 `sweep_config.json` 및 checkpoint 존재 확인.
