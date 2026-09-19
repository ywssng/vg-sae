# VG-SAE idea-stage 비판 리뷰 — Round 1 raw response

작성일: 2026-09-19. 검토 대상은 `critical_review_brief.md`의 기존 연구 방향이다.
후속 분석 중 전달받은 near-prior group ablation은 추가 탐색 근거로 검토했으며,
진행 예정인 6,000-step convergence probe와 posterior backup pilot 결과는
이번 판정에 포함하지 않았다. 본문은 주 에이전트가 편집한 합의문이 아닌
리뷰어의 원응답이다.

- reviewer_model: `gpt-6-astra`
- reasoning_effort: `ultra`
- canonical_agent_identity: `/root/critical_ultra`
- review_independence: `same-family`
- acceptance_status: `provisional`
- review_mode: fresh critical research review, composed into `idea-stage/IDEA_REPORT.md`
- continuation_verdict: **PROCEED WITH CAUTION**

## 판정

**문제 자체는 제한된 후속 연구를 계속할 가치가 있다. 현재 근거로 VG 특유의
실패 원인이나 새로운 해결책이 발견됐다고 판정할 수는 없다.** 이 판정은 다음
세 개 이하의 저비용 판별 실험을 진행하라는 뜻이며, 논문 기여의 수용 판정이
아니다. 특히 B04의 복제 원인설, B08의 prior scaling 해결책, B02/B03의 간단한
readout 개선을 확정한 상태로 논문 작성을 시작해서는 안 된다.

연구가치는 **6/10**, 기존 코드로 다음 판별 실험을 구현할 가능성은 **9/10**,
현재 top-venue 논문 준비도는 **2/10**이다. 연구가치 점수는 현재 관찰의 신규성
점수가 아니라, 이 저장소에서 짧은 실험으로 잘못된 해석을 배제하고 원인을
좁힐 수 있다는 판단이다. 범용 SOTA 방법으로서의 현재 근거는 없다.

차별점의 경계는 한 문장으로 다음과 같다.

> 복제로 dropout 분산 비용을 줄이는 원리와 soft-to-hard 정보 손실은 알려져
> 있으며, 남은 기여 후보는 **학습된 VG-SAE의 normalized gate KL·amplitude·
> 분산 항이 어떤 조건에서 실제 경성 표현의 품질과 어긋나는지, bias·일반적인
> 평균화 이득·미수렴·추가 추론 계산의 효과를 통제하여 예측하고 검증하는 것**이다.

이 조건부 VG 분석이 성공한다면 진단 및 메커니즘 연구가 될 수 있다. 현재는
첫 조건인 잘 정의된 현상과 원인 후보를 분리하는 단계다. 큰 posterior family,
새로운 encoder stack, hierarchical prior를 동시에 도입할 근거는 없다.

## 직접 확인한 사실과 검증 범위

읽은 주요 자료는 `src/sae_model.py`, `src/sae_train.py`, `src/sae_data.py`,
두 pilot script와 protocol, `brainstorm_candidates.json`의 11개 후보 전체,
`code_audit.md`, 원본 replication/training JSON 및 Stage-2 CSV다.

주요 수치의 파일 경로는 다음과 같다.

- `outputs/idea_discovery_20260919/replication/results.json`
- `outputs/idea_discovery_20260919/training/seed0/results.json`
- `outputs/idea_discovery_20260919/training/seed1/results.json`
- `idea-stage/evidence/pilots/pilot_summary.json` — 후속 탐색 group ablation
- `outputs/runs/stage2_synthsaebench16k_l0calibrated_sae4096_train200m_test25m_beta_learned_seed0/summary/last/final_metrics.csv`
- `outputs/runs/stage2_synthsaebench16k_l0calibrated_sae4096_train200m_test25m_seed0/summary/last/final_metrics.csv`

`free_energy`는 mean reconstruction energy와 Bernoulli variance energy를
합하고, entropy weight가 1일 때 normalized Bernoulli KL을 더한다.
`encode_inference`는 `m > 0.5`와 원래 amplitude `a`를 사용한다.
normalized decoder, amplitude softplus, 학습 가능한 pre-bias와 beta도
briefing과 일치한다. 이 부분에서 수식 구현 오류를 발견하지 않았다.

리뷰 과정에서 다음을 독립 실행했다.

1. 기존 replication 16개 조건을 실제 model 함수로 다시 계산했다.
   분산·loss 닫힌 식과 `hard_active_count=0`을 확인했다.
2. `.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_idea_discovery_replication.py -q`
   결과는 **20 passed**, 기존 의존성의 deprecation warning 1개였다.
3. 원본 checkpoint 세 개(seed 0의 fixed-pi width 32/128, seed 1의 width 128)를
   CPU에 로드하고 동일 NumPy data seed로 재평가했다. 각 checkpoint에서
   보고된 모든 summary statistic의 최대 절대 차이는 각각
   `2.38e-7`, `1.91e-6`, `3.81e-6`이었다.
4. 이 세 checkpoint에서 추가로 `top-(ma)` support에 원래 `a`를 붙인
   readout과 hardening distortion의 decoder 교차항을 계산했다. 이는 리뷰 중
   추가한 탐색 계산이며 사전 계획된 pilot 결과로 표시해서는 안 된다.

Stage-2 gamma 1.99의 CSV도 직접 읽었다. hard/mean L0는
`30.253607889236196 / 543.1683862860689`, hard/posterior-mean EV는
`0.7750422983074959 / 0.8609611262131668`이었다. 인접 JumpReLU의
L0/EV는 `29.880485654043277 / 0.8041284475361851`이었다.
이 숫자 자체는 briefing과 일치한다. 한 seed, calibration에 재사용된 평가
stream, 완전히 같지는 않은 achieved L0 조건이므로 확증적 architecture
순위로 쓰면 안 된다. Stage-2 전체 stream을 재실행하지는 않았다.

## 가장 중요한 우려 3개

### 1. posterior-mean EV와 확률적 기대 복원 위험을 구분해야 한다

현재 `expected_ev`는 `decode(m*a)`의 EV다. 그러나 학습되는 Gaussian 항은
`E_q ||x-decode(s*a)||²`이므로 두 양이 다르다. 다음 표기에서

\[
\mu=b+D(ma),\quad z_h=1[m>0.5]a,\quad
V=\tfrac12\sum_j m_j(1-m_j)a_j^2\|d_j\|^2,
\]

테스트 집합의 centered energy를 \(T=\mathbb E\|x-\bar x\|^2\)라 하면

\[
\mathrm{EV}_{\mathrm{stochastic}}
=\mathrm{EV}_{\mathrm{mean}}-\frac{2\mathbb E V}{T}.
\]

이는 샘플링으로 추정해야 하는 값이 아니라 현재 mean-field 모델에서 정확히
계산되는 analytic stochastic reconstruction EV다. raw JSON으로 계산하면
**10개 학습 조건 모두 hard EV가 stochastic reconstruction EV보다 높다.**

| optimizer seed / width / prior | posterior-mean EV | hard EV | stochastic reconstruction EV |
|---|---:|---:|---:|
| 0 / 32 / fixed-pi | 0.905439 | 0.880477 | 0.855955 |
| 1 / 32 / fixed-pi | 0.903660 | 0.878230 | 0.853433 |
| 0 / 128 / fixed-pi | 0.901415 | 0.851032 | 0.842000 |
| 1 / 128 / fixed-pi | 0.903137 | 0.852344 | 0.841571 |
| 0 / 128 / fixed-count | 0.857426 | 0.805405 | 0.795836 |
| 1 / 128 / fixed-count | 0.864917 | 0.813460 | 0.803133 |

따라서 이번 pilot은 ‘좋은 확률적 기대 복원을 hard deployment가 망친다’는
강한 해석을 지지하지 않는다. 평균 코드는 여러 불확실한 기여를 동시에
사용하므로 개별 sparse action보다 좋은 제곱오차를 얻을 수 있다. 그 차이는
정확한 posterior에서도 생길 수 있다. 이 파일럿에서 확률적 목적함수가
hard reconstruction보다 좋은 risk를 숨겼다는 진술은 사실과 반대다.

문제 정의는 **실제 원하는 sparse action의 위험, support quality, inference
cost가 어떤 objective 및 approximation 요소에 의해 제한되는가**로 좁혀야
한다. mean EV는 유용한 진단값이지만 동일 sparsity의 달성 가능한 benchmark나
oracle 성능이 아니다. stochastic sample 역시 hard code와 expected active
count가 다르므로 이 세 EV만으로 sparsity 공정 비교를 완성하지 않는다.

hard/mean 손실 차이도 distortion 하나와 같지 않다. \(r=x-\mu\),
\(\delta=b+Dz_h-\mu\)라 하면

\[
\mathbb E\|x-(b+Dz_h)\|^2-\mathbb E\|x-\mu\|^2
=\underbrace{\mathbb E\|\delta\|^2}_{H}
 -2\mathbb E\langle r,\delta\rangle.
\]

JSON의 분해는 이 항등식과 약 `1e-8` EV 이내로 일치한다. 잔차 교차항은
전체 gap의 약 **34–52%**다. 이 부분을 모두 decoder duplication으로 부르면
원인 설명의 절반 가까이를 생략한다.

### 2. 복제 구성은 맞지만, 실제 학습 원인과 신규성의 연결이 없다

고정 pi 복제 구성에서 variance가 `0.9236320124`에서 `0.0144317502`로
줄고 KL이 수치적으로 0인 것은 맞다. fixed-prior-count 구성에서는
`0.375`에서 `0.498046875`로 증가한다. 하지만 이 구성은 decoder bias가
없는 상수 입력의 feasibility 결과다. 자유 bias가 있는 같은 상수 입력에는
bias-only 해가 있다.

더 강하게 말하면, 제시된 zero-residual, `m=pi<0.5`, `a>0` 구성은
일반적으로 stationary point도 아니다. frozen beta에서 다른 변수를 고정하면

\[
\frac{\partial F}{\partial m_j}
=\frac{\beta}{2}(1-2\pi)a_j^2\|d_j\|^2>0,\qquad
\frac{\partial F}{\partial a_j}
=\beta\pi(1-\pi)a_j\|d_j\|^2>0.
\]

이 사실은 construction을 반박하지 않는다. 다만 optimizer가 그 해를 선택했다거나
그 해가 더 좋은 sparse solution을 이겼다는 결론은 construction에서 나오지
않는다. width를 달리한 loss 곡선은 fixed-width objective 내부에서 학습된
최적점의 안정성을 증명하지도 않는다.

학습 pilot에서는 fixed-pi width 32→128에 따라 분산 energy가
seed 0에서 `0.095810→0.115036`, seed 1에서 `0.097249→0.119203`으로
**증가**한다. KL은 0이 아니며 총합 약 `3.44–3.93`이다. maximum cosine이나
participation 증가만으로 알려진 variance dilution 경로가 실행됐다고
판정할 수 없다. 폭이 커지면 nearest-neighbor cosine이 커질 기회도 늘고,
사용하지 않는 열이나 상수 성분도 geometry 지표에 들어간다.

리뷰어가 확인한 조금 더 직접적인 신호는 있다. \(\epsilon=(s-m)a\)라 할 때
\(H=\mathbb E\epsilon^TD^TD\epsilon\)의 off-diagonal 기여는 fixed-pi
seed 0 width 32에서 `-0.005689`, width 128에서 `+0.069252`였고,
seed 1 width 128에서는 `+0.055338`이었다. 폭 128에서 각각 H의
`54.1% / 46.8%`다. 이는 경성화 오차의 기하가 변한다는 탐색 신호다.
그것이 복제 때문인지, 공통 평균 성분 때문인지, 일반 correlated coding
때문인지는 아직 모른다. 실제 H는 이 세 조건 모두 `2V`보다 작다.

문헌 중복은 직접적이다.

- [Cavazza et al., AISTATS 2018](https://proceedings.mlr.press/v84/cavazza18a/cavazza18a.pdf)의
  Eq. 13, Proposition 1, Sec. 5는 factor 복제에 따른 dropout regularizer 감소와
  width-dependent retention 교정을 이미 다룬다. 일반 복제 정리와 폭별
  retention 조정은 이번 연구의 신규성이 될 수 없다.
- [SoftSAE, Sec. 3](https://arxiv.org/html/2605.06610v1)는 작은 비영 soft weight에
  정보가 남는 문제와 마지막 학습 단계의 hard Top-K 전환을 명시한다.
  soft/hard mismatch 최초 발견이나 hardening 자체의 신규성 주장은 불가하다.
- [A Dominant Diffuse Phase, Sec. 3–6](https://arxiv.org/html/2609.10299v1)는
  ReLU/L1 synthetic SAE에서 좋은 복원과 낮은 recovery, 조밀한 code가 함께
  나타나는 현상을 측정한다. 이 연구가 VG를 검증한 것은 아니지만 단순히
  ‘복원은 좋은데 feature가 틀리다’라는 발견의 신규성은 크게 낮춘다.

이 세 원문은 직접 확인했다. 다른 확률적 SAE와 Bayesian model-selection
문헌 전체를 이 리뷰에서 완전 검색한 것은 아니므로, 남은 VG-specific gap의
부재나 최초성을 보증하지 않는다.

### 3. 실제 해석을 제한하는 미수렴과 대조군의 빈틈이 있다

마지막 1,000→1,199 step 구간에서도 fixed-pi beta는 약 17–18에서 24–25로
증가하고 training loss는 약 1 이상 감소한다. 예를 들어 seed 0 width 128은
beta `17.7139→24.3411`, loss `0.78435→-0.41430`이다. 이 기록으로
converged representation의 폭 의존성이나 objective optimum을 논할 수 없다.
noise SD 0.05의 물리적 precision은 400이지만 learned beta가 이를 재현해야
한다는 가정 역시 부적절하다. 현재 beta는 표현 및 추론 오차도 흡수한다.

데이터/dictionary seed는 하나이고 optimizer seed만 둘이다. 마지막 checkpoint를
사전 고정한 점은 좋지만, 학습 길이의 영향을 확인해야 한다. 이번 결과는 짧은
학습 예산에서의 재현된 관찰이며 일반적인 VG failure regime가 아니다.

fixed-count의 width 128 hard EV는 `0.805405 / 0.813460`으로 fixed-pi의
`0.851032 / 0.852344`보다 나쁘고 gap도 닫히지 않았다. 이것은 단순한
prior scaling 해결책에 불리한 근거다. 하지만 hard L0도
`1.303 / 1.385`에서 `1.052 / 1.059`로 낮아져 원인별 실험은 아니다.
‘전체 occupancy를 맞추면 복제를 교정한다’는 방법 주장은 현재 지지되지 않는다.

readout 대조에도 구분이 필요하다. `top_ma_same_count`는 support를 바꾸면서
amplitude를 동시에 `a→ma`로 바꾼다. 그러므로 default와의 차이 전체를
support ranking의 실패로 해석하면 안 된다. `same_support_mean`과의 비교는
support 변경만 분리하며 실제 차이가 매우 작다. 빠진 `top-ma support + a`
대조를 리뷰어가 세 checkpoint에 추가 계산한 결과는 다음과 같다.

| 조건 | default hard EV | top-ma support + a EV | top-ma support + ma EV |
|---|---:|---:|---:|
| seed 0, width 32, fixed-pi | 0.880477 | 0.880761 | 0.869650 |
| seed 0, width 128, fixed-pi | 0.851032 | 0.849893 | 0.836685 |
| seed 1, width 128, fixed-pi | 0.852344 | 0.851670 | 0.840571 |

따라서 큰 readout 악화는 주로 `a→ma` 변화와 함께 나타난다. 순위 변경의
큰 이득은 이 추가 대조에서도 없다. 이 결과가 NNLS, residual-aware choice,
posterior refinement 등 모든 hard-decision 방법의 가능성을 반박하지는 않는다.
fixed K=2도 Bernoulli 생성의 **평균** 활성 개수를 매 입력에 강제하는 것이므로
true-support count oracle이 아니다.

추가 group ablation도 강한 정보 누출 주장을 약화한다. width 128 fixed-pi에서
near-prior(log-odds ±0.25) 약 102개를 제거하면 EV loss는
`0.028331 / 0.026815`이지만, 그 group의 학습 집합 평균 decoded contribution을
대신 넣으면 `0.007107 / 0.008425`로 줄어든다. decoded variation은 입력
centered energy의 `0.003462 / 0.002718`이다. raw contribution의 상당 부분이
상수 성분인 것이다. ‘near-prior group이 전부 무해하다’도 아니지만 ‘작은
gate가 풍부한 입력 정보를 숨긴다’는 주장도 이 결과로 지지되지 않는다.
반면 모든 subthreshold feature의 mean-preserving removal loss는
`0.039533 / 0.042675`여서 **near-prior floor와 consequential subthreshold
features를 구분할 필요**는 남는다. 이 ablation은 결과를 본 뒤 추가한 탐색이다.

## 허용되는 주장과 현재 지지되지 않는 주장

| 질문 | 현재 허용되는 주장 | 현재 지지되지 않는 주장 / 필요한 근거 |
|---|---|---|
| 구현의 수식 | 실제 mean-field energy와 Bernoulli KL, hard readout은 확인한 식과 일치한다. | 전체 실험의 integrity certification 또는 모든 모델 경로의 무결성 보장. |
| prior floor | `a=0`에서 unit-entropy objective의 optimum은 `m=pi`; 큰 expected count는 혼자서 정보량을 뜻하지 않는다. Softplus 모델에서는 `a→0` 극한도 구분한다. | 저장된 모든 inactive feature가 정확히 prior에 있고 모두 정보가 없다는 단정. |
| 복제 toy | bias-disabled constant-input family가 mean-perfect, KL≈0, hard-zero를 허용하고 fixed-pi variance가 1/r로 감소한다. | 새로운 dropout theorem, stationary/global optimum, 실제 학습 원인, default-bias 데이터에서의 보편적 실패. |
| 학습 width pilot | 한 data seed·두 optimizer seed·1,200 steps에서 width 32→128의 mean/hard EV gap이 약 0.025→0.051로 커졌다. | converged width law, 실제 duplication 경로, 모든 dataset/SAE로의 일반화. |
| mean 대 hard | 같은 checkpoint의 posterior-mean EV가 hard EV보다 높다. | 실제 stochastic expected risk보다 hard risk가 나쁘다는 주장. 이번 10개 결과는 반대다. |
| geometry | 세 checkpoint의 hardening distortion 교차항이 좁은 모델과 넓은 모델에서 달랐다. | near-duplicate decoder가 그 변화를 일으켰다는 인과 또는 geometry 지표의 novelty. |
| prior scaling | 이번 설정에서 fixed-count가 mean/hard EV를 개선하지 못했다. | prior scaling의 보편적 무효, matched-L0에서의 인과, sparsity-invariant 개선책. |
| 간단한 readout | same-support ma 및 기록된 top-ma+ma는 성능을 개선하지 못했고, 추가 top-ma+a 대조의 개선도 작거나 음수였다. | 모든 sparse projection/NNLS/refinement가 실패함, 또는 inference-only 변경으로 dictionary recovery가 좋아짐. |
| group deletion | mean-preserving ablation 후 near-prior와 전체 subthreshold group의 영향이 다르다. | unweighted count 또는 raw group energy가 정보량·확률 calibration을 직접 측정함. |
| Stage-2 | 지정된 seed/평가 stream에서 VG hard EV가 인접-L0 JumpReLU보다 낮다. | 통계적으로 확정된 일반 architecture 순위, mean EV를 hard-L0 baseline과 직접 비교한 우월성. |
| posterior 해석 | `m`은 현재 deterministic input-dependent amplitude 아래의 variational gate이다. | 전체 learned SAE의 calibrated support probability, epistemic uncertainty, 최초 probabilistic SAE. |
| 다음 연구의 기여 | 위 혼동들을 통제한 VG-specific objective/decision 메커니즘은 아직 검사할 가치가 있다. | 벌써 새 방법이나 충분한 top-venue contribution이 확보됐다는 결론. |

## 다음 단계: 최대 3개의 값싼 core block

아래 비용은 새로 측정한 runtime이 아니라 작업 계획용 상한 추정이다.
기존 tiny 모델의 실제 1,200-step 학습은 모델당 약 6초였으므로 대형 sweep을
시작할 이유가 없다. 각 block의 결과가 원인 가설을 구분하지 못하면 다음
architecture를 추가하는 대신 해당 가설을 버린다.

### Block 1 — 정확한 risk accounting과 학습 길이 확인

기존 10개 결과에 mean/stochastic/hard EV, H와 residual cross, 실제 L0를
함께 놓는다. fixed-pi width 32/128 및 fixed-count width 128을 두 seed에서
사전 고정된 더 긴 학습 예산으로 반복한다. 결과를 본 뒤 가장 좋은 step을
고르지 않고, 마지막 구간의 loss/beta/각 EV 변화를 보고한다. 새 모델을
선택하거나 임계값을 맞출 때 calibration split을 쓰고 최종 판단에는 새
holdout을 둔다. 의미 있는 효과가 남으면 두 번째 data/dictionary seed로
소수 조건만 재확인한다. Stage-2는 새 stream의 2,048–4,096개 예제로
frozen-checkpoint risk/group audit만 한다.

- 성공 조건: 학습 길이를 늘려도 남는 특정 sparse-action 문제와 그 크기를
  정확한 risk 기준으로 명시할 수 있다. 단순 mean-hard gap의 존속만으로는 부족하다.
- 반증/중단: gap이나 width 차이가 계속 줄어들어 초기 관찰이 학습 중간 현상으로
  설명되면 permanent objective failure를 중단한다. 두 번째 dictionary에서
  사라지면 현재 조건의 사례로만 남긴다.
- 예산: tiny training과 bounded checkpoint audit 합계 **약 1 GPU-hour 이내**를
  우선 상한으로 잡는다. 6,000 steps도 자동으로 수렴을 뜻하지 않는다.

### Block 2 — 학습 checkpoint에서 local replication preference만 검사

주 에이전트가 후속 refinement로 제안한 clone perturbation은 폭 상관보다
직접적인 다음 검사다. frozen beta에서 한 atom의 gate를 동일하게 복제하고
각 복제의 Bernoulli 변수는 독립으로 유지하며 amplitude를 정확히 1/r로
나눌 수 있는 function-level 개입이라면

\[
\Delta F=(r-1)K_j-\beta(1-1/r)V_j
=(r-1)(K_j-\beta V_j/r),
\]

여기서 \(K_j=\mathbb E\mathrm{KL}(q_j\|p_j)\),
\(V_j=\tfrac12\mathbb E[m_j(1-m_j)a_j^2\|d_j\|^2]\)이다.
따라서 r=2의 선호 조건은 \(\beta V_j>2K_j\)다. 이 식 자체는 알려진
dropout duplication을 normalized KL까지 기록한 산술적 확장이며 새로운
일반 정리로 포장하지 않는다.

실제 linear-softplus encoder는 `a/r`을 항상 같은 형태로 표현하지 못한다.
amplitude bias를 `-log(r)`만큼 바꾸는 제안은 작은 amplitude에서의 근사다.
실제 개입은 \(\Delta\mathrm{recon},\Delta V,\Delta KL\)을 모두 별도로
계산하여 mean drift가 apparent objective gain을 만든 경우를 구분해야 한다.
atom 선택은 학습/calibration 데이터에서 고정하고 새 holdout에서 평가한다.
amplitude/firing이 비슷한 무작위 atom 대조를 둔다.

**exact clone은 mean reconstruction과 hard reconstruction을 모두 보존한다.**
일부 입력에서 해당 atom이 firing하면 hard active count만 증가한다. 따라서
이 개입의 positive endpoint는 ‘현 checkpoint에서 objective가 표현 복제를
선호한다’이며, sparse-decision 손상 자체는 아니다. 긍정 결과가 있더라도
실제 학습 경로나 feature 품질 저하까지 주장하려면 이후 동일 폭·계산량의
짧은 continuation 대조가 필요하다. 복제 가능성 재확인만으로 논문 기여가
완성되지는 않는다.

- 성공 조건: selection에 쓰지 않은 데이터에서도, mean-drift가 충분히 작고
  normalized KL을 포함한 실제 loss 분해가 replication preference를 설명한다.
- 반증/중단: 이득이 거의 모두 reconstruction drift에서 오거나, 의미 있는
  기여를 하는 atom에서는 KL 비용이 이겨 preference가 없으면 현재 B04
  메커니즘을 주가설에서 내린다. objective preference가 있어도 그것이
  harmless bias/nonfiring mass에만 있으면 hard-quality 원인이라고 부르지 않는다.
- 예산: checkpoint 몇 개의 forward 재평가로 **약 0.25 GPU-hour 이내**부터 시작.

### Block 3 — 고정 dictionary에서 decision/amplitude와 inference 한계 분리

현재 체크포인트에서 입력별 실제 support count를 유지하여 support 선택과
amplitude readout을 분리한다. 원래 support 및 top-ma support 각각에
`a`와 같은 예산의 nonnegative amplitude refit을 적용하는 작은 2×2 대조로
충분하다. 현재 `ma` 결과는 shrinkage 대조로 그대로 남긴다. 사용하지 않은
calibration split에서 필요한 설정을 고정하고 새 holdout에서 EV, support
F1, 실제 L0, runtime을 측정한다. 해당 후처리는 가능한 deterministic
baseline에도 같은 예산으로 제공한다.

그 결과가 dictionary 고정 상태에서 거의 개선 여지가 없음을 보이거나,
posterior inference가 의심될 경우에만 frozen `a,D,b,beta`에서 1/3/10회의
순차 VG gate 갱신을 작은 subset에 적용한다. loss 감소와 hard-quality
변화를 따로 읽는다. 이미 별도로 진행 중인 exact-posterior backup은
작은 known model의 reference로만 사용하고 전체 learned SAE calibration의
증거로 전이하지 않는다. 새 family/encoder 구현은 이 block에 포함하지 않는다.

- 성공 조건: support selection, amplitude refit, gate approximation 중 어떤
  요소가 matched-count 성능을 제한하는지 반복되는 방향성이 나온다.
- 반증/중단: 이득이 일반 NNLS와 추가 계산만으로 설명되면 새 VG readout
  방법 주장을 중단한다. free energy만 좋아지고 hard 결과가 나빠지면 그
  조건부 objective/decision 불일치를 기록하며 ‘더 정확한 posterior가 더
  좋은 SAE’라는 전제를 버린다.
- 예산: tiny width 128 및 Stage-2의 512개 이하 subset에서 시작하여
  **약 0.5 GPU-hour 이내**를 우선 상한으로 둔다.

## 후보 선택과 논문 방향에 대한 결론

B01의 문제 진단을 중심에 두고 B04는 검증할 메커니즘 후보, B03/B07은
원인 분리 도구로 쓰는 구성이 가장 단순하다. B08은 이번 pilot에서 이미
개선책으로서 부정적이다. B05는 분리된 known-model backup으로 가치가
있지만, B06의 pair posterior나 B09/B10의 새로운 prior까지 동시에
개발하면 무엇이 기존 문제를 해결했는지 판단하기 어려워진다. B11의
precision 문제는 현재의 미수렴 및 misspecification 통제에 포함할 수 있고
독립 방법 기여라고 미리 약속할 필요는 없다.

top-venue 방향을 계속 택하려면 알려진 soft/hard 현상을 VG에서 재관찰했다는
수준을 넘어야 한다. 최소한 **VG objective의 조건부 예측 → 실제 learned
checkpoint에서의 검증 → bias/mean averaging/미수렴 대안의 배제 → matched
sparsity와 계산량에서의 실질적 영향**이 연결되어야 한다. 이 연결이 없다면
정직한 재현·진단 보고서는 남길 수 있어도 독립적인 강한 방법 논문의 중심
기여로는 부족하다. 현재 자료는 이 연결을 아직 제공하지 않는다.

## 적용한 SCOPE LIMITS — 원문 그대로

```text
=== SCOPE LIMITS (these bound what you PROPOSE, never what you look for) ===
Report anything that is actually wrong here — including a rare-looking case, if
this repo actually produces it. Then keep the fix in scope:
1. This is a RESEARCH-WORKFLOW tool, not a security paper. Verification is
   welcome; over-defense is not. Assume a cooperating operator on their own
   machine — a malicious local user is NOT in the threat model.
2. Do NOT propose SHA / hash / content-fingerprint / digest-binding schemes.
   Reporting a real defect in hashing code that already exists is fine.
3. NO speculative machinery: do not add feature flags, migration frameworks,
   compat layers, wrappers, pins, or similar mechanisms unless evidence shows
   a current repo defect they fix or an explicit existing invariant they must
   preserve. "Load-bearing", "compatibility", and "not scaffolding" are labels,
   not evidence. Point to the failing path/artifact or invariant, and check the
   proposal's factual premises, such as whether a named package version exists.
4. NO corner-case obsession: exotic encodings, symlink races, RTL text and
   millisecond races are out of scope unless you can show the case arises here.
5. Where a rubric or checklist is genuinely needed, do not over-mechanize
   judgement. A clear sentence a human reads beats a scored table nobody
   maintains.
Exception: code that runs remote commands, starts a network service, or installs
an MCP server runs on the user's machine with their credentials — trust-boundary
findings there are in scope and the default is strict.
Say plainly when something is correct. Do not manufacture findings.
Be brutally honest. If, after genuinely trying to break it, the work
holds up and is ready, say so clearly.
```
