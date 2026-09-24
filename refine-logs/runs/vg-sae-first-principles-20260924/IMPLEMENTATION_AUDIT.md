# SAE 구현 대조 결과

2026-09-24. 기존 코드·설치된 의존성을 논문과 공식 GitHub에 대조했다. 이전 결과의 성능을 근거로 구현이 맞다고 판정하지 않았다. 검토: GPT-6 Astra ultra `/root/vg_implementation_audit`, `/root/baseline_audit`; same-family provisional.

## 발견과 수정

| 항목 | 실제 문제 | 수정·독립 검증 |
|---|---|---|
| VG entropy | `softplus(logit)−m*logit`이 BF16 positive tail에서 cancellation. logit7의 entropy/gradient가0이 됨. scalar probability clamp도 float32 logit−16에서 entropy gradient를 지움 | logits의 양쪽 tail에 안정적인 식, FP16/BF16 posterior를 FP32로 승격. 독립 finite difference 및 BF16 autocast 검증 |
| VG profiled β | total energy와 per-coordinate risk를 각각 floor하면서 loss와 beta_eff가 서로 다른 floor 사용. 동일 tiny-energy batch 복제만으로 beta가 달라짐 | per-coordinate mean risk에 단일 floor, beta_eff는 그 역수. 복제 불변성, floor 양쪽 finite difference 검사 |
| BatchTopK inference export | expected_average_only_in activation scaling에서 decoder norm만 fold하고 norm-weighted EMA threshold는 유지해 support가 달라짐 | norm-weighted 경로의 threshold도 scaling factor로 나눔. best/last 상태 및 rescale on/off 변환 불변성 검사 |

이 변경은 일반 범위 VG 목적함수나 baseline public alias를 바꾸지 않는다. Low-precision entropy와 floor 근방, 해당 BatchTopK 정규화 조합의 이전 실행은 영향 여부를 재평가해야 한다. 이번 캠페인은 수정 후 FP32/no input normalization으로 새로 실행한다.

## VG의 수식과 확장 범위

[Soh et al. v1](https://arxiv.org/html/2509.06383v1), 제공 PDF p.3 Eq6의 prior는 exp(−γs)이고 prose도 positive gamma가 sparsity를 강화한다고 명시한다. 따라서 코드의 **+γΣm**가 맞다. Eq9/13의 minus는 이 prior와 부호가 불일치한다. 반면 [Kappen–Gómez 원문](https://arxiv.org/pdf/1109.0486)은 prior exp(+γs)이므로 minus가 맞다. 두 gamma 관례를 구분했다.

- Independent Bernoulli expected energy `½||x−D(m*a)||² + ½Σm(1−m)a²||d||²`는 정확하다.
- SAE의 sample별 prior normalizer `L softplus(−γ)`와 latent sum/batch mean이 맞다.
- Learned β의 Gaussian normalization `−d/2 log(β/(2π))`, logβ gradient `βE−d/2`가 맞다.
- Floor 밖 profile의 β*=d/(2 mean E), 두 objective의 상수차 `d/2*(1+log(2π))`가 맞다. Floor 아래 log-risk는 flat이며 constrained-precision likelihood의 선형 구간이 아니다.
- Scalar fixed-gamma objective가 prior normalizer를 생략하는 것은 최적화상 허용되지만 γ 간 절대 loss 비교나 γ 학습으로 확장하면 normalizer가 필요하다.
- 전역 selector 회귀를 입력별 selector와 학습 dictionary로 옮기는 것은 의도적인 SAE 확장이다. Input-dependent point amplitude에 normalized prior/entropy가 없으므로 full generative ELBO라고 부르지 않는다.
- Soft predictive mean, full stochastic reconstruction risk, hard inference를 별도로 평가한다.

Kappen/Gómez를 원개발자로 명시한 [DMLT garrote wrapper](https://github.com/fieldtrip/fieldtrip/blob/49619ecb0e5b941db33fcc8401cbf5266eb0e2c4/external/dmlt/%2Bdml/garrote.m)와 같은 revision의 [regression solver](https://github.com/fieldtrip/fieldtrip/blob/49619ecb0e5b941db33fcc8401cbf5266eb0e2c4/external/dmlt/external/vg/regression.m)에서 original gamma, beta update, variance/entropy를 대조했다. Soh 논문 자체의 공식 GitHub는 확인하지 못했다. 위 코드를 Soh 구현이라고 소개하지 않는다.

## Baseline은 어느 논문의 어느 변형인가

설치 metadata와 upstream 소스를 확인했다. SAELens6.47.0, commit [`8be14080485952f729ed58d674bcddf9778e0aa4`](https://github.com/decoderesearch/SAELens/tree/8be14080485952f729ed58d674bcddf9778e0aa4/sae_lens/saes). Shell SSL 문제로 원격 전체 파일의 bytewise 일치까지 확인한 것은 아니다. 프로젝트 alias identity와 설치 commit test는 통과했다.

| 모델 | 검토 결과 | 이번 실험에서의 사용 |
|---|---|---|
| ReLU/L1 | decoder norm을 곱하는 RI-L1, unconstrained decoder, initial norm .1. [Anthropic April 2024](https://transformer-circuits.pub/2024/april-update/index.html)에 대응. 손계산 loss와 decoder gradient 확인 | SAELens ReLU(RI-L1)로 명시 |
| TopK | sample TopK+ReLU, norm-weighted ranking, unconstrained decoder, aux coefficient1. Detached residual/dead-only aux loss 확인. [OpenAI code](https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py), [원논문](https://arxiv.org/html/2406.04093v1)의 unit decoder/통상 aux1/32와 recipe 차이 | SAELens norm-weighted TopK로 명시; 원논문 전체 recipe 재현이라고 하지 않음 |
| Gated | 처음 제안된 frozen-decoder auxiliary와 다름. [JumpReLU v2 Appendix D](https://arxiv.org/html/2407.14435v2#A4)의 Gated RI-L1는 auxiliary decoder unfreezing을 명시하므로 pinned code는 근거 있는 변형 | 원논문이 틀렸다고 고치거나 detach하지 않음; 주캠페인에서 제외 |
| BatchTopK | batch전체 floor(B*k) selection, EMA threshold inference. Training class eval()만으로 inference 방식으로 전환되지 않으나 프로젝트는 official export 사용. [원논문](https://arxiv.org/html/2412.06410v1)의 threshold 추정과 recipe 차이 | scale folding bug 수정; 주캠페인에서 제외 |
| JumpReLU | log-threshold/rectangle STE의 핵심은 확인. [v2 Appendix J](https://arxiv.org/html/2407.14435v2#A10)의 pre-ReLU가 upstream에는 없어 θ<bandwidth/2에서 backward 차이. 기본 θ=.01, bandwidth=.05 | paper-exact라 부르지 않음; 주캠페인에서 제외 |

프로젝트의 `sae_sweep_eval.py`에는 L1만 GMM threshold support를 쓰면서 native reconstruction을 함께 보고하는 기존 지표 경로가 있다. 이번 runner는 전 방법에 native `code>0`를 사용해 그 혼합을 피한다. Nonorthogonal true dictionary에는 orthogonal-only mixing energy를 쓰지 않으며 signed Hungarian matching을 별도로 계산한다.

## 검증 근거

변경 전 전체 tests:342 passed. 수정 후 focused VG/adapter:66 passed; baseline analytic/conversion:29 passed; exact conditional inference 신규9와 기존 posterior4:13 passed. 캠페인 평가 신규4 tests는 별도로 통과했다. 전체 최종 검증 결과는 EXPERIMENT_RESULTS에 기록한다.

새 테스트는 upstream 함수를 다시 호출해 동일 결과인지 보는 것에 그치지 않는다. 8-state enumeration의 full normalized F와 모든 parameter gradient, tail finite difference, analytic RI-L1 loss, TopK auxiliary detach, Gated RI-L1 gradient, JumpReLU window, BatchTopK budget/export 및 scale invariant를 검사했다. 이 검증은 모델 수식·계산 경로의 근거이며 원 논문의 성능 재현을 뜻하지 않는다.
