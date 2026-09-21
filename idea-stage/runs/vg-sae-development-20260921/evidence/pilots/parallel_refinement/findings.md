# P3b: 한 번의 병렬 VG gate 보정 탐색 후속

**Cal-selected eta=.5는 같은 actual L0에서 EV·F1을 개선했지만, 현재 구현은 사전 native latency3배 기준을 통과하지 못했다.** 동일한 추가 decoder product2회를 쓰는 amplitude PG보다 error는 조금 높고 EV·F1은 높다. 따라서 지표 간 tradeoff 신호가 남으며 배포용 VG gate 확장을 채택했다는 결론은 지원하지 않는다.

이 실행은 P3의 순차 보정 결과를 확인한 뒤 같은 finite-correction 경로 안에서 추가한 **탐색 후속**이다. P3의 사전 확증 실험이거나 독립적인 네 번째 개발 가설로 제시하지 않는다. 이 결과를 본 뒤 추가 eta, optimizer, 구현·benchmark 재시도를 수행하지 않았다. VG-SAE 원 연구 목표는 유지한다.

## 동결한 조건과 calibration 선택

- P1 calibration에서 선택하고 P3에 쓴 checkpoint5개를 그대로 사용했다: exponential VG gamma3, BatchTopK k4, JumpReLU .5, TopK k4/k6. 각 checkpoint hash와 P3 matching이 일치했다. 원 TRAIN에서 구한 beta18.15495570109264를 재사용했다.
- Fresh cal512(seed2026092194), test512(seed2026092195). P1/P3와 표본이 독립이며 학습 dictionary는 같다. Cal/test ground truth는 평가·calibration 선택에만 쓰고 gate/PG/NNLS 추론에 넣지 않았다.
- Eta in{0,.25,.5,1}에서 count_a의 최소 cal error, F1>=base-.01, 평균 conditional F<=base+1e-10 조건으로 선택했다. Eta0은 fallback. **Eta=.5**를 동결한 다음 test를 생성했다.
- Frozen selection timestamp2026-09-21T05:53:50.635263+00:00, SHA256 `cf4cff158edaabba911e494960ec9bfb71e51d8996cf87bd58cb6d3b2a42a9bc`.

| Cal eta, count_a | Error ↓ | F1 ↑ | EV ↑ | Conditional F ↓ |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.710346 | 0.327390 | 0.466302 | 42.181608 |
| 0.25 | 0.704813 | 0.337328 | 0.475732 | 30.654450 |
| 0.5 | 0.696062 | 0.341421 | 0.491442 | 24.243135 |
| 1.0 | 0.708386 | 0.334697 | 0.471160 | 26.547653 |

## 모든 사전 지정 test arm

Native PG는 source hard support에서 한 번의 nonnegative projected gradient를 수행한다. 입력별 support Gram trace의 역수를 step으로 사용한다. Count gate는 base native K_i를 그대로 유지하고 원 a를 쓴다. NNLS는 모든 모델에 동일한 float64 CPU solver/maxiter 정책을 적용한다. Latency는 common wrapper의 encode/postprocessing/readout/decode와 CPU/GPU transfers를 포함하며, cal batch128에서 warm-up2회 후11회 median이다.

| 모델 | Eta | Readout | Error ↓ | F1 ↑ | EV ↑ | Actual L0 | Candidate L0 | Median ms/128 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| batchtopk_k=4.0 | 0.0 | native_a | 0.879065 | 0.200395 | 0.312001 | 3.714844 | 3.714844 | 0.1919 |
| batchtopk_k=4.0 | 0.0 | native_pg | 0.866784 | 0.200395 | 0.338009 | 3.714844 | 3.714844 | 0.4330 |
| batchtopk_k=4.0 | 0.0 | native_nnls | 0.819776 | 0.206615 | 0.450594 | 3.279297 | 3.714844 | 4.4960 |
| jumprelu_l0_coefficient=0.5 | 0.0 | native_a | 0.884022 | 0.180825 | 0.198640 | 2.750000 | 2.750000 | 0.1804 |
| jumprelu_l0_coefficient=0.5 | 0.0 | native_pg | 0.865612 | 0.180825 | 0.236750 | 2.750000 | 2.750000 | 0.4261 |
| jumprelu_l0_coefficient=0.5 | 0.0 | native_nnls | 0.783108 | 0.185364 | 0.428723 | 2.392578 | 2.750000 | 3.6216 |
| topk_k=4 | 0.0 | native_a | 0.742486 | 0.287533 | 0.469163 | 3.990234 | 3.990234 | 0.2929 |
| topk_k=4 | 0.0 | native_pg | 0.714276 | 0.287573 | 0.510186 | 3.988281 | 3.990234 | 0.5351 |
| topk_k=4 | 0.0 | native_nnls | 0.672361 | 0.292470 | 0.556715 | 3.751953 | 3.990234 | 4.9053 |
| topk_k=6 | 0.0 | native_a | 0.728513 | 0.305155 | 0.512104 | 5.863281 | 5.863281 | 0.2951 |
| topk_k=6 | 0.0 | native_pg | 0.704738 | 0.305230 | 0.545947 | 5.859375 | 5.863281 | 0.5156 |
| topk_k=6 | 0.0 | native_nnls | 0.639453 | 0.312751 | 0.614268 | 5.437500 | 5.863281 | 5.5381 |
| vgsae_gamma=3.0 | 0.0 | native_a | 0.706038 | 0.338201 | 0.444171 | 3.250000 | 3.250000 | 0.1702 |
| vgsae_gamma=3.0 | 0.0 | native_pg | 0.675913 | 0.338201 | 0.488605 | 3.250000 | 3.250000 | 0.4166 |
| vgsae_gamma=3.0 | 0.0 | native_nnls | 0.618339 | 0.339595 | 0.598405 | 3.183594 | 3.250000 | 4.6531 |
| vgsae_gamma=3.0 | 0.0 | count_a | 0.706038 | 0.338201 | 0.444171 | 3.250000 | 3.250000 | 0.4224 |
| vgsae_gamma=3.0 | 0.25 | count_a | 0.700428 | 0.344918 | 0.473088 | 3.250000 | 3.250000 | 0.7267 |
| vgsae_gamma=3.0 | 0.5 | count_a | 0.678962 | 0.347255 | 0.502693 | 3.250000 | 3.250000 | 0.7436 |
| vgsae_gamma=3.0 | 1.0 | count_a | 0.683271 | 0.336449 | 0.486404 | 3.250000 | 3.250000 | 0.5764 |
| vgsae_gamma=3.0 | 0.25 | count_nnls | 0.612991 | 0.345636 | 0.605915 | 3.210938 | 3.250000 | 5.2378 |
| vgsae_gamma=3.0 | 0.5 | count_nnls | 0.609544 | 0.350066 | 0.599115 | 3.142578 | 3.250000 | 5.2525 |
| vgsae_gamma=3.0 | 1.0 | count_nnls | 0.621288 | 0.342857 | 0.579078 | 3.000000 | 3.250000 | 5.1385 |

## 개발 판단

- **선택된 gate 정책:** source error .706038→.678962(3.8349% 감소), EV .444171→.502693(+.058522), F1 .338201→.347255(+.009054). Actual/candidate L0는 모두3.25로 같다. 5% error 기준은 미달이지만 EV+.01 기준은 넘는다. Fixed conditional F는47.037616→22.477082로 줄었다.
- **비용:** source native .170206ms 대비 selected .743575ms는 **4.3687배**다. 사전3배 기준에 미달하며 count ranking baseline(.422416ms) 대비1.7603배를 대신 써서 기준을 바꾸지 않는다. ETA1의 더 짧은 latency(.5764ms)도 test로 재선택하지 않았다.
- **Generic PG 대조:** source PG는 .416559ms, error .675913, EV .488605, F1 .338201, actual L0=3.25다. Selected gate가1.785배 느리고 error는0.4511% 높지만 EV는+.014089, F1은+.009054다. Gate 또는 PG가 모든 지표에서 우세하다고 표현하지 않는다. 둘의 추가 decoder products가2회라는 사실은 전체 compute나 wall-clock 일치를 뜻하지 않는다.
- **NNLS 이후 mask 신호:** source NNLS error .618339→selected-mask NNLS .609544로1.4223% 개선했으며 cal에서도0.6890% 개선했다. Test EV 개선은+.000711로 작고 cal은+.010180이다. Test F1도 .339595→.350066으로 개선했으나 actual L0가3.183594→3.142578로 달라진다. Candidate L0만3.25로 같으므로 exact actual-L0 효과로 쓰지 않는다.
- **NNLS 비용 계약:** source NNLS4.6531ms, selected-mask NNLS5.2525ms는 별도 amplitude-refit 정책이다. 이 비율이 작다고 native 대비3배 배포 기준을 통과한 것으로 다시 정의하지 않는다. Latent error 관점에서는 더 강한 amplitude fitting이 큰 역할을 한다.
- **P3와의 관계:** 벡터화 경로가 실용적으로 훨씬 작은 overhead일 수 있음을 보여, P3의992배가 모든 VG correction의 하한이라는 해석을 배제한다. P3와 P3b의 표본·timing session이 다르므로 두 실행의 품질/속도 차이를 paired causal effect로 쓰지 않는다. 이번 결과만으로 새 architecture나 teacher를 채택하지 않는다.

## Conditional diagnostics

모든 아래 지표는 원래 a와 보정된 확률 m_eta로 계산했다. PG/NNLS coefficient는 넣지 않았다. Risk는 표본별 half SSE이며 Gaussian/prior constants를 포함한 fixed-beta F다. Simultaneous update에는 일반적 단조 감소 보장이 없고, 아래 감소는 이 데이터의 관측값이다.

| Test eta | Conditional F | Mean residual half SSE | Variance half SSE | Stochastic half SSE | Expected L0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 47.037616 | 4.264686 | 1.069673 | 5.334359 | 51.414327 |
| 0.25 | 31.305875 | 3.403835 | 1.159952 | 4.563787 | 53.489381 |
| 0.5 | 22.477082 | 2.964305 | 1.094222 | 4.058528 | 55.564435 |
| 1.0 | 25.348466 | 3.349209 | 0.494737 | 3.843946 | 59.714543 |

## 모델·검증·한계

| 모델 | Inference parameter count | 추가 parameter | P1 CAL controls |
| --- | ---: | ---: | ---: |
| batchtopk_k=4.0_seed=0 | 264320 | 0 | 41 |
| jumprelu_l0_coefficient=0.5_seed=0 | 264320 | 0 | 32 |
| topk_k=4_seed=0 | 263296 | 0 | 128 |
| topk_k=6_seed=0 | 263296 | 0 | 128 |
| vgsae_gamma=3.0_seed=0 | 395392 | 0 | 33 |

- 독립 수치 tests **4 passed**: eta0 identity, 원래 m를 고정한 excluded-atom loop oracle, saturated gate의 stable mixture ranking, eta=.5/1에서 F가 증가하는 명시적 Jacobi 반례, non-unit/correlated D의 per-support-trace PG known solution·SSE 감소·지원집합·빈 support. Gauss-Seidel 보장을 Jacobi에 가져오지 않았다.
- Main timer **2.25454초**, GPU2 allocation **.0006263h**. Python import 시간은 main timer 밖이다. NVIDIA RTX A6000, batch128, 전체22 arm, 새 학습/추가 parameter 없음. 결과 선택이나 timing을 바꾸기 위한 재실행 없음.
- 한 frozen training/dictionary seed, exponential/noiseless 한 조건, fresh test512개의 탐색 결과다. Source method의 HPO/control/parameter budget도 서로 다르다. 최종 baseline 우위·seed robustness·novelty를 확증하지 않는다.
- 현재 native latency3배 조건의 실패를 모든 vectorized/optimized implementation의 불가능성으로 확대하지 않는다. 반대로 향후 최적화 가능성을 현재 기준 통과로 간주하지 않는다.
- 실행 전 별도 같은 계열 에이전트의 읽기 전용 코드 검토에서 blocking issue를 발견하지 않았다. 전체 결과와 해시 검토는 `independent_review.md`에 기록한다.

```bash
.venv/bin/python -B -m pytest tests/test_vg_development_parallel_refinement.py -q -p no:cacheprovider
.venv/bin/python -B scripts/run_vg_development_parallel_refinement.py --device cuda:2
```

재실행은 기존 결과를 덮어쓰지 않도록 별도 새 하위 output 경로가 필요하다. 이번 실행 범위에서 추가 변형·최적화·재측정은 하지 않는다. 원 P3 source/protocol/results는 수정하지 않았다.

작은 산출물: `summary.json`, `findings.md`, `test_metrics.json/csv`, `calibration_metrics.json/csv`, `latencies.json`, `frozen_selection.json`, `protocol.json`, `completion.json`, fresh `calibration.npz`/`test.npz`.
