# P3 exact joint-support pilot 결과

판정: **matched-L0에서 mean-field family restriction의 추가 손해는 미해결(unresolved)**이다. 정해진 grid에서 matching coverage가 부족했고, eligible 비교는 사전 joint 성공 기준을 충족하지 않았다. 이는 VG-SAE 개발 전체의 부정 판정이 아니다.

54개 primary fit을 각각 2000 updates × batch 128로 완료했다. 500/1000/2000의 162개 calibration checkpoint와 선택 ID를 저장한 뒤 test를 생성하여 모든 54개 final fit을 한 번씩 평가했다. Optional 18 mixed-init fit은 test 전 전체 유예했다. CPU 2 Torch threads의 본 grid+평가 wall time은 549.83초(9.16분), GPU 사용은 0 GPUh다. Probe 및 import/test 명령 시간은 이 본 grid wall time에 포함하지 않는다. `requested_thread_hours_upper_proxy`는 실제 CPU 사용시간 계측값이 아니다.

## 사전 기준과 coverage

Exact-vs-optimized-MF의 calibration-matched pair는 목표 1.8에서 0/6, 2.0에서 2/6, 2.2에서 2/6이었다. 18개의 목표/부호/world 비교 중 4개만 target ±.15 및 pair difference ≤.10을 함께 만족했다. 이 4개는 모두 dictionary/F1/mixing joint 성공 기준에 미달했다. 반복되는 같은 checkpoint를 독립 반복으로 세지 않는다. Wrong-sparsity robustness 또는 covariance mechanism의 우위를 주장할 근거는 확보하지 못했다.

Gamma grid는 0/4/8로 사전 고정됐으며 test 이후 확장하지 않았다. 아래 gamma 0 표는 사전 등록된 동일-gamma total effect이며, matched-L0 결과로 재명명하지 않는다.

| Gaussian rho | World | Amortized cosine / F1 / L0 | Optimized MF cosine / F1 / L0 | Exact MAP cosine / F1 / L0 |
|---|---:|---|---|---|
| -0.4 | 210 | 0.6008 / 0.7331 / 2.634 | 0.9997 / 0.9841 / 2.063 | 0.9996 / 0.9851 / 2.059 |
| -0.4 | 211 | 0.6003 / 0.7499 / 2.421 | 0.5579 / 0.6384 / 3.461 | 0.5576 / 0.7126 / 2.972 |
| -0.4 | 212 | 0.9998 / 0.9343 / 2.275 | 0.9997 / 0.9908 / 2.032 | 0.9997 / 0.9914 / 2.029 |
| +0.4 | 210 | 0.9996 / 0.9660 / 2.143 | 0.9995 / 0.9650 / 2.148 | 0.9995 / 0.9587 / 2.175 |
| +0.4 | 211 | 0.6571 / 0.7508 / 3.086 | 0.9996 / 0.9989 / 1.994 | 0.9996 / 0.9653 / 2.133 |
| +0.4 | 212 | 0.6626 / 0.7542 / 3.093 | 0.6606 / 0.7554 / 2.918 | 0.9996 / 0.9566 / 2.186 |

Gamma 0의 +.4/world212에서 exact가 MF 대비 cosine +.3390, mixing −.4260, native F1 +.2011을 보였다. 하지만 hard L0는 2.186 대 2.918로 달랐다. Common marginal readout에서도 exact F1=.9681, L0=2.136으로 dictionary와 support 신호가 남지만 밀도 차이가 여전히 커서 intrinsic covariance gain의 증거로 확정할 수 없다. 나머지 5개 gamma 0 world의 exact/MF cosine 차이 절댓값은 최대 .00034였다.

## Readout을 분리한 해석

−.4/world211/gamma0에서 exact native MAP F1=.7126 대 MF .6384지만, exact marginal>.5 F1=.6390이다. Dictionary cosine은 .5576 대 .5579로 같다. 이 native support gain은 MAP readout에 따른 것으로 dictionary training 개선이 아니다. +.4/world211/gamma0에서도 exact MAP F1=.9653 대 MF .9989이나 exact marginal F1=.9990이다. Common marginal control을 생략하면 반대 방향의 과장도 발생한다.

Amortized 대신 optimized MF가 두 gamma 0 world에서 좋은 dictionary basin에 도달했지만, 다른 world에서는 거의 같거나 더 나빴다. Exact가 추가로 다른 basin에 도달한 것은 한 world다. 이는 residual/support inference를 후속 학습 개선 후보로 남길 수 있는 screening 신호이며, 세 world·고정 gamma·밀도 차이를 넘어선 우위나 scalable SAE는 입증하지 않는다.

## Solver와 수치 검증

18개 MF fit의 최저 batch convergence는 99.15%로 95% 기준을 넘었다. 4,608,000 MF sample presentations 중 비수렴 sample 81개(81 batches)를 기록했다. 모든 sample과 2000 updates를 유지했고, 비수렴 sample은 detached terminal q에서 고정-q partial gradient로 학습했으며 converged envelope라고 부르지 않는다. Gamma 0 final 비교는 calibration/test의 selected MF solver failure가 없다. Gamma 4/8의 네 final test fit은 각각 비수렴 sample 1개를 가져 해당 비교의 기전 귀속은 미해결이다. Intermediate/final calibration failure도 원본 JSON에 보존했다. 세 시작점은 global MF optimum의 인증이 아니다.

수치 tests 8개 통과: product-state variance 및 complete F identity, 32-state risk/MAP tie order, orthogonal exact factorization, exact log-partition/envelope gradient, coordinate monotonicity/stationarity/finite difference, resolved MF envelope finite difference, known-D/code identity, shared initialization/frozen beta/core objective equality. 기존 dependency deprecation warning 하나만 있었다.

모든 checkpoint SHA256, protocol/runner SHA256, selection manifest SHA256, world/condition 내 9개 fit의 초기 decoder/amplitude/bias hash와 first-batch hash를 재확인했다. Test ID는 사전 54개 ID와 일치하며 중복 평가가 없다.

## 범위와 재현

Beta=10은 명시적으로 frozen log_beta이며 P1 profiled-beta와 직접 인과 비교하지 않는다. Exact conditional posterior는 입력별 point amplitude를 조건으로 하며 full generative ELBO나 semantic posterior가 아니다. J=5에서 32-state enumeration은 exponential 비용을 감춘 scalable 방법이 아니다. QR 및 유한 train set의 shuffled cycling, fit당 256000 presentations는 원 논문의 online 15M-sample 학습과 다르다. Gaussian rho는 binary support Pearson correlation이 아니다.

주 원본: `selection_manifest.json`, `calibration_results.json`, `test_results.json`, `training_timing_solver.json`, `paired_comparisons.json`. `final_metrics.csv`는 54개 final arm의 평탄한 표다. `integrity_and_summary.json`, `numerical_gates.json`, `runtime_probe.json`, `completion.json`에 검증·자원 근거를 저장했다. Source는 `scripts/run_sbw_joint_pilot.py`, 수치 tests는 `tests/test_sbw_joint_pilot.py`, 사전 protocol은 `idea-stage/runs/vg-sae-sparse-but-wrong-20260922/pilots/joint_protocol.md`다.

## 이후 scientific review에서 보완한 해석

원래 frozen metrics와판정은유지한다. `mixing_energy`는signed-Hungarian할당후leakage라
pure sign/mapping오류도포함한다. 예를들어rho−.4/world210/amortized/gamma0의
signedcosine .60075에도absolutecosine은.99966이고 sign-invariant multi-component energy는
.00049다. OptimizedMF의 개선을 이 사례에서다중feature의기하학적demixing이라고쓰지않는다.
반면rho+.4/world212의MF→exact는absolute cosine .96034→.99959와다중성분energy
.07763→.00081도함께개선해부호와실제혼합의변화가모두있다.
이는사후의모든54모델projection분해이며새성공판정이나모델선택을하지않았다.

같은gamma의한world total effect는실제신호다. L0불일치가이총효과를없애지는않는다.
다만고유한covariance효과나밀도변화와분리된기전,다른world일반화는여전히미검증이다.
