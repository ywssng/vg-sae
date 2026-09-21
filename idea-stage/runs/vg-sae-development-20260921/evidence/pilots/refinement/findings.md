# P3 결과: 조건부 VG gate 보정의 품질·비용

Cal에서 선택한 **1 sweep + count_a**는 독립 test에서 같은 actual L0를 유지하며 hard latent error를 **5.0123%** 낮췄다. 그러나 순차 보정의 warmed latency가 기본 추론의 **991.94배**였고, 더 저렴한 기본 support NNLS 진폭 보정이 더 낮은 latent error를 냈다. **현재 순차 보정은 배포용 확장으로 채택하지 않는다.** 조건부 hard-quality 신호는 남지만, teacher/distillation 후속은 generic refit/capacity 대조를 갖춘 제한된 개발 후보일 뿐이다.

## 사전 범위와 선택

- P1의 frozen CAL cap4/8에서 정한 exponential checkpoint 5개를 재사용했다. VG gamma3, BatchTopK k4, JumpReLU .5는 두 cap에서 중복되며 독립 반복으로 세지 않는다. TopK는 k4와 k6다. 새 training, 새 module, 새 parameter는 없다.
- 원 TRAIN 8,196개를 원 config/seed로 재생성하고 frozen dictionary/probabilities와 일치시켰다. VG beta는 원 TRAIN expected energy 합 28,892.6069904에서 **18.15495570**으로 한 번 고정했다. Test beta profiling은 없다.
- P3 전용 calibration 512개(seed2026092192), test512개(seed2026092193)를 사용했다. P1 평가 표본과 독립이며 같은 학습 dictionary를 유지한다. 모든 model 선택은 P1 CAL, sweep 선택은 P3 CAL만 사용했다.
- CAL count_a error/F1: sweep0 **.703256/.316499**, sweep1 **.684877/.326829**, sweep3 **.687315/.327690**. F1 조건을 충족한 후보에서 error가 가장 낮은 sweep1을 동결했다. Test에서 sweep3 error가 아주 조금 낮아도 재선택하지 않았다.
- `frozen_selection.json`은 2026-09-21T05:33:02.825185+00:00에 저장했고 이후 test를 생성했다. SHA256 `39d0f94f9f5bd43868730f5de034aca4c7395bb87f63aa7f72a7f136da3e6dfd`.

## 사전에 정한 모든 test arm

Hard error는 Hungarian-aligned latent relative L2 error다. F1과 actual L0는 **실제 양수 code** 기준이다. Candidate L0는 NNLS에 허용한 support 수다. 각 latency는 calibration batch128의 전체 encode/readout/decode와 후처리 및 GPU↔CPU 전송을 포함한 1회 warm-up 후 5회 median이다. Model load, matching, TRAIN beta fit, truth 평가는 제외한다. NNLS는 모든 방법에서 동일 CPU float64 SciPy 알고리즘과 maxiter=max(3K,1)을 사용했다.

| 모델 | Sweep | Readout | Hard error ↓ | F1 ↑ | Hard EV ↑ | Actual L0 | Candidate L0 | Median ms / 128 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| batchtopk_k=4.0 | 0 | native_a | 0.895058 | 0.192137 | 0.331736 | 4.083984 | 4.083984 | 0.2012 |
| batchtopk_k=4.0 | 0 | native_nnls | 0.831618 | 0.198594 | 0.452892 | 3.585938 | 4.083984 | 6.3607 |
| jumprelu_l0_coefficient=0.5 | 0 | native_a | 0.932831 | 0.165859 | 0.091438 | 3.281250 | 3.281250 | 0.2088 |
| jumprelu_l0_coefficient=0.5 | 0 | native_nnls | 0.803833 | 0.174399 | 0.406041 | 2.562500 | 3.281250 | 9.6029 |
| topk_k=4 | 0 | native_a | 0.770145 | 0.270662 | 0.438750 | 3.988281 | 3.988281 | 0.3161 |
| topk_k=4 | 0 | native_nnls | 0.698016 | 0.275441 | 0.525740 | 3.716797 | 3.988281 | 4.4533 |
| topk_k=6 | 0 | native_a | 0.746727 | 0.296906 | 0.495681 | 5.880859 | 5.880859 | 0.3370 |
| topk_k=6 | 0 | native_nnls | 0.662609 | 0.304145 | 0.591798 | 5.476562 | 5.880859 | 4.8904 |
| vgsae_gamma=3.0 | 0 | native_a | 0.737010 | 0.311650 | 0.417488 | 3.185547 | 3.185547 | 0.2158 |
| vgsae_gamma=3.0 | 0 | count_a | 0.737010 | 0.311650 | 0.417488 | 3.185547 | 3.185547 | 0.4100 |
| vgsae_gamma=3.0 | 0 | count_ma | 0.729325 | 0.311650 | 0.427739 | 3.185547 | 3.185547 | 0.4235 |
| vgsae_gamma=3.0 | 0 | native_nnls | 0.651088 | 0.312324 | 0.565965 | 3.144531 | 3.185547 | 5.3133 |
| vgsae_gamma=3.0 | 0 | count_nnls | 0.651088 | 0.312324 | 0.565965 | 3.144531 | 3.185547 | 5.1376 |
| vgsae_gamma=3.0 | 1 | native_a | 0.684060 | 0.406239 | 0.600117 | 6.378906 | 6.378906 | 213.6080 |
| vgsae_gamma=3.0 | 1 | count_a | 0.700069 | 0.319042 | 0.500653 | 3.185547 | 3.185547 | 214.0127 |
| vgsae_gamma=3.0 | 1 | count_ma | 0.698174 | 0.319042 | 0.497516 | 3.185547 | 3.185547 | 216.2278 |
| vgsae_gamma=3.0 | 1 | native_nnls | 0.615939 | 0.409702 | 0.714193 | 6.240234 | 6.378906 | 220.0076 |
| vgsae_gamma=3.0 | 1 | count_nnls | 0.644513 | 0.320178 | 0.585280 | 3.138672 | 3.185547 | 221.9336 |
| vgsae_gamma=3.0 | 3 | native_a | 0.681720 | 0.414557 | 0.604037 | 6.021484 | 6.021484 | 641.8867 |
| vgsae_gamma=3.0 | 3 | count_a | 0.700056 | 0.324069 | 0.508204 | 3.185547 | 3.185547 | 642.9580 |
| vgsae_gamma=3.0 | 3 | count_ma | 0.698408 | 0.324069 | 0.505314 | 3.185547 | 3.185547 | 639.7320 |
| vgsae_gamma=3.0 | 3 | native_nnls | 0.612216 | 0.414860 | 0.717095 | 6.009766 | 6.021484 | 652.7262 |
| vgsae_gamma=3.0 | 3 | count_nnls | 0.645761 | 0.324212 | 0.589745 | 3.179688 | 3.185547 | 646.9360 |

## 해석

- **Gate만의 효과:** 원 a를 유지한 selected count_a는 error .737010→.700069, F1 .311650→.319042, EV .417488→.500653이다. Base와 refined의 actual/candidate L0는 모두 3.185546875다. 입력별 K_i를 base native mask에서 고정하는 실험적 readout이며, 배포용 global threshold와는 다르다.
- **Native threshold의 sparsity 이동:** sweep1 native_a는 error .684060, F1 .406239, EV .600117이지만 L0가 3.185547→6.378906으로 증가했다. 이를 같은 sparsity의 개선으로 쓰지 않는다.
- **Generic amplitude control:** base VG support NNLS는 error .651088, EV .565965, 5.3133ms다. 이는 selected gate correction(.700069, .500653, 214.0127ms)보다 낮은 error와 높은 EV를 주며 약40.28배 빠르다. NNLS가 일부 계수를 0으로 만들어 actual L0는3.144531로 조금 낮아졌다. Base NNLS의 F1 .312324는 selected gate .319042보다 낮으므로 모든 지표에서 우세하거나 gate 이득 전부를 설명한다고 쓰지 않는다. 동일 wall-clock budget 비교를 시행했다고 주장하지 않는다.
- **Refit 후 gate의 잔여 이익:** 같은 NNLS를 refined count support에 적용하면 .651088→.644513로 error가1.0098% 더 낮고 EV가 .565965→.585280로 .019315 높다. Candidate L0는 같지만 actual L0는3.144531→3.138672로 달라 정확한 actual-L0 일치 결과로 부르지 않는다. Calibration에서는 동일 NNLS 이후 error가 0.629138→0.635017로 **0.9344% 악화**되어 test와 방향이 다르다. EV/F1 개선은 두 split에 남지만 latent-quality 근거가 불안정하므로 teacher를 채택했다는 결론은 지원하지 않는다.
- **ma control:** base count_ma는 .729325로 native .737010보다 소폭 좋지만, selected count_ma .698174와 selected count_a .700069의 차이도 작다. 이 결과만으로 posterior mean을 새 해결책으로 주장하지 않는다.
- **비용 판정:** 214.0127ms/0.2157507ms =991.94배로 사전3배 기준을 크게 초과한다. Zero-sweep ranking까지 포함한 count_a baseline 대비도521.96배다. 강한 deployable inference 주장은 지원하지 않는다. Full width1024의 순차 Python/GPU launch 구현이므로 이것이 최적화된 모든 가능 구현의 하한이라는 주장도 하지 않는다.
- **Baseline 범위:** 모든 baseline의 native와 NNLS arm을 보고했다. TopK k6 NNLS error는 .662609지만 actual L0는5.476562여서 VG NNLS3.144531과 exact matched-L0 우위라고 하지 않는다. 원 sweep의 HPO/optimizer/parameter budget 차이는 남는다.

## Fixed conditional objective와 risk

아래 값은 각 sweep의 확률 m와 **원래 amplitude a**로 평가했다. Hard code나 NNLS coefficient를 stochastic VG objective에 혼합하지 않았다. Gaussian/prior constants를 포함하고 beta는 모든 sweep에서18.15495570이다. 단위는 표본별이며 risk는 half SSE다.

| Sweeps | Conditional free energy ↓ | Mean residual half SSE | Bernoulli variance half SSE | Stochastic half SSE | Expected L0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 47.219501 | 4.263154 | 1.092368 | 5.355522 | 51.342178 |
| 1 | 13.778371 | 2.820683 | 0.510671 | 3.331353 | 58.282482 |
| 3 | 11.672421 | 2.827168 | 0.463557 | 3.290725 | 57.269783 |

Gate sweep1은 초기 dense decoder product1회 + 좌표 dot product1024회 + residual rank1 update2048회를 추가한다. Sweep3은 좌표 부분이3배다. 각 readout은 마지막 decode를 포함한다. NNLS 내부 product/iteration 수는 SciPy에서 제공하지 않아 측정값처럼 꾸미지 않는다.

## 모델 크기와 기존 탐색 수

| 모델 | Inference parameter count | 추가 parameter | P1 방법별 CAL controls |
| --- | ---: | ---: | ---: |
| batchtopk_k=4.0_seed=0 | 264320 | 0 | 41 |
| jumprelu_l0_coefficient=0.5_seed=0 | 264320 | 0 | 32 |
| topk_k=4_seed=0 | 263296 | 0 | 128 |
| topk_k=6_seed=0 | 263296 | 0 | 128 |
| vgsae_gamma=3.0_seed=0 | 395392 | 0 | 33 |

## 검증·재현·제한

- Correctness tests **4 passed**: exact Bernoulli support enumeration 기반 coordinate별 objective 비증가, residual 재구성, 마지막 좌표 autograd stationarity, zero/full/tie/saturation K_i, NNLS known solution·support·SSE·empty/rank-deficient 사례. 별도 같은 계열 에이전트가 설계·구현·결과를 읽기 전용 검토했고, 23개 arm의 유한성 및 script/protocol/선택/data/5 checkpoint 해시 일치를 확인했다.
- 기본 shell `python`에는 pytest가 없어 최초 명령은 실패했다. 프로젝트 `.venv/bin/python`으로 위 테스트를 통과했다. 외부 라이브러리 deprecation warning 및 metadata용 gradient tensor→scalar warning이 있었으나 실행 실패/결과 NaN은 없었다.
- Main runtime **63.1055초**, GPU2 allocation **.0175293h**, NVIDIA RTX A6000. Baseline/cal/timing45.44초 안에 끝났고, test 전체 후 완료했다. Environment/version과 전체 latency samples는 JSON에 보존했다.
- 512개 test, 한 frozen training/dictionary seed, exponential/noiseless 한 조건이다. Selected error 개선은5% 기준을 겨우 넘고 CAL에서는2.61%였으므로 통계적 확증/seed robustness로 해석하지 않는다. Source conditions가 과거 결과를 보고 정해졌다는 P1의 제한도 남는다.
- Sequential coordinate update는 알려진 VG 조건부 최적화이며 새 정리/최초 iterative SAE라는 주장을 하지 않는다. 이 결과는 VG-SAE 원 연구 목표의 실패 판정이 아니다. Public source API/config와 checkpoint를 수정하지 않았다.

```bash
.venv/bin/python -B -m pytest tests/test_vg_development_refinement.py -q -p no:cacheprovider
.venv/bin/python -B scripts/run_vg_development_refinement.py --device cuda:2
```

기존 output을 덮어쓰지 않도록 runner는 protocol.json이 있으면 중단한다. 재실행은 같은 refinement 디렉터리 안의 새 하위 경로를 `--output-dir`로 지정해야 한다.

원자료: `test_metrics.json/csv`, `calibration_metrics.json/csv`, `latencies.json`, `checkpoint_manifest.json`, `protocol.json`, `frozen_selection.json`, `summary.json`, `completion.json`.
