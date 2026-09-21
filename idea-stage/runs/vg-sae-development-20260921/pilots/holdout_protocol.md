# P1: 새 calibration/test 표본을 통한 기존 VG-SAE 선택 검증

사전 등록일: 2026-09-21. 결과를 보기 전에 작성한다.

목표는 기존 VG-SAE의 양성 신호를 새 표본에서 확인하고 후속 방법 개발의 operating point를 정하는 것이다. 새로운 논문 주제로 전환하지 않는다.

- 원본: profiled Stage 1 exponential/skew .5와 constant/skew .5, d128/true1024/learned1024, training seed0, `last` checkpoint. 방법은 VG, L1, TopK, BatchTopK, JumpReLU, Gated다. 각 273 controls의 전체 grid를 평가한다.
- 원본 artifact에는 true dictionary tensor가 직접 저장되어 있지 않다. 저장된 config와 original seed0로 원래 데이터를 복원하고 cached original test support/latents와 bitwise 일치 검증한다. 그 dictionary와 feature probabilities를 새 출력에 동결한다. 새 split seed로 dictionary를 재생성하지 않는다.
- Calibration 2048, test 4096. Base seeds 2026092101/2026092102. NumPy SeedSequence는 `[base_seed, source_offset, component_offset]`; source offset은 exponential0/constant1, component offset은 support11/amplitude17/noise23이다. 원래 amplitude 법칙/scale/noise를 유지한다.
- 방법/조건별 L0 cap 4/8/16 아래에서 calibration hard latent relative error 최소 모델을 선택한다. 동률은 calibration hard MSE, actual calibration L0, 원본 fixed control order 순이다. 해당 cap이 가능한 모델이 없으면 N/A. 정확한 matched-L0 비교가 아니며 actual L0와 test cap 초과를 기록한다.
- L1 threshold는 저장된 original-train fitted GMM 값을 사용한다. 없을 때만 original train activations로 fit한다. 새 calibration/test로 fit하지 않는다.
- Decoder matching은 checkpoint weight와 true dictionary만의 Hungarian assignment이며 checkpoint별 한 번 계산한다. `src/sae_sweep_eval.py`의 signed latent error, hard reconstruction, support micro F1 정의를 따른다. 이 실험의 width는 동일하므로 unmatched union 항목은 없다.
- 모든 calibration 선택을 파일로 기록하고 SHA-256으로 동결한 뒤 test 생성/추론을 시작한다. Test metric으로 재선택하지 않는다. 선택된 checkpoint 경로와 matching을 후속 P3에 전달한다.
- 먼저 checkpoint 하나의 calibration 속도를 측정한다. 예상 전체 비용 .5 GPU-hour 이하이면 GPU2에서 진행한다. Timeout은 2시간이다. 기존 학습이나 checkpoint를 바꾸지 않는다.
- 원본 조건 선택은 이전 곡선 관측의 영향을 받았다. 방법별 grid/optimizer/parameter budget은 동일하지 않다. 한 training seed의 fresh-sample 검증이며 SOTA나 training-seed robustness의 증거로 취급하지 않는다.

산출물은 `outputs/vg_sae_development_20260921/holdout/`의 protocol/environment/source manifest, raw calibration metrics, frozen selection, selected test metrics, split samples와 selected mappings이다. 저비용 단위 검증은 fixed dictionary와 independent RNG, train-only L1 threshold, cap/tie selection 및 test-field 무관성을 확인한다.
