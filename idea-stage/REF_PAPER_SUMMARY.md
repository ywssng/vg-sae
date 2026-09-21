# Reference Paper Summary — Sparse but Wrong

David Chanin, Adrià Garriga-Alonso, ICML 2026. 기준은 사용자 local PDF와 같은
[arXiv v4, 2026-07-07](https://arxiv.org/abs/2508.16560v4)다.
[ICML 공식 페이지](https://icml.cc/virtual/2026/poster/63049)에서 학회·저자를 확인했다.
PMLR 306은 PDF에 표시돼 있으며 이번 조회에서 proceedings index는 404였다.

## What They Did

SAE의 L0가 적절하지 않을 때 재구성이 좋은 모델이 실제 feature는 잘못 학습할 수 있는지
synthetic truth와 실제 모델 activation에서 검사했다. 낮은 L0에서는 함께 나타나는 feature를
혼합해 MSE를 줄이는 경향, 높은 L0에서는 다른 퇴화·혼합을 보였다. Decoder pairwise
absolute cosine인 `c_dec`를 L0 선택의 보조 지표로 제안하고 sparse probing과의 관계를 봤다.

주 실험은 BatchTopK와 JumpReLU다. L1 전체의 실패나 VG 우위를 증명한 논문이 아니다.
논문이 제공하는 문제와 **L1 대안으로 VG-SAE를 개발하자는 사용자 가설**을 구분한다.
L1 amplitude shrinkage와 selection/magnitude 분리의 직접 선행은
[Gated SAE](https://arxiv.org/abs/2404.16014) 등에서 따로 확인한다.

## Key Results

- §3.1: 5개 orthogonal features, 평균 true L0 약 2에서 BatchTopK 1.8의 feature 혼합.
  Ground-truth 초기화에도 나타나므로 단순 random-init local minimum만의 현상이 아니다.
- §3.3–3.4: 같은 낮은 L0에서 혼합 dictionary가 true dictionary보다 낮은 MSE를 얻는다.
  따라서 sparsity–reconstruction 우위만으로 feature identity를 검증할 수 없다.
- §3.5: 너무 낮은 L0에서 시작하면 나중에 L0를 바꿔도 회복이 불완전한 toy 결과.
  High-to-low continuation은 알려진 동기이며 VG만의 새로운 원리가 아니다.
- §3.7: 넓은 coefficient 범위에서 적정 L0 부근을 유지한 결과는 JumpReLU다.
  공식 notebook의 `l1` 변수명에도 불구하고 실제 recipe는 tanh sparsity JumpReLU다.
- §4: 일부 Gemma/Llama layer에서 `c_dec`와 sparse probing이 관련된다.
  LLM의 true semantic feature를 관찰해 복구했다고 입증한 것은 아니다.

## Limitations & Open Questions

Mean true L0와 per-sample K는 다르다. Appendix F의 정리는 orthonormal truth,
tied TopK, unit norm과 누락된 활성 feature event에 관한 범위다. 이를 VG objective 전체의
정리로 옮기지 않는다. Decoder pairwise cosine과 decoder-input projection histogram도
서로 다른 지표다. `c_dec`의 작은 값이나 넓은 최소 구간만으로 정답 dictionary를 보장하지 않는다.

Toy의 구조, width, superposition, coactivation에 따라 적절한 L0와 지표 해석이 달라진다.
Appendix I의 자동 L0 조절도 실제 LLM에서는 여러 hyperparameter에 민감했다.
입력의 true support를 모르는 실제 사용 단계에서 어떤 선택 신호가 작동하는지는 열려 있다.

## Potential Improvement Directions

1. VG의 확률적 support와 point amplitude를 학습하고, L1 대비 단순 shrinkage 개선을
   넘어 dictionary 방향과 support identity에 이득이 남는지 Gated/JumpReLU로 대조한다.
2. Positive/negative/independent firing 조건에서 full hyperparameter 곡선을 평가하고,
   올바른 활성 수를 모를 때 유효한 recovery 영역과 선택 민감도를 검증한다.
3. 필요한 학습 경로·prior 개선을 작은 파일럿으로 검사한다. 고정 gamma나 gamma 학습이
   자동으로 true L0를 찾는다고 가정하지 않고 dense/collapse와 잘못된 혼합을 보고한다.

주 연구는 원래 VG-SAE 개발이다. 진단 metric이나 참조 논문 반박을 별도 주제로 채택하지 않는다.

## Codebase and Reproduction Scope

공식 [저장소](https://github.com/chanind/sparse-but-wrong-paper/tree/d5886b540dc5b9cac4f76e6db2b0cce1b0b7c585)는
commit `d5886b540dc5b9cac4f76e6db2b0cce1b0b7c585`로 읽었다. Small toy는
`notebooks/small_toy_model_experiments.ipynb`, generator는
`sparse_but_wrong/toy_models/get_training_batch.py`다.

Small toy: d20, truth/SAE width5, marginal p=.4, latent Gaussian hub correlation
±.4, magnitude=max(0,Normal(1,.15)). 이 Gaussian correlation은 binary Pearson correlation이 아니다.
원본은 방향을 optimization으로 orthogonalize하고 online fresh examples를 쓴다.
QR와 고정 train/cal/test를 쓰는 작은 pilot은 **paper-inspired test**로 표시한다.

본문 batch500, Appendix/helper batch1024; 기본 15M samples와 notebook schedule 간에도
차이가 있다. 50-feature 빈도 수식, RNG, normalization·bias·JumpReLU variant를 함께 기록한다.
짧은 pilot을 논문의 원래 budget을 재현한 결과로 부르지 않는다.

상세 원문 위치·설정 차이·수식 범위는
[분석 메모](runs/vg-sae-sparse-but-wrong-20260922/evidence/ref_paper_analysis.md),
서지·코드 hash는 [metadata](runs/vg-sae-sparse-but-wrong-20260922/evidence/ref_paper_metadata.json)에 있다.
