# 통합 계획의 문헌·코드 근거

확인: 2026-09-29. 원문 method/appendix의 표적 조회이며 전면적인 systematic novelty audit은 아니다. 아래 1–4는 이번에 primary source를 다시 확인했고, 5–7은 9월23일 프로젝트의 검증된 문헌 노트를 재사용한다. 확인한 내용을 계획의 가설과 구분한다.

| 문헌 | 확인 위치와 역할 | 계획에서의 구분 |
|---|---|---|
| Hyungjoon Soh, Dongha Lee, Vipul Periwal, Junghyo Jo. *Variational Garrote for Statistical Physics-based Sparse and Robust Variable Selection*. arXiv2025 | [v1 §III.1.4 및 App.A–B](https://arxiv.org/html/2509.06383v1): realization 평균 이후의 selection uncertainty와 piecewise density template, 비음수 mixture | 모형·곡선은 선행이다. Original variable identity는 고정되어 있고 SAE learned identity는 대응 절차가 필요하다. 혼합 가중치 정규화만으로 Bayesian posterior/신뢰구간을 얻었다고 하지 않는다. |
| David Chanin, Adrià Garriga-Alonso. *Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders*. v4,2026; 로컬 bib는 ICML2026 학회판 기록 | [v4 §3.6,§4,App.G](https://arxiv.org/html/2508.16560v4): decoder cosine와 density/feature mixing 문제 | c_dec는 같은 dictionary 내 기하학적 비교 신호다. Toy의 minimum과 real-LM elbow를 동일한 보편적 selector로 취급하지 않는다. |
| Walter Nelson, Theofanis Karaletsos, Francesco Locatello. *Toward Identifiable Sparse Autoencoders*.2026 | [원문](https://arxiv.org/pdf/2605.31245), §3.3/Thm3.6,§4; [metadata](https://arxiv.org/abs/2605.31245)는 ICML2026 표기 | Hungarian matching과 code stability는 선행이다. 그 논문의 모형·조건을 VG에 자동 적용하거나 본 계획의 식별성 보장으로 사용하지 않는다. |
| Gleb Gerasimov, Timofei Rusalev, Nikita Balagansky, Daniil Laptev, Vadim Kurochkin, Daniil Gavrilov. *Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in Sparse Autoencoders*. arXiv2026 | [v1 §3.2–4,§6](https://arxiv.org/html/2606.12138v1) | Feature correspondence와 subspace 재현은 다른 대상이다. Seed stability 자체가 새 기여이거나 모든 unmatched feature가 틀렸다고 주장하지 않는다. |
| Yin Lu, Xuening Zhu, Tong He, David Wipf. *Sparse Autoencoders, Again?* ICML2025 | [학회판](https://proceedings.mlr.press/v267/lu25w.html), 기존9월23일 노트 | VAEase와 probabilistic sparsity를 비교하되 새로운 full baseline 구현을 이번 core에 넣지 않는다. |
| Kion Fallah, Christopher J. Rozell. *Variational Sparse Coding with Learned Thresholding*. ICML2022 | [학회판](https://proceedings.mlr.press/v162/fallah22a.html), 기존9월23일 노트 | Variational sparse inference 자체의 선행. VG의 이산 support/point amplitude 가정과 구분한다. |
| Dmytro Velychko, Simon Damm, Asja Fischer, Jörg Lücke. *Learning Sparse Codes with Entropy-Based ELBOs*. AISTATS2024 | [학회판](https://proceedings.mlr.press/v238/velychko24a.html), 기존9월23일 노트 | 모형의 재유도·entropy·noise/scale 처리에서 얻는 설명력의 선례. Bernoulli support estimator와 동등하다고 하지 않는다. |

Historical free-energy/autoencoder 연결은 9월23일 IDEA_REPORT의 Hinton–Zemel1993 primary link를 사용한다. 이번 계획에서 새 BibTeX를 기억으로 만들지 않았고 기존 bib도 수정하지 않았다. Full paper drafting 단계에서 학회판 metadata와 exact citation context를 다시 검증한다.

## 본 계획에서 추가하는 해석

Known within/between variance decomposition을 aligned SAE gates에 적용하고, global-selection template의 exchangeability 가정을 `(input,feature)` 전체 좌표로 옮긴다. 이때 learned correspondence, 서로 다른 native densities, heterogeneous firing, stable-wrong basins가 생긴다. Single-template/abstention protocol과 그 실제 density/recovery 효용이 후보 기여다. 신규성이 확정되거나 결과가 이미 나왔다는 뜻은 아니다.

## 로컬 구현 인계

- `src/evaluate.py:68`: selection uncertainty helper의 첫 축은 호출자가 정한다.
- `src/evaluate.py:85`: 기존 kernel는 재사용할 수 있다.
- `src/evaluate.py:117`: NNLS, 정규화 weights, zero-total uniform fallback은 새 confidence/abstention으로 사용하지 않는다.
- `src/sae_sweep_eval.py:343`: `paper_style_sigma_sel`은 input 평균이다. 새 cross-run measure와 구분해야 한다.
- `src/sae_inference.py`: exact/MF/factorized F의 Phase1 기준.
- `scripts/run_first_principles.py`: 기존 학습·world/seed/평가 뼈대. 새 ensemble bank와 blind selector는 미구현이다.
- `README.md:244`: Stage2 calibration stream 재사용 제한. Phase2 external validation에는 fresh stream이 필요하다.
- `docs/stage3_real_activations.md`: 큰 default grid와 접근·logging 조건. 이번 계획은 실행을 시작하지 않았다.
