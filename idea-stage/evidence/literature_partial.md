# VG-SAE idea-discovery 문헌 조사 인계

작성일: 2026-09-19. 사용자 지시에 따라 추가 조사를 중단하고, 이미 확인한 사실만 보존한다. 이는 완료된 체계적 문헌 조사나 최종 novelty 판정이 아니다. 후속 ultra 검토의 입력으로 사용한다.

## 수행 범위와 제한

- `.agents/project-memory.md`, `.agents/skills/research-lit/SKILL.md`를 읽었다.
- `refs/`의 PDF 3개에서 `pdftotext -f 1 -l 3 ... -`를 실행했다. VG와 SynthSAEBench 첫 3쪽의 본문을 읽었다. Sparse but Wrong은 첫 3쪽을 추출했으나 대량 출력 제한 때문에 이 대화에서 첫쪽 중심으로 확인했다.
- 사용 가능한 도구 메타데이터에서 Zotero/Obsidian 도구가 발견되지 않았다. 사용자 소장 컬렉션 전체를 확인한 것은 아니다.
- 웹 검색과 arXiv 공식 abstract 페이지 열람을 수행했다. 아래에서 abstract-only와 본문 확인을 구분한다.
- arXiv API helper `python3 Auto-claude-code-research-in-sleep/tools/arxiv_fetch.py search 'all:"sparse autoencoder" AND (all:probabilistic OR all:uncertainty OR all:calibration)' --max 10`를 실행했으나 sandbox DNS 오류로 실패했다. 권한 확장 재시도는 사용자 permission 변경에 따른 중단으로 이 shard에서는 결과를 얻지 못했다. Root는 별도로 helper 성공을 보고했다.
- **최근 3–6개월의 명시적인 날짜 범위 검색은 아직 완료하지 못했다.** 2026-05 논문, 2026-07 개정판과 저널 논문, 2026-09 Zenodo 검색결과까지 발견했으나, 이는 2026-09-19까지의 완전한 concurrent-work coverage를 뜻하지 않는다.
- DOI/arXiv canonical ID로 아래 중복을 합쳤다. OpenReview Probabilistic TopK는 PDF hash URL과 forum ID가 동일 논문으로 보이지만 저자 및 최종 accept/reject 상태는 미확인이다.

## 검증된 주요 출처

### 1. Variational Garrote for Statistical Physics-based Sparse and Robust Variable Selection

- dedup_key: `arxiv:2509.06383`
- 저자: Hyungjoon Soh, Dongha Lee, Vipul Periwal, Junghyo Jo.
- 2025 preprint, v1 2025-09-08. [공식 arXiv](https://arxiv.org/abs/2509.06383).
- source_kind: local_pdf_first_3_pages + primary_arxiv_abstract.
- 기여: binary support 변수를 mean-field variational inference로 처리하고 현대 자동미분으로 최적화한다. 불필요한 변수가 들어오는 구간에서 generalization과 selector uncertainty의 전이가 나타난다고 보고한다.
- 프로젝트 관련성: VG-SAE의 variance term, selector entropy, sparsity 선택 진단의 직접적 출발점이다.
- 한계: regression의 고정 설명변수에서 얻은 전이를 학습 dictionary와 amortized encoder가 있는 SAE에서 보장하지 않는다. 이 논문 자체가 uncertainty를 이용한 변수 개수 진단을 제안하므로 그 개념만 SAE에 옮기는 것은 신규성 근거가 약하다.
- 본문 주의: local PDF Eq. (6)는 exp(-gamma s)인데 Eq. (9)에는 -gamma sum(m)가 보인다. 구현 및 reproduction notes의 부호 검증을 우선해야 한다. 이 shard는 코드를 재검증하지 않았다.

### 2. Variational Garrote for Sparse Inverse Problems

- dedup_key: `arxiv:2603.12562`; 추가 DOI `10.1016/j.jocs.2026.102980`.
- 저자: Kanghun Lee, Hyungjoon Soh, Junghyo Jo.
- 2026-03-13 preprint. 검색에서 2026년 Journal of Computational Science 출판 페이지도 확인.
- [arXiv](https://arxiv.org/abs/2603.12562), [출판사 원문 페이지](https://www.sciencedirect.com/science/article/pii/S1877750326001985).
- source_kind: primary_arxiv_search_metadata + primary_publisher_abstract_and_intro.
- 기여: signal resampling/denoising, sparse-view CT에서 VG와 shrinkage regularizers를 비교하며 prior–data alignment를 강조한다. 저널 페이지는 LASSO, Elastic Net, SCAD, MCP 비교를 명시한다.
- 한계: fixed transform inverse problems이며 learned LLM feature recovery가 아니다. 단순히 VG가 L1보다 shrinkage가 작다는 주장은 이미 가까운 선행연구가 있다.

### 3. Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders

- dedup_key: `arxiv:2508.16560`.
- 저자: David Chanin, Adrià Garriga-Alonso.
- preprint 2025-08-22, v4 2026-07-07. 로컬 첫쪽에 ICML 2026/PMLR 306 표기 확인.
- [공식 arXiv](https://arxiv.org/abs/2508.16560).
- source_kind: local_pdf_first_page + primary_arxiv_abstract.
- 기여: L0가 너무 낮거나 높으면 correlated feature가 섞이며, 올바른 L0 탐색을 위한 proxy metric을 제안한다. toy에서는 정답 L0, LLM에서는 sparse probing 정점과 일치한다고 보고한다.
- 한계/novelty 위험: **ground-truth 없이 L0를 고르는 새로운 지표**라는 주장에는 이미 직접 경쟁자가 있다. 해당 proxy의 정확한 정의와 실패 조건을 본문에서 읽고 구현해야 한다. 아직 이 shard는 proxy 공식을 읽지 못했다.

### 4. SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data

- dedup_key: `arxiv:2602.14687`.
- 저자: David Chanin, Adrià Garriga-Alonso.
- preprint 2026-02-16, v2 2026-07-13. 로컬 첫쪽은 ICML 2026 Mechanistic Interpretability Workshop 표기.
- [공식 arXiv](https://arxiv.org/abs/2602.14687).
- source_kind: local_pdf_first_3_pages + primary_arxiv_abstract.
- 기여: correlation, hierarchy, superposition, Zipfian firing을 함께 제어하며 ground-truth dictionary와 firing을 제공한다. Gaussian copula로 support를 생성하고 rectified Gaussian으로 amplitude를 생성한다. 16k canonical benchmark와 조정 가능한 generator를 제공한다.
- 첫 3쪽에서 확인한 공식 체크포인트: `https://huggingface.co/decoderesearch/synth-sae-bench-16k-v1` (링크 자체는 아직 열지 않음).
- 중요한 부정증거: reconstruction 개선이 feature recovery 개선을 뜻하지 않는다. Matching Pursuit가 superposition noise에 맞춰 reconstruction만 개선하는 실패를 보고한다.
- 프로젝트 적용: 현재 단순 independent spike-and-slab에서 보이는 VG 이득이 correlation/hierarchy/Zipf로 이동하는지 확인해야 한다. 자체 작은 synthetic generator를 SynthSAEBench 표준 결과라고 표현하면 안 된다.

### 5. Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders

- dedup_key: `arxiv:2505.11756`.
- 저자: David Chanin, Tomáš Dulka, Adrià Garriga-Alonso.
- 2025-05-16, v2 2025-09-26. [공식 arXiv](https://arxiv.org/abs/2505.11756).
- source_kind: primary_arxiv_abstract.
- 기여: true feature 수보다 dictionary가 좁고 feature가 상관될 때 reconstruction objective가 correlated directions의 혼합을 유도한다. 개선된 Matryoshka 변형도 제안한다.
- 한계/적용: support posterior를 부드럽게 만든다는 사실만으로 width bottleneck이나 hedging이 사라지지 않는다. width sweep과 dictionary matching을 분리해야 한다.

### 6. Analysis of Variational Sparse Autoencoders

- dedup_key: `arxiv:2509.22994`.
- 저자: Zachary Baker, Yuxiao Li.
- 2025-09-26, v2 2025-10-01. [공식 arXiv](https://arxiv.org/abs/2509.22994).
- source_kind: primary_arxiv_abstract.
- 기여: Gaussian posterior sampling 및 standard normal KL을 넣은 TopK vSAE를 Pythia-70M residual activations에서 평가한다.
- 강한 부정증거: core metrics에서 standard SAE보다 낮으며, KL pressure와 dead features 증가를 원인으로 보고한다. robustness/independence 일부 이점은 있으나 naive variationalization이 interpretability 개선을 보장하지 않는다.
- 적용: VG의 Bernoulli selector와 analytic expected reconstruction은 이 Gaussian TopK vSAE와 구조가 다르다. 그 차이가 실증적으로 작동함을 보여야 하며 단지 'variational'이라는 명칭으로 우위를 주장하면 안 된다.

### 7. Probabilistic TopK Sparse Autoencoder for Interpreting the Activations of Large Language Models

- dedup_key: `openreview:zMIIHeKivz`.
- 저자: PDF는 Anonymous authors. 정확한 저자 및 최종 출판 상태 미확인.
- 시기: ICLR 2026 submission 표기의 2025 공개 PDF 검색결과.
- [PDF](https://openreview.net/pdf?id=zMIIHeKivz), [동일 제목 hash PDF](https://openreview.net/pdf/4aa8500bc5bab9945032ad8bb18af908b414ad52.pdf), [forum](https://openreview.net/forum?id=zMIIHeKivz).
- source_kind: primary_openreview_pdf_search_extraction. Forum 열람은 browser verification으로 차단됨.
- 기여(abstract에서 확인): binary Concrete distribution으로 TopK SAE에 probabilistic gating을 추가한다. dead component gradient starvation을 줄이고 feature presence confidence와 coefficient magnitude의 상관을 높이는 목적이다.
- **가장 직접적인 novelty 위험:** 'probabilistic gate로 feature confidence를 제공하는 SAE'는 이미 등장했다. calibration 정의, 평가 지표, 실제 Brier/ECE/support NLL 실험 여부는 아직 본문 확인하지 않았다. 미확인을 '그 논문은 calibration을 평가하지 않았다'라는 결론으로 바꾸면 안 된다.

### 8. Variational Sparse Paired Autoencoders (vsPAIR) for Inverse Problems and Uncertainty Quantification

- dedup_key: `arxiv:2602.02948`.
- 저자: Jack Michael Solomon, Rishi Leburu, Matthias Chung.
- 2026-02-03, v3 2026-07-03. [공식 arXiv](https://arxiv.org/abs/2602.02948).
- source_kind: primary_arxiv_abstract + first_version_pdf_search_extract.
- 기여: observation VAE와 quantities-of-interest sparse VAE를 latent mapping으로 연결하여 inverse problems와 uncertainty를 평가한다.
- 검색 PDF는 spike-and-slab prior 및 input-dependent encoder distribution을 명시한다. [검색된 v1 PDF](https://pdfs.assets.alphaxiv.org/2602.02948v1.pdf)는 secondary-hosted 원문이므로 실제 수식 비교는 공식 v3에서 재확인해야 한다.
- 적용/한계: 분야는 CT/inpainting/heat equation으로 SAE mechanistic interpretability와 다르다. 그래도 'spike-and-slab variational autoencoder 자체'는 신규성 주장이 될 수 없다.

### 9. Are Sparse Autoencoder Benchmarks Reliable?

- dedup_key: `arxiv:2605.18229`.
- 저자: David Chanin.
- 2026-05-18 preprint. [공식 arXiv](https://arxiv.org/abs/2605.18229).
- source_kind: primary_arxiv_abstract.
- 기여: evaluation reseed noise, synthetic ground-truth correlation, training trajectory discriminability의 세 축으로 SAEBench를 감사한다.
- 최근 직접 관련 부정증거: canonical TPP/SCR은 여러 기준에서 실패한다고 보고한다. sae-probes k-sparse probing이 비교한 지표 중 가장 신뢰할 만하지만 동일 architecture 변형 간 작은 차이는 여전히 구분하기 어렵다고 한다.
- 적용: stage3에서 단일 SAEBench 점수 몇 개 개선만으로 feature quality 우위를 판정하지 않는다. ground-truth synthetic evidence와 seed uncertainty를 우선한다. 이 preprint의 비판 역시 후속 재현 없이 확정된 학계 합의로 쓰지 않는다.

### 10. Improving Dictionary Learning with Gated Sparse Autoencoders

- dedup_key: `arxiv:2404.16014`.
- 저자: Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Tom Lieberum, Vikrant Varma, János Kramár, Rohin Shah, Neel Nanda.
- 2024-04-24, v2 2024-04-30. [공식 arXiv](https://arxiv.org/abs/2404.16014).
- source_kind: primary_arxiv_abstract.
- 기여: selection과 magnitude estimation을 분리하여 L1 shrinkage를 해결한다.
- novelty 위험: 'selection/amplitude 분리가 새로운 아이디어', 'L1보다 shrinkage를 줄인다'만으로는 부족하다. Gated는 필수 비교군이다.

### 11–13. 제목과 canonical ID만 직접 확인한 핵심 baseline 문헌

아래 공식 arXiv 페이지를 열어 제목과 ID를 확인했으나, batch 출력 제한 때문에 저자/abstract 세부내용은 아직 이 shard에서 읽지 못했다. 최종 metadata 생성 전 추가 확인할 것.

| dedup_key | 제목 | 공식 URL | 상태 |
|---|---|---|---|
| arxiv:2407.14435 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders | https://arxiv.org/abs/2407.14435 | 제목 확인, 메타데이터/본문 추가 확인 필요 |
| arxiv:2406.04093 | Scaling and evaluating sparse autoencoders | https://arxiv.org/abs/2406.04093 | 제목 확인, 메타데이터/본문 추가 확인 필요 |
| arxiv:2412.06410 | BatchTopK Sparse Autoencoders | https://arxiv.org/abs/2412.06410 | 제목 확인, 메타데이터/본문 추가 확인 필요 |

## 검색결과로 발견했으나 추가 검증이 필요한 leads

- 원래 **The variational garrote** (Kappen)의 논문 페이지가 검색에 나왔으나 정확한 canonical ID/DOI/저자를 확보하지 못했다. 2025 논문의 reference [30]을 따라가면 된다.
- **Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training**, [arXiv:2510.08855](https://arxiv.org/abs/2510.08855). 검색 abstract는 probabilistic statistical threshold masking과 Gemma-2-2b absorption 감소를 말한다. 공식 페이지를 아직 열지 않았다.
- **Improving Sparse Autoencoder with Dynamic Attention**, [CVPR 2026 공식 페이지](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Improving_Sparse_Autoencoder_with_Dynamic_Attention_CVPR_2026_paper.html). 검색결과에 Dongsheng Wang, Jinsen Zhang, Dawei Su, Hui Huang, pp. 41996–42006; sparsemax를 활용한 input-dependent concept count가 명시된다. 페이지를 직접 열지 않았으므로 metadata 재검증 필요.
- **SoftSAE**: Marcin Mazur alphaXiv 프로필 검색결과에서 differentiable Soft Top-K로 input-dependent k를 학습하는 논문 설명을 발견했다. 정식 제목/ID/시기/저자 미확인. 자동 sparsity 선택 신규성 조사에서 찾아야 한다.
- **A Perturbation Ladder for Sparse Autoencoder Dictionaries: Seed, Optimizer and Precision Against a Measured Baseline**, [Zenodo](https://zenodo.org/records/21971551), 검색결과상 2026-09-02. peer review 상태와 저자 미확인. 주제는 seed/optimizer/precision perturbation 후 dictionary matching이다. 직접 관련 동시작업 후보이지만 아직 근거로 사용하지 말 것.
- **Train Sparse Autoencoders Efficiently by Utilizing Features Correlation**, [arXiv:2505.22255](https://arxiv.org/abs/2505.22255) 검색결과의 alphaXiv 설명에 KronSAE가 등장. 공식 abstract와 저자 추가 확인 필요.
- **SAEBench**: 잘못 기억한 `2503.09524`를 공식 페이지로 열어 물리학 논문임을 확인했으므로 **이 ID는 사용 금지**. 정확한 SAEBench ID를 제목 검색으로 확보해야 한다.

## 검색 trace

완료된 웹 검색 query:

1. `variational garrote sparse autoencoder spike slab probabilistic uncertainty calibration`
2. `"Sparse but Wrong" "SynthSAEBench"`
3. `"sparse autoencoder" uncertainty calibration 2026`
4. `"Probabilistic TopK Sparse Autoencoder"`
5. `"SynthSAEBench" arxiv`
6. `"Variational Garrote for Sparse Inverse Problems"`
7. `"Sparse Autoencoder" "calibration" "2026" uncertainty`

다음 query는 권한 승인 중 tool call이 중단되어 **실행 완료 근거가 없다**:

- `"SAEBench" arxiv`
- `"The variational garrote" Kappen`
- `"SoftSAE" dynamic top k sparse autoencoder`
- `"sparse autoencoder" "uncertainty" after:2026-03-19 before:2026-09-20`

## 현재까지의 gap 가설 — 검증된 novelty 결론이 아님

1. **VG posterior probability의 의미 검증.** Bernoulli mean-field gate를 만든다고 support probability가 calibrated되지는 않는다. Known dictionary / learned dictionary를 분리하고, Brier/support NLL/reliability/risk–coverage를 비교하면 '좋은 reconstruction'과 '믿을 수 있는 support'를 구분할 수 있다. 다만 Probabilistic TopK의 calibration 관련 실험을 먼저 읽어 차별점을 확인해야 한다.
2. **Sparsity 진단의 범위 조건.** VG regression의 entropy transition과 SAE의 true L0 선택이 연결되는지, correlated/hierarchical/Zipf 환경에서 언제 깨지는지 밝히는 문제는 구체적이다. Sparse but Wrong의 proxy를 필수 경쟁자로 둔다. 'entropy가 자동으로 true L0를 알려준다'는 현재 근거가 없다.
3. **Expected reconstruction와 실제 hard inference의 차이.** 학습 시 posterior mean reconstruction이 좋아도 hard gate 배치에서 support 또는 reconstruction이 악화될 수 있다. 이것을 shrinkage, posterior uncertainty, amortization error로 분해하는 연구 가능성이 있다. 최근 direct competitors의 soft/hard evaluation 관행은 아직 추가 조사 필요하다.
4. **상관된 feature의 불확실성과 dictionary 오류 분리.** Frozen true dictionary에서의 posterior ambiguity와 jointly learned dictionary에서의 feature hedging은 다른 실패다. VG의 이득이 어느 쪽에 있는지 확인하면 단순 architecture 비교보다 진단적 가치가 있다. Mean-field approximation이 correlated alternatives에 과신할 가능성도 가설로 포함한다.
5. **지원하지 않는 broad claim을 좁힐 필요.** Variational SAE 최초, probabilistic gating 최초, shrinkage 해결 최초, automatic L0 최초라는 방향은 위 선행연구들 때문에 약하다. 새 benchmark 전체를 만들기보다 SynthSAEBench 위에 support uncertainty와 soft-to-hard decision evaluation을 추가하는 것이 현재 뼈대와 맞지만, 최종 방향은 pilot evidence 및 경쟁 논문 본문 검토가 결정해야 한다.

## Root에서 전달받은 코드 관찰 — 이 shard 미검증

Root의 메시지: selector prior `sigmoid(-gamma)` 때문에 amplitude≈0에서도 expected L0가 높게 나올 수 있다. 특정 Stage2 artifact에서 gamma 1.99, expected L0≈543, prior floor≈492, hard L0≈30, hard EV≈0.775, expected EV≈0.861을 관찰했다고 한다. 경로 및 run config는 root가 보유하며 이 shard는 숫자를 재검증하지 않았다. 이 현상을 곧바로 dense cheating이라고 부르지 말아야 한다.

후속 targeted search 후보: prior-floor-adjusted information/active-count measure, posterior-prior KL와 support evidence, sparse Bayesian model의 inactive slab identifiability, posterior mean vs median-probability/hard selection, amortized mean-field spike-and-slab dictionary learning. 이 쿼리들은 아직 실행하지 않았다.
