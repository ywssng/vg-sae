# Sparse but Wrong 기준 연결 문헌 갱신

확인일: 2026-09-22. 연구 목표는 L1 진폭 벌점에 의존하지 않는 VG-SAE 개발과 실제 feature 복원이다. 이 문서는 문헌 근거를 제공하며 후보 선택·신규성 판정·실험 결과를 대신하지 않는다. 기존 32개 registry를 재사용하고 AFA, MDL-SAE, Anthropic JumpReLU recipe 3개를 추가했다. 참조 논문 정독은 별도 `REF_PAPER_SUMMARY.md`가 담당한다.

## 현재 목표에서 문헌이 말하는 것

[Sparse but Wrong](https://arxiv.org/abs/2508.16560)의 잘못된 L0와 feature 혼합 문제는 “L1을 다른 함수로 바꾸면 해결된다”는 결론이 아니다. 주요 비교인 BatchTopK와 JumpReLU도 이미 L1 진폭 shrinkage를 피한다. 따라서 VG의 support/amplitude 분리는 출발점이며, **고정 gamma를 포함한 희소성 설정의 변화에도 참 feature와 firing pattern을 유지하는지**가 추가 검증 대상이다. `c_dec`는 decoder 열 사이의 pairwise cosine 기반 통계다. 입력에 대한 decoder projection, support firing correlation, 참 dictionary와의 정렬은 각각 다른 양이다.

## 핵심 15개 논문

`현재 확인`은 이번 원문·metadata 확인, `재사용`은 09-21에 기록한 본문 독해를 뜻한다. arXiv만 확인된 논문은 정식 학회 채택으로 올려 적지 않았다. 서지사항 전체와 독해 범위는 `literature_sources.json`에 있다.

| 논문·저자·상태 | 확인된 방법 또는 결과 | 현재 fixed-gamma VG-SAE와의 관계 / unknown L0의 한계 |
|---|---|---|
| [The Variational Garrote](https://doi.org/10.1007/s10994-013-5427-7), Kappen & Gómez, Machine Learning 2014 (preprint 2011) | Binary selection의 mean-field 분포와 point coefficient를 결합. 현재 metadata + 기존 본문 재사용. | support/amplitude 분리와 VG free energy 자체의 직접 선행. 알려진 predictor를 고르는 회귀와 학습 dictionary 복원은 다르며 sparsity prior는 남는다. |
| [VG for Statistical Physics-based Sparse and Robust Variable Selection](https://arxiv.org/abs/2509.06383), Soh, Lee, Periwal & Jo, 2025 preprint | 자동미분 VG와 희소 회귀 실험; selection uncertainty의 전이 신호 제안. local PDF 1–3쪽·metadata 현재 확인. | unknown support 크기 선택의 동기가 이미 존재한다. 이 전이 신호가 overcomplete learned dictionary에서 true L0를 추정한다는 결과는 없다. |
| [VG for Sparse Inverse Problems](https://arxiv.org/html/2603.12562v1), Lee, Soh & Jo, 2026; [J. Computational Science 100, 102980](https://doi.org/10.1016/j.jocs.2026.102980) | 고정 transform/forward operator의 복원에 VG 사용. §2.4·3에서 gamma sweep과 minimum generalization error 비교 확인. | Bernoulli gate, variance, entropy, beta profiling이 선행. gamma 자동 추정 증거가 아니며 sparse prior/data 정합성이 중요하다. 저널 인쇄월은 2026-10, online 날짜는 확인되지 않았다. |
| [Improving Sparse Decomposition of Language Model Activations with Gated Sparse Autoencoders](https://proceedings.neurips.cc/paper_files/paper/2024/hash/01772a8b0420baec00c4d59fe2fbace6-Abstract-Conference.html), Rajamanoharan et al., NeurIPS 2024 | Gate와 magnitude를 분리하고 sparsity penalty를 gate 쪽에 적용해 shrinkage를 완화. | “selection/amplitude 분리”만으로 신규성 주장 불가. gate penalty 강도를 고르는 문제는 남으므로 강한 baseline이다. arXiv의 이전 제목은 Improving Dictionary Learning…이다. |
| [Jumping Ahead](https://arxiv.org/abs/2407.14435), Rajamanoharan et al., 2024 preprint | JumpReLU threshold와 STE, L0 penalty. 현재 metadata·기존 방법 독해 재사용. | L1을 벗어나는 직접 경쟁. learned threshold와 penalty coefficient는 true L0 추정 보장이 아니다. 아래 Anthropic variant와 objective를 구분해야 한다. |
| [Scaling and evaluating sparse autoencoders](https://proceedings.iclr.cc/paper_files/paper/2025/hash/42ef3308c230942d223c411adf182c88-Abstract-Conference.html), Gao et al., ICLR 2025 (preprint 2024) | TopK로 입력별 sparse count를 직접 제어하고 dead latent·대규모 학습을 개선. 공식 proceedings 현재 확인. | MSE/L0 frontier 외 feature-quality 지표도 제시한다. 지정 k는 true count를 알려주지 않으며 잘못 지정한 sparsity 비교의 기준이다. |
| [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410), Bussmann, Leask & Nanda, 2024 preprint | Batch 전체에 TopK budget을 적용해 입력별 활성 수를 허용. | 입력별 L0 변동은 자동적인 참 sparsity 추정과 다르다. 평균 k 설정의 민감도가 이번 anchor의 직접 비교 대상이다. |
| [Evaluating and Designing SAEs by Approximating Quasi-Orthogonality](https://arxiv.org/html/2503.24277v2), Lee, Davies, Canby & Hockenmaier, COLM 2025 | Approximate Feature Activation(AFA): 입력 norm과 latent norm의 관계로 top-AFA 활성 수를 정함. §4·7·Appendix A 현재 확인. | unknown k를 다루는 직접 경쟁. λ_AFA가 남고 높은 L0가 한계로 명시된다. norm matching과 ε_LBO는 true feature identity·support의 보증이 아니다. |
| [Interpretability as Compression: … MDL-SAEs](https://arxiv.org/html/2410.11179v1), Ayonrinde, Pearce & Sharkey, 2024 preprint | 설명 길이로 width/L0를 고르는 관점과 MNIST 사례. §2·5·Appendix A 현재 확인. | sparsity 수치 대신 모델 선택 기준을 두는 직접 선행. coding scheme·복원 허용오차 선택이 필요하며 원문은 true generating factor 보장을 하지 않는다. |
| [Feature Hedging](https://arxiv.org/html/2505.11756v2), Chanin, Dulka & Garriga-Alonso, NeurIPS 2025 MI workshop | 참 feature보다 좁은 dictionary에서 상관 feature가 MSE 때문에 섞이는 현상. 본문과 workshop PDF 현재 확인. | support prior를 바꾸어도 누락 feature의 복원 압력이 남을 수 있다. width와 firing correlation을 별도로 다뤄야 한다. 모든 SAE에서 같은 실패가 필연적이라는 일반 정리는 아니다. |
| [Sparse Autoencoders, Again?](https://proceedings.mlr.press/v267/lu25w.html), Lu, Zhu, He & Wipf, ICML 2025 | VAEase의 Gaussian uncertainty gate, trainable decoder noise. 공식 proceedings 현재 확인; 세부 수식은 09-21 독해 재사용. | uncertainty 기반 gate와 noise learning의 직접 경쟁. Bernoulli inclusion + point amplitude와는 다르다. 확인된 LLM 설정(d=512, latent=300)은 overcomplete SAE recovery 증거가 아니다. |
| [Variational Sparse Coding](https://proceedings.mlr.press/v115/tonolini20a.html), Tonolini, Jensen & Murray-Smith, UAI 2019 / proceedings 2020 | amortized spike-and-slab, analytic KL, sampled likelihood, spike warm-up. 공식 proceedings 현재 확인; 본문 재사용. | Bernoulli support와 continuous slab의 amortized sparse coding은 선행. VG point amplitude는 Gaussian slab의 zero-variance finite-KL 극한으로 정당화되지 않는다. |
| [Learning Sparse Codes with Entropy-Based ELBOs](https://proceedings.mlr.press/v238/velychko24a.html), Velychko, Damm, Fischer & Lücke, AISTATS 2024 | Laplace prior·Gaussian posterior의 analytic entropy ELBO와 amortized/correlated approximation. proceedings 현재 확인; §1–4·부록 재사용. | analytic objective, entropy, noise 최적화 자체는 신규하지 않다. 현재 VG의 discrete support와 amplitude 조건부 risk라는 차이를 정확히 적어야 한다. |
| [SoftSAE](https://arxiv.org/html/2605.06610v1), Stępień, Mazur, Tabor & Spurek, 2026 preprint | MLP가 입력별 k̂를 예측하고 soft TopK 학습 뒤 hard TopK 추론. §3·Appendix E 현재 확인. | adaptive selection의 직접 경쟁이지만 target K, k_max=2k와 k-loss를 사용한다. 동적 배분과 true L0 자동 발견을 구분한다. 큰 dictionary의 계산비용도 보고한다. |
| [Toward Identifiable Sparse Autoencoders](https://arxiv.org/abs/2605.31245), Nelson, Karaletsos & Locatello, 2026; ICML 2026은 arXiv 저자 comment 확인 | TopK 변형 iSAE, dictionary 안정성과 approximate restricted-isometry 기반 code 식별성 주장. 현재 abstract/metadata까지 확인. | feature identity/seed stability의 최근 직접 비교 대상. 이론의 전체 가정·unknown-L0 범위는 본문 미확인으로 남기며 무조건 복원 보장으로 인용하지 않는다. |

## 실제 JumpReLU baseline의 구분

[Conerly, Cunningham, Templeton, Lindsey, Hosmer & Jermyn의 공식 Anthropic recipe](https://transformer-circuits.pub/2025/january-update/index.html)는 2025년 technical post다. 같은 JumpReLU activation을 사용하지만 decoder norm을 곱한 activation의 **tanh sparsity penalty**, inactive feature의 pre-act loss, threshold 이외 parameter로도 흐르는 STE를 명시한다. 이는 Rajamanoharan et al.의 L0-loss variant와 동일하지 않다. 비교 시 명칭만 맞추지 말고 loss·gradient·dead-feature recipe를 기록한다. 이 recipe도 sparsity coefficient를 사용하므로 unknown L0의 정답을 제공하는 기준으로 삼지 않는다.

## 이번 목표에서의 연결과 남은 근거

L1 shrinkage의 대안이라는 큰 방향은 Gated, JumpReLU, TopK 및 variational sparse coding과 겹친다. 현 VG-SAE의 구체적 조합은 Bernoulli support, 입력별 point amplitude, 선형 decoder에서의 analytic expected risk, normalized support KL, learned/profiled precision이다. 이를 **full generative ELBO 또는 calibrated semantic posterior**로 확대해 설명할 근거는 없다. 관련 구성요소가 선행에 존재한다는 점과 이 조합이 실제 correlated-feature 복원에서 어떤 이득을 주는지는 별개 질문이다.

미지의 L0를 다루려면 세 항목을 나눠야 한다. BatchTopK/SoftSAE는 정한 budget 안의 **입력별 배분**, AFA/MDL은 **활성 수·모델 선택 규칙**, VG 고정 gamma는 **선택 prior의 강도**다. AFA·MDL·VG의 uncertainty/transition 신호도 selector 후보이지 ground truth가 아니다. 참 support와 dictionary를 아는 합성 데이터에서 selection rule을 먼저 검증하고, 실제 activation에서는 label/probe/causal utility가 제공하는 제한된 검증을 구분할 필요가 있다. Ground-truth recovery로 고른 oracle 결과와 실제 사용 가능한 rule의 결과를 섞으면 unknown-L0 문제를 해결했다는 결론을 낼 수 없다.

Width 부족, 잘못된 sparsity, amortization, 최적화는 모두 다른 실패 원인이다. 09-21 registry의 [Compute Optimal Inference and Provable Amortisation Gap](https://proceedings.mlr.press/v267/o-neill25a.html), [Variational Sparse Coding with Learned Thresholding](https://proceedings.mlr.press/v162/fallah22a.html), Probabilistic TopK lead를 삭제하지 않았다. 다만 수치 없는 새로운 방법 주장을 위해 끌어오지 않는다. Probabilistic TopK는 저자·최종 심사 결과가 계속 미확인이며, “확률 gate를 처음 사용한다”는 넓은 주장에 대한 중복 가능성으로만 기록한다.

## 최근 6개월 확인과 검색 범위

검색 창은 **2026-03-22–2026-09-22**다. arxiv_fetch.py의 날짜 제한 검색과 adaptive/identifiable/description-length 검색이 각각 12개 metadata 결과를 반환했다. 두 결과를 합쳐 canonical arXiv ID로 중복 제거한 수는 `literature_queries.json`에 기록했다. broad query의 응용 논문은 핵심 표에 억지로 넣지 않았다. exact phrase, mechanism, alternative spelling, prior family, time-window, venue-verification 및 reference chaining을 사용했고 원문 abstract는 저장하지 않았다. Zotero/Obsidian은 노출된 도구가 없어 미사용이며 `refs/`의 local VG PDF와 기존 32개 기록을 재사용했다.

최근 직접 관련 문헌은 SoftSAE(5월), iSAE(5월), [Are Sparse Autoencoder Benchmarks Reliable?](https://arxiv.org/abs/2605.18229)(5월), [A Dominant Diffuse Phase in the SAE Phase Diagram](https://arxiv.org/abs/2609.10299)(9월)다. 마지막 논문은 학습된 SAE의 diffuse solution과 objective minimizer를 구분해야 한다는 최근 preprint다. 현재는 abstract 수준의 저자 보고로만 사용하며 Sparse but Wrong의 정리나 VG-SAE 실험을 반박·입증한 것으로 취급하지 않는다. VG inverse(3월 13일)는 정확한 6개월 창 밖이지만 직접 선행이므로 포함했다.

추가 발견한 SOSAE(2507.04644)는 title/authors metadata만 확인한 후속 lead다. Diagnosing and Fixing Latent Recovery in Sparse Autoencoders(OpenReview Hl3rEn7S4P)는 primary PDF 검색 추출까지의 lead로 남겼다. AFA는 COLM 공식 accepted list, TopK는 ICLR 공식 proceedings, Feature Hedging은 저자명이 있는 workshop PDF로 venue를 갱신했다. iSAE/Probabilistic TopK/MDL의 OpenReview direct access는 challenge에 막혔으므로 심사 상태를 추정하지 않았다. 이 검색은 범위가 정해진 문헌 갱신이며 신규성이나 문헌 완전성의 증명이 아니다.
