# VG-SAE: 이산 선택의 변분 원리와 SAE의 추론·학습

사용자가 선택한 주제는 VG 기반 SAE 개발이며, 중심 기여 의도는 통계물리학의 모형 가정과 변분원리로 기존 SAE의 목적함수·선택·불확실성을 이해하는 것이다. 성능 우위는 가능한 기여이며 이번 초기 캠페인의 필수 판정 조건은 아니다.

명시한 Bernoulli support prior p(s_j=1)=sigmoid(−γ), Gaussian observation noise precisionβ, point amplitude a, decoder D에 대해 conditional support free energy를 분석한다. Factorized q에서 expected squared reconstruction은 predictive-mean error와 Bernoulli variance correction으로 분해된다. Prior cost와 entropy가 선택 확률을 결정한다.

Amplitude1·고정dictionary의 작은 생성모형에서는 exact posterior를 전부 계산할 수 있어 normalized Bayesian 기준과 actual support labels가 모두 존재한다. 이 control에서 mean-field와 encoder를 분리한다. 이후 joint SAE에서는 amplitude가 입력에 의존하므로 같은 식을 full generative ELBO나 calibrated semantic posterior라고 부르지 않는다.

핵심 후보 주장 C1은 overlap에 따른 조건부 posterior dependence와 factorized inference의 한계, C2는 frozen 추론오차와 joint 학습의 영향을 구분하는 설명이다. Known identities를 새 정리라 주장하지 않고 3-world 결과는 초기 evidence로 보고한다. 실험 조건·반증 기준·실행 budget은 같은 폴더의 EXPERIMENT_PLAN 및 config가 정한다.

관련 이론: [VG](https://arxiv.org/abs/2509.06383), [Sparse but Wrong](https://arxiv.org/abs/2508.16560), [VAEase](https://arxiv.org/abs/2506.04859), [Entropy-Based ELBOs](https://arxiv.org/abs/2311.01888). 새 baseline 전체 재구현이나 대규모 LLM benchmark는 이번 범위에 넣지 않는다.
