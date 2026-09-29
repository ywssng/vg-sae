# Phase 1 보강: interaction, response, ensemble의 구분

2026-09-30. 문헌에서 영감을 얻은 프로젝트 모형의 설명·평가 계획이다. 아래 대수 관계는 알려진 원리의 적용이며 새 실험 결과나 신규 정리로 제시하지 않는다. 기존9월24일의135 exact cells/9frozen/54joint fits는 그대로 보존한다.

## 1. Conditional VG energy와 RBM의 차이

Fixed `r=x-b`, `u_j=a_j d_j`, beta>0, independent prior `p(s_j=1)=sigmoid(-gamma)`에서:

\[
\widetilde Z_x(\gamma)=\sum_s \exp[-\tfrac\beta2\|r-\sum_j u_js_j\|^2-\gamma N(s)].
\]

이는 conditional posterior의 partition sum이다. Normalized prior의 gamma-dependent factor는 posterior 정규화에서 취소되지만 generative evidence를 gamma로 미분할 때는 취소하지 못한다. Input-dependent point amplitude를 쓰는 SAE의 objective를 full generative evidence로 바꾸지 않는다.

Energy의 전개는

\[
E_x(s)=\tfrac\beta2\|r\|^2+
\sum_j(\gamma+\tfrac\beta2\|u_j\|^2-\beta u_j^Tr)s_j+
\beta\sum_{i<j}u_i^Tu_j s_i s_j.
\]

Log posterior의 pair term은 음수 부호다. Off-diagonal Gram interaction과 prior independence가 posterior dependence를 결정한다. RBM의 bipartite energy·hidden conditional factorization과 같은 모형이라고 부르지 않는다. Orthogonal Gram과 product prior일 때의 factorization을 correlated-support prior로 자동 확장하지 않는다.

## 2. Exact·MF·재학습 response

\[
-\partial_\gamma\mathbb E[N\mid x]
=\operatorname{Var}(N\mid x)
=\sum_j m_j(1-m_j)+2\sum_{i<j}\operatorname{Cov}(s_i, s_j\mid x).
\]

Density response는 위 값을K로 나눈다. 같은 식의 finite-difference 확인은 기존 Phase1에서 이미 수행됐다. 후속은 covariance의 diagonal/off-diagonal 기여를 분리해 현재의 overlap 결과를 해석하는 재평가다. 새 독립seed 또는 새 발견으로 세지 않는다.

Factorized free energy의 interior stationary point에서, 고정D, a, beta의 Hessian은

\[
H_{ii}=\frac1{m_i(1-m_i)},\qquad H_{ij}=\beta u_i^Tu_j\ (i\ne j).
\]

Stable differentiable branch에서

\[
\frac{d m}{d\gamma}=-H^{-1}\mathbf 1,
\qquad -\frac{d\sum_j m_j}{d\gamma}=\mathbf1^TH^{-1}\mathbf1.
\]

이 response는 factorized q의 `Var_q(N)=sum m(1-m)`와 일반적으로 다르다. Boundary/singular Hessian/branch switch를 별도 상태로 기록한다. Suggested numerical contract: fixed-point residual<=1e-6, 모든m이(1e-8,1−1e-8), smallest eigenvalue>1e-6인 branch에서만 inverse-Hessian comparison을 해석한다. 이 범위 밖 sample을 제거하지 않고 status로 남긴다. 같은branch continuation의 finite difference를 사용하며 여러start 중해가바뀐 차이를 local derivative로 부르지 않는다.

Joint D/a/encoder 및 learned/profiled beta를 gamma별로 바꾼 response에는 다른 total-derivative 항이 들어간다. Retraining instability도 posterior variance와 다른 확률 공간이다.

## 3. 기존 orthogonal frozen 결과의 의미

Unit D, a=1, b=0의 orthogonal control에서 exact logit은 `beta*d_j^T*x-beta/2-gamma`다. Linear gate로 표현할 수 있으므로 기존0.36nats 잔여KL을 구조적 capacity 한계로 읽지 않는다. Analytic-parameter witness 및 저장된 checkpoint의target 일치를 먼저 확인한다. 추가 학습은 필요성이 확인될 경우 기존3orthogonal fits의고정4k/8k continuation으로한정하고, 최대3×6000=18000 추가updates를 별도계상한다. 이번 문서 작업에서는 실행하지 않았다.

## 4. 회전 대조 / 학습0

R^TR=I인 ambient transform에서 `x'=Rx`, `b'=Rb`, `D'=RD`로 바꾸고 a를 고정하면 squared residual, Gram, support posterior와conditional response는동일하다. Amplitude/gate encoder를 같이 쓸 때에는 input weights를 `W'=W R^T`로 변환해 같은 a, m을 보존해야 한다.

반면 coordinate-wise decoder sparsity/participation은달라질수있다. 따라서 PRL2017의특정 visible basis에서정의된 sparse connection p를 VG의activation density로대체하지않는다. Adam을회전된데이터에서새로학습하는실험과 이함수 보존변환을구분한다.

Contract: 사전 고정한 dense orthogonal R과 identity를 같은tiny fixtures에적용하고 float64 posterior/energy차이tol1e-8를확인한다. 이는모형의sanity이며 새로운 성능 실험이아니다.

## 5. Prior dependence의후속oracle / SHOULD

Main prior는 계속independent다. 확장이필요하면 K=8, a=1, p=.25, beta8에서 첫pair의decoder cosine c_D∈{0,.9}, support co-firing c_s∈{0,.6}를독립조작한다. Bernoulli pair는

`P11=p²+c_s*p*(1-p)`, `P10=P01=p*(1-p)*(1-c_s)`, `P00=(1-p)²+c_s*p*(1-p)`.

나머지support는독립으로두어 expected count=Kp를유지한다. 3blockedworlds×4teacher cells×2student prior choices=24target evaluations(중복된independent-prior대조표시), training fits0. 각 target의normalized exact/MF를비교하고, 서로다른posterior간의KL을같은inference target의근사오차로혼동하지않는다.

Matched prior exact와independent prior exact의차이는 model misspecification이고, 각target에서exact/MF차이는approximation이다. 이는 Hou–Huang의 global-weight posterior를재현하는실험이아니다. 새target normalization·gradient 검증이필요하므로density pilot의착수 gate로강제하지않는다.

## 6. Amplitude composition / 선택진단

\[
PR_3(z)=\frac{(\sum_j |z_j|^3)^2}{\sum_j |z_j|^6}.
\]

Zero code는0. Unit-direction effective coefficients를사용하고hard L0와함께표시한다. 같은hard count에서일부coefficient가압도하면PR3가작아질수있다. 이는dominant-amplitude count이며true support·회복된feature수·semantic compositionality가아니다. RBM의large-N scaling이나`L~1/p`를이숫자의해석근거로직접이식하지않는다.

## 7. 세단계검증의경계

알려진D에서의recoverability, 학습한D의identifiability, encoder의inference quality, human interpretability를서로대체하지않는다. Compressed-sensing의order relation만으로우리finiteK와학습된D의고정threshold를정하지않는다. Bernoulli constant-amplitude finite support의구별문제와continuous-amplitude worst-case recovery도구분한다.

원문 위치/버전과채택판정은 `PI_SYNTHESIS.md` 및두agent 입장문을따른다. Existingvariance deletion 결과는beta반응을포함한total effect로유지한다. Fixed-beta term-isolation이새주장에필요하면동일x/D/a/beta의counterfactual부터계획하며profiled삭제결과를직접인과효과로소급변경하지않는다.
