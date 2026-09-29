# ARIS: PI 통합 초안의 최종 과학·문헌 검토

2026-09-30. 검토 대상은 같은 run의 PI_SYNTHESIS.md, PHASE1_PHYSICS_BRIDGE.md, PAPER_PLAN.md, FINAL_PROPOSAL.md, EXPERIMENT_PLAN.md다. 문헌 범위·수식·확률 공간·공동 권고 반영에 집중했다. 실행 수와 분기 감사는 scientific scientist가 별도로 맡는다. 새 학습·코드 수정·실험은 하지 않았다.

## 판정

**공동 권고와 과학적 해석이 일치한다. 수식 부호나 normalizer 오류, RBM과 VG의 등가 주장, soft/hard ensemble 혼동은 발견하지 못했다.** 아래 회전 요약의 조건 한 문장만 명시하면 된다. 이는 이미 정확한 상세 수식을 바꾸는 요청이 아니라 짧은 요약 두 곳의 누락을 고치는 수정이다. 새 모델·실험 확대가 필요하지 않다.

## 수정할 한 항목

**[P2] 회전 요약에도 amplitude 고정 조건을 유지해야 한다.**

- PI_SYNTHESIS.md:33은 x,b,D를 같은 R로 회전하면 conditional posterior가 보존된다고만 쓴다.
- FINAL_PROPOSAL.md:46도 같은 요약이다.
- PHASE1_PHYSICS_BRIDGE.md §4에는 a 고정 및 encoder conjugation 조건이 정확하게 들어 있다.

Input-dependent `a(x)`를 원래 encoder에 그대로 두고 입력만 `Rx`로 바꾸면 일반적으로 `a(Rx) != a(x)`여서 support likelihood가 달라진다. 따라서 두 요약에 아래 조건을 짧게 추가한다.

> “x,b,D를 같은 orthogonal R로 회전하고 a를 고정하거나 encoder도 함수보존변환하면 조건부 support posterior가 보존된다.”

권고한 영문식 encoder 변환 `W'=W R^T`는 column-vector convention 및 bias/centering의 함께 변환 아래 맞다. 새 Adam 학습을 요구하지 않는다. Scientific peer에게 같은 조건 누락을 전달했고, 실행 audit와 독립적인 해석 수정임을 알렸다.

## 확인한 문헌 범위

1. Nature2026의 metadata/abstract/Perspective와 2503.01824v1 공개 저자본의 상세 수식을 구분했다. 출판본 전체의 동일성을 확인하지 못했다는 제한이 PI_SYNTHESIS, PAPER_PLAN, FINAL_PROPOSAL에 있다. 읽지 않은 출판본을 읽었다고 승격한 곳은 없다.
2. PRL2017의 arXiv 본문·저자 출판본 PDF·supplement 확인은 팀 전체의 읽기 기록과 부합한다. ARIS는 supplement §III A–B를, scientific는 저자 출판본 PDF를 확인했다. 전체 replica 계산을 재증명했다는 주장은 없다.
3. PRL2020의 q를 receptive-field weight correlation으로, 학습 posterior를 global weights에 대한 것으로 구분했다. Support density 또는 per-input posterior와 혼동하지 않았다.
4. Eq. (8)의 fixed-D squared-error+L1 convexity 지적은 공개 preprint의 그 식에 한정되어 있다. 최종 Nature판의 동일 문장을 확인했다는 주장이 없고, joint dictionary learning 및 L0 문제와 구분되어 있다.
5. 세 논문으로 신규성·성공·일반 identifiability를 확정하지 않고 기존 VG/variational SAE/stability 문헌을 유지했다.

## 확인한 수식과 가정

- `p(s_j=1)=sigmoid(-gamma)`에 대응하는 energy의 단항은 `+gamma*N`이다. Energy의 pair coefficient는 `+beta*u_i^T*u_j`, log posterior에서는 음수다. 교차항에 불필요한 factor 2가 붙지 않았다.
- `Ztilde_x=sum_s exp[-beta*error/2-gamma*N]`를 conditional partition으로 쓴 것이 맞다. Normalized prior의 `-K*log(1+exp(-gamma))`는 posterior 정규화에는 취소되지만 marginal evidence에는 남는다는 설명도 맞다.
- Fixed x,D,a,beta에서 `-∂gamma E[N]=Var(N)`와 diagonal/covariance 분해가 맞다. 여기서 m은 exact marginal이다. 이 결과를 joint/profiled path의 total derivative나 retraining variance와 같다고 하지 않는다.
- Factorized free energy의 m-Hessian은 diagonal likelihood curvature와 Bernoulli variance correction이 상쇄되어 `H_ii=1/[m_i(1-m_i)]`, off-diagonal은 `beta*u_i^T*u_j`가 된다. Stable interior nonsingular branch의 `dm/dgamma=-H^{-1}1` 및 count response도 맞다. 경계·singularity·branch switch를 별도 상태로 다룬다.
- Numeric residual/eigenvalue cutoffs는 suggested contract로 표시되어 있으며 정리의 보편적 상수처럼 제시하지 않는다. 기존 exact field-response 검증을 새 결과로 중복 계산하지 않는다.
- Orthogonal D에서 factorization이라는 결론에는 product support prior 조건이 유지되어 있다. Co-firing prior가 생겼을 때 같은 결론을 자동 적용하지 않는다.
- Pair Bernoulli teacher의 P11/P10/P01/P00는 합이 1이고 두 marginal p를 보존한다. 제안된 p=.25,c_s=.6에서는 각 확률이 유효하다. 이 후속 대조를 Hou–Huang의 global-weight posterior 재현이라고 부르지 않는다.
- PR3와 hard L0, true support count, semantic compositionality가 분리되어 있다.

## 확인한 primary와 challenger 의미

- `E_opt + rho_hard + U_hard`가 operational primary다. `E_joint`는 independent training datasets와 optimizer가 함께 변하는 결합 ensemble이며 순수 data variance로 부르지 않는다.
- 원 Soh의 soft-total+data-realization에서 ensemble과 observable 두 축을 바꿨다는 가설이 명시되어 있다.
- Stable-correct/incorrect의 flatness 반례는 hard-between에 적용하며 동일 soft m의 within uncertainty가 남을 수 있음을 명시했다.
- Soft 후보는 native soft density로 별도 mapping·coverage·alignment를 만들고, hard-matched soft plot을 별개 진단으로 부른다. Hard finite-R correction을 soft-total에 곱하지 않는다.
- `U_total=U_within+V_between`는 보정 없는 finite-ensemble identity로 유지된다. Bayesian calibration, semantic truth, population confidence interval로 승격하지 않는다.
- Shared reference를 ensemble에서 제외하고 alternative common anchor에서 reference membership도 변한다는 제한을 추가했다. 이 대조를 순수 anchor effect로 부르지 않은 것은 정확하다.
- Primary 변경은 개발 결과와 이유를 남긴 새 version 및 untouched confirmation을 요구한다. 결과가 좋은 challenger만 남기거나 기존 primary의 성공으로 이름을 바꾸는 경로는 없다.

## 공동 권고의 반영

작은 전체 곡선 horizon screen, shared-reference challenger, 4k-adaptive freeze와 basic17 우선, prior2×2의 SHOULD 하향, PR3 선택진단, same-bank/L1 결과 무관 보고, 큰 objective ablation의 조건부화가 모두 유지됐다. VG-SAE와 density inference 목표가 diagnostic-only 연구나 RBM 재현으로 바뀌지 않았다. C2a/C2b 및 estimate/deployment 구분도 보존됐다.

새 substantive 충돌은 발견하지 못했다. 어떤 ensemble/readout이 실제로 유효한지는 경험적 미해결로 남아 있고, 이 검토는 그 성공을 보증하지 않는다. 위 한 문장 수정 후 과학·문헌·수식 측면에서 canonical 반영을 진행해도 된다.
