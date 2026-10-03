# VG-SAE 통합 논문 계획: Phase 1에서 Phase 2까지

**2026-10-03 현재 상태.** Dot-cloud 이관본의 10월1–2일 최종 결과를 반영한 [연구 상태 감사](refine-logs/runs/vg-sae-colab-continuation-20261003/RESEARCH_STATE_AUDIT.md)가 아래 과거 실행 상태보다 우선한다. BR5는 `measurement_unresolved`, W2는 gated, C2a/C2b는 unentered다. 실제 LLM activation은 사용자가 직접 진행할 후속 범위로 남긴다. 기존 음성 학습 분기를 반복하지 않고, [새 CPU 검증 계획](refine-logs/runs/vg-sae-colab-continuation-20261003/EXECUTION_PROTOCOL.md)에 따라 이론·수치 증거의 열린 항목을 진행한다.

2026-09-30 문헌·토론 반영판. 상태: **Phase 1 초기 근거 보유 / Phase 2 계획·미실행**. 사용자가 요청한 원래 density-estimation 목표와 최근 first-principles 검증을 한 논문으로 연결한다.

**작업 제목:** *Variational Garrote Sparse Autoencoders: From Selection Free Energy to Density Inference*.

**이번 개정:** 세 지정 논문과 기존 문헌을 검토한 두 과학자의 토론을 반영했다. Phase 1의 관측량을 더 정확히 정의하고, Phase 2 앞에 ensemble·soft/hard·학습 예산을 구분하는 작은 bridge를 넣었다. 큰 확인 실험은 그 뒤에 protocol을 잠근 후 진행한다. 검토 자체는 새 실험 결과가 아니다.

**논문 유형:** 모형·추론 설명을 기반으로 한 SAE 방법 및 추정 연구. 투고처 미확정이며 기존 PRE 스타일 원고를 기준으로 구성한다. 아래10쪽은 내부 본문 예산으로 공식 venue page limit이 아니다. References/appendix는 별도다.

## 논문이 답할 하나의 질문

**VG의 선택모형에서 얻는 불확실성을 명확히 구분하면, SAE의 적절한 활성 밀도를 정답 없이 추정하고 그 적용 조건을 설명할 수 있는가?**

논문의 흐름은 **잘못된 L0에서도 좋은 재구성이 가능함 → VG-SAE의 모형과 자유에너지 → interaction·추론·관측량의 구분 → 어떤 training ensemble에서 밀도 신호가 생기는가 → density 추정과 별도 recovery 검증**이다. Phase 1은 모형이 무엇을 말하는지 확인하고, Phase 2는 그 관측량이 실제 선택에 유용한지 별도로 시험한다. 이 둘의 연결을 가정하지 않고 검증하는 것이 논문을 묶는 논리다.

원래 제목 *L0 Is Not a Free Parameter*와 중심 동기는 보존한다. 다만 제목이 단정하는 추정 효용은 Phase 2 결과를 보고 선택한다. 이번에는 기존 LaTeX 원고를 덮어쓰거나 논문 결과를 미리 채우지 않는다.

## 주장–근거–그림 대응

| 주장 | 근거/필요 실험 | 현재 상태와 제한 | 위치 |
|---|---|---|---|
| C1: VG-SAE의 선택 자유에너지 항과 추론 근사의 역할을 구분할 수 있다 | 9월24일 exact135 cells/frozen9/joint54, 수식·구현 감사 | 기존 작은 조건의 근거 있음. 새 response/rotation 재평가는 계획. Known identity는 새 정리 아님. F 개선≠marginal accuracy 개선 | §3, Fig2 |
| C2a: specified synthetic에서 generating density를 blind하게 추정할 수 있다 | ensemble/readout bridge 후 protocol lock, 새 36worlds, 여러 true densities, blind matching과 error/abstention | 미검증. Expected generating count가 유일한 good operating count라고 가정하지 않음 | §4–5, Fig1/3, Table1 |
| C2b: 그 추정으로 recovery가 좋은 operating point를 선택한다 | 같은 VG bank의 reconstruction/c_dec/raw-U/Vpost selectors, independent test NMSE/regret, all-world fallback policy | 미검증. Oracle는 evaluator 전용. Accepted-only 성능으로 전체 성공을 주장하지 않음 | §5, Fig3/4, Table1 |
| 범위 분석 | 결과 무관 same-bank/L1 비교; objective/TopK/분포 확장은 사전 조건부 | 계획. 독립된 세 번째 논문 기여로 늘리지 않고 C1–C2 연결을 설명 | §5–6, Fig4, appendix |

C2a와 C2b는 주된 기여 C2의 서로 다른 endpoint다. C1은 이를 해석하기 위한 기반 기여다. 큰 LM 성능 우위를 연구 성립의 필수조건으로 만들지 않는다.

## 문헌에서 가져오는 구조와 경계

- **Klindt et al.**의 구분을 따라 representation 식별성, 알려진 D의 sparse inference, dictionary 학습, human interpretability를 분리한다. 저자본의 가정·차원·global optimum 조건을 일반 LM/SAE의 보장으로 확대하지 않는다.
- **Tubiana–Monasson**에서 weight connectivity, hard occupancy, dominant-amplitude count가 다른 변수라는 점을 반영한다. PR3는 필요할 때의 보조 지표이며 RBM의 compositional phase 또는 L~1/p를 VG에 적용하지 않는다.
- **Hou–Huang**에서 prior/likelihood matching, truth-overlap와 self-consistency, 데이터 수의 역할을 참고한다. Global RF-weight posterior와 per-input support posterior, SGD outcomes는 다른 대상이다.
- **Soh의 원래 식과의 차이:** soft-total 및 data realizations에서 출발한 template를 현재의 fixed-data hard-between curve에 옮기는 과정 자체가 가설이다. 측정한 확률 공간과 density축을 먼저 고정한다.

상세 source/version/access-depth와 채택·유보 판단은 `refine-logs/runs/vg-sae-physics-literature-20260930/PI_SYNTHESIS.md`를 따른다. Nature의 세부 수식은 공개2025저자본을 읽었으며2026출판본 전체와 동일하다고 단정하지 않는다.

## 불확실성과 response를 연결하는 이론 항목

1. **Conditional free energy:** Bernoulli support, Gaussian observation, input-dependent point amplitude의 가정과 normalized prior를 쓴다. Full generative ELBO는 적절히 정한 고정-amplitude control에서만 말한다.
2. **Expected risk:** mean-code reconstruction과 Bernoulli variance correction의 합. 실제 sampling risk와 hard-code risk는 구분한다.
3. **Inference errors:** 같은 target에서 exact→MF와 trained encoder→best-found MF를 구분한다. Best-found를 global optimum으로 부르지 않는다.
4. **Within/between decomposition:** 정렬된 gate에서 `mbar(1-mbar)=mean m(1-m)+Var_r(m)`. 알려진 대수 항등식의 적용이며 semantic identity 보장 아님.
5. **Interaction과 response:** support energy의 pair term은 beta*a_i*a_j*d_i^T*d_j다. Exact count variance는 diagonal+covariance이고 stable MF response는1^T H^{-1}1이다. Gamma별 재학습 곡선과 구분한다. 고정 모형의 회전 대조로 coordinate weight sparsity와 occupancy의 차이도 드러낸다.
6. **Template transfer 가정:** 아래 pooled-support 계산을 본문에 짧게 제시하고 상세 전개는 appendix에 둔다. True slot correspondence와 selection exchangeability가 왜 필요한지 설명한다.

고정된 `(x,j)` 좌표에서 truth active fraction을 q, repeat별 선택확률을 pi[x, j]라 하자. True active/inactive 그룹의 pi 평균을 mu1/mu0, 그룹 내 분산을 v1/v0라 두면:

`rho=q*mu1+(1-q)*mu0`

`U=q*mu1*(1-mu1)+(1-q)*mu0*(1-mu0)-q*v1-(1-q)*v0`.

Under-selection에서 false positive가 없고(mu0=0), true-active 좌표의 pi가 균일하면(v1=0), `U=rho/q*(q-rho)`가 된다. Over-selection에서 false negative가 없고(mu1=1), inactive 좌표의 pi가 균일하면(v0=0), `U=(rho-q)*(1-rho)/(1-q)`가 된다. 이 계산은 Soh의 가정을 입력별 support에 적용한 충분조건 설명이며 **학습된 SAE가 이 조건을 만족한다는 정리도, 새 일반 식별성 정리도 아니다**. Heterogeneous rates는 해당 분산항을 만들고, FP/FN과 잘못된 정렬은 더 큰 편차를 만든다. Native density matching은 fixed-field response와 다른 estimand다.

`q`는 유한 평가 좌표의 true fraction이지만 Phase2 primary error의 target은 generating expected density다. Input sample noise 때문에 둘이 다를 수 있어 둘 다 기록한다. Oracle truth는 이 가정의 사후 진단에만 사용한다.

## 먼저 검증할 ensemble bridge

| 대상 | 바꾸는 것 | 해석 |
|---|---|---|
| Fixed-model posterior | 같은x, D, a, beta에서 support s | 모형 내 conditional uncertainty |
| E_opt | 같은 teacher/dataset에서 initialization·batch order | optimizer-given-data instability; 현재 operational primary |
| E_joint | 같은 teacher에서 dataset와 optimizer를 함께 변경 | joint-data-and-optimizer challenger; 순수 data variance 아님 |

Hard-between은hard rho축, soft-total은별도로matching한soft rho축에서분석한다. 같은 model을재사용해within-soft/between-soft를분해한다. Stable-correct와stable-wrong hardcode 모두U_hard=0일수있지만, 동일soft m에는within uncertainty가남을수있다. 따라서 flatness로correctness를판정하지않는다.

첫bridge는2teacher worlds와공유reference로238–350 distinctfits를제안한다. 모든 fit의2k/4k/8k 전체곡선을보존하며먼저같은basic17controls를비교한다. 4k에서만고른extended bank는보조다. 이비용은단계별제안이고새 학습은아직없다. Sharedreference의data 비대칭을추가 training없는alternative-anchor 진단으로기록한다.

36-world 확인실험은bridge와나머지개발후의조건부 후속이다. 같은 bank의비교selector와작은 L1은main결과에상관없이모두보고한다. 어떤challenger가좋아보여도확증결과를보고primary를바꾸지않는다. 개발근거에따른변경은새 version과untouched-world 확인을요구한다.

## Phase 2의 방법을 한 문단으로

같은 training data에서 여러 초기화로 VG-SAE를 학습한다. 정답 없이 얻은 native hard density로 checkpoint를 맞추고, 별도 reference decoder에 feature를 정렬한다. 같은 입력에서 선택이 얼마나 반복되는지 곡선을 만든다. 전체 곡선을 단일 VG template에 맞춰 밀도를 추정하되 coverage·fit·식별성이 부족하면 보류한다. 추정한 밀도에서 실제 checkpoint를 골라 untouched test의 feature recovery를 평가한다. 원래 NNLS mixture의 normalized weights를 posterior나 confidence interval로 읽지 않는다. 새 trainable network는 추가하지 않는다.

정확한 target/control grids, tolerance, bootstrap, abstention, seeds, 판정·예산은 저장소의 `refine-logs/EXPERIMENT_PLAN.md`가 단일 기준이며 이 run 폴더에도 동일 사본을 보존한다. 큰 흐름의 새로운 아이디어와 수치 protocol을 여러 파일에 중복 정의하지 않는다.

## 본문 구성: 7개 절, 총10쪽

| 절 | 분량 | 문단의 역할과 반드시 포함할 내용 | 근거/연결 |
|---|---:|---|---|
| §1 Introduction | 1.25 | 재구성–feature identity 문제 → 왜 density가 추론 대상인가 → VG 접근 → 두 단계와 두 endpoint → 관측 결과 범위 | Sparse but Wrong, VG, Fig1 |
| §2 Related Work | 1.00 | (1) free-energy/variational sparse coding, (2) density/feature identity, (3) cross-seed alignment·identifiability. 단순 논문 나열 대신 공통점과 남는 간격 설명 | LITERATURE_GROUNDING |
| §3 Selection model and inference | 1.75 | 가정·interaction → exact/MF/encoder → response와ensemble 구분 → template 충분조건. Phase1 Fig2를 여기서 해석 | 기존 Phase1, 위 수식 |
| §4 Density inference from repeated SAE fits | 1.75 | 추정 대상 → world/repeat/splits → native density matching → truth-free alignment → profile fitting/보류 → deployment. Algorithm1 | Phase2 protocol |
| §5 Experiments | 3.00 | setup .4쪽, ensemble/horizon bridge .8쪽, density+recovery main1.2쪽, small L1·핵심 failure .6쪽. 큰 benchmark 표를 필수로 강제하지 않음 | Fig3/4, Table1 |
| §6 Scope and limitations | .75 | stable-wrong, granularity/duplicates, capacity, sample/optimization uncertainty, Phase2가 실패한 조건과 남는 기여 | 사전 결과해석표 |
| §7 Conclusion | .50 | 실제 지지된 C1/C2만 정리. 성공과 실패 양쪽에서 다음 구체적 질문 제시 | 결과 확정 뒤 작성 |

Abstract는150–220단어를 목표로 하되 지금은 구성만 정한다. 첫 문장: density 선택 문제. 다음: VG-SAE의 모형과 uncertainty-informed estimator. 다음: Phase1의 실제 observation과 Phase2의 **미입력 결과 자리**. 마지막: 확인된 적용 범위. Phase2 수치·성공은 실험 전 작성하지 않는다.

## 그림·표 계획

| ID | 내용/비교 | 데이터 | 우선순위 |
|---|---|---|---|
| Fig1 | 논문의 핵심: uncertainty curve·추정 density와 independently evaluated recovery optimum의 관계 | Phase2 planned curve_points/density_estimates/selection_metrics | 핵심, 미생성 |
| Fig2 | exact/MF/encoder·covariance/response와 mean/stochastic/hard risk를 압축 | `outputs/first_principles_20260924/`의 기존 CSV/그림 | 기존 원본 존재; 새 response 분해·rotation은 계획 |
| Fig3 | bridge의ensemble/readout/horizon 차이와 main의density error·abstention·paired regret | Phase2 planned bridge_summary/world_summary | 핵심, 미생성 |
| Fig4 | 실행한small L1·selector/기전 비교, stable-correct/incorrect 또는 정렬 실패 사례 | mechanism subset와 지정 failure controls | 핵심, 미생성 |
| Table1 | C2a error+coverage, C2b all-world NMSE/regret, comparator별 비용. 6cell 및 pooled 집계 | Phase2 planned summary | 핵심 |
| TableA1 | 모든 seed/control, failures, native density coverage, original-paper/SAELens recipe 차이 | manifest/history/protocol | 부록 |

**Fig1 구체 구성:** 왼쪽은 모형→반복학습→blind estimator→평가의 작은 diagram. 가운데는 사전 지정 world의 achieved hard density–U 점, single-template fit, q_hat, stability band 또는 abstention reason. 오른쪽은 동일 world의 recovery NMSE와 hard-MSE curve. Truth/recovery-optimum 선은 evaluator overlay로만 그리며 selector input이 아님을 표시한다. Fig3에 모든world가 들어가므로 좋은 사례만 고를 수 없다. 사전 지정world가 실패하면 실패 그림을 그대로 쓴다.

**Fig1 캡션 초안:** “VG-SAE의 선택 불확실성에서 operating density를 추정하는 절차와 독립 평가. 추정기는 정답 feature와 test recovery를 사용하지 않는다. 정답 및 hindsight recovery optimum은 평가용으로만 겹쳐 표시하며, 추정 보류도 결과에 포함한다.” 성공을 미리 전제하는 문장은 결과 확인 뒤에만 추가한다.

Theory comparison은 기존 정리의 bound를 새로 개선한 논문이 아니므로 허구의 bound table을 만들지 않는다. 대신 §2에서 모델 가정·uncertainty 정의·density 선택 여부를 비교하는 짧은 표를 사용한다. 긴 derivation, complete tests, mixture audit, L1/TopK 상세, conditional Stage2/3는 appendix로 보낸다.

## 문헌 배치

- §1: Soh VG와 Sparse but Wrong가 추론 도구와 문제 동기를 제공하고, Klindt 등의 framework는 단계별 가정·평가의 위치를 잡는다.
- §2: Hinton–Zemel의 historical free energy, VAEase, Learned Thresholding, Entropy-Based ELBOs와 다른 latent/amplitude/posterior 선택을 비교한다.
- §2/4: Toward Identifiable Sparse Autoencoders와 Unstable Features, Reproducible Subspaces를 통해 matching·stability가 이미 있는 도구임을 인정하고, 본 논문의 density-inference 질문을 구분한다.
- §3: Soh의 원래 모형·template를 귀속하고, Tubiana–Monasson의 order parameter 구분과 Hou–Huang의 matching 조건을 비교한다. Conditional response와optimizer ensemble은같다고하지않는다.
- §5: pin된 SAELens recipe와 데이터 artifact를 기록한다. Method별 original paper citation은 최종 사용 family가 정해진 뒤 공식 논문과 대조한다.

기존 URL/metadata는9월29일 `LITERATURE_GROUNDING.md`, 이번세논문의source ledger와범위는9월30일 `PI_SYNTHESIS.md` 및agent독립입장문에있다. 신규성은 후보 수준이며 정식 novelty audit 완료로 표시하지 않는다.

## 결과에 따른 논문 주장

| C2a 생성 밀도 | C2b recovery 선택 | 허용되는 결론 |
|---|---|---|
| 지지 | 지지 | 지정 조건에서 density inference와 유용한 selection이 함께 성립. 두 단계의 가장 강한 연결 |
| 지지 | 비지지 | count를 맞히는 것만으로 좋은 representation을 보장하지 않음. 실용적 selection 주장은 제한 |
| 비지지 | 지지 | operating-density heuristic 효용은 있으나 generating-density estimator 주장은 비지지 |
| 비지지 | 비지지 | 이 VG→SAE template transfer의 실패 범위를 보고. Phase1 성공을 C2 성공으로 대체하지 않으며 논문 기여의 충분성은 재평가 |

추정 보류가 많으면 별도 failure result다. 실패 때문에 원래 연구를 다른 clone 진단 연구로 바꾸지 않는다. Negative 결과의 출판 가치를 자동 보장하지도 않는다.

## 기존 계획에서 유지·보완·유보한 것

- 유지: VG-SAE, first-principles 의도, 원래 cross-realization density estimation, synthetic truth 검증, learned/profiled beta, 장기 Stage2/3 방향.
- 보완: conditional interaction/response, 확률공간과readout 구분, teacher→dataset→optimizer 계층, full-curve horizon, learned coordinate대응, native density matching, 보류·all-world policy.
- 유보: NNLS mixture를 Bayesian posterior로 해석, uncertainty 최솟값만으로 density 선택, LLM true feature count 주장, 거대한 phase diagram/HPO, 새 teacher/encoder/learned gamma.
- 단계화:0-fit 물리대조와238–350fit bridge →개발·lock →조건부core synthetic/작은 L1 →필요한기전·분포확장 →조건부SynthSAEBench/LM. 현재 문서 작성은 이 실행을 시작하지 않는다.

## 다음 작업과 인계

1. Phase1 기존결과에interaction/response를연결하고0-fit axis·회전·퇴화대조를구현.
2. 확률공간을구분한bank schema와238–350fit bridge, 나머지개발후config lock.
3. 조건부P2-B와작은 L1 fresh evaluation, 결과–주장표갱신.
4. Fig1/3/4·Table1 생성 후 기존 원고를 이 outline에 맞춰 개정.

최신 실행 상태는 저장소의 `refine-logs/EXPERIMENT_TRACKER.md`가 기준이다. 현재 Phase2는 전부TODO다. 전체 계획의 비용은 measured timing 이전의 제안이고, 사용자에게서 받은 실행 승인을 임의로 확대하지 않는다.


## 검토 반영

9월29일의3라운드READY는당시계획의준비도판정이었다. 이번에는요청한ARIS scientist와scientific-skills scientist가독립문헌검토, 실제교차반론, 공동 권고와endorsement를남겼고PI가채택·수정했다. 기존 점수를새설계의성공 근거로옮기지않는다.

같은모델계열의협업이므로cross-family 검증이라고하지않는다. 방법론provenance에Scientific Agent Skills를명시했으며물리학주장의근거로쓰지않는다. 토론과PI판정은 `refine-logs/runs/vg-sae-physics-literature-20260930/PI_SYNTHESIS.md`를따른다. 어떤ensemble/readout이효과적인지는실험 전 미확인이다.
