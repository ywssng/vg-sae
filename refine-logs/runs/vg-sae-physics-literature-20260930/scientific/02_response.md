# Scientific 교차반론 및 입장 수정

2026-09-30. 상대의 `aris/01_position.md`를 읽고 ARIS scientist와 직접 메시지로 논의했다. 독립본 `01_position.md`는 당시 입장으로 보존한다. 이 문서는 동의·수정·남은 이견의 기록이며 새 실험 결과가 아니다.

## 강한 동의

ARIS가 명시한 probability-space 구분과 Gram interaction→covariance→MF susceptibility의 연결에 동의한다. 특히 stable branch에서 `1ᵀH⁻¹1`이 optimized MF response이고 `Σm(1−m)`와 일반적으로 다르다는 계산은 기존 Phase 1의 설명력을 높인다. 모든 계산은 fixed D,a,beta 조건과 Hessian의 안정성을 명시해야 한다. 신규 물리 정리로 포장할 필요가 없다.

Converged-correct flat-hard-U 반례를 양쪽이 독립적으로 제기했다. 따라서 현 C0의3 gamma보다 전체17-control bank의2k/4k/8k 검사를 먼저 해야 한다는 데 동의한다. 첫 screen은 기존 M1의 family별 한 world, 총136/최대200fits로 시작할 수 있다. ARIS의544/800fits는 최초 진입의 최소조건이 아니라 두 ensemble 정책을 비교하는 개발 단계 재배분 상한으로 이해한다.

현재 C2a generating density와 C2b recovery selection 분리, blind selector, native readout, 실패·보류 보존, 36 fresh worlds의 독립 단위는 유지한다. 큰 ablation이 자동으로 필수라는 전제는 수정한다. 모든 source의 용어를 그대로 SAE에 옮기지 않는다는 점과 compressed-sensing guarantee를 joint SAE 식별성에 붙이지 않는다는 점도 동의한다.

ARIS가 지적한 Perspective preprint Eq.(8)의 오류에도 동의한다. 표시된 fixed-dictionary squared-error+L1 objective는 coefficient에 대해 convex다. NP-hard sparse L0 또는 joint dictionary-learning 문제와 구별하여 그 문장을 우리 논문에 복사하지 않는다.

## 실제 이견 1: 어느 ensemble을 primary로 삼을 것인가

내 독립본은 현재 optimizer-given-data estimator를 primary로 명시적으로 제한하고 data resampling을 secondary로 제안했다. 상대는 repeat마다 독립 training data를 둔 challenger가 원 VG template의 전이를 더 직접 시험한다고 주장했다. PI 의제에 따라 [Soh §III.1.4, Eqs.(18)–(19)](https://arxiv.org/html/2509.06383v1)를 직접 추가 읽은 후, **독립 data challenger를 반드시 비교해야 한다**는 방향으로 내 입장을 강화한다. 원문은 dataset realizations를 언급하며 soft ensemble uncertainty를 정의한다.

다만 repeat마다 dataset과 initialization을 함께 바꾸는 challenger의 U는 순수 data variance가 아니다. `optimizer | fixed dataset`과 `data + optimizer`의 두 정책 비교다. 동일 initialization ID를 두 arm에서 pairing하면 contrast가 명료해지지만, variance component를 분리한 것은 아니다. Pure data/optimizer 분해를 논하려면 같은 D에서 여러 datasets×여러 optimizer seeds의 작은 crossed/nested 설계가 필요하다. 첫 화면에서는 정확한 이름을 붙인 두 arm으로 충분하고, crossed design 전체를 선행 필수로 삼지 않는다.

내 잔여 이견은 **아직 실행하지 않은9월29일 계획의 primary를 무조건 보존할 이유는 없다**는 점이다. 원 template에 충실한 data policy를 주가설로 변경하는 것도 현재 계획 보완 요청 안에서 과학적으로 가능하다. 반면 ARIS는 seed-only primary를 유지하고 data policy를 명시적 challenger로 두어 post-selection을 막는 쪽을 선호한다. 두 입장은 density 목표에 대한 이견이 아니라 설계의 연속성 대 원문 ensemble의 충실성에 대한 우선순위 차이다.

공동으로 지킬 수 있는 기준은 다음과 같다. 두 가설의 이름·metrics·개발 결과를 모두 공개하고, 개발에서 더 작은 truth error를 보였다는 이유만으로 실패한 arm을 지우지 않는다. Confirmatory data를 보기 전에 primary와 전체 protocol을 잠근다. Primary를 바꾸면 새로운 method/protocol version이며 이전 가설의 pilot 실패·제한을 그대로 남긴다. 지금 문헌만으로 어느 ensemble이 잘 작동한다고 합의하지 않는다.

## 상대와 PI가 놓치지 말아야 할 observable 변경

Soh의 soft `mbar(1−mbar)`와 현재 corrected hard-between U는 ensemble 축 외에도 관측량이 다르다. Stable-correct **hard** masks가 같은 경우 U_hard는0이다. 같은 soft m이 모든 repeat에서 반복되어도 0<m<1이면 soft-total에는 within term이 남는다. 따라서 flat-hard 반례를 original soft-total 전체의 소실로 확장하면 안 된다.

Pilot은 같은 saved models에서 다음을 모두 기록하되 서로 같은 q estimator라고 이름 붙이지 않아야 한다.

- `rho_hard`와 hard-between curve.
- `rho_soft=mean(m)`와 soft-total curve, within-soft와 between-soft 분해.
- 각 curve를 계산한 ensemble policy와 optimization horizon.

특히 hard-density matching으로 checkpoint를 선정했더라도 soft template fit의 x축까지 hard density로 사용하면 원 template의 계산과 달라진다. Soft curve의 native mean occupancy와 실제 coverage를 따로 저장한다. 다양한 조합을 계산하는 것은 failure analysis이며 best combination을 final primary로 몰래 바꾸는 허가가 아니다.

## 실제 이견 2: correlated-prior exact 검사의 우선순위

내 독립본은 geometry×co-firing 24 target cells를 학습0인 초기 gate로 제안했다. ARIS는 기존 independent-prior exact fixture의 covariance/response를 먼저 재평가하고 새 prior target은 후속으로 두자는 의견이다.

이 지적을 수용한다. 학습0이어도 새 prior normalization·MF target 구현·검증 비용은 존재한다. 해당2×2는 density pilot 착수를 막는 gate에서 **제한된 Phase 1 기전 추가**로 내린다. 독립 support 적용 범위를 넓히거나 correlated stress 실패를 설명해야 할 때 실행한다. 필요성 자체는 유지한다. Decoder overlap만 바꾸는 기존 실험으로 support co-firing의 영향을 측정했다고 할 수 없기 때문이다.

내가 독립본에 넣지 않았던 PI의 orthogonal ambient rotation control을 채택한다. Fixed x→Rx,D→RD에서 isotropic likelihood/Gram/posterior는 보존되지만 coordinate weight sparsity는 바뀔 수 있다. 이것은 RBM connection sparsity를 VG occupancy에 그대로 대입할 수 없음을 보여주는 작은0-fit control이다. 변환된 encoder까지 함수 보존하도록 명시하거나 fixed conditional model에서만 계산해야 하며 Adam 재학습 trajectory의 회전 불변성을 주장하지 않는다.

## Full-curve horizon 비교의 구체 계약

가장 단순한 primary horizon check는 **기본17개의 동일 gamma**를 모든 run에서8k까지 학습하여2k/4k/8k를 저장하고, 각 horizon의 achieved density와 U를 다시 계산하는 것이다. Horizon별 missing coverage는 결과이고 숨기지 않는다. q를 직접 비교할 때는 동일 retained-target 교집합 결과와 각 시점의 전체 eligible status를 함께 낸다. 교집합에서만 성능이 좋아진 경우를 전체 안정성으로 쓰지 않는다.

Adaptive controls는 별도 진단으로 둔다. Pilot의 규정 horizon4k에서만 현재 label-free midpoint rule로 최대8개의 controls를 추가하며, 새 control도8k까지 학습해 모든 snapshot을 얻는다. 이4k-selected extended bank를 고정하여 모든 horizon을 평가한다. 8k에서 보이는 coverage hole 때문에 추가 controls를 계속 붙이지 않는다. Basic17 결과와 extended 결과를 분리하므로 adaptive selection의 영향을 볼 수 있다. 이 방법은 horizon마다 서로 다른 bank를 만든 후 q만 비교하는 혼란을 줄인다. 비용에는 모든8k continuation을 포함한다.

이는 제안이며 fixed4k adaptation이 모든 horizon을 공정하게 최적화한다는 뜻은 아니다. 동일 gamma/budget 변화의 estimand를 우선한다. PI가 더 복잡한 horizon-union rule을 택한다면 candidate union·추가 fit cap·tie ordering을 사전에 정해야 한다.

## Ablation/baseline의 사전 branch rule

후속 비용을 조건부로 만들면서도 유리한 comparator만 남기는 일을 막으려면 결과에 따른 행동을 미리 정해야 한다.

| 개발/bridge 상태 | 사전 후속 행동 | 주장 처리 |
|---|---|---|
| Coverage/shape 구현이 측정 불가능 | A0/coverage 문제만 해결하고 lock 갱신. 큰 main/ablation은 시작하지 않음 | estimator 성능을 평가했다고 하지 않음 |
| Recovery가 좋고 hard-U가 flat, 또는 horizon에 따라 signal 소실 | fixed-data hypothesis 한계로 기록. 이름 붙인 data+optimizer challenger와 soft diagnostics를 계획대로 완료 | 물리적 density 신호나 broad C2a 주장 보류 |
| 한 policy가 측정 가능하고 budget 민감도가 허용 범위 | 그 policy를 새 protocol에서 명시적으로 잠그거나, 기존 seed-primary를 유지한 채 둘 다 보고하는 설계 중 PI가 선택. Fresh main 이전 고정 | 개발 success는 confirmatory 근거가 아님 |
| Main 실행 진입 | 기존 same-bank reconstruction/c_dec/raw-U/within-peak/uniform comparator는 결과에 상관없이 전부 평가 | 불리한 comparator 삭제 금지 |
| VG 고유성 주장을 유지 | 사전 정한 small L1 bank를 main 결과가 좋거나 나빠도 평가 | Generic stability 성분이면 그렇게 귀속 |
| 세 objective ablation의 full bank | 첫 실행 전에 subset·methods·endpoint·진입조건을 잠금. Claim에 필요 없으면 시작 전 scope에서 제거 | 나중에 유리한 ablation만 primary 근거로 선택하지 않음 |

Main 결과가 나쁜 뒤 원인을 설명하려고 추가한 실험은 명시적 exploratory follow-up이다. C2b superiority의 comparator나 test bank를 바꾸지 않는다. 광범위 ablation을 줄이는 것은 주장도 좁히는 결정이지 성공 기준 완화가 아니다.

## 지금 공동으로 권고할 수 있는 최소안

1. 새 architecture 없이 conditional interaction, exact covariance, stable MF response, rotation control을0-fit bridge로 명시한다.
2. Stable-correct와 stable-wrong를 함께 A0에 넣는다. Stability는 correctness 판정기가 아님을 데이터로 확인할 수 있게 한다.
3. 기본17-control full-curve horizon screen을 family별 한 개발 world에서 먼저 수행한다. 현재 C0 세 gamma는 대체한다.
4. 같은 population의 independent-training-data challenger를 label-free 동일 evaluation bank에서 비교한다. Pure data variance로 부르지 않는다.
5. Hard/soft observables와 각 native density axis를 저장한다. 주방법의 선택은 confirmatory 전에 version을 잠그며 모든 bridge arm을 공개한다.
6. Existing M1 정도의 development budget에서 두 정책을 비교하는 재배분을 고려한다. 7k–10k 전체 bank는 최초 실행 단위가 아니다.
7. Original density-estimation 목표와 C2a/C2b를 유지하고, large LM·thermodynamic phase diagram·new prior model은 후속으로 둔다.

현재 미해결은 ensemble primary의 우선순위다. 이견을 감추어 어떤 것이 과학적으로 확정됐다고 쓰지 않는다. PI가 기존 protocol의 연속성을 택해 seed-primary를 유지하면 그 결정에 따를 수 있지만, source fidelity 때문에 향후 data policy가 primary가 되어야 할 가능성은 명시적으로 열어둬야 한다.


## 직접 재논의 후 해소된 쟁점과 최종 공동 권고

위 본문은 토론 과정의 실제 이견을 기록한다. 이후 ARIS의 직접 회신을 받아 다음 operational 합의에 도달했다.

- 현재 미실행 계획의 seed-only estimator를 **운영상 primary로 유지**할 수 있다. 이것이 이론적으로 우월하거나 영구 고정된 정의라는 뜻은 아니다. Independent-data challenger를 별도 가설로 반드시 공개하며, primary를 바꿀 경우 새 protocol version과 untouched-world 확증을 요구한다. 이 조건 아래 내가 제기한 primary 변경 가능성 우려는 해소됐다.
- 두 arm은 `optimizer-given-data`와 `joint-data-and-optimizer`로 명명한다. 순수 variance component 분해는 하지 않는다. Stable-correct flat 반례는 hard-between에 관한 것으로 한정한다. Soft-total과 hard-between은 자기 native density axis를 사용한다.
- Basic17 gamma를8k까지 학습해2k/4k/8k를 보존한다. 최대8 adaptive controls는 규정4k에서만 결정한다. 새 control도 처음부터8k까지 학습하고 모든 snapshots를 저장한다. Basic17 결과를 먼저 보고하며 extended bank는4k에 조건부로 선택된 보조 진단이다. Horizon에 대칭적으로 공정한 bank라고 하지 않는다. 모든 continuation/adaptive 비용을 합산한다.
- 현재 same-bank selectors는 main의 성공 여부와 무관하게 모두 보고한다. VG 고유성 주장을 유지하면 사전 고정한 small L1 비교도 결과와 무관하게 실행한다. Broad objective ablation은 사전 branch와 claim scope에 따라 조건부로 두며 불리한 결과를 없애지 않는다.
- Correlated-prior2×2는 density pilot의 필수 착수 gate에서 후속 기전검사로 이동한다. Rotation은 fixed-model algebraic0-fit control로 채택하고 새 Adam rotation grid로 확대하지 않는다.

양측 사이에 남은 operational 이견은 없다. **어느 ensemble/observable이 실제로 generating density를 예측하는지,3-repeat screen의 신호가 fresh worlds에서 유지되는지, finite-budget dependence가 얼마나 큰지는 여전히 미확인 과학적 질문**이다. 합의가 이 결과들을 확정한 것으로 쓰면 안 된다. PI는 이 합의의 근거와 비용을 확인하여 canonical plan에 채택·수정할 책임을 갖는다.

## 교차 토론 뒤 추가 원문 확인

ARIS가 알려준 [Tubiana–Monasson 저자 supplement](https://www.phys.ens.fr/~monasson/Articles/a105-si.pdf)를 웹 도구로 다시 시도했으나 내 요청에서는 접근 실패했다. 대신 상대가 원 URL에서 확보한 저장소 내부 source PDF의 추출문을 읽어 §III A–B, D 및 F를 확인했다. Participation-ratio exponent3의 근거가 strong/weak activation scaling과 동등 magnetization 가정에 의존한다는 점이 직접 명시되어 있다. 초기 독립본의 “supplement 미확인”은 당시 읽기 상태이며 지금은 위 선택적 section까지 확인했다. 전체 training/sampling 방법을 감사한 것은 아니다.
