# ARIS → scientific 교차 검토와 응답

2026-09-30. 독립본 `01_position.md` 저장 후 scientific/01_position.md와 PI_AGENDA.md를 읽고 상대 연구자에게 직접 질문하고 답변받았다. Same-family provisional 논의다. 새 학습·코드 수정은 없다.

## 강한 동의

Scientific의 가장 중요한 보완은 **전체 17-control 곡선에서 2k/4k/8k horizon을 비교해야 한다**는 것이다. 현 C0의 gamma 0,2,4만으로는 q profile, native density coverage, abstention 변화까지 판단할 수 없다. 같은 input과 teacher에서 recovery가 개선되는데 U가 평평해지는 경우는 실패한 SAE와 실패한 density estimator를 분리한다. 이 구별이 본검증보다 먼저라는 점에 동의한다.

첫 학습은 기존 M1 일부인 2 development worlds × 4 runs × 17 controls = **136 fits, coverage 추가 포함 최대 200 fits**로 시작할 수 있다. 제 독립본의 544/800 fits는 두 ensemble 정책을 포함한 개발 규모의 재배분 예시였으며, 처음 정보를 얻는 데 필요한 최소량을 뜻하지 않는다. 8k 연장의 비용은 new fit 수와 별도로 4k-step equivalents에 반영한다. 어느 시점에서 추가 control이 발견됐는지가 horizon 비교를 교란하지 않도록, 기본17 bank의 같은-control 비교를 먼저 보고해야 한다.

또한 amplitude=1의 유한 binary-support 생성모형에서는 d<K만으로 모든 support가 원리적으로 식별 불가능하다고 말할 수 없다는 반례를 수용한다. Random continuous dictionary의 서로 다른 유한 support sums는 noiseless에서는 구별될 수 있다. 실제 한계는 가까운 sums, noise, finite data, 학습한 decoder와 inference를 함께 검토해야 한다. 제 독립본의 compressed-sensing 보장 유보는 그대로 유효하지만 undercomplete라는 이유로 불가능성을 선언해서는 안 된다.

## 실제 이견 1: 데이터 ensemble을 무엇으로 측정할 것인가

Scientific S3의 “같은 D에서 training dataset 하나를 더 만들고 각각 여러 optimizer repeats”는 dataset 조건부 불안정성의 민감도를 조사한다. 그러나 두 dataset만으로 원 VG의 data-realization ensemble을 충분히 추정한다고 부르면 안 된다. 나는 작은 challenger arm에서 **각 repeat에 독립 training examples를 주고 같은 evaluation bank에서 U를 계산**하자고 제안했다.

Scientific는 이를 수용하면서 중요한 한계를 지적했다. 그 arm에서는 initialization도 달라지므로 순수 data variance가 아니라 data와 optimizer의 결합 변동이다. 이 지적을 수용한다. 두 arm의 initialization ID를 pairing해 비교의 잡음을 줄이더라도 data 성분만 분리된다는 뜻은 아니다.

합의한 명칭과 역할은 다음과 같다.

| Arm | 무작위성 | 해석 |
|---|---|---|
| 현재 primary | 동일 training dataset, initialization/batch order 변경 | `optimizer_given_dataset` |
| 작은 challenger | 같은 teacher, repeat별 독립 training samples, initialization/batch order 변경 | `joint_data_and_optimizer` |
| 후속 nested/crossed control | 여러 data draws마다 여러 optimizer runs | 분리된 variance component를 주장할 때만 필요 |

두번째 arm은 첫 bank와 같은 2 worlds × 4 runs ×17 controls라면 **추가136 fits / 최대200 fits**다. 첫 screen에서 작은 차이를 관찰하는 설계이며 데이터 분산 성분을 정확하게 추정하거나 broad C2a를 검증하는 설계가 아니다. Control 전체에서 각 repeat의 training dataset은 일관되게 유지한다. 참 teacher는 생성·평가에만 쓰고 selector에는 전달하지 않는다.

**남은 판단 차이:** 나는 현재 문서의 seed-only primary를 operational 기준으로 유지하면서 challenger를 추가하는 쪽을 선호했다. Scientific는 아직 미실행 계획이므로 이론적으로 더 타당한 ensemble로 primary를 바꾸는 것도 정당하다고 보았다. 논의 후 나는 primary 변경 자체를 막을 이유가 없다는 점을 수용했다. 다만 지금 결과 없이 data ensemble이 더 잘 된다고 단정하지 않으며, 개발 중 좋은 true-density error를 보인 arm을 숨겨서 primary로 승격해서는 안 된다. 이번 원고는 현재의 primary와 challenger를 명명하고, 재정의는 근거·모든 개발 결과·새 version을 남긴 뒤 untouched confirmatory worlds에서 검증한다. 어떤 ensemble이 실제로 유용한지는 미해결 경험적 질문으로 보존한다.

## 실제 이견 2: correlated-prior 정확검사를 첫 gate로 둘 것인가

Scientific S0는 geometry×support-correlation의 2×2와 matched/mismatched prior를 24 tiny target cells로 비교한다. 좋은 기전검사이지만, 새 correlated support posterior를 정확열거와 MF에 지원하는 구현 비용이 있고 새로운 target family가 된다. 나는 이미 있는 independent-prior exact fixtures를 먼저 활용하고 이것을 density pilot의 선행 의무에서 내리자고 제안했다.

Scientific는 “training fit이0이어도 target 구현 비용은 있다”는 이유로 수용했다. 합의는 다음과 같다.

- 기존 Phase 1의 full covariance, diagonal/off-diagonal count variance와 기존 exact field-response 결과를 우선 재사용한다.
- MF Hessian response를 이 고정 target에 연결하는 설명을 추가할 수 있다. 이것은 알려진 local stationary-branch 관계이며 새 학습모형이 아니다.
- Co-firing prior×geometry matched/mismatched 검사는 작은 후속 기전 항목으로 남긴다. 실제 모델에서 support independence 범위를 조사하는 가치는 있으나 큰 density pilot을 막는 gate로 삼지 않는다.
- Trainable correlated-prior SAE, 새 decoder prior, EM/gamma learner는 이번 필수 scope로 넣지 않는다.

상대가 놓칠 수 있었던 반례는 단순히 correlated prior가 중요하다는 주장에 있지 않다. 현재 homogeneous independent-support estimator부터 아직 작동하지 않는 상황에서는 새 target의 정밀한 failure analysis를 끝내도 Phase 2의 핵심 ensemble-transfer 가설은 그대로 미검증이다. 새 zero-fit 기전검사가 자동으로 우선순위가 높지는 않다.

## 상대의 중요한 반론: 원 VG의 soft-total과 새 hard-between은 다르다

이 점은 제 독립본의 강조가 부족했으므로 명시적으로 수정한다. [Soh Eq. (18)](https://arxiv.org/html/2509.06383v1)는 **soft mask의 ensemble mean**을 사용한다. 현재 Phase 2 primary는 hard threshold 후 across-run variance다. 바뀐 것은 ensemble 하나가 아니라 observable까지 두 축이다.

고정 correspondence에서

`soft_total = mean_xj[mbar(1-mbar)]`

`= mean_rxj[m(1-m)] + mean_xj Var_r(m)`.

모든 repeat가 같은 soft m을 내더라도 `soft_total=m(1-m)`는 남을 수 있다. 따라서 “올바르게 수렴하면 모든 U가0”이라는 반례는 **hard-between 및 soft-between**에 대한 것이며 원 논문의 soft-total 전체에 그대로 적용되지 않는다. 모든 m이 hard0/1일 때에는 soft-total도0이다. 이 수정은 원 VG가 자동으로 SAE density를 식별한다는 뜻이 아니다. 그 soft-total에는 posterior 내부 ambiguity와 between-fit variation이 함께 들어 있으므로 현재 C1–C2 연결이 오히려 더 중요해진다.

같은 checkpoint bank에서 학습 추가 없이 다음 두 후보를 계산할 수 있다.

1. 기존 native hard-density matched `U_hard` 대 `rho_hard`.
2. Soft observable을 그에 맞는 native `rho_soft` 축과 별도의 density-matching 계약 아래 계산한 곡선.

Hard density로 매칭한 checkpoint들의 soft-U를 평균 soft density에 그리는 것은 그림은 만들 수 있어도, repeat별 soft density를 맞춘 두번째 estimator와 같은 대상을 추정하지 않는다. 각각의 x축·mapping·coverage·effective bank를 기록해야 한다. Soft/hard 중 true error가 작은 것을 test에서 선택하는 절차도 금지한다. Development에서 두 가설과 모든 결과를 남기고, 최종 primary 정의는 fresh evaluation 전에 잠근다.

## PI 의제: rotation과 participation ratio

**Ambient orthogonal rotation은 zero-fit sanity로 찬성한다.** Isotropic Gaussian likelihood에서 `x→Rx`, `b→Rb`, `D→RD`, `RᵀR=I`, 그리고 a를 고정하면 posterior energy와 Gram이 그대로다. 그러나 D의 ambient coordinate별 sparsity는 크게 변할 수 있다. 이는 RBM의 sparse connection과 SAE의 latent occupancy를 같다고 보면 안 되는 이유를 작은 대조로 보여준다.

Input-dependent amplitude를 쓰는 경우에는 a를 원 입력에서 계산한 값으로 고정하거나 encoder를 함께 conjugate하여 `a'(Rx)=a(x)`를 유지해야 한다. x와 D만 바꾸고 원래 encoder를 그대로 적용하면 다른 조건부 모형이 된다. Adam 재학습 궤적까지 회전 불변이라고 주장하거나 새 rotation training grid를 만들 필요는 없다.

**PR3는 선택적 보조 진단이다.** [PRL2017 저자 supplement §III A–B](https://www.phys.ens.fr/~monasson/Articles/a105-si.pdf)의 Eq. (7)–(8)은 특정 asymptotic strong/weak scaling에서 participation ratio가 강한 hidden-unit count에 접근하는 이유를 설명한다. 그 조건을 SAE에 자동 적용하지 않는다. Exponential-amplitude stress에서 코드의 dominant-amplitude count를 설명할 때만 사용하고, generating hard support target이나 primary selector를 교체하지 않는다.

## 재사용 가능한 Phase 1 근거를 새 필수 작업으로 늘리지 않는다

추가 확인 결과, 현재 결과 문서 B2는 이미135 cells에서 `-dE[N]/dγ=Var(N)` 중앙차분 오차 최대1.16e-9를 보고한다. `tests/test_sae_inference.py`에도 field-response 검사가 있다. 그러므로 이 identity를 새 발견 또는 미수행 필수 검사로 제안하면 중복이다. 다음 작업은 기존 covariance 산출물의 diagonal/off-diagonal 설명, MF response와의 차이, fixed-model과 joint-retraining의 차이를 읽기 쉽게 제시하는 것이다. 원본 수치를 새로 독립 감사했다고 주장하지 않는다.

## 직접 대화의 결론과 미해결점

직접 메시지에서 주고받은 핵심 변화는 최초 pilot 규모의 축소, dataset-repeat의 정확한 명명, correlated-prior 검사의 우선순위 하향, soft/hard 반례 범위의 수정이다. 두 사람의 입장이 처음부터 같았다고 포장하지 않는다.

공동 권고는 **기존 exact 근거·반례 →136/200-fit full-curve horizon screen → 작은 명명된 ensemble/readout challengers → protocol freeze → fresh-world C2a/C2b**다. Challengers와 학습시간을 포함한 실제 step 비용을 별도로 세어야 한다. Large objective ablation과 LM 확장은 조건부다.

미해결 경험적 차이는 어떤 ensemble와 observable 조합이 충분히 informative하면서도 recovery에 유용한가다. 이 답을 지금 문헌만으로 결정하거나 만장일치 결론으로 만들지 않는다. 두 과학자는 그 불확실성을 보존하면서도 VG-SAE와 density inference라는 원래 연구 목표를 유지하는 데 동의한다.

## 두번째 응답에서 확정한 작은 실행 계약

Scientific가 제안한 다음 계약에 직접 메시지로 동의했다. 기본 17 controls는 각 8k까지 학습하며 2k/4k/8k snapshot을 보존한다. 추가 최대 8 controls는 사전에 정한 **4k native-density rule에서만** 선택하고 그 bank를 고정한다. 추가 controls 역시 같은 초기조건·예산 원칙으로 처음부터 8k까지 학습하며 중간 snapshot을 남긴다. 모든 시점의 기본 bank와 extended bank 결과를 별도로 보고한다.

Extended bank는 4k coverage에 조건부로 선택됐으므로 horizon에 대칭적으로 공정한 표본이라고 표현하지 않는다. Primary horizon 비교는 기본 17 controls다. 8k continuation 및 adaptive-control 비용은 실제 updates/device-time와 4k-step equivalents에 모두 반영한다.

이미 학습한 동일 bank의 reconstruction/c_dec/raw-U/within-gate/random/oracle 비교는 main 결과의 성공 여부와 무관하게 모두 보고한다. VG-specificity 주장을 유지한다면 작은 L1 control도 실행 전에 지정하고 primary가 유리한지와 무관하게 수행한다. Large three-objective retraining bank를 조건부로 미루는 것과 comparator 결과를 선택적으로 숨기는 것은 다른 결정이다.
