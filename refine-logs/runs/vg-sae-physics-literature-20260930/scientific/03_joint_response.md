# Scientific scientist의 공동 권고안 확인

2026-09-30. `aris/03_joint_recommendation.md` 전체와 마지막 alternative-anchor 추가를 읽었다. **공동 권고안에 동의하며 남은 operational 이견은 없다.** 이 문서는 두 scientist의 토론 합의이고 실증 결과나 외부 독립 검증이 아니다.

확인한 핵심은 다음과 같다.

- VG-SAE first-principles와 원래 density-estimation 질문을 유지한다. C2a generating density와 C2b recovery selection을 분리한다.
- Exact conditional covariance와 stable-branch MF response를 설명에 추가하고, rotation은 함수 보존 조건 아래0-fit 검사로 둔다. Prior2×2는 후속 기전검사다.
- 운영상 primary는 E_opt+hard-between으로 유지하되, E_joint는 별도 명시적 challenger다. 어느 것이 실제로 유용한지는 미확인이다. Method를 변경할 수 있지만 새 version과 untouched confirmation이 필요하다.
- Shared reference0를 ensemble에서 제외하여 두 worlds에서 총238 기본/350 최대 distinct fits라는 계산은 맞다. 모든 trajectory가8k이면4000-step equivalents는476/700이다. 중간2k/4k snapshot은 새 fit이 아니다. Device-hour 실측이나 power 보장으로 읽지 않는다.
- Basic17 gamma를 모든 horizon에서 우선 비교한다. Adaptive≤8 controls는4k-native-hard 규칙에서만 선택하고 freeze하며, 추가 control도8k와 중간 snapshots를 저장한다. Extended bank가4k에 조건부라는 한계를 명시한다.
- E_joint는 data+optimizer 결합 variation이다. Paired initializations가 variance component 식별을 보장하지 않는다. 동일 teacher·공통 평가 stream을 유지하고 datasets와 optimizer seeds를 별도 metadata로 기록한다.
- Reference0의 dataset이 E_opt와 겹치는 asymmetry는 존재한다. 기존 E_joint repeat1을 alternative common anchor로 쓰는 제안은 추가 학습 없이 이를 점검할 수 있다. E_opt 3 repeats 유지, E_joint에서 새 anchor 제외/old reference0 추가로 R=3 유지라는 구성에 동의한다. 이 sensitivity로 더 좋은 anchor를 사후 primary로 고르지 않는다.
- Soft-total/within/between에는 별도 native-soft density와 coverage contract가 필요하다. Stable-correct flat 반례는 hard-between에 대한 것이며 soft-total에는 within variance가 남을 수 있다.
- Same-bank comparator 전체와 VG-specificity 주장에 필요한 사전 고정 small L1은 결과와 무관하게 보고한다. 큰 objective ablation은 사전 branch를 둔 후속 검증이며 불리한 결과를 숨기는 장치가 아니다.

Canonical plan 반영 시 용어 한 가지를 특히 맞춰야 한다. 기존 `world = D + 하나의 fixed train stream` 정의는 E_joint를 담기에 부족하다. 상위 independent world는 teacher dictionary와 고정 생성 조건을 가리키고, 그 아래 training-dataset realization ID와 optimizer-repeat ID를 분리해야 한다. Shared align/select/test와 reference dataset ID도 manifest에 명시하면 된다. 같은 teacher의 여러 training datasets를 새로운 독립 world처럼 n에 더하지 않는다. 이는 공동 권고를 바꾸는 이견이 아니라 그 권고를 실행 가능한 schema로 옮길 때 필요한 명료화다.

과학적 불확실성은 남는다. 두-world screen의 결과가 36-world confirmation에서 유지될지, optimized hard instability가 finite-budget 현상인지, data resampling이 template를 회복시키는지, soft/hard 중 어떤 관측량이 유효한지는 실행 전 알 수 없다. 이 질문을 구별하는 bridge가 공동 권고의 목적이다.
