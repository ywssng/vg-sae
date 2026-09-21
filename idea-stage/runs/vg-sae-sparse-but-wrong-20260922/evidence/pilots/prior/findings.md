# P2: normalized scalar-prior adaptation 결과

72개 모델(3 worlds ×3 densities ×2 copula signs ×4 arms), CPU에서 동일 조건으로 학습·평가했다.
Primary learned init2와 fixed2는 사전 고정했고 다른 초기값을 test로 선택하지 않았다.

| p | Copula rho | Arm | Signed cosine | F1 | Hard L0 | EV | Final gamma |
|---|---:|---|---:|---:|---:|---:|---:|
| 0.2 | -0.4 | fixed | 0.90928 | 0.82315 | 1.59469 | 0.99765 | 2.00000 |
| 0.2 | -0.4 | learned | 0.89204 | 0.50638 | 3.77976 | 0.99672 | 0.03995 |
| 0.2 | +0.4 | fixed | 0.99998 | 0.99999 | 0.99563 | 0.99810 | 2.00000 |
| 0.2 | +0.4 | learned | 0.99941 | 0.83398 | 1.49744 | 0.99154 | 0.64065 |
| 0.4 | -0.4 | fixed | 0.78609 | 0.57078 | 5.00000 | 0.99931 | 2.00000 |
| 0.4 | -0.4 | learned | 0.78095 | 0.57078 | 5.00000 | 0.99938 | 0.11669 |
| 0.4 | +0.4 | fixed | 0.99935 | 0.58917 | 4.79822 | 0.99946 | 2.00000 |
| 0.4 | +0.4 | learned | 0.97390 | 0.57113 | 5.00000 | 0.99949 | 0.03159 |
| 0.6 | -0.4 | fixed | 0.43395 | 0.75038 | 5.00000 | 0.99936 | 2.00000 |
| 0.6 | -0.4 | learned | 0.45773 | 0.75038 | 5.00000 | 0.99937 | 0.34690 |
| 0.6 | +0.4 | fixed | 0.69037 | 0.75053 | 5.00000 | 0.99927 | 2.00000 |
| 0.6 | +0.4 | learned | 0.64478 | 0.75053 | 5.00000 | 0.99941 | 0.32302 |

**판정: 사전의 자동 prior 적응 개선 기준을 만족하지 못했다.** 0/18 paired cells가 개별 joint threshold를 통과했다. 각 density/sign에 요구한2/3 worlds 및 initialization 안정성의 전체 조건을 만족하지 못한다.

Reconstruction EV가 .99 이상이어도 true feature 방향과 hard support가 좋지 않은 조건이 있다.
Gamma의 self-consistency 식은 이론적으로 true L0 보장이 아니다. 별도의 full-train 진단에서
primary18 cells 중14개는 최종 |dF/dgamma|>2여서, 이번 결과를 잘못된 EB 정상점으로
수렴한 사례라고 해석할 수 없다. 유한 학습 예산의 gradient recipe 비지지로 기록한다.
이는 bounded gradient-learned scalar prior와 해당 finite noiseless regime의 부정 결과다.
후속 hierarchical prior나 slab 방법 전반을 기각하거나 VG-SAE 주 연구를 진단 논문으로 바꾸는 근거가 아니다.

Realized L0의 변화는 이 적응의 의도된 total effect다. 같은 L0에서 intrinsic mixing을
줄였다는 주장은 target p=.2/.6의 fixed-gamma grid가 없어 검증하지 않았다.
Source p=.4의 GPU P1 control을 섞지 않고 fixed controls도 같은 CPU에서 재학습했다.

학습 process wall time 합계 800.54초, 최대 개별 process 269.85초, test process 합계 1.78초. World3개는 병렬이므로 합계가 실제 경과시간은 아니다. GPU 사용0.

Code와 scalar-prior gradient3tests는 통과했다. 독립 code audit에서 blocking 수식 오류를 찾지 못했다.
Per-update beta-floor는 미측정이며 test/evaluation snapshot의 floor 정보를 그 대용으로 사용하지 않는다.

`mixing_energy`는 signed-Hungarian 할당 후 leakage이며 순수 sign/mapping mismatch도
포함한다. `posthoc_geometry_and_stationarity.json`의 absolute cosine 및 sign-invariant
다중성분 energy로 이를 실제 다중 feature 혼합과 구분한다. 이는 모든 고정 모델에 적용한
사후 설명용 분해이고 사전 성공 기준·모델 선택·학습은 변경하지 않았다.
