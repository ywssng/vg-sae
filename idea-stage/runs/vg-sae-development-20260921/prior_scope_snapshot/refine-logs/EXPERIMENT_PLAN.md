# VG-SAE 실험 계획

날짜: 2026-09-19. [최종 제안](FINAL_PROPOSAL.md)의 Problem Anchor와 C1/C2를
그대로 실행한다. 아래 B1–B3은 **아직 실행하지 않은 후속 연구**다.
이미 실행한 파일럿은 [tracker](EXPERIMENT_TRACKER.md)에서 별도로 구분한다.

## Claim Map

| 주장 | 최소 설득 근거 | 반대 설명 | 블록 |
|---|---|---|---|
| C1: learned VG의 일부 atom에서 실제 표현 가능한 복제가 loss상 유리 | cal에서 고른 atom의 held-out ΔF와 S=beta ΔV+ΔK가 모두 음수, mean/hard drift 작음 | 출력 증가로 reconstruction만 개선, fixed-gamma floor·미수렴·상수 bias | B1 |
| C2: 주입한 clone의 이후 feature 손상이 variance 항에 조건부로 의존 | high/low margin × variance on/off에서 t0+ 대비 변화의 상호작용, 중복 불변 AP 및 cutoff robustness | 주입 직후 손상, 대칭 유지, raw L0·matching의 지표 변화, 일반 최적화 차이 | B2, B3 |

반드시 배제할 주장: 알려진 duplication 식 자체의 신규성, 원래 학습에서 자발적인
복제 발생, 일반적인 posterior calibration, VG의 baseline 대비 전반적 우위.
C1 양성/C2 비지지도 유효한 결과다. 성공을 새 architecture나 보조 loss 추가로
만들지 않는다.

## Paper Storyline

핵심은 국소 objective의 예측 → 실제 parameter 개입 → feature 품질 변화의
조건부 검증이다. mean/sample/hard risk를 구분하고, 구성이 가능한 것과 실제
학습 현상을 구분한다. C1이 없으면 B2/B3을 실행하지 않는다.

메인 후보: C1의 sign/drift/prevalence 표, C2의 paired interaction 표와
true-feature별 변화. Appendix: 초기 부정 파일럿, beta 학습 민감도,
known-posterior calibration 진단, 원래 Hungarian 지표.
Cut: 새로운 hierarchical prior, pairwise gate 모델, 대규모 real-LM sweep,
무관한 LLM teacher·distillation. 이 연구에 frontier 학습 primitive는 필요 없다.

## 공통 데이터와 재현 규칙

- 새 independent synthetic data seeds100/101/102, 각 optimizer seeds0/1/2.
- 각 data seed마다 단일 dictionary를 생성하고 train8192/cal2048/test8192를
  서로 다른 표본으로 분할한다. test8192는 B1 gate용4096과 B2 최종4096으로
  미리 나누어 adaptive gate에 본 표본을 최종 확인에 재사용하지 않는다.
  d16, true features32, width128, true density2/32,
  exponential amplitude, noise SD.05. 현재 탐색 seed20260919와 분리한다.
- source VG는 기존 normalized decoder, learned beta, entropy weight1,
  variance on, gamma=log15, LR.003/batch256/weight-decay0을 사용한다.
- source checkpoints6000/8000/10000의 cal component·beta·clone margin stability를
  [proposal의 실행 규칙](FINAL_PROPOSAL.md)으로 확인하고 필요하면14000까지 한 번 연장한다.
- 3 data world가 최상위 재현 단위다. 9 optimizer runs를9개 독립 데이터셋으로
  취급하지 않는다. 각 world의 평균과 seed 범위를 모두 보고한다.
- source checkpoint와 atom 선택, epsilon, thresholds는 cal에서 끝내고 test는
  동결된 설정의 평가에만 쓴다. 이전 test를 재튜닝에 사용하는 재실험은 새 탐색이다.

## B1 — Frozen checkpoint의 실제 clone 선호

**C1 / MUST-RUN / Main.** 현재 코드에서 고정 beta/gamma로 atom을 복제한다.
ideal function-level 복제와 actual encoder parameter 복제는 다른 variant다.

| Variant | 조작 | 비교 목적 |
|---|---|---|
| source | 개입 없음 | 기준 |
| ideal clone | m복사, a/r, D복사 | ΔF=(r−1)K−beta(1−1/r)V 해석식 검증 |
| feasible high | cal signed margin이 큰 atom의 softplus bias−log2 복제 | 실제 구현 가능한 유인 |
| feasible low | amp/firing/drift가 matched된 낮은 margin atom 복제 | 같은 조작을 받는 대조 |

Signed margin M=beta E[V]/2−E[K]로 정렬한다. cal에서 amplitude log차이<=.25,
firing차이<=.01, normalized drift차이<=5e-5인 첫 high/low pair를 고정한다.
9 source 중 matching 실패 수를 숨기지 않는다. optimizer seed 하나라도
matching 실패하면 해당 data world는 inconclusive로 두며 평균에서 빼지 않는다. 미매칭에서 가장 좋아 보이는
test atom으로 바꾸지 않는다.

주 지표는 ΔF, S=beta ΔV+ΔK, mean/hard decoded drift, 선택된 atom의 실제
firing과 contribution이다. hard/mean/sampled EV와 raw L0도 병기한다.
기준은 epsilon_F=epsilon_S=1e-4 nat/sample, normalized mean/hard drift<=1e-4.
실제 B2의 perturbation을 포함한 t0+ 초기상태에 이 조건을 재적용한다.

**Go:** 선택한 high의 ΔF와 S가 모두 기준 이하이고 drift가 작으며3개 data-world
각각의 optimizer 평균이 이를 통과. operational decision이며 유의성 검정이 아니다.
**No support:** 충분한 matched candidates가 있었으나 기준 불통과.
**Inconclusive:** cal matching/안정성/coverage가 없거나 feasible drift가 너무 큼.
재구성 변화만으로 얻은 이득은 C1 양성으로 인정하지 않는다.

표/그림: atom별 ideal prediction vs 실제 ΔF, 세 loss 성분, 전체 eligible 분모와
prevalence. 가장 좋은 한 atom만 제시하지 않는다.

## B2 — 같은 개입 뒤의 조건부 feature 품질 변화

**C2 / MUST-RUN iff B1 passes / Main.** source 하나에서 high/low feasible clone을
만들고 variance on/off로 분기한다. 네 arm 모두 width129이며 beta/gamma를 source
값으로 고정한다. variance off는 posterior의 동일한 구현이 아닌 명시적 ablation이다.

- 모든 arm Adam state reset, LR3e-4, batch256, weight decay0, 2000updates,
  동일 minibatch 순서, 동일 decoder normalization.
- clone pair에 서로 반대인 작은 seeded perturbation을 넣어 analytic loss의
  permutation symmetry를 깬다. epsilon1e-5/1e-4/1e-3 중 cal drift를 양쪽에서
  통과하는 가장 큰 값을 선택하고 on/off에 동일하게 복사한다.
- t0−, perturbation 포함 t0+, T를 저장한다. ΔQ=Q(T)−Q(0+)가 학습 변화다.
- 주 Q는 true-feature별 grouped-max gate AP 평균이며 ranking endpoint다. positive cosine으로
  nearest truth에 할당하고, 동일 duplicate를 추가해도 점수가 불변임을 검증한다.
  AP는 threshold-free이며 native.5는 hard EV/L0/grouped F1에 적용한다.
- primary cutoff.8에 더해 .75/.85를 같은 artifact에서 재계산한다. 모든 cutoff에서
  같은 손상 조건을 통과해야 robust feature-ranking harm 문구를 사용한다. cutoff-crossing만으로
  생긴 결과는 threshold-sensitive로 남긴다. best-positive cosine/coverage 변화도 보고한다.
- raw latent L0, unmatched firing, hard EV, 기존 Hungarian 결과는 보조다.
  native 결과가 primary; L0-matched 결과는 cal에서 공통 budget을 달성할 수 있을 때만
  보조 평가한다. 허용치는 target의2%와0.05 중 큰 값. ties로 불가능하면 보간하지 않는다.

D_on=ΔQ_high,on−ΔQ_low,on; D_off도 같은 방식; I=D_on−D_off.
high/on의 ΔQ<=−.01, D_on<=−.01, I<=−.01이 세 cutoff와 세 data-world 평균에서
같은 방향으로 나타날 때만 제한된 손상을 지지한다. 경성 support 손상 문구는 같은 score를.5에서 자른 grouped F1도 동일 high/on,
D_on,I의−.01 및 cutoff/data-world 조건을 만족할 때만 쓴다. AP만 통과하면
ranking 결과로 제한한다. AP와 EV가 보존되고 raw L0만
늘면 의미적 손상이 아닌 표현 비용 증가다. 인위적으로 주입한 초기조건에 대한
결과를 자연 학습의 일반 원인으로 확대하지 않는다.

표/그림: 네 arm의 t0+/T paired 변화, data-world별 interaction, cutoff sensitivity.
이 블록이 단순화/삭제 검사도 담당한다. 기존 variance 항 on/off 외 새 모듈이 없다.

## B3 — SynthSAEBench의 한 조건 반복

**C1/C2 / MUST-RUN before external-generalization claim.** B1/B2가 남을 때만 실행한다.
기존 pinned generator/revision과 기존 width4096 체크포인트 형식을 재사용한다.
coefficient 선택에 사용했던 기존 평가 stream은 confirmatory test로 쓰지 않는다.
새 stream seed/offset을 고정하고 cal64k/test64k에서 시작한다. test는 B1용32k와
B2 최종32k로 미리 나눈다.

support positive가 cal에서20회 이상인 true-feature 집합을 먼저 고정하며,
frequency bin과 포함률을 보고한다. test 미관찰 feature는 AP undefined로 남긴다. eligible 집합의5%를 넘으면
semantic 비교는 inconclusive이며, 이하라도 누락률과 동일 분모 범위를 보고한다.
위 cutoffs/분해/perturbation/optimizer 규칙을 그대로 적용하고 별도 유리한 threshold를
고르지 않는다. throughput 먼저 측정하고 memory나 비용 한계면 stream 크기를
줄였다는 사실과 coverage 부족을 기록한다. Stage2의 MCC 정의와 Stage1 AP를
같은 metric으로 혼합하지 않는다.

현 주장은 baseline superiority가 아니므로 baseline sweep을 억지로 붙이지 않는다.
새 개선방법을 제안하게 되는 후속 연구에서는 Gated/JumpReLU/BatchTopK를3개
baseline family로 고정하고, 동일 postprocess/학습 토큰/검증 예산을 제공해야 한다.

## Run Order and Budget

| 순서 | 목표 | 작업 | Go/Stop | 예산 상한 |
|---|---|---|---|---|
| R001 | 지표·개입 구현 검사 | loss identity, exact clone invariance, feasible drift, duplicate AP sanity | 수치 오차 또는 지표 비불변이면 수정 | CPU +0.05GPUh |
| R002 | B1 저비용 screening | 기존6000-step checkpoint는 debug만, 이후 새 data3×train3 source와 cal matching | C1 criteria 불통과 시 C2 중단 |1.0GPUh |
| R003 | B2 paired continuation | 통과 source마다4arms×2000steps | I/semantic criteria로 조건부 지지·비지지 |2.0GPUh |
| R004 | B3 외적 반복 | pinned SynthSAEBench 한 조건, 새 holdout | 반복 실패 시 범위 축소 |4.0GPUh |

다음 연구의 budget ceiling은 합계7.05 GPU-hours다. 실제 소요시간 예측값이 아니다.
현재 작은 모델은 6000steps 약28–34초였으나 Stage2의 대형 모델·generator 시간은
별도 benchmark해야 한다. 기존 idle A6000 자원을 전제로 하며 외부 자원 구매는 없다.
인간 라벨링은 필요 없다. checkpoint와 대형 array는 outputs 아래에만 둔다.

## 첫 세 작업과 구현 상태

1. **R001:** 새 clone intervention runner와 duplicate-invariant metric을 구현한다.
   현재 제공된 pilot script는 clone experiment를 구현한 것으로 간주하지 않는다.
2. **R002:** cal/test 분리와 source stability를 추가한 B1을 실행한다.
3. **R003:** B1이 실제 통과한 경우에만 paired continuation을 실행한다.

현재 그대로 재실행 가능한 명령은 `idea-stage/pilots/`의 기존 파일럿들이다.
아직 없는 후속 runner 이름을 실행 명령으로 제시하지 않는다.
주요 연결 지점: `src/sae_model.py`, `src/sae_train.py`, `src/sae_sweep_eval.py`,
`src/synthsaebench_sweep.py`. 실험 전용 코드부터 만들고 기본 API 의미를 바꾸지 않는다.

## 위험과 범위

알려진 dropout/soft leakage의 재현만 남을 수 있다. positive 결과가 없는 부분을
새 module·더 큰 sweep으로 덮지 않는다. source가 안정적이지 않거나 matched control이
없으면 그것 자체가 현재 방법의 실행 한계이며 ‘효과 없음’으로 바꾸지 않는다.
B05 exact-posterior 연구는 독립 대안이다. B1 실패 시 B05로 옮기는 선택은 새 연구
문제이므로 현재 C1/C2의 성공으로 합산하지 않는다.
