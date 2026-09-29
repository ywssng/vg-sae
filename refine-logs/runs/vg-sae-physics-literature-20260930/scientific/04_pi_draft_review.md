# PI 통합 초안 최종 검토 — 실행 계약과 비용

2026-09-30. 검토 범위는 같은 run root의 `EXPERIMENT_PLAN.md`, `EXPERIMENT_TRACKER.md`, `PI_SYNTHESIS.md`, `PAPER_PLAN.md`, `FINAL_PROPOSAL.md`다. Canonical 문서는 수정하지 않았다. 새 실험이나 GPU 작업은 실행하지 않았고 산술만 독립 계산했다.

**판정: 공동 권고를 올바르게 반영했다. 주요 과학적·비용 오류는 발견하지 못했다.** 아래 세 군데의 좁은 명료화/수정을 권고한다. 추가 실험 범위는 필요 없다.

## 좁은 수정 사항

### 1. BR5의 horizon gate가 적용되는 bank를 명시

`EXPERIMENT_PLAN.md` BR2는 basic17을 primary horizon check로 지정하고 extended를4k-selected 보조 진단으로 둔다. 그러나 BR5-3의 shape-valid 전환 및 q 변화 gate 문장에는 어느 bank의 status/q인지 명시되지 않았다. 특히 basic17은 coverage 부족인데 extended만 valid인 경우, 한 구현은 진행하고 다른 구현은 보류할 수 있다.

권고: BR5에 “우선 basic17의 status/q로 gate를 평가한다”를 명시하고, basic17 coverage 부족일 때는 **horizon stability 미확인**으로 기록하도록 정한다. Extended 결과를 보고 계속 개발하는 것은 가능하지만, 그것만으로 basic17 horizon gate 통과를 선언하지 않는다. 만약 extended를 formal gate에 사용하려면4k-selected bank라는 조건을 유지한 별도 사전 규칙을 명시한다. 이것은 새 threshold나 새 fit을 요구하는 것이 아니라 이미 합의한 우선순위를 실행 가능하게 만드는 편집이다.

### 2. Hard occupancy의 primitive를 한 줄로 정의

A2의 `native hard gate m>.5`와 A5의 hard masks는 일관되지만 PI synthesis에는 hard-code라는 요약 용어도 나온다. 정확한 관측량을 다음처럼 한 줄로 고정하면 기존 epsilon-based effective-code L0와 섞이지 않는다.

`H[r,x,j]=1[m[r,x,j]>.5]`, `z_hard=H*a`, `rho_hard=mean(H)`.

모델의 현재 nonnegative amplitude는 `src/sae_model.py`의 softplus이므로 정확산술에서는 양수다. 현재 모델에서 발견한 중대한 mismatch는 아니다. 실제 numerical zero/비정상값이나 decoder zero 등의 진단을 위해 code effective L0를 보조로 기록할 수 있지만, 그 값으로 primary gate occupancy를 조용히 바꾸지 않는다는 뜻을 명확히 하면 충분하다. 이 검토는 모델 코드를 수정하지 않았다.

### 3. Legacy milestone 이름 한 곳 수정

`EXPERIMENT_PLAN.md` C1에 “Primary bank612/900 fits는 M2에 이미 포함”이라는9월29일 명칭이 남아 있다. 현재는 **W3**이다. 숫자는 맞고 중복 계상도 없지만 미래 실행자가 잘못된 milestone을 찾지 않도록 이름을 바꾼다.

추가로 total4930/7250은 **W1–W4 전체 누적치**임을 비용 요약에서 일관되게 쓰면 읽기 쉽다. W2–W4만의 새 fit 합은4692/6900이다. 현재 식은 정확하며 이 부분은 숫자 오류가 아닌 표현 명료화다.

## 독립 산술 확인

| 항목 | 기본 | 최대 | 확인 |
|---|---:|---:|---|
| W1a:2×4×17/25 | 136 | 200 | reference 포함 |
| W1b:2×3×17/25 | 102 | 150 | reference 재사용 |
| W1 distinct fits | 238 | 350 | 중복 없음 |
| W1의4k equivalents, 전부8k | 476 | 700 | 중간snapshot 추가 계상 없음 |
| W2:6×4×17/25 | 408 | 600 | 나머지 개발teachers |
| W3:36×6×17/25 | 3672 | 5400 | fresh main |
| W4:6×6×17/25 | 612 | 900 | small L1 |
| W1–W4 distinct total | 4930 | 7250 | 합계 일치 |
| W2–W4 distinct subtotal | 4692 | 6900 | H_star 적용 부분 |

`h=H_star/4000`에서 step-equivalents는 `476+4692*h` / `700+6900*h`다. H_star4000이면5168/7600,8000이면9860/14500으로 문서와 일치한다.

조건부X1/X2/X3의 기본 합은1836+576+918=3330, 최대 합은2700+576+1350=4626이다. 모두 추가한 distinct total8260/11876, H_star4000의 equivalents8498/12226도 정확하다. Frozen3fits에 각6000updates를 연장하면18000updates=4.5 equivalents이며 선택 시에만 더한다. Prior24 target과 rotation을 training0으로 표기하면서 CPU/evaluation time을 별도 기록한 것도 적절하다.

10–40초/4k-fit와25% overhead 가정에서 W1은 약1.65–9.72GPUh, H_star4000 core는약17.94–105.56GPUh, H_star8000 core는약34.24–201.39GPUh다. 문서의약1.7–9.8/18–106/35–202 범위는 반올림한 계획 견적으로 이해할 수 있다. 실제 timing 미측정, first-wave10GPUh 제안cap, 기존160GPUh 자동증액 금지, 이전2GPUh승인과 구분이 명시되어 있다. 단순 update 비례 시간이 항상 실제 runtime과 같다는 보장은 없으므로 문서처럼 method별 실측으로 다시 계산해야 한다.

## 프로토콜 일관성 확인

- **반복 단위:** teacher_world_id → training_dataset_id → optimizer_repeat_id 계층이 명시됐다. 동일 teacher의 data draws나 shared reference를 독립world n에 더하지 않는다. 36-world 집계와 cell별6world 평가, world별 repeat 평균의 사용은 일치한다.
- **Reference:** Reference0는독립 initialization을 갖고 두 arm의ensemble에서제외된다. E_joint 추가분에 reference를다시계상하지않는다. Fixed-data arm과reference dataset이같다는asymmetry를공개한다. Alternativecommonanchor에서reference와membership이함께바뀐다고명시한것도정확하다.
- **Pairing:** Arm/control을 optimizer seed에넣지않고 같은initialization/batch-index를pairing한다. E_joint의각repeat는독립dataset을갖되동일repeat의모든controls에서dataset을고정한다. 이로순수data variance를식별했다고주장하지않는다.
- **Horizon/adaptation:** BR 기본17은모두8k,2k/4k/8k를저장한다. 추가최대8controls는4k hard-density만으로결정하고추가fit도8k까지학습한다. 8k hole을보고다시추가하지않고soft challenger 때문에fit를늘리지않는다.
- **관측량:** Hard-between과soft-total의native density matching/축이분리됐다. Soft의within/between은uncorrected finite-ensemble identity이고hard의R correction을soft전체에적용하지않는다. R3development에서R5 bootstrapgate를잘못적용하지않는다.
- **본검증:** H_star4000은후보이며BR와개발후잠금, 변경시version/timing/untouched확인을요구한다. Snapshot을test-best로선택하지않는다. C2a/C2b, estimate/deployment/fallback, scientific abstention/infrastructure failure를분리한다.
- **조건부 비교:** Same-bank selectors는모두보고하고W3에진입하면사전L1 W4도결과와무관하게수행한다. X1의methods/subset/endpoints는추가학습전에고정한다. Negative main을보고시작한추가진단은exploratory로표시하여기존success기준을바꾸지않는다. 현재계약에서comparator cherry-picking을허용하는규칙은발견하지못했다.
- **Legacy 숫자:** 과거7240/10376과C0의24개연장은과거계획으로명시했고현재예산에중복가산하지않았다. Phase1의135cells/9+54fits도새학습수에넣지않았다.

## 최종 의견

위의 bank-gate 문구와 hard primitive, M2→W3 명칭만 정리하면 실행 계약과 비용 면에서 canonical 반영에 동의한다. 두-world bridge가 broad validity를 증명한다거나, 계획의 합의가실험성공을뜻한다는과장은없다. 추가model/실험scope는권고하지않는다. 문헌·수식에대한ARIS의별도검토와함께PI가최종반영하면된다.
