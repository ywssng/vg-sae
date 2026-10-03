# VG-SAE 프로젝트 메모리

사용자와 확정한 결정 및 다음 작업에 필요한 맥락을 보관한다. 행동 지침은
[AGENTS.md](../AGENTS.md)에 두고, 여기에는 사실과 결정의 근거를 간결하게
남긴다. 현재 코드와 사용자 정정으로 확인하면서 갱신한다.

## 구현 복잡도와 검증 범위에 대한 사용자 결정 — 2026-10-03

- 사용자는 “각 구현 단계에서 과도한 테스트와 확인된 요구 없이 선제적으로 복잡도를 늘리는 구현을 금지한다”를 작업 원칙으로 명시했다.
  행동 규칙은 [AGENTS.md의 검증 범위](../AGENTS.md#검증-범위)에 반영했다.

## Dot-cloud 이관 연구 재개 — 2026-10-03

- 사용자는 `research-import-20261002T131813Z/START_HERE_KO.md`와 기존 계획에
  따라 연구를 이어가되, 실험·수치 검증은 Colab CLI CPU에서 수행하고 실제
  LLM activation은 직접 진행하겠다고 지정했다. 로컬은 파일·코드·전송·Git
  작업에만 사용한다. GPU는 필요성이 입증될 때만 최소 사용량 종류를 고려한다.
- 이관본 `project/`가 base/delta를 합친 작업 복사본이다. 보존본은 수정하지
  않는다. 상세 최신 보고서가 root의 9월30일 계획과 이관 요약보다 우선한다.
  현재 상태표와 후속 계획은
  `refine-logs/runs/vg-sae-colab-continuation-20261003/RESEARCH_STATE_AUDIT.md`에 있다.
- W1 350 fits/1,050 snapshots와 21-fit·9-fit 개발 진단, fresh precision-reset,
  G1/G2와 후속 수리 진단이 이관되었다. BR5는 measurement_unresolved,
  W2는 gated, C2a/C2b는 unentered다. 완료된 음성 분기를 자동 반복하지 않는다.
- G1의 detailed protocol/result는 orthogonality·unit amplitude 구조를 주고
  D/q/v를 학습한다. START_HERE의 known-dictionary 요약은 이와 충돌한다.
  Known-D exact-mixture control과 G1을 구분한다. G2는 짧은 native retention에서
  mask가 변하지 않았고 복구 방법으로 입증되지 않았다.
- Known two-atom population proof는 joint scalar variance 아래 fixed finite
  crossover의 limiting activity bracket과 eventual prelimit transfer를 다룬다.
  Native VG, finite sample 또는 실용 추정기의 성공으로 해석하지 않는다.
  기존 20-cell 표의 primary image-unresolved 12개와 replay-unresolved 17개는
  서로 다른 판정이다. 새 terminal audit와 explicit finite-noise bound는 별도
  version으로 진행 중이며 완료 결과가 아니다.
- Colab Standard CPU의 실제 연결과 torch CPU-only를 확인했다. Drive 마운트는
  `mount failed`로 종료되어 짧은 작업은 explicit runtime storage와 즉시 fetch를
  사용한다. 새 source manifest/CPU guard/runtime storage 검증 47개가 Colab에서
  통과했다(Python 3.13.15, torch 2.11.0+cpu). 전체 프로젝트의 Python 3.14 및
  SAE 검증과는 다르다. 원격 결과는 `.colab/runs/`에 회수했다.

## Colab CLI 실험 실행 준비 — 2026-10-02

- 사용자는 이 서버에서 학습하는 대신 자신의 Colab Pro를 CLI로 이용하고,
  중단 시 작업을 보존·재개할 수 있게 설정하도록 요청했다.
- 공식 CLI 0.7.4를 Git 제외 경로 `.colab/`에 설치했다. `scripts/colab`과
  `scripts/colab_cli_local.py`가 token·세션·로그·캐시 저장 위치를 격리한다.
  설치·로그인·실행·복구 안내는 `docs/colab.md`다.
- `scripts/colab_experiment.py`와 `scripts/colab_worker.py`는 소스/계획 해시를
  확인하고 Drive에 작업별 상태·로그·결과를 저장한다. 완료 작업은 건너뛰지만
  미완료 작업은 처음부터 재실행한다. step 단위 optimizer/RNG 복구와 자동
  런타임 재할당은 미구현이다. 기존 런타임 생존 여부가 불명확하면 재실행하지 않는다.
- `configs/colab_smoke.json`은 VG regression의 작은 2-seed 인프라 검증용이다.
  전체 SAE/SAELens 환경이나 논문 실험 결과를 검증한 것으로 해석하지 않는다.
- 관련 오케스트레이션·복구 테스트 15개와 shell 문법·CLI 버전 확인을 통과했다.
  이 작업에서 로컬 학습은 실행하지 않았다. 10월2일에는 사용자 Google 로그인과
  실제 Colab 실행이 미검증이었다. 이후의 연결·실행 검증은 위 10월3일 기록을 따른다.

## 물리 문헌과 두 과학자 토론 — 2026-09-30

- 사용자는 Klindt et al.의 Nature2026 Perspective, Tubiana–Monasson PRL2017,
  Hou–Huang PRL2020 및 기존 문헌을 바탕으로 전체 논문과 Phase 1·2를 검토하고,
  ARIS agent와 scientific-skills agent가 토론하며 PI가 감독하도록 요청했다.
- 독립 입장, 실제 교차 반론, 공동 권고·동의, PI 초안 재검토는
  `refine-logs/runs/vg-sae-physics-literature-20260930/`에 있다. 핵심 판단은
  `PI_SYNTHESIS.md`, 수식·재평가 계획은 `PHASE1_PHYSICS_BRIDGE.md`다.
  Nature 세부 가정은 공개 저자본을 읽었으며 출판본 전체와의 일치는 미확인이다.
- Phase 1에 conditional Gram interaction, exact covariance 대 stable MF response,
  amplitude 고정/encoder 변환을 포함한 ambient rotation 대조를 반영했다.
  기존 exact response 결과는 재사용하며 새 실험이나 정리로 세지 않는다.
- Phase 2의 현재 primary는 fixed-data optimizer ensemble의 hard-between이다.
  원 VG의 soft-total/data-realization과 다른 두 전이를 별도로 검증한다.
  Stable-correct hard code도 flat U를 만들 수 있고, 같은 soft m에는 within
  uncertainty가 남을 수 있다. 어떤 ensemble/readout이 실제 유익한지는 미확인이다.
- 최초 계획은 두 teacher와 공유 reference의 E_opt136–200 fits 및 E_joint
  추가102–150 fits, 총238–350 fits를8k까지 학습하는 bridge다. 모든2k/4k/8k
  곡선을 보존하고, adaptive controls는4k hard coverage로만 고른다.
  Teacher/dataset/optimizer를 구분하고 E_joint를 순수 data variance라 하지 않는다.
- 기존7k–10k fits를 처음부터 필수 실행하지 않는다. 큰 본검증은 bridge·개발·
  lock 뒤 조건부이며, same-bank/L1 비교는 main 결과와 무관하게 보고한다.
  Primary/horizon/grid 변경은 새 version과 untouched confirmation을 요구한다.
- 이번에는 문헌·계획만 갱신했으며 학습·GPU·모델 코드 변경은 없다. 두 agent의
  operational 합의는 성능/신규성 확정이 아니다. 9월29일READY9.27 점수는 해당
  과거 설계에만 적용한다. Canonical PAPER_PLAN/EXPERIMENT_PLAN은 현재 갱신본이다.

## Phase 1–2 통합 논문 계획 — 2026-09-29

- 사용자는 최근의 모형·추론 검증을 Phase 1, 원래의 밀도 추정을 Phase 2로
  구분하고, 후속 Phase 2를 refine하여 하나의 논문 계획으로 작성하도록 요청했다.
  이번 작업은 계획·검토이며 새 학습 실험을 실행하지 않았다.
- 통합 논문 계획은 `PAPER_PLAN.md`, 현재 방법·실행 계획은
  `refine-logs/FINAL_PROPOSAL.md`와 `refine-logs/EXPERIMENT_PLAN.md`다.
  전체 이력은 `refine-logs/runs/vg-sae-unified-paper-20260929/`에 있다.
- Phase 1 원본은 `refine-logs/runs/vg-sae-first-principles-20260924/`에
  보존했다. 이전 메모리의 canonical 경로는 당시 범위를 가리키므로, 과거 계획은
  해당 run 경로에서 읽는다. `EXPERIMENT_RESULTS.md`는 Phase 1 결과만 담는다.
- Phase 2 제안은 truth-free feature 정렬, native hard-density matching,
  cross-run uncertainty의 단일 template fit과 추정 보류다. 생성 밀도 오차와
  recovery 선택 효용, estimate 상태와 deployment 상태를 각각 구분한다.
  이 알고리즘·threshold·예산은 작성된 제안이며 실증적 지지가 아니다.
- 기존 `paper_style_sigma_sel`은 input 축의 평균으로, 새 cross-run 지표를
  대신하지 못한다. 기존 NNLS의 normalized weights도 Bayesian posterior나
  confidence interval로 재사용하지 않는다.
- 같은 GPT-6 Astra reviewer와 3라운드 검토 후 계획 준비도 READY/9.27을
  기록했다. Same-family provisional 판단이며 신규성·실험 성공의 확정이나
  새 compute 실행 승인이 아니다. Phase 2 tracker는 미실행 상태다.

## 실험 설계·실행 승인과 첫 캠페인 — 2026-09-24

- 사용자는 first-principles 연구 의도에 맞춘 experiment-plan/bridge 실행과,
  기존 SAE 코드의 원 논문·공식 GitHub 대조를 명시적으로 요청했다. 아래09-23의
  문헌평가 전용 범위는 그때의 요청이며 이번 구현·실험을 금지하지 않는다.
- 새 계획과 감사·결과는
  `refine-logs/runs/vg-sae-first-principles-20260924/`에 있다. 기존 계획은 참고만
  했고, canonical proposal/plan/contract를 새 설명 중심 목표로 갱신했다.
- VG entropy tail/BF16 posterior, profiled risk floor와 beta 일치, BatchTopK
  activation-scale folding threshold를 수정했다. Gated는 published RI-L1 변형,
  JumpReLU는 upstream pre-ReLU/STE 편차가 있어 paper-exact로 부르지 않는다.
- 정확열거135 cells, frozen encoder9 fits, joint54 fits를3 seeds로 완료했다.
  결과는 `outputs/first_principles_20260924/`에 있다. 실제 assigned GPU wall
  합계약.15GPUh,2GPUh 상한 이내. 최종 tests378개와 compileall 통과.
- Orthogonal conditional posterior는 MF와 일치했고 overlap.95의 matched모형은
  평균MF reverse-KL약.141nats였다. Frozen encoder의 orthogonal 잔여KL약.36은
  표현 가능한 gate의 유한 최적화 오차를 포함하며 구조적 한계로 부르지 않는다.
- Frozen overlap.95의 gate refinement는 reverse-KL을 낮추지만3 seeds 모두
  Brier/marginal NLL/marginal error를 악화했다. Variational objective 개선과
  marginal posterior 확률 정확도 개선을 같은 주장으로 합치지 않는다.
  Brier/NLL을 calibration만 측정하는 지표로 표현하지 않는다.
- Joint gamma2에서 variance 삭제는 mean MSE를 낮췄지만 full stochastic/hard
  risk를 크게 높였다. 이는 profiled beta까지 반응한 objective 삭제의 total effect다.
  Semantic calibration, 자동 true-L0, SOTA, thermodynamic phase transition은
  주장하지 않는다. 이전 toy 결과와 합쳐 독립 seed 수를 늘리지 않는다.

## 연구 기여의 명확화 — 2026-09-23

- 사용자가 밝힌 중심 의도는 **통계물리학 기반 first principles에서 SAE를
  유도하고 기존 SAE를 다른 관점으로 이해하는 VG-SAE 연구**다. 성능 우위나
  Sparse but Wrong의 feature 혼합 해결만을 기여의 필수 조건으로 요구하지 않는다.
- 기존 변분 SAE 논문을 AlphaXiv로 확인하고 이 방향과의 정렬을 평가하도록
  요청했다. 코드 분석, 추가 구현·실험은 이번 범위에 포함하지 않는다.
- 새 판단 기록은
  `idea-stage/runs/vg-sae-first-principles-20260923/IDEA_REPORT.md`에 있다.
  아래 첫 주제평가의 feature-recovery 중심 기준은 현재 의도를 전부 표현하지
  않는다. 원리적 유도·가정 명료화·설명력도 유효한 기여 기준으로 다룬다.
- 에이전트의 문헌 판단은 연구 지속 가치 YES이며, 구체적 유도의 신규성을
  확정한 결과나 사용자가 승인한 과학적 결론으로 저장하지 않는다.

## 사용자 범위 정정 — 2026-09-23

- 이번 요청은 **Variational Garrote 기반 SAE 개발 자체의 연구 주제 적합성과
  계속할 가치**를 판단하는 것이다. 기존 코드 분석을 출발점으로 추가 구현이나
  실험을 수행하는 요청이 아니다. 09-22 brief의 파일럿·개발 단계는 이번 요청에
  적용하지 않는다.
- 문헌 검토는 사용자가 제공한 Sparse but Wrong와 Soh et al.의 Variational
  Garrote 두 논문에서 출발한다. 주제를 다른 진단 연구로 바꾸지 않는다.
- 이번 문헌 평가와 별도 에이전트 비판은
  `idea-stage/runs/vg-sae-topic-suitability-20260923/IDEA_REPORT.md`에 기록한다.
  에이전트의 판단은 연구 지속 가치 YES이며, 현재 구현의 우위나 최종 기여의
  신규성이 확인됐다는 뜻이 아니다. 사용자에게 확정받은 과학적 결론으로
  취급하거나, 이 판단만으로 과거 실험계획을 자동 실행하지 않는다.

## 확정된 결정 — 2026-09-07

- 사용자는 이 프로젝트에서 작업하는 Codex의 모델로 GPT-6 Astra를 선택했다.
  [`.codex/config.toml`](../.codex/config.toml)에 `model = "gpt-6-astra"`를
  기록했으며, reasoning effort는 프로젝트에서 지정하지 않는다.
- 모델 선택의 대상은 프로젝트의 Codex이다. SAE 실험은 로컬 모델의 내부
  activation을 사용한다. 관련 구현은
  [`src/gpt2_activations.py`](../src/gpt2_activations.py)와
  [`src/real_activations.py`](../src/real_activations.py)에서 확인한다.
- 사용자는 Astra의 프롬프팅 권장사항을 프로젝트 지침과 메모리에 반영해 달라고
  요청했다. 승인된 작업의 자율적 완료, 필요한 경우에만 질문, 명확한 지침
  우선순위, 간결한 설명, 도움이 되는 병렬 위임, 변경 영향에 맞는 검증이
  확정된 협업 선호다. 구체적인 적용 규칙은 `AGENTS.md`가 기준이다.

## 연구 맥락의 확인 위치

- 실행 환경과 실험 흐름: [`README.md`](../README.md),
  [`pyproject.toml`](../pyproject.toml), [`configs/`](../configs/).
- 수식·정규화·gradient·재현성 선택:
  [`REPRODUCTION_NOTES.md`](../REPRODUCTION_NOTES.md), [`tests/`](../tests/).
- Stage 3 모델과 activation 실험:
  [`docs/stage3_real_activations.md`](../docs/stage3_real_activations.md).
- 실험 결과를 해석할 때는 해당 run에 저장된 seed, config, checkpoint 및
  평가 방식을 함께 확인한다. 메모리에 숫자를 옮길 때도 결과 경로와 조건을
  근거로 남긴다.

## VG-SAE idea-discovery — 2026-09-19 (문제 설정은 09-21 정정으로 대체)

- 사용자는 현재 프로젝트의 구현을 출발점으로 `idea-discovery`를 실행하고,
  이 작업의 아이디어 생성·신규성·비판·방법 리뷰를 GPT-6 Astra의 `ultra`
  effort로 수행하도록 지정했다. 프로젝트의 전역 모델 설정을 바꾼 요청은 아니다.
- 통합 보고서는 [`idea-stage/IDEA_REPORT.md`](../idea-stage/IDEA_REPORT.md),
  후속 방법과 실험 계획은 [`refine-logs/`](../refine-logs/)에서 확인한다.
  28개 문헌과 11개 후보를 검토했으며, 리뷰는 same-family provisional이다.
- 당시 에이전트의 선택은 learned VG checkpoint에서 복제의 loss 선호와 feature 품질의
  관계를 검사하는 조건부 진단 연구다. 정확한 support posterior를 이용한
  gate 대상 검증을 대안으로 남겼다. 일반 dropout 복제 원리와 폭별 prior는
  기존 연구이므로 새 정리나 해결책으로 주장하지 않는다.
- 파일럿 코드와 사전 계획은 `scripts/idea_discovery_*`, `idea-stage/pilots/`,
  작은 원본 결과는 `idea-stage/evidence/pilots/`에 보존한다. 기존 Stage-2
  결과는 한 seed와 calibration stream 재사용 조건을 함께 읽어야 한다.
- `expected_ev`라는 파일럿 필드는 posterior-mean reconstruction EV다.
  stochastic reconstruction risk에는 Bernoulli variance가 추가되며 hard
  inference와도 구분한다. 큰 expected L0나 mean-hard 차이만으로 실제 복제,
  calibration 실패, objective 실패를 확정하지 않는다.
- 후속 clone intervention과 duplicate-invariant 평가 구현·확증 실험은 아직
  계획 단계다. 파일럿의 성공과 후속 C1/C2의 검증 완료를 혼동하지 않는다.

## 사용자 정정과 원래 연구 목표 — 2026-09-21

- **Variational Garrote 기반 새로운 SAE를 개발하는 것 자체가 사용자의 원래
  연구 아이디어**다. 이 저장소는 그 연구를 지금까지 구현하고 실험한 작업물이다.
- 요청의 목적은 VG-SAE에서 별개의 새 연구 주제를 찾는 것이 아니라, 진행 중인
  VG-SAE 방법 개발 및 연구 파이프라인을 다듬는 것이다. 09-19의 복제 진단 주제
  선택은 에이전트의 범위 오해였으며 현재 연구 목표로 사용하지 않는다.
- 새 실행의 고정 목표·제약은 [`RESEARCH_BRIEF.md`](../RESEARCH_BRIEF.md)에 있다.
  후보 생성은 같은 방법의 개선·검증 경로 안에서 수행한다. 진단 실험은 방법
  개발을 돕는 도구이며 주 연구를 대체하지 않는다.
- 기존 수치·문헌·코드는 조건을 재확인해 재사용한다. 기존 판단이나 리뷰 점수를
  변경된 목표의 검증으로 가져오지 않는다. `ultra` reviewer 지정은 유지한다.

## 원래 연구 목표로 재실행한 결과 — 2026-09-21

- 현재 실행 ID는 `vg-sae-development-20260921`이다. 이전 목표의 문서 사본은
  `idea-stage/runs/vg-sae-development-20260921/prior_scope_snapshot/`에 보존했다.
  최신 proposal/plan/contract는 VG-SAE 자체의 방법 개발을 기준으로 한다.
- 새 P1은 기존 546 checkpoints를 독립 calibration에서 선택한 뒤 fresh test로
  평가했다. Exponential의 coefficient/F1 신호와 input fidelity 손실이 함께 있었다.
  같은 L0 상한 아래 한 training world의 결과이며 exact-L0 또는 multi-seed 우위가 아니다.
- P2의 36개 작은 학습에서 moment beta 초기화의 일관된 이득은 없었다.
  일부 조건에서는 학습 연장으로 EV가 개선돼도 latent recovery/F1는 악화했다.
  다음 본 비교는 모든 방법에 같은 checkpoint 후보 기회를 주고 calibration recovery와
  L0로 선택한다. 기존 core beta 기본값과 기본 추론은 변경하지 않았다.
- P3 순차 보정과 P3b 탐색 병렬 후속은 일부 품질 개선을 보였으나 사전 latency 기준을
  넘었다. Generic amplitude control 뒤의 추가 이득과 비용을 더 검증해야 하므로
  solver/teacher를 기본 방법에 채택하지 않았다.
- Objective는 input-dependent point amplitude를 조건으로 한 support variational
  risk다. Full generative ELBO나 calibrated semantic posterior라고 부르지 않는다.
  Global learned beta와 minibatch profiling의 stochastic objective를 구별한다.
- 다음 B1/B2/B3는 아직 계획이다. 독립 paired training worlds와 동일 HPO/checkpoint
  기회, profiled-beta 조건에서 variance/entropy 각각의 objective coupling 검증,
  기존 Stage2/3 saved checkpoint의 fresh 평가를
  순서대로 진행한다. 구체적인 grids·실행 상한·미구현 prerequisites는
  `refine-logs/EXPERIMENT_PLAN.md`와 `EXPERIMENT_TRACKER.md`가 기준이다.
- 이번 새 runner 네 개의 관련 수치/선택 테스트 14개가 통과했다. 전체 repository
  test suite를 이 실행에서 다시 돌렸다고 기록하지 않는다. 작은 evidence는
  `idea-stage/runs/vg-sae-development-20260921/evidence/`에 있다.

## 새 핵심 연구 동기 — 2026-09-22

- 사용자는 ICML 2026의 Sparse but Wrong (Chanin & Garriga-Alonso)을 기준으로
  idea-discovery를 다시 실행하도록 요청했다. L1 sparsification 외의 방법론으로
  VG-SAE를 개발하는 것을 주목적으로 삼는다.
- 논문의 직접 L0/feature-mixing 결론과 사용자의 VG 대안 가설을 구분한다.
  주 연구는 여전히 VG-SAE 개발이며 별도 진단 주제로 변경하지 않는다.
- 현재 run ID는 `vg-sae-sparse-but-wrong-20260922`다. 이전 09-21 계획은
  해당 run의 `prior_development_snapshot/`에 보존했다. 기존 결과·review를
  새 문제의 확증으로 재사용하지 않는다.
- 사용자는 보고서를 별도 브라우저 없이 대화 안에서 읽는 방식을 선호한다.

## Sparse but Wrong 기반 재실행 결과 — 2026-09-22

- 참조 논문 v4/ICML2026의 직접 문제는 wrong L0와 feature identity이며, L1 대체는
  사용자의 VG 개발 가설이다. 논문이 L1 전반을 기각하거나 VG를 제안한 것으로 쓰지 않는다.
- 현재 보고서와 contract는 세 새 pilot을 통합했다: P1 225 fits, P2 72, P3 54.
  합계351은 독립 seed 수가 아니다. 각 pilot은3 paired worlds의 짧은 toy 검증이다.
- P1은 target2의7/9 covered cases에서 거의 완벽한 VG 복원을 보였고, simple L1
  rescale 후에도4/5 matched pairs의 coefficient NMSE가 낮았다. Strong-baseline
  joint 우위는 미확인이고 target1.8 및 Jump의 coverage가 부족했다.
- P2는 finite-budget scalar-prior recipe 비지지다. 대부분 gamma gradient도 아직
  크므로 잘못된 EB 정상점으로 수렴했다고 표현하지 않는다. P3의 한 total-effect
  rescue는 보존하되 density/readout/부호 효과와 고유 covariance 효과를 구분한다.
- Frozen mixing_energy는 signed-assignment leakage로 pure sign/mapping 오류도
  포함한다. 실제 다중성분 혼합은 absolute cosine/전체 projection과 함께 해석한다.
- Dense-offset 구성은 true decoder와 dense code로 profiled loss를 낮출 수 있는
  parameter family다. Profiled branch만 epsilon floor를 사용한다. Global learned
  branch에는 같은 energy clamp가 없다. 실제 SGD 원인이나 recovery 보장은 미입증이다.
- 다음 개발은 같은 VG의 precision 정책 대조와 baseline/selector calibration이다.
  Gamma 학습이나 exact32를 새 기본 구성으로 채택하지 않았다. 후속248-fit core
  roadmap은 미실행이며 큰 source 확장은 조건부다.
- Scientific continuation은 PROCEED WITH CAUTION, proposal은REVISE/7.85,
  same-family provisional이다. 점수의 변화만으로 연구 목표를 바꾸지 않는다.
  최종 실행 규칙과 남은 작업은 최신 refine-logs/EXPERIMENT_PLAN.md 및 tracker를 읽는다.

## 갱신 원칙

- 다음 작업에 도움이 되는 확정된 결정·사용자 정정·재현성 맥락을 갱신한다.
  일시적인 진행 로그는 필요한 결정, 검증 결과, 남은 작업으로 압축한다.
- 변경 가능한 환경 상태, 브랜치 상태, 모델 목록은 필요할 때 다시 확인한다.
  개인 설정값이나 미완료 작업을 영구적인 프로젝트 정책으로 기록하지 않는다.
- 이 파일은 저장소에서 관리하는 프로젝트 메모리다. `AGENTS.md`의 시작 지침을
  통해 읽으며, Codex의 자동 생성 메모리 기능에 필수 규칙의 보존을 의존하지
  않는다.
