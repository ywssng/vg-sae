# Research Contract: VG-SAE physics-to-density bridge

2026-09-30. 사용자 요청은 세 지정 논문 및 기존 문헌을 읽고 ARIS/scientific 두 agent가 토론하여 전체논문과Phase1/2를보완하는것이다. 이번작업은문헌·수식·계획이며실험을시작하지않았다.

## 고정 목표

VG-SAE를 명시적 선택모형/변분원리에서 이해하고, 불확실성에서 적절한 밀도를 추정해 feature recovery로 검증한다. 별도 clone 진단 주제로 바꾸지 않는다. C1 모형·추론 설명, C2a generating density, C2b recovery 효용을 구분한다.

## 현재 결정

- Conditional posterior/response, training-data variation, optimizer-given-data variation을 구분한다.
- Operational primary는 E_opt+hard다. E_joint 및 soft는 사전 지정 challenger이며원문과의차이를검증한다. 향후변경은새version+untouched확증을요구한다.
- Teacher world 아래 training_dataset_id, optimizer_repeat_id, reference_id를 따로 저장한다. Data draws를 독립teacher n에더하지않는다.
- Initial paired bridge는238/350distinctfits, 모두8k;4k-step equivalents476/700이다. Basic17의2k/4k/8k를우선보고하며adaptive bank는4k에서만선택한다.
- 큰36-world 본검증은bridge·개발·lock뒤의조건부후속이다. Main진입시same-bank selectors와smallL1은결과에상관없이보고한다.
- Estimate/deployment status 분리, truth-free selection, all-world fallback과실패보존, C2a/C2b분리를유지한다.
- Prior/likelihood matching의통찰을사용하되RBM=Nishimori=VG라는등가나semantic posterior·phase transition을주장하지않는다.

## 근거와 산출물

Phase1 실제근거는 `refine-logs/runs/vg-sae-first-principles-20260924/`. 새논문은 `PAPER_PLAN.md`, exact실험protocol은 `refine-logs/EXPERIMENT_PLAN.md`, 최신논의는 `refine-logs/runs/vg-sae-physics-literature-20260930/PI_SYNTHESIS.md`다. 기존9월29일계획은해당run에보존했다.

Operational 합의와계획검토는완료했다. 어떤ensemble/readout이효과적인지는미검증이다. 이전점수·기존seed·제안GPU예산을새확증이나실행승인으로재사용하지않는다.
