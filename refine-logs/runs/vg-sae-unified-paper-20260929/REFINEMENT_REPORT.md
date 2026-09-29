# VG-SAE Phase 1–2 refinement report

이번요청은 원래밀도추정 Phase2를 Phase1 기반 위에 refine하여 하나의논문계획으로 작성하는것이다. 새연구실험은 실행하지 않았다.

# Phase 1–2 통합 계획 검토 요약

2026-09-29. 실제 secondary GPT-6 Astra ultra reviewer와3차례 검토했다. 최종 **READY/9.27**, 대상은 계획의 준비도다. Same-family provisional, calibration:none. Phase2는 미실행이며 review가 가설의 성공·신규성 확정·GPU 실행승인을 뜻하지 않는다.

## Problem Anchor

- 연구 문제: Variational Garrote 기반 SAE를 명시한 통계물리적 모형과 변분원리에서 유도·이해하고, 그 모형의 밀도 응답과 독립 재학습 간 선택 불확실성으로 적절한 sparse operating density를 추정할 수 있는지 검증한다.
- 핵심 병목: 재구성 오차와 sparsity만으로 올바른 feature 분해나 밀도를 선택할 수 없다. 한 모델 내부의 posterior uncertainty, 모델 간 선택 불안정성, 실제 feature recovery 사이의 연결은 별도로 검증해야 한다.
- 유지할 목표: VG-SAE 자체의 개발과 이해를 중심에 둔다. Phase 1은 모형·추론·목적함수의 근거, Phase 2는 정답을 보지 않는 밀도 추정과 그 유용성 검증이다. Phase 1 성공이 Phase 2 성공을 함의한다고 가정하지 않는다.
- 비목표: 별도 복제 진단 주제로 전환, 모든 SAE를 VG의 특수형으로 선언, SOTA를 유일한 성공 조건으로 강제, 파라미터 없는 보편적 true-L0 발견, semantic posterior calibration 또는 thermodynamic phase transition의 무근거 주장.
- 제약: 이번 요청은 계획 작성·검토다. 새 학습·GPU 실험은 실행하지 않는다. 기존 결과와 원고는 보존한다. 후속 비용은 제안이며 9월 24일의 2 GPUh 승인 범위를 확장한 것으로 취급하지 않는다.
- 완료 기준: 원래 밀도 추정 목표를 포함한 하나의 논문 서사, 실행 가능한 추정 절차, 주장별 실험·반증 기준, Phase 1의 실제 근거와 Phase 2의 미검증 가설을 구분한 계획을 제공한다.


## 수정과 해결

| Round | 핵심 지적 | 반영 | 결과 |
|---|---|---|---|
| 1 | native-density curve 미정, template 전이 가정 부족, 보류/선택편향/예산 불명확 | density matching을 primary로 고정, pooled-support 계산, finite profile/abstention, all-world fallback, exact fit budget | REVISE7.95 |
| 2 | deployment coverage 실패가 유효한 C2a estimate를 지움; edge-case 규칙 부족 | estimate/deployment status 분리, 빈curve fallback, zero norm/tie/개별repeat drift/loop stop | REVISE8.965 |
| 3 | 수정된전체 proposal/protocol/outline 재평가 | 새component나benchmark를 추가하지 않고 같은목표·수치계약 유지 | READY9.27, remaining blocker 없음 |

## 최종 판단의 한계

- Phase1의 조건부 설명과 Phase2의 생성밀도/선택효용을 한논문으로 연결한다.
- Template transfer, achievable native coverage, stable-wrong, finite-R 변동, 실제GPU timing은 남은 경험적위험이다.
- 판정은 연구계획 준비도이며 결과의지지나 출판가능성의 보장이 아니다.
- 원래연구를 다른진단주제로 바꾸거나 SOTA를 유일한성공조건으로 강제하지 않았다.
- 숫자threshold와compute는 계획제안이다. 실행은 이번작업에 포함하지 않았다.

## 기록

`refine-logs/runs/vg-sae-unified-paper-20260929/round-1-review.md`, `round-2-review.md`, `round-3-review.md`에 full raw response가 있다. `round-1-refinement.md`와 `round-2-refinement.md`에 전체수정제안을 보존했다. 최종 논문구성은 저장소루트의 `PAPER_PLAN.md`, 실행프로토콜은 `refine-logs/EXPERIMENT_PLAN.md`다.


# Score history

모델: GPT-6 Astra ultra. 동일 reviewer를 이어서 사용했다. calibration:none, review_independence:same-family, acceptance_status:provisional. 점수는 **계획 준비도**이며 empirical support/novelty/paper acceptance가 아니다.

| Round | Fidelity | Specificity | Contribution | Frontier | Feasibility | Validation | Venue | Weighted | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.8 | 7.1 | 7.6 | 9.0 | 6.9 | 7.5 | 7.8 | 7.95 | REVISE |
| 2 | 9.8 | 8.4 | 9.0 | 9.3 | 8.8 | 8.5 | 8.9 | 8.965 | REVISE |
| 3 | 9.8 | 9.4 | 9.0 | 9.3 | 8.8 | 9.4 | 9.1 | 9.27 | READY |

Weights: .15/.25/.25/.15/.10/.05/.05. Threshold9, max5rounds. Round3에서READY/no-blocker로 종료했다.

## 주요 변경

1. 기존 first-principles 방향과 원래 density-estimation 목적을 동일 Problem Anchor에 고정.
2. Within-model uncertainty와 between-fit instability, 생성밀도와 recovery operating point를 분리.
3. 원고의 NNLS mixture를 단일template profile로 단순화하고 coverage/식별성/보류를 명시.
4. Native density matching과 truth-free feature correspondence를 통해 실제실행 인터페이스 정의.
5. Estimate와 deployment status를 분리하고 all-world policy와 fallback 이유를 기록.
6. 기존Phase1 결과·코드는 재사용하고 큰Stage2/3는 조건부확장으로 보존.

## 원문 review



# Round 1 raw method and integrated paper-plan review

<details open>
<summary>Full raw review of round-0-initial-proposal.md</summary>

Review date: 2026-09-29. Reviewer role: fresh method/outline reviewer. Review target: the FULL `round-0-initial-proposal.md`; subsequent protocol revisions were not scored. I read the root instructions, project memory, the local research-refine and paper-plan skills, the Phase 1 results report, the old manuscript's density-inference section, and targeted existing evaluator code. No new experiments were run. No human-curated good/bad proposal exemplars were supplied.

CALIBRATION: none

Evidence status: Phase 1 has existing small controlled results; Phase 2 has no results. This is a same-family provisional judgment of planning quality, not a novelty certification, independent empirical replication, or acceptance prediction.

## Overall judgment

The proposal preserves the user's actual research question and can support ONE paper. Its most useful move is to place the learned SAE's selection variables, inference errors, and between-fit instability in a single chain while refusing to identify these quantities with semantic correctness. The dominant prospective contribution is the truth-free transfer of a VG selection-curve estimator to learned SAE coordinates, with an explicit validity domain. Phase 1 supplies modeling and inference premises; it does not prove that transfer.

The current document is not yet an executable estimator specification. It still leaves the primary density-matching rule, numerical identifiability/abstention contract, recovery policy under abstention, and finite cost unresolved. These are central method decisions, not optional experiment polishing. A second, conceptual gap is that the borrowed template needs explicit sufficient assumptions after moving from a global fixed support to input-dependent supports and learned feature identities. The minimal fix is a short conditional derivation/assumption statement plus a frozen protocol, not a new trainable module or a large benchmark suite.

## Seven-axis scores

| Axis | Weight | Score / 10 | Reason |
|---|---:|---:|---|
| Problem fidelity | 0.15 | 9.8 | VG-SAE derivation/understanding and the original density-estimation objective are both retained; no substitution by a standalone diagnostic topic. |
| Method specificity | 0.25 | 7.1 | Splits, repeats, medoid alignment, finite-R correction, single-template profile fitting, and native deployment are specified, but the primary curve and acceptance set remain underdetermined. |
| Contribution quality | 0.25 | 7.6 | One plausible main contribution with sensible supporting analysis; the assumptions that make the curve location informative about generating density still need a crisp statement. |
| Appropriate frontier usage | 0.15 | 9.0 | SAE-specific alignment and native inference are appropriate. A foundation-model component would not improve this bottleneck; no trendy component is needed. |
| Feasibility | 0.10 | 6.9 | Tiny individual models are plausible, but the core plus development already exceeds the stated rough fit count, and no finite campaign cost cap or stop policy exists. |
| Focused validation | 0.05 | 7.5 | Three connected blocks and separate C2a/C2b endpoints are good; ablations/transfer are too open-ended and selective abstention remains unresolved for C2b. |
| Venue readiness | 0.05 | 7.8 | A coherent model-to-estimator paper is plausible, but exact positioning, claim-evidence mapping, and the conditional success/failure narrative need tightening. Venue is correctly left uncommitted. |

Weighted composite = 0.15(9.8) + 0.25(7.1) + 0.25(7.6) + 0.15(9.0) + 0.10(6.9) + 0.05(7.5) + 0.05(7.8) = **7.95 / 10**.

GAP: The gap to READY is primarily closure of the estimator contract and its assumptions. Existing Phase 1 evidence is sufficient to motivate the proposed bridge, but cannot establish the density template's applicability. A high-quality revision would explicitly identify what is inherited from Soh, what is an algebraic identity, what is a new SAE-specific protocol, and what is an untested empirical hypothesis; fix the actual native-density curve used by the primary estimator; set a compact numerical acceptance rule that detects flat/ambiguous profiles; make C2b an all-world policy evaluation with an honest abstention decomposition; and give exact fit arithmetic and a bounded execution proposal. The absence of Phase 2 results is expected at this planning stage and is not itself a reason to deny planning readiness.

## Blocking issues and minimum fixes

### B1 — CRITICAL: Define one primary native-density curve

The text first computes U at a common gamma and then says that common-density checkpoint selection will be compared, leaving the primary rule to development. These are different estimands. Common gamma with varying achieved densities can turn density mismatch into instability; common density uses different gamma values across repetitions and estimates reproducibility conditional on achieved sparsity. The latter is a defensible operating-density experiment, but the paper must name it and not call it the exact field-response curve.

Minimum fix: fix a truth-free target-density ladder, choose the closest native final checkpoint in each repeat using X_align, specify absolute density tolerance, how many repeats must cover a target, whether the same checkpoint may be reused across adjacent targets, and how reused/near-duplicate density points are collapsed or weighted. Form the selected checkpoint tuple first and align that tuple. Compute its achieved rho and U only on X_select. If any of these are tuned on development worlds, record the finite alternatives and a freeze point. Common-gamma curves can remain a labeled sensitivity analysis. No interpolation, mask threshold retuning, or ground-truth matching should enter the primary pipeline.

### B2 — CRITICAL: State the support and alignment assumptions behind the template

The template is imported from a global selection problem. In a SAE, true support varies with x, learned identities can split or merge, and selection rates can differ across true atoms and samples. Averaging a nonlinear kernel over input-specific densities generally does not equal evaluating the kernel at average densities. The uncertainty decomposition alone does not close this gap.

Minimum fix: give the short sufficient-conditions calculation using fixed correctly corresponding feature slots. For example, if selection probabilities over repeats are homogeneous within true-active and true-inactive (x,j) groups, with no false positives below q or no false negatives above q, the existing kernel follows with q equal to the pooled true-active fraction. State clearly that these are strong inherited/exchangeability assumptions, not established properties of learned SAEs and not a novel identity. If rates vary, write the averaged expression and show where variance/heterogeneity terms can distort the curve. The empirical claim is then that the proposed estimator remains useful under a specified degree of assumption violation. Use the planned controlled homogeneous generator first; one focused departure analysis is enough. No new theorem about generic identifiability is necessary.

### B3 — CRITICAL: Specify finite-grid identifiability and abstention numerically

With U = A f(rho;q) + b, A = 0 makes every q equivalent. A positive but small A, or two well-separated profile minima, can also make q practically unidentified despite seven points, bilateral coverage, and a small residual. A small residual alone rewards flat curves. Nonnegative A and b are constraints rather than meaningful signal checks.

Minimum fix: fix the q grid, density domain, fit weighting, tie rule, absolute/density-relative coverage criteria, a signal-to-residual or signal-size floor, and an explicit profile ambiguity criterion. A compact choice is a predeclared near-minimum profile set and a maximum allowed width/separation; require that its relevant region satisfies bilateral coverage. Specify how alignment quality and matching ambiguity trigger abstention, including duplicated/near-zero atoms and non-finite training outcomes. Keep endpoint all-off/all-on records as diagnostics, but exclude them from admissible q and do not let their low U certify a solution. State bootstrap replicate count, how failed bootstrap fits are counted, and how too few usable bootstrap replicates cause abstention. This remains an operational reliability rule, never a certificate of feature correctness.

### B4 — CRITICAL: Close C2b under abstention and define the evaluator

C2a correctly counts abstaining worlds in the denominator. C2b currently lacks a deployed policy when the estimator abstains. Evaluating recovery only on accepted worlds could produce a favorable result by selecting easy worlds. The stated minimum effect is also not numerical. The coefficient-NMSE oracle must be completely defined to make regret reproducible.

Minimum fix: report (i) coverage, (ii) accepted-only density/recovery diagnostics, and (iii) an all-world deployment policy with one fixed truth-free fallback, such as reconstruction-only selection on abstention. The fallback is a policy component and does not turn an abstention into a successful density estimate. Freeze the exact coefficient NMSE normalization, hard-code readout, decoder scale convention, signed truth-matching rule, and tie handling. Define the finite-bank hindsight oracle per repeat on X_test and aggregate over repeats before world-level comparisons. Regret to the same finite-bank oracle must be nonnegative up to numerical tolerance; a signed difference between two selectors' regrets may be negative. Do not label this hindsight oracle a population-optimal or uniquely correct L_oper.

Set the numerical minimum recovery effect and which comparisons must pass. Predeclare equal cell weighting or another aggregation rule, the paired world resampling scheme, and handling of multiple comparator tests. An interval strictly on the favorable side of zero plus a minimum effect is clearer than the current 'does not exceed zero' phrasing. Report each family/density cell so a pooled success cannot conceal an entire failed regime.

### B5 — IMPORTANT: TopK need not have the estimator's required resolution

At width 16, an exactly-k-sparse TopK family has rho = k/16. Near q = 1/16 or 2/16, it cannot supply three strictly interior density points below q. ReLU-induced zeros can make actual density smaller than k/16, but relying on these accidental zeros does not ensure the required coverage. Therefore imposing the identical template eligibility rule can make low-density TopK fail structurally rather than scientifically.

Minimum fix: keep TopK as a recovery-family comparator with a fully specified native count ladder and equal per-fit training budget. Do not require a C2a head-to-head estimate from an ineligible curve or count that ineligibility as evidence that VG's estimator is superior. Applying the template to eligible TopK curves can be an explicitly secondary analysis. No continuous mask interpolation is needed to rescue this comparison.

### B6 — IMPORTANT: Give exact counts and a finite execution proposal

Using the stated upper bounds, confirmatory training is 2 families × 3 densities × 8 worlds × 6 repeats × 17 controls × 2 methods = 9,792 fits. Development is 2 families × 2 worlds × 4 repeats × 17 controls × 2 methods = 544 fits. Together these are 10,336 fits before any ablations, timing runs, extensions, or failed retries, rather than fewer than 10k. At 4,000 updates each this is 41,344,000 optimizer updates. TopK may have fewer distinct native controls, but that must be counted explicitly.

Minimum fix: enumerate per-method controls and exact block counts, set finite ablation subsets and a total fit/update cap, and specify one future timing gate that converts measured throughput into a proposed GPU-hour budget before any larger execution. If the cap cannot support the full matrix, shrink optional transfer or secondary ablations first, rather than silently treating repeat counts as world counts. Training failure/retry rules must be bounded and retained in the denominator. This review does not authorize new compute; planning must remain separate from execution approval.

## Nonblocking improvements

1. **Alignment estimand and audit.** Signed cosine Hungarian to a truth-free medoid is a good primary choice for nonnegative codes. Describe how columns are normalized, dead/zero atoms are treated, and deterministic ties are resolved. The finite-R correction is valid conditional on correspondences; estimated matching is part of the estimator and may artificially stabilize or destabilize U. Recomputing alignment in bootstrap is appropriate sensitivity analysis but does not prove unbiasedness. A small permutation/dead-atom/duplicate-atom unit control plus a truth-alignment diagnostic is sufficient.
2. **Bootstrap terminology.** Repeat resampling quantifies conditional optimization sensitivity within a fixed world and fixed data splits. World-level resampling quantifies variation over sampled worlds. Neither by itself captures all sample-level uncertainty; avoid advertising a universal 95% confidence guarantee from R = 6. Empirical interval coverage can be evaluated on synthetic worlds without relabeling bootstrap output as posterior mass.
3. **Native deployment and bank fairness.** Define whether each baseline chooses per-repeat checkpoints or one common control, and use that interface consistently in its oracle comparison. Matching q to achieved X_select density is acceptable; test density drift and quantization error should remain separate outputs. Report selected-checkpoint quality separately from total bank-building cost.
4. **Scope of beta/variance ablations.** The Phase 1 findings are correctly characterized as conditional or total-effect evidence. Keep only ablations that test why the density estimator behaves differently; rerunning the entire Phase 1 matrix is unnecessary.
5. **Negative-result route.** Prestate the four possible C2a/C2b outcomes and the corresponding allowed claims. Accurate generating density is not equivalent to useful recovery selection, and neither follows from improved reverse-KL. A failed density hypothesis may still produce a useful integrated account of this VG-SAE construction, but publication-level novelty must then be reassessed honestly rather than guaranteed by the plan.

## Integrated paper outline review

| Outline dimension | Score / 10 | Minimum fix |
|---|---:|---|
| Logical flow | 8.5 | Add one explicit bridge paragraph from conditional support inference to across-fit selection instability; state what does and does not transfer. |
| Claim-evidence alignment | 8.0 | Add a compact matrix with C1 existing evidence and caveats, C2a/C2b planned metrics, acceptance rules, and exact figure/table locations. |
| Completeness of necessary analyses | 7.0 | Include native-density mismatch, profile identifiability/abstention, and all-world fallback recovery; these are core analyses, not new independent contributions. |
| Positioning relative to prior work | 7.5 | Supply a citation-role scaffold separating original VG, the borrowed Soh density kernel, prior variational SAE/sparse coding, and current SAE density/identity problems. Avoid claiming novelty from standard entropy/free-energy identities. |
| Page feasibility | 8.0 | The stated sections sum exactly to 10 pages, but the three-page experiment section cannot carry all proposed extensions. Reserve main space for one compact Phase 1 figure, a main density/recovery figure, and the all-world table; move details and conditional transfer to appendices or a follow-up. |

The working title is appropriately conditional and broader than an unearned true-L0 claim. Retain the existing PRE-style orientation unless the user later chooses a venue. The 10-page internal budget is correctly not presented as an official venue limit.

Suggested minimal narrative map:

- Introduction: one problem—how a model-derived selection description can inform a SAE operating-density choice—and two distinct empirical endpoints.
- Model and inference: derivation, conditional target, known response identities, exact/MF/encoder distinction, existing Phase 1 evidence compressed around the premises needed by the estimator.
- Density inference: alignment, native density, transferred template assumptions, estimator and failure handling. This is the prospective method contribution.
- Experiments: existing inference/objective checks, fresh synthetic C2a/C2b results, then one mechanism/assumption sensitivity. Keep worlds and repeats visibly distinct.
- Discussion: what successful and failed C2 outcomes license; stable-but-wrong, capacity mismatch, nonconstant amplitudes/noise, and finite optimization budget.

The hero figure proposal is strong: a curve panel plus an independently evaluated recovery panel makes co-localization testable without hiding truth in the selector. The caption should explicitly label truth/recovery optima as evaluator-only overlays and show coverage or abstention. A predetermined illustrative world plus an all-world summary prevents favorable-example selection. A blank planned figure specification is appropriate now; no Phase 2 numbers should be filled in.

## Simplification opportunities

1. Make one matched-native-density estimator primary; retain common-gamma curves as one sensitivity and remove undecided parallel primary variants.
2. Keep the single-template model. Leave the old NNLS mixture as a historical appendix audit if it clarifies the old manuscript, not a competing primary method or posterior estimator.
3. Use TopK as a recovery comparator and gate all large transfer studies behind the primary result. The core claim does not require immediate GPT-scale training, a large baseline zoo, or a new diagnostic project.

## Modernization opportunities

NONE. The relevant modernization is accurate native SAE inference, learned-coordinate alignment, and transparent estimator evaluation. An LLM, diffusion model, teacher, or auxiliary trainable component would add scope without addressing the current bottleneck.

## Drift warning

NONE in the proposal. The immutable Problem Anchor is preserved. Requests for an independent clone-diagnosis paper, universal SOTA superiority, or replacing the original density hypothesis with only posterior diagnostics would cause drift and should be rejected. The fixes above strengthen the requested VG-SAE theory/inference-to-density paper.

## Verdict

**REVISE — 7.95 / 10.** The direction is promising and coherent, with one dominant prospective contribution. Resolve B1–B4 and the bounded design issues before marking the plan READY. Do not add neural components or enlarge the benchmark list to chase review score. A ready plan would still contain untested Phase 2 hypotheses; readiness is not a claim that those hypotheses are true.

</details>


# Round 2 raw method and integrated paper-plan review

<details open>
<summary>Full raw review of revised proposal, protocol, and outline</summary>

Review date: 2026-09-29. Reviewed in full: this run's `FINAL_PROPOSAL.md`, `EXPERIMENT_PLAN.md`, `PAPER_PLAN.md`, and `LITERATURE_GROUNDING.md`; also read the response table in `round-1-refinement.md`. The score concerns the revision as read before the parent agent receives this review. No experiment or training was run. This review does not independently recertify every cited paper or bibliographic record.

CALIBRATION: none

Status: same-family provisional planning review. Phase 2 remains unexecuted, as appropriately disclosed. Missing Phase 2 outcomes are not themselves a planning-readiness failure.

## Anchor and method assessment

The original Problem Anchor is preserved. This is now a coherent VG-SAE paper plan linking conditional modeling/inference to truth-free density estimation and separately measured feature recovery. It does not replace the original research with clone diagnosis, impose SOTA as the only contribution, or claim known identities as new theorems.

The revision is substantially more specific. Native-density matching is primary; the pooled-support calculation identifies the conditions that recover the borrowed kernel; a single profile fit and finite reliability gates replace ambiguous mixture-posterior language; a fixed fallback prevents accepted-only recovery claims; and the independent reference cleanly separates alignment from the ensemble used in U. The dominant contribution is sharper. The neural mechanism remains small, although the experimental protocol is necessarily longer because it defines the estimator rather than merely naming it.

One claim-defining issue remains: A6 currently allows failure to find a sufficiently close deployable checkpoint to erase an otherwise valid C2a estimate. This couples generating-density accuracy to the bank's deployment resolution, contrary to the declared separation of C2a and C2b. It should be fixed before freezing the plan. Several small implementation contracts can be completed in the same revision without adding experiments.

## Seven-axis scores

| Axis | Weight | Score / 10 | Assessment |
|---|---:|---:|---|
| Problem fidelity | 0.15 | 9.8 | The model-to-density question and user constraints remain intact. |
| Method specificity | 0.25 | 8.4 | Nearly executable numerical contract; estimate-versus-deployment status is still conflated, and a few degenerate-case conventions are missing. |
| Contribution quality | 0.25 | 9.0 | Focused candidate contribution, explicit inherited assumptions, and appropriate non-novelty statements. Actual empirical and novelty strength remain conditional on later evidence. |
| Appropriate frontier usage | 0.15 | 9.3 | Current SAE identity/stability work is relevantly positioned; no unrelated foundation-model module is added. |
| Feasibility | 0.10 | 8.8 | Exact counts, bounded adaptation, bank reuse, timing assumptions, and a proposed cap are now present. Development must still establish that coverage is achievable. |
| Focused validation | 0.05 | 8.5 | All-world policy and minimal mechanisms are well chosen, but C2a reporting must be insulated from deployment failure. |
| Venue readiness | 0.05 | 8.9 | Claims, figures, negative outcomes, citations, and a feasible internal page budget are substantially improved. |

Weighted composite = 0.15(9.8) + 0.25(8.4) + 0.25(9.0) + 0.15(9.3) + 0.10(8.8) + 0.05(8.5) + 0.05(8.9) = **8.965 / 10**, reported as **8.97 / 10**.

GAP: Most Round 1 concerns are resolved. The remaining gap is not a missing model, benchmark, or Phase 2 result. It is one status/estimand distinction plus a small set of deterministic edge-case rules. Keep a valid density estimate as a C2a output when the bank cannot deploy near it, and let only the operational C2b policy fall back in that case. Then make the comparator and alignment degenerate cases explicit. With these changes, the plan can reach READY while transparently retaining its untested template-transfer and finite-budget risks.

## Resolution of Round 1 issues

| Round 1 issue | Status | Evidence in revision |
|---|---|---|
| B1 primary native-density curve | Resolved | A3 fixes target grid, tolerance, deterministic bounded control adaptation, and independent reference; A5 uses measured select density and handles duplicates/drift. |
| B2 transferred template assumptions | Resolved as a plan | Pooled active/inactive rates and heterogeneity terms provide the missing sufficient-conditions explanation; empirical applicability remains explicitly open. |
| B3 profile identifiability and abstention | Resolved as a plan | Finite q grid, bilateral coverage, signal/residual/profile-width criteria, conditional bootstrap, and alternate reference are specified. These are operating rules, not guarantees. |
| B4 recovery under abstention | Mostly resolved | All-world reconstruction fallback, exact hard-code NMSE, same-bank hindsight oracle, and primary statistical hierarchy are clear; C2a/deployment status must still be separated. |
| B5 TopK resolution | Resolved | TopK is a recovery comparator and structural ineligibility is not scored as VG density superiority. |
| B6 finite cost/counts | Resolved as a proposal | Exact block arithmetic, bounded retries, timing-dependent estimates, and a proposed cap replace the vague fit count. No execution authorization is inferred. |

## Remaining blocker and minimum fix

### R2-B1 — IMPORTANT, blocks freezing: separate estimate validity from deployment coverage

A6 states that if any selected repeat has `abs(rho_selected-q_hat)>delta(q_hat)`, the world selector abstains, and then C2a counts abstention as failure/nonreporting. This discards a valid fitted q because of a separate finite-checkpoint-bank limitation. For example, an accurate q_hat near 0.0625 can lie between the native ladder targets 0.045 and 0.0675; the latter differs by 0.005, while delta(0.0625)=0.003125. The target-coverage routine does not promise a native checkpoint within this smaller radius of every possible fitted q. Thus a valid and accurate density estimate can be converted into a C2a failure even in an ideal template case.

Minimum fix: emit at least two explicit statuses.

- `estimate_status`: valid or abstain under A3–A5; a valid q_hat remains available for C2a regardless of deployment granularity.
- `deployment_status`: native selection available, or fallback required because the estimate abstained or native coverage is inadequate.

C2a report/error/success criteria should use `estimate_status`. C2b all-world policy should use the fixed reconstruction fallback whenever deployment is unavailable. Report estimate coverage, deployment coverage conditional on valid estimation, and total fallback rate separately, with reason codes. Keep failed deployment's quantization error and q_hat in the output rather than nulling the estimate. If desired, add an explicitly secondary joint 'accurate estimate and deployable checkpoint' rate; do not silently redefine C2a as that joint endpoint.

No extra training is required. If later choosing bounded post-fit control refinement instead, it would need a separate finite rule and compute accounting; the simpler status split already resolves the claim issue.

## Small contract fixes to close now

1. **Raw-U and posterior-uncertainty comparators on empty curves.** A6 defines their objective on retained common targets but does not state what happens when no target survives matching/drift checks. Give both a deterministic reconstruction fallback and a reason flag. State that a selected target deploys its already matched per-repeat checkpoint tuple. If the intended behavior is nearest-checkpoint remapping, specify the density source and tie rule. These are secondary comparators, but all-world rows should not disappear or invent a target.
2. **Zero-norm and dead-feature alignment convention.** 'Preserve all slots' is correct but does not define norm folding for a zero decoder column. Define a numerical norm floor, whether a zero column remains zero, the corresponding folded code convention, and the signed-cosine value used for matching. Preserve its slot and record degeneracy. Also give a deterministic exact-tie rule for Hungarian assignments that does not perturb non-tied optima. This supports the stated permutation/duplicate controls and avoids a platform-dependent or NaN matching path.
3. **Density drift aggregation.** A5 says align/select density drift exceeding `2*delta(t)` excludes a target; explicitly state whether this is checked per repeat/reference or only on an ensemble average. Per-run checks are more faithful to the same-native-density interpretation because opposing drifts can cancel in the average. The independent reference's role in the check should also be explicit.
4. **Development acceptance and bootstrap.** The R=3 development caveat is appropriate: a size-3 bootstrap has all three distinct repeats only 6/27 of the time, so the confirmatory valid-refit threshold cannot be applied there. For R=5, the distinct-repeat gate alone retains about 90.24% of draws before any profile failures, making 80% total validity a demanding but possible rule. Keep this as a disclosed reliability choice. Development can inspect curve/profile behavior and timing but should not report its coverage as the confirmatory estimator's calibrated coverage.
5. **Adaptive loop closure.** The bounded midpoint rule is largely clear. One final sentence can say it stops when the added-control cap is reached or all unmet targets lack an untried bracket midpoint; excluded targets remain excluded. This avoids an implementation looping over permanently unaddable targets. This is a minor clarification, not a reason to change the design.

The first three are small concrete interface decisions. They do not justify another baseline, new estimator family, or larger data campaign.

## Arithmetic and statistical-design checks

The revised fit arithmetic is internally correct:

- Development: 8 worlds × 4 runs × 17/25 controls = 544/800 fits.
- Main VG: 36 × 6 × 17/25 = 3,672/5,400 fits.
- Three extra ablation methods on six reused worlds: 6 × 6 × 3 × 17/25 = 1,836/2,700 additional fits. The primary bank is already counted.
- L1: 612/900; TopK: 576; combined 1,188/1,476.
- Core total: 7,240/10,376 distinct fits; 24 extra 4,000-update segments give 7,264/10,400 step-equivalents.
- Three transfer conditions × three new worlds × six runs × 17/25 = 918/1,350; with core, 8,158/11,726 fits and 8,182/11,750 step-equivalents.
- The stated 10–40 device-second assumption and 25% overhead produce approximately 25.2 to 144.4 GPU-hours over the core base/max extremes, consistent with the rounded 25–145 range. These are explicitly unmeasured planning assumptions.

The six mechanism IDs correspond to the first three p=0.125 worlds in each family under the stated ordering. Reusing these worlds does not inflate the independent-world denominator. Primary estimates are generated before truth access; per-repeat averages precede world-level analysis; paired stratified bootstrap preserves the stated unit of inference.

The C2b superiority-plus-noninferiority hierarchy is acceptable as a predeclared conjunction: reconstruction-only is the primary improvement target, c_dec is a guard, and other comparisons are explanatory. The plan appropriately does not claim power for six worlds per cell. Report the actual intervals and all cell summaries even when a global criterion passes. Bootstrap sensitivity bands for q are correctly distinguished from calibrated confidence/credible intervals.

## Integrated outline review

| Outline dimension | Score / 10 | Assessment and minimum fix |
|---|---:|---|
| Logical flow | 9.5 | The model → uncertainty distinction → transferred estimator → recovery chain is explicit. No new section is needed. |
| Claim-evidence alignment | 9.2 | Existing versus planned evidence and C2a/C2b are clearly mapped. Propagate the estimate/deployment distinction into the matrix and Fig3/Table1. |
| Necessary analyses present | 8.8 | Template assumptions, native density, profile failures, all-world policy, and mechanisms are present. Add deployment coverage as a separate output, not a new study. |
| Positioning | 9.0 | The plan credits known kernels, variational objectives, Hungarian matching, and stability. Literature grounding is appropriately labeled a targeted primary-source review, not a full novelty certification. |
| Page feasibility | 9.0 | Seven sections sum to 10 pages. Phase 1 is compact and large transfer is conditional. Keep most protocol cutoffs and all control-level tables in the appendix. |

The PRE-style orientation and venue-neutral internal page budget are appropriately preserved. The paper does not need an invented bound-comparison table. The proposed assumption/observable/density-selection table is more honest. The figure plan is feasible if Fig1's illustrative world remains small and Fig3/Table1 carry the all-world evidence. The accepted/failed/negative outcome table is particularly useful and should survive later drafting.

For the final paper, describe the theoretical content as a model-based derivation and conditional analysis unless an actually new theorem is established. Do not turn the homogeneous/no-FP/no-FN derivation into a generic learned-SAE identifiability guarantee. An accurate posterior approximation, a stable fitted representation, an accurate generating count, and a good recovery operating point remain four distinct observations.

## Simplification opportunities

1. Resolve deployment failure with separate status fields and the existing fallback, not additional training or a second estimator.
2. Keep numeric reliability gates in the protocol/appendix and present the main algorithm with only their purposes in the paper body.
3. Retain the current conditionally staged transfer scope; no further benchmark expansion is needed for the core planned claim.

## Modernization opportunities

NONE. Current SAE-specific stability and identifiability literature is incorporated in the appropriate role. Adding a teacher, learned prior, diffusion model, or LLM is not warranted by these remaining issues.

## Drift warning

NONE. The original VG-SAE and density-estimation objective is preserved. The proposed status split strengthens the user's intended distinction rather than changing the research question.

## Verdict

**REVISE — 8.97 / 10.** One claim-defining status issue and a few small deterministic interface conventions remain. A further large redesign is unnecessary. After these are fixed consistently across proposal, protocol, and outline, planning readiness can be assessed without waiting for Phase 2 results. Any later READY judgment would remain conditional on development feasibility and would not certify empirical success or publication novelty.

</details>


# Round 3 raw method and integrated paper-plan review

<details open>
<summary>Full raw final planning review</summary>

Review date: 2026-09-29. I read the updated full `FINAL_PROPOSAL.md`, `EXPERIMENT_PLAN.md`, and `PAPER_PLAN.md`, and the change record in `round-2-refinement.md`. The applicable instructions, project memory, Phase 1 report, old manuscript context, and literature scaffold were read in the preceding rounds. No Phase 2 experiment, new training, or implementation verification was performed in this review.

CALIBRATION: none

Review status: same-family provisional. The verdict below concerns **planning readiness**, not scientific success, confirmed novelty, code correctness, publication acceptance, or permission to spend the proposed compute budget.

## Verdict and anchor check

**READY — 9.27 / 10 for planning. No remaining blocking issue identified.**

The original Problem Anchor is preserved. Phase 1's model, inference, and objective analysis supplies the assumptions and observational distinctions used to motivate Phase 2. Phase 2 retains the original density-estimation question and separately tests whether the estimated operating point improves feature recovery. The plan does not turn into a standalone clone-diagnosis project or impose SOTA as the sole measure of research value.

The dominant prospective contribution remains one focused method question: whether a truth-free selection-uncertainty curve can transfer from the VG setting to learned SAE coordinates, under explicitly examined conditions. The conditional free-energy explanation supports this question. Known algebraic identities, the inherited kernel, standard matching, and generic seed stability are appropriately excluded from novelty claims.

## Seven-axis scores

| Axis | Weight | Score / 10 | Assessment |
|---|---:|---:|---|
| Problem fidelity | 0.15 | 9.8 | Preserves VG-SAE development, first-principles understanding, and the original density-inference endpoint. |
| Method specificity | 0.25 | 9.4 | The bank, split roles, density mapping, alignment, profile fit, reliability checks, two statuses, deployment, and fallback interfaces are sufficiently concrete to implement. |
| Contribution quality | 0.25 | 9.0 | One clear candidate contribution and one supporting analysis; claims are bounded by inherited assumptions and planned empirical evidence. |
| Appropriate frontier usage | 0.15 | 9.3 | Uses current SAE identity/stability context appropriately and avoids irrelevant foundation-model additions. |
| Feasibility | 0.10 | 8.8 | Exact bounded fit counts, timing gates, bank reuse, failure rules, and proposed cost cap are present. Actual coverage and throughput remain development-stage uncertainties. |
| Focused validation | 0.05 | 9.4 | Generating-density inference, deployment availability, and all-world recovery are now cleanly separated; failure and negative outcomes remain observable. |
| Venue readiness | 0.05 | 9.1 | The integrated narrative, evidence matrix, figures, conditional title, and internal page budget are sufficiently specified for the next stage. |

Weighted composite = 0.15(9.8) + 0.25(9.4) + 0.25(9.0) + 0.15(9.3) + 0.10(8.8) + 0.05(9.4) + 0.05(9.1) = **9.27 / 10**.

GAP: No further planning revision is required by this review. The remaining distance from a successful paper is evidential and operational: implement the frozen interfaces, pass the deterministic contracts, establish development coverage and throughput, lock the protocol, and then obtain fresh Phase 2 outcomes. The known template may fail under learned correspondence or heterogeneous selection; the strict reliability gates may yield low report rates; native bank quantization may limit deployment; and six worlds per cell may leave broad statistical intervals. These are disclosed hypotheses and limitations the planned evaluation is designed to measure, not hidden deficiencies that can be resolved by adding prose or modules. READY does not imply that any success threshold has been met.

## Resolution of the final review concerns

| Round 2 concern | Final assessment |
|---|---|
| Valid density estimate erased by deployment coverage failure | Resolved. A5 controls `estimate_status`; A6 separately sets `deployment_status`. A valid q_hat survives quantization failure and remains eligible for C2a. C2b uses the fixed fallback with separate reasons. |
| Empty-target raw-U and within-uncertainty policies | Resolved. Both specify reconstruction fallback when no retained target exists, preserving all-world outputs. |
| Zero and near-zero decoder normalization | Resolved. Positive norms use reconstruction-preserving direction/coefficient folding; exact zeros use zero direction/cosine/effective coefficient. Native masks remain separately observed, avoiding a silent change to the density estimand. |
| Assignment ties | Resolved. Equal-score assignments have a deterministic index-based tie rule, and the plan distinguishes tie-free permutation invariance from duplicate/tie sensitivity. |
| Density drift cancellation | Resolved. Each ensemble repeat and the reference are checked separately against the drift bound. Opposing drifts cannot cancel in an ensemble average. |
| Adaptive midpoint loop termination | Resolved. The loop stops on its finite control cap or a pass with no addable midpoint; only actual new fits consume the adaptation budget. |
| R=3 development versus R=5 bootstrap acceptance | Correctly distinguished. Development is not reported as confirmation of the R=5 reliability procedure. |

The status split is the key scientific improvement in this final revision. It preserves the distinction between estimating an expected generating count and finding an already-trained native model near that estimate. The all-world policy then measures the actual utility of the combined estimator and fallback, with fallback outcomes not misattributed to successful density-guided deployment.

## Estimator and evaluation consistency

The current algorithm has a coherent data flow: fixed training worlds and independent optimization runs; held-out native-density matching; truth-free signed matching to a separate reference; a common-input uncertainty curve; a finite-grid single-template fit with explicit abstention; and final evaluation with ground truth isolated from selection. The proposed oracle is correctly a same-bank finite-test hindsight benchmark, not a deployable selector or a population optimum.

The finite-ensemble soft decomposition, corrected hard between-run statistic, conditional fixed-field response, and learned native-density curve remain distinct. The pooled support calculation states sufficient homogeneity/no-FP/no-FN conditions and identifies how heterogeneous rates change the expected uncertainty. It does not prove these conditions for a learned SAE. Alignment sensitivity and bootstrap variability are reported without pretending they certify correct feature identity or calibrated posterior uncertainty.

The native TopK resolution limitation is handled by using it as a recovery-family comparator. Template ineligibility is not counted as VG superiority. L1 and the planned small ablation subset can examine whether any gain is VG-specific or a generic stability effect without becoming separate research projects. Primary and ablation banks use finite native controls and fixed final training budgets; no test-best checkpoint, seed, or threshold is selected.

The C2b inference hierarchy remains clear: meaningful improvement against reconstruction-only plus a c_dec noninferiority guard, with other comparisons labeled explanatory. All-world results, estimate-accepted diagnostics, deployment-available diagnostics, and fallback reasons are separated. No generation-density success is inferred from recovery improvement, and no recovery success is inferred from a more accurate posterior approximation.

## Counts and scope

No experimental-count changes were introduced in this revision. The previously verified arithmetic remains consistent: core 7,240/10,376 distinct fits, or 7,264/10,400 4,000-update equivalents including development extensions; optional small transfer adds 918/1,350 fits. The 25–145 GPU-hour range is explicitly based on unmeasured throughput assumptions, with a proposed 160 GPU-hour cap. It is not described as an approved budget, measured runtime, or extension of the old 2 GPU-hour authorization.

The plan bounds retry behavior and does not silently replace failed worlds or reinterpret repeats as independent worlds. Larger SynthSAEBench and LM studies remain conditional and separately costed. Their absence from the core is appropriate; a large LM benchmark is not needed merely to make this proposal sound current.

## Integrated outline assessment

| Outline dimension | Score / 10 | Assessment |
|---|---:|---|
| Logical flow | 9.5 | The progression from reconstruction ambiguity through model/inference and uncertainty to density estimation and recovery is coherent. |
| Claim-evidence alignment | 9.4 | C1 existing evidence and C2 planned endpoints are separated; figures and outcome-dependent claims map to the correct evidence. |
| Necessary analyses present | 9.3 | Template assumptions, alignment, native density, abstention, deployment limitations, and all-world policy behavior are covered. |
| Positioning relative to prior work | 9.0 | Inherited theory/tools are credited; the candidate difference is the explicit SAE density-transfer protocol and its measured validity domain. |
| Page feasibility | 9.0 | The seven-section allocation sums to 10 internal pages, and detailed protocol/full grids/large transfer belong in the appendix or later work. |

The existing PRE-style orientation is respected, and no official venue page limit is invented. The working title is appropriately conditional on later evidence. The hero figure uses evaluator-only truth/recovery overlays, and preselected examples plus all-world summaries reduce favorable-case selection. The four-way C2a/C2b outcome table gives an honest route for mixed or negative results without substituting Phase 1 success for the missing Phase 2 claim.

The next drafting stage should simply follow the existing protocol when populating tables: report estimation coverage and deployment availability with distinct labels. This is an application of the now-specified output contract, not a new requested analysis or an additional blocker.

## Simplification opportunities

No further scientific component needs deletion or addition. Keep the detailed numeric reliability rules in the protocol and appendix while presenting their roles succinctly in the main algorithm. Retain the single-template primary estimator and the current conditional transfer scope.

## Modernization opportunities

NONE. A new teacher, encoder, learned prior, LLM, or diffusion component would not resolve the remaining empirical uncertainty more directly than the planned validation.

## Drift warning

NONE. The original Problem Anchor remains intact. The plan still advances VG-SAE understanding and the original density-estimation hypothesis in one paper.

## Remaining action items and final interpretation

No blocking planning fix remains. Implementation, development feasibility checks, protocol locking, and fresh experiments are the next planned stages; they have not happened in this task. Do not treat this READY review as their completion or as an automatic authorization to execute the proposed budget. If later development changes the estimator or horizon, record a new frozen version before exposing confirmatory outcomes, as the plan already requires.

**Final verdict: READY, 9.27 / 10 — planning readiness only, same-family provisional.**

</details>
