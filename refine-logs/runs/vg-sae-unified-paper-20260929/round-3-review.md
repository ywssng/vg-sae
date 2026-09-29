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
