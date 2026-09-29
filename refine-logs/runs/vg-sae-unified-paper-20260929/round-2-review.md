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
