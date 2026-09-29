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
