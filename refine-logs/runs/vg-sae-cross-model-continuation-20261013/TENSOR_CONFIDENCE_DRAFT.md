# Third-cumulant confidence with an unknown mean — provisional draft

2026-10-13. Independent Astra derivation; pending further Opus/Gemini review. No numerical evaluation, code implementation, test, data analysis, Colab allocation or training was performed. This is the only file written in this task.

## Target and decision

Supply a rigorous finite-sample tensor confidence event for `THIRD_CUMULANT_GEOMETRY_CANDIDATE.md`, including the unknown mean, and assess whether its simplest explicit confidence constructions can support a useful geometry bound at n<=8192.

**Result.** A translation-invariant unbiased estimator from independent triples, its exact Hilbert-space variance, and elementary Chebyshev/median-of-means confidence balls are derived below. At K=16 and the declared p interval, these particular global Frobenius confidence formulas cannot certify small dictionary error at n<=8192, even when granted the exact variance and exact cumulant amplitude. This does not establish an information-theoretic sample lower bound or failure of another tensor estimator, another tensor norm, sharper concentration, a structured likelihood method, or VG-SAE.

The unknown mean is resolved exactly. The remaining obstruction is the size of this explicit global confidence radius, rather than a missing centering term. No sampling job is justified merely to evaluate that already inadequate radius.

## 1. Contract and invariant object

Let

```
X=offset+D_* S+sigma G,
S_j iid Bernoulli(p), 0<p<1/2,
D_* in O(K),  G~N(0,I_K), independent of S.
```

The offset and D_* are unknown. The population mean is mu. The target is the full third central cumulant tensor

```
T=E[(X-mu)^{tensor 3}]=kappa sum_j d_*j^{tensor 3},
kappa=p(1-p)(1-2p)>0.
```

Tensor norms below are the ordinary Frobenius norms over all ordered indices; an off-diagonal symmetric entry is counted with its ordinary multiplicity. The variance calculation uses isotropic Gaussian noise, so the centered coordinates are independent in the unknown teacher basis. The population cumulant identity itself extends to independent Gaussian noise with general covariance, but the explicit variance formula below must not be carried over unchanged to that larger contract.

Sigma, or a valid structural upper bound on it, is needed to calibrate the proposed moment bound. The estimator itself uses neither sigma, p, mu nor D_*. Supplying a sigma bound for confidence construction is distinct from claiming that the third-cumulant population identity needs known noise.

All input rows are iid from this contract. Grouping into triples and blocks is fixed without looking at the data. The geometry event will apply to every orthogonal Q on the same event, including data-dependent Q; this does not remove the separate selection requirements for a clean-risk/SURE certificate.

## 2. Unbiased independent-triple estimator

For three independent inputs x1,x2,x3, let xbar=(x1+x2+x3)/3 and define

```
h(x1,x2,x3)=(3/2) sum_{r=1}^3 (xr-xbar)^{tensor 3}.       (T1)
```

This is invariant to adding the same vector to all three inputs. Thus the unknown offset and mean do not have to be estimated separately, and there is no plug-in centering bias left unaccounted for.

Write Sym for averaging a tensor over its index permutations. The equivalent expansion is

```
h=(1/3) sum_r xr^{tensor 3}
  -(1/2) sum_{r!=s} Sym(xr^{tensor 2} tensor xs)
  +2 Sym(x1 tensor x2 tensor x3).                         (T2)
```

To verify unbiasedness, translate all inputs by -mu and write Zr=Xr-mu. Independence and E Zr=0 kill every cross term in (T2), while the first term has expectation T. Therefore

```
E h(X1,X2,X3)=T.                                         (T3)
```

The factor 3/2 in (T1) is essential: the uncorrected third sample-centered moment of a sample of size three is biased. This kernel is the third-order central-moment k-statistic on a triple, expressed without assuming that terminology as a proof.

Use N=floor(n/3) disjoint triples to obtain independent Hilbert-valued tensors H1,...,HN with E Hr=T. At most two input rows are unused. Forming all overlapping triples would not give independent replication units; it would be a different U-statistic analysis.

## 3. Exact variance, including all tensor multiplicities

By orthogonal invariance of the Frobenius norm, the calculation may be done in the teacher basis without supplying that basis to the estimator. A centered coordinate is Z=S-p+sigma G. Define

```
nu=p(1-p), s2=sigma^2,
v=E Z^2=nu+s2,
kappa=E Z^3=nu(1-2p),
m4=E Z^4=nu(1-3nu)+6s2 nu+3s2^2,
m6=E Z^6=nu(1-5nu+5nu^2)
             +15s2 nu(1-3nu)+45s2^2 nu+15s2^3.
```

Direct expansion of (T2) gives the following component variances:

| Index pattern | Number of ordered entries | Variance of one entry |
|---|---:|---:|
| a=a=a | K | m6/3-v m4/2+7 kappa^2/6+11v^3/2 |
| a=a!=b and permutations | 3K(K-1) | v m4/2+3v^3/2 |
| a,b,c all distinct | K(K-1)(K-2) | 3v^3/2 |

Covariances between tensor entries are not needed to compute the mean squared Frobenius error, which is the sum of the entry variances. Consequently

```
V_H := E||H-T||_F^2
 = (K/3)m6
   +[K(3K-4)/2] v m4
   +(7K/6) kappa^2
   +K(3K^2/2+4) v^3.                                   (V1)
```

For auditability, the entry calculations can be checked as follows. For three distinct coordinates, write an entry as `sum_rst c_rst Z_ra Z_sb Z_tc`. The coefficients are 1/3 when all row indices are equal, -1/6 when exactly two are equal, and 1/3 when all are distinct. The sum of their squares is `3/9+18/36+6/9=3/2`, proving the last row of the table.

For a repeated pair a,a,b, condition on the a-coordinate observations. The coefficient of Z_1b is

```
(1/3)Z_1a^2-(1/6)Z_2a^2-(1/6)Z_3a^2
 -(1/3)Z_1a Z_2a-(1/3)Z_1a Z_3a+(2/3)Z_2a Z_3a.
```

Its mean square is `m4/6+v^2/2`. Multiplying by v and summing the three row choices gives the middle row. For a=a=a, expand

```
(1/3)sum_r Z_r^3-(1/2)sum_{r!=s}Z_r^2 Z_s+2Z_1Z_2Z_3
```

and subtract kappa^2. The resulting first row agrees with (V1). These are algebraic derivations; the formula has not been machine-tested or externally reviewed in this task.

A useful exact nonnegative sub-contribution is

```
V_H >= (3/2)K(K-1)(K-2)v^3.                             (V2)
```

Unlike a generic upper envelope, (V2) is a lower bound on the actual variance of this particular estimator. It comes from genuinely noisy mixed tensor coordinates, even though their population third moments are zero.

## 4. Valid calibration when p is unknown

A confidence procedure cannot plug the hidden p into (V1) and claim to be label-free. One may use a deterministic upper bound `Vbar>=V_H` over the declared structural p/noise range, or an independent confidence construction for those nuisance parameters with its own failure budget.

A simple fully explicit bound for K>=2 is obtained using

```
v <= vU=1/4+s2,
kappa^2 <= 1/108,
m4 <= m4U=1/12+(3/2)s2+3s2^2,
m6 <= m6U=1/12+(15/12)s2+(45/4)s2^2+15s2^3.
```

Here `E(S-p)^4=nu-3nu^2<=1/12`; because |S-p|<=1, its sixth moment is no larger than its fourth. The bound on kappa^2 follows by maximizing `nu^2(1-4nu)` over nu in [0,1/4]. All coefficients in (V1) are nonnegative for K>=2, so replacing v,m4,m6,kappa^2 by these upper bounds gives a valid Vbar. A supplied noise upper bound may replace sigma throughout.

This particular upper bound is intentionally conservative. The budget diagnosis below grants the oracle value Vbar=V_H, so it does not depend on the looseness of these nuisance bounds. Observed sample variance alone is not declared a valid variance upper bound.

## 5. Two rigorous Hilbert-space confidence constructions

### 5.1 Mean of the independent triples

Set `T_hat=N^{-1}sum_r Hr`. Independence in the finite-dimensional Hilbert space gives

```
E||T_hat-T||_F^2=V_H/N.
```

Markov's inequality applied to the squared norm yields

```
P(||T_hat-T||_F>epsilon_mean)<=alpha,
epsilon_mean=sqrt(Vbar/(N alpha)).                      (C1)
```

This is a valid finite-sample confidence ball without coordinatewise union bounds, bounded-input assumptions or an assumption that Gaussian cubes have exponential tails. It is a variance-based bound, not a normal approximation.

### 5.2 A fully specified Hilbert median-of-means alternative

Choose a fixed odd b with

```
b >= (32/9)log(1/alpha), l=floor(N/b)>=1.
```

Compute b independent block means M1,...,Mb, each from l triples, and set `r0=sqrt(8Vbar/l)`. Each block has probability at most 1/8 of lying farther than r0 from T. Hoeffding bounds the chance of at least half bad blocks by `exp(-9b/32)`.

For each block mean Mi, let Ri be the distance to its `(b+1)/2`-th nearest block mean, including itself. Select the Mi with smallest Ri, with a predetermined index tie rule, and call it T_hat_MOM. If more than half the block means are within r0 of T, any good center has Ri<=2r0. The selected center therefore has a radius-2r0 cluster containing a good center, so it is within 3r0 of T. Hence

```
P(||T_hat_MOM-T||_F>epsilon_MOM)<=alpha,
epsilon_MOM=3sqrt(8Vbar/l).                             (C2)
```

This finite medoid definition avoids relying on unspecified numerical accuracy of a geometric-median optimizer. It is only an analytic candidate; no implementation is requested here.

The constants matter: at alpha=1/20, (C1) has radius squared `20Vbar/N`, whereas (C2) has `72Vbar/l>=72Vbar/N`, even before paying b>1. Thus this elementary MOM proof is more conservative at the present fixed confidence level. Robustness at very small alpha does not make it automatically the tightest available formula at alpha=.05.

## 6. Conversion to dictionary geometry

On either tensor event, the existing global positive-permutation theorem applies. With unknown kappa, write

```
kappa_hat=||T_hat||_F/sqrt(K),
kappa_L=(||T_hat||_F-epsilon_T)/sqrt(K).
```

If kappa_L>0, the reported radius for a full-rank square decoder D with polar factor Q is

```
R_D=||D-Q||_F
    +[||T_hat-kappa_hat T_Q||_F+2epsilon_T]/kappa_L.      (G)
```

Its necessary floor is `2epsilon_T/kappa_L`. On the valid tensor event, kappa_L<=kappa, so that floor is at least `2epsilon_T/kappa`.

For an intentionally favorable budget calculation, grant the exact kappa instead. Then the sharper valid formula uses only one epsilon_T:

```
R_D_known=||D-Q||_F
         +[||T_hat-kappa T_Q||_F+epsilon_T]/kappa
         >=epsilon_T/kappa.                            (G-oracle)
```

If even this smaller floor is too large, estimating kappa via (G) cannot create a useful guarantee on the valid confidence event. No truth information from this oracle planning comparison is supplied to a claimed estimator.

The same tensor-norm event is uniform over Q; there is no extra bank penalty in this geometry argument. A separate encoder-risk bound still needs its own sampling/selection contract. Unknown physical offset also remains relevant to any subsequent coefficient-risk/bias bridge even though the tensor estimator is translation-invariant.

## 7. Analytic n<=8192 budget diagnosis

Fix the optimistic single-world confidence alpha=1/20, K=16, and p in [1/16,1/4]. In particular this covers the three fixed reference values 1/16,1/8,1/4. No p values, seeds or models are searched. The following obstruction does not require evaluating sigma numerically and remains true at sigma=0; adding Gaussian variance cannot reduce (V2).

For 3<=n<=8192, 1<=N=floor(n/3)<=2730. For n<3 this estimator is undefined and no certificate is issued. From (V2),

```
V_H >=5040 v^3 >=5040 nu^3,
kappa^2=nu^2(1-2p)^2,
nu/(1-2p)^2 >=15/196   for p in [1/16,1/4].
```

The last ratio is increasing on (0,1/2); it equals 15/196 at p=1/16. Consequently, even with the exact variance and exact kappa,

```
(epsilon_mean/kappa)^2
 >=5040*15/[196*(2730/20)]
 =1800/637 >1.                                         (B1)
```

Every report of the known-kappa confidence formula (G-oracle) therefore has radius greater than one for any observed array. The unknown-kappa version has an even larger floor on its valid confidence event. The MOM bound (C2) is larger than (C1) at this alpha and cannot repair the conclusion.

A useful absolute coefficient-NMSE diagnostic scale epsilon_code=.02 would, through the earlier bridge and under the most favorable C=0, B=0, kappa_D=1 assumptions, already require a dictionary-error certificate at most .1. This is only a scale diagnostic: .02 in the original C2b protocol remains a paired selector improvement. The radius lower bound in (B1) is much larger even than that favorable geometry allocation.

For the precise proposed formula, a necessary input count to report geometry radius at most rho in the known-kappa case is

```
N >= Vbar/(alpha kappa^2 rho^2),
n >=3 ceil(Vbar/(alpha kappa^2 rho^2)).                  (B2)
```

This is a necessary condition for this confidence expression to fit below rho, not a necessary sample size for dictionary recovery by all methods. For rho=1/10, (V2) and the same p interval imply the conservative necessary count

```
n >=3 ceil(5400000/7).
```

No decimal evaluation is needed to see that this far exceeds 8192. The exact V_H in (V1) would make the count larger. Nonzero polar discrepancy, observed tensor residual, simultaneous-world correction, uncertainty in nuisance parameters, clean-risk allowance or bias allowance cannot improve this formula's necessary floor.

This is a statement about the displayed variance-based confidence recipes. It is not a claim that every dataset produces a large actual tensor error, that a Frobenius confidence ball from every possible analysis must be this large, or that a structured tensor estimator cannot exploit the orthogonal model more efficiently.

## 8. What is resolved and the next discriminating question

Resolved provisionally:

- unknown-mean centering is handled exactly without a separate mean estimate;
- a fresh-input unbiased tensor estimator and an exact variance formula are specified;
- valid finite-sample Hilbert confidence constructions and all constants are explicit;
- their failure to meet the present budget is established analytically even under an optimistic oracle calibration.

Not resolved:

- an affordable confidence construction in a norm or structured family that directly controls dictionary geometry;
- whether a full-data U-statistic, a projected tensor estimate, directional/contracted moment bounds, or a normalized likelihood can meet the required precision;
- the companion clean-risk/bias confidence bound, the original hard-between identification law, and C2a/C2b outcomes.

Do not count overlapping triples as independent to increase N. A full-data U-statistic would require its own Hoeffding-decomposition or other dependence-aware variance/concentration argument. Likewise, projecting onto the orthogonal-tensor model changes the statistical analysis; its benefit must be proved rather than inferred from a smaller visual residual.

The smallest useful next theory question is whether a confidence bound targeted to the orthogonal tensor family can control the **dictionary geometry directions** without paying for all mixed third-order entries in the ambient Frobenius norm. The exact mixed-entry contribution (V2) identifies what the current recipe pays for. It does not prove that all of that noise is necessary for identifying D_*.

No new Colab audit, training sweep, retained-array recomputation or additional toy pair follows from this draft. Keep BR5 unresolved, W2 gated and C2a/C2b unentered. Classical orthogonal tensor/moment recovery remains a same-information comparator, and no VG-specific novelty or density-estimator success is claimed.

## Provenance and review status

Read the latest remote AGENTS.md, project memory, Git status and `THIRD_CUMULANT_GEOMETRY_CANDIDATE.md`; the population geometry theorem has earlier primary/Astra and focused Opus review, but the new estimator, variance formula and confidence argument in this file are provisional. No messages were sent to other chats in this task, and no existing research file or code was changed.

Applied the previously read project-local ARIS formula-derivation plus scientific-critical-thinking, physicist-analyst and physics-guide-solver: preserve the tensor target, account for centering and sampling units, distinguish exact variance from a confidence upper bound, and distinguish a formula-specific budget obstruction from an information-theoretic claim.

Scientific Agent Skills procedural attribution: Kassis et al. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065). Metadata was checked earlier in this same review session; the skill is procedural support, not evidence for these mathematical claims.

## Focused review completion

Opus 5.5 xhigh accepted the unbiased triple kernel, mixed-coordinate variance lower bound (V2), Chebyshev mean-confidence argument, and exact budget constants (B1)-(B2); see OPUS_TENSOR_BUDGET_THEOREM_REVIEW.md and receipt. Primary independently checked this algebra. Vbar must be a certified upper bound on V_H; V2 is used only to lower-bound the necessary radius, never to calibrate coverage. The full variance V1 and MOM construction remain outside that focused review and provisional; Gemini is unavailable. The Frobenius radius >1 is not universally vacuous: the comparison here is with the specified small geometry target rho=.1. No suggested simulation was run because the decisive bound is algebraic.
