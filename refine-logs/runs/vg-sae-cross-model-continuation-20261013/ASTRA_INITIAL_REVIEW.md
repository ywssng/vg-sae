# Astra independent derivation package — 2026-10-13

## Target and strongest actionable proposal

Build a label-free finite-sample upper confidence bound for the original model's **full clean stochastic risk**, retaining the divergence term, and supply a class-level dictionary argument under an explicit information contract. The substantive new analytic step is a local one-/two-bit geometry lemma below. It avoids the existing zero-pattern condition, which is structurally inapplicable to part of the actual C2 roster.

Do not run another pairwise native witness or a gamma/seed search. The smallest next decision is an independent algebra audit of this lemma and the resulting symbolic nonvacuity inequalities. A checkpoint or fresh-input audit is justified only after those inequalities admit a meaningful margin and a credible sample size. No current checkpoint is certified here.

## Status

**COHERENT AFTER EXTRA ASSUMPTIONS / FINITE-SAMPLE CERTIFICATE DERIVED; PRACTICAL NONVACUITY UNESTABLISHED.**

This is an independent Astra research review requested as extra high. It is not an Opus/Gemini review or a three-model consensus. It provides elementary sufficient inequalities, without a mathematical novelty claim. No numerical work, training, tests, existing-array calculation, model modification or confirmation-world access was performed.

The target remains the original native estimator. The confidence calculation is an analysis/audit, not an added training penalty. If its X-based moment interval were used as a replacement density selector, that would be a separately versioned method and would require the same-information classical controls.

## Invariant object, notation and assumptions

Let `S_j iid Bernoulli(p)`, `0<p<1/2`, `K=d>=2`, `Y=D_* S`, where `D_*` is orthogonal and the teacher origin is known to be zero. Let `X=Y+sigma Z`, `Z~N(0,I_d)` independent of S. Sigma is supplied or justified independently; setting a learned reciprocal residual equal to sigma is insufficient.

For a frozen native model, unit decoder columns form a square invertible D. Define

```
s = ||D||_op,  kappa = sigma_min(D)>0,  B = ||b||,
h(X)=m(X)*a(X),  f(X)=b+D h(X),
Zcode_i=B_i a_i(X), B_i|X ~ Bernoulli(m_i(X)) independently,
v_i(X)=m_i(X)(1-m_i(X))a_i(X)^2,
C=E ||Y-b-D Zcode||^2,
J=E sum_i KL(Bernoulli(m_i(X)) || Bernoulli(r)),
r=sigmoid(-gamma),  M=Kp.
```

Conditional gate draws are independent of latent truth given X. Expectations are over fresh inputs and gates. C includes the stochastic variance, not only the mean reconstruction error. The native mathematical sigmoid/softplus encoders have finite affine weights and all the required Gaussian integrability.

A finite bank of H models is fixed before the audit inputs are sampled. The bank may depend arbitrarily on training/development data, but no audit sample is reused to change the bank or construct unaccounted models. A simultaneous confidence statement permits selecting among that bank after seeing the audit. It does not legalize adaptive reuse of the old select arrays.

The theorem is square-family only. In the actual plan, family O has d=K=16; family R has d=8,K=16 and has no positive global kappa. The theorem supplies no family-R claim. Fixed matched beta=1/sigma^2 is the analytic regime; the original operational learned-beta bank does not become matched simply because its initialization is 400.

## Derivation strategy and map

1. Express C as the expectation of an observable Stein-corrected scalar, retaining divergence and stochastic-code variance.
2. Obtain a finite-bank confidence bound with an explicit moment envelope. Separately bound J and, if p is not supplied, its structural moment interval.
3. Use small C to make nonnegative mean codes close to an affine transform of S outside a small exceptional set.
4. Flip one or two independent source bits while fixing the others. Negative coefficients and two significant coefficients force extra activation, with probability penalties depending only on p and p^2.
5. Low J plus small stochastic variance caps total activation; rank then forces the remaining large coefficients into a positive permutation.
6. Convert C, dictionary geometry and the observed bias B into hard coefficient NMSE. State the conditions under which the full chain is nonvacuous.

No approximation is used in Steps 1–6. The later discussion of realistic sample-size feasibility is a limitation, not an asserted certificate.

## 1. Exact observable risk and objective identity

Write the per-input normalized native objective, omitting only the common Gaussian normalizer, as

```
ell(x) = beta/2 [||x-f(x)||^2 + sum_i v_i(x)] + j(x),
j(x)   = sum_i KL(Bernoulli(m_i(x)) || Bernoulli(r)).
```

Stein integration by parts conditional on Y gives

```
psi(x) = ||x-f(x)||^2 + sum_i v_i(x)
         -d sigma^2 + 2 sigma^2 div f(x),
E psi(X) = C.
```

At matched `beta sigma^2=1`, this is exactly

```
psi(x)=2 sigma^2 [ell(x)-j(x)+div f(x)-d/2],
G=E ell= C/(2 sigma^2)+J-E div f+d/2.
```

Thus the requested risk certificate is a confidence bound for psi, not for ell alone, and not merely an empirical divergence mean. No derivative of `sum v_i` is added: that same term already appears in the two risks. Correctness of this population identity does not itself provide any finite-sample upper bound.

## 2. Finite-bank finite-sample theorem

### 2.1 An explicit, conservative native moment envelope

Absorb the model's input centering into its affine biases. Let `a(x)=softplus(W_a x+c_a)` and `m(x)=sigmoid(W_m x+c_m)`. Define

```
A0 = ||softplus(c_a)||_2,
L  = ||W_a||_op,
Gm = ||W_m||_F,
La = sum_i ||row_i(W_a)||_2,
F0 = B+s A0,                  F1=s L,
D0 = Gm A0/4+La,              D1=Gm L/4.
```

Softplus is 1-Lipschitz, sigmoid derivative is at most 1/4, and all decoder columns are unit. Hence for `u=||x||`,

```
||a(x)|| <= A0+L u,
||f(x)|| <= F0+F1 u,
sum v_i <= (A0+L u)^2/4,
|div f(x)| <= D0+D1 u.
```

The last line follows by expanding the trace as a sum over decoder columns and using Cauchy–Schwarz on the gate derivative term. Consequently `|psi(x)|<=c0+c1 u+c2 u^2`, where

```
c0=F0^2+A0^2/4+d sigma^2+2 sigma^2 D0,
c1=2F0(1+F1)+A0 L/2+2 sigma^2 D1,
c2=(1+F1)^2+L^2/4.
```

Because `||Y||^2<=K`, Gaussian conditioning gives the exact upper bounds

```
Q2=K+d sigma^2 >= E||X||^2,
Q4=K^2+2 sigma^2(d+2)K+sigma^4 d(d+2) >= E||X||^4.
Vpsi=3(c0^2+c1^2 Q2+c2^2 Q4) >= Var(psi(X)).
```

These bounds are deliberately explicit and can be very loose. A smaller Vpsi is allowed only if it is independently proved under the same information contract; an empirical sample variance alone is not automatically a valid upper bound. Supplying a p interval permits sharper input moments, but does not remove the need to justify the variance envelope.

### 2.2 Confidence construction and proof

Fix error probability alpha. Choose an odd block count

```
b >= (32/9) log(3H/alpha),   l=floor(n/b)>=1.
```

Discard at most `b-1` samples and form b disjoint block averages of psi for each model. Let `med_psi` denote their median. Put

```
e_C=sqrt(8 Vpsi/l),
C_U=max(0,med_psi+e_C).
```

A block mean misses C by more than e_C with probability at most 1/8 by Chebyshev. Independence of blocks and Hoeffding's inequality bound the probability that at least half are bad by `exp(-9b/32)`. Thus the displayed C_U is a genuine upper confidence bound, including for unbounded Gaussian inputs. It is not a clipped-data surrogate.

For the same model define `jmax=max(log(1/r),log(1/(1-r)))`. Pointwise `0<=j(X)<=K jmax`, so

```
J_U=mean(j)+K jmax sqrt(log(3H/alpha)/(2n))
```

is a one-sided Hoeffding upper bound. Cross-model dependence and dependence between J_U and C_U do not invalidate the union bound.

If p is not supplied, the known-origin orthogonal binary contract gives the data-only identity

```
E[||X||^2-d sigma^2]=Kp.
```

Let `V0=2Q4+2d^2 sigma^4`, an upper bound on the variance of this scalar, and use the same block construction to obtain a median `med_0`. Then

```
I_p = [(med_0-sqrt(8V0/l))/K,
       (med_0+sqrt(8V0/l))/K] intersect [0,1/2]
```

contains p except on an event of probability at most alpha/(3H). Empty intervals or failure to remain inside the intended open range mean that a usable certificate is unavailable. By a union bound, all H risks, all H KL terms and this single p interval are valid together with probability at least `1-alpha`. No independence across features or across models is used in this confidence argument; the independent sampling units are input rows.

The p identity supplies precisely the information also available to a classical moment baseline. It does not validate the original hard-between point estimator and must not be advertised as a new VG density success.

## 3. New local-flip geometry theorem

### 3.1 Why the previous zero-pattern route cannot cover the C2 roster

The imported average-risk theorem assumes `g0=(1-p)^K-p>0`. For K=16, `(7/8)^16<1/8`, so it fails at confirmation p=.125, and it also fails at p=.25 and development p=.1875. This is a structural failure at C=0, not an issue resolved by more samples or a better SURE interval. It does not establish failure of VG-SAE; it establishes inapplicability of that sufficient theorem.

### 3.2 Statement

Fix `e>0, eta>0, V>0`, and set

```
t=eta+2e,
u=eta^2/(eta^2+V),
k=KL(Bernoulli(u)||Bernoulli(r)),
g(p)=min(1-2p,p(1-p)),
a_K(p)=(K-1)/p+1/p^2.
```

Require `u>r`, so k>0, and

```
s sqrt(K(K-1)) t < 1,                                      (G1)
J/k + C/V + a_K(p) C/(kappa^2 e^2) < Kp+g(p).              (G2)
```

Then some positive permutation P obeys

```
||D P-D_*||_op <= ||D P-D_*||_F
               <= epsilon_D := 2 sqrt(K)(K-1)t.            (G3)
```

This conclusion concerns all native models satisfying (G1)–(G2), not a chosen pair or a conveniently recovering checkpoint. The prior r need not equal p. It is fixed and normalized exactly as in J. Gamma<=0 is not excluded algebraically, but then making u>r and (G2) useful may be much harder.

### 3.3 Proof

Let `A=D^{-1}D_*`, `c=-D^{-1}b`, and

```
Ggood={||h(X)-AS-c||_infinity<=e},
delta=P(Ggood^c)<=C/(kappa^2 e^2).
```

Indeed, the mean reconstruction part of C bounds `E||D(h-AS-c)||^2`, and Markov gives the displayed probability bound. Every coordinate of h is nonnegative. Let `L_eta=E sum_i 1{h_i>eta}`.

**Local-flip lemma.** In a row of A, choose one or two coefficients with absolute value greater than t. Orient their bits so that switching an oriented bit from 0 to 1 increases `AS+c`. A positive coefficient has exciting probability p; a negative coefficient has exciting probability 1-p. Let z be the probability that the chosen bits are all unexcited, conditional on the other source bits. Independence makes z constant across the other-bit configurations.

For fixed other bits, call the unexcited row value F0. If F0<-e, every noisy input with those chosen bits unexcited is bad because h_i>=0. The probability, over other bits, of such a bad baseline is therefore at most delta/z. Outside those baselines, whenever any chosen bit is exciting the row value exceeds `-e+t=eta+e`. On Ggood this implies h_i>eta. Consequently

```
P(h_i>eta) >= (1-z)(1-delta/z)-delta
            =1-z-delta/z.                                (L)
```

This inequality remains true when its right-hand side is negative. It requires neither shared noise coupling between the flipped inputs nor a uniform noise radius.

Because `sigma_min(A)=1/s`, every row has Euclidean norm at least 1/s and thus has an entry of magnitude at least `1/(s sqrt(K))`. Condition (G1) makes this greater than t. Apply (L) to one such coefficient: every row has activation probability at least `p-delta/p`.

A row with a negative coefficient less than -t has activation probability at least `1-p-delta/p`, gaining `1-2p` over p. A row with two coefficients of absolute value greater than t has exciting-union probability at least `2p-p^2`, while z>=p^2, so its activation probability is at least `p+p(1-p)-delta/p^2`. Therefore, if either defect exists in any row,

```
L_eta >= Kp+g(p)-[(K-1)/p+1/p^2] delta.                    (L2)
```

Independently, when `h_i>eta` and `v_i<=V`, the identity `v_i=h_i^2(1-m_i)/m_i` implies `m_i>u>r`, and hence its gate KL is at least k. Since unit decoder columns give `E sum_i v_i<=C`,

```
L_eta <= J/k+C/V.                                         (L3)
```

Equations (G2), (L2) and (L3) rule out both defects. Thus every row of A has exactly one entry greater than t, every such entry is positive, and all other entries have absolute value at most t.

Let N retain only those large entries. Then

```
||A-N||_op <= ||A-N||_F <= sqrt(K(K-1))t < 1/s=sigma_min(A).
```

N must be invertible. Since it has exactly one nonzero in every row, those entries occupy distinct columns, hence a permutation. After reordering, write its positive matched diagonal entries as alpha_j. The identity `DA=D_*`, unit column norms and the bound on off-diagonal entries imply

```
|alpha_j-1| <= (K-1)t,
||d_j-d_*j|| <= 2(K-1)t.
```

Summing squared column bounds proves (G3). QED.

**Bias contract.** This theorem does not deduce a small bias from reconstruction. B=||b|| is directly observable under the specified known-zero-origin teacher contract and is retained explicitly in the recovery inequality. Unknown teacher offset would require another premise; centering the data without accounting for the latent mean does not supply it.

## 4. Hard coefficient NMSE and finite-sample certificate

For `H_i=1{m_i>1/2}` and binary truth T_i, pointwise

```
(H_i a_i-T_i)^2 <= 2 E_B[(B_i a_i-T_i)^2|X,T_i].
```

For H_i=1 use m_i>1/2; for H_i=0 use 1-m_i>=1/2. Combining with the L2 triangle inequality and the singular-value bound gives

```
NMSE_hard <= 2/(Kp kappa^2)
             [sqrt(C)+B+epsilon_D sqrt(Kp)]^2.            (R)
```

On the simultaneous confidence event, replace C by C_U, J by J_U and require (G1)–(G2) for every p in I_p. The right side of (R), maximized over the same interval, is then a population upper confidence bound, valid simultaneously for the entire frozen bank and any subsequent bank selection. This remains conditional on the structural model and matched noise contract; it does not certify those assumptions from the same data.

This is not a paired C2b advantage certificate. The plan's .02 is the minimum improvement over reconstruction-only, with a decoder guard and world-level inference. Even a valid absolute NMSE bound below .02 supplies no lower bound on the comparator. Nor does this theorem establish hard-between branch membership, homogeneous repeat-selection propensities, generating-p accuracy of the template, or confirmation-world performance.

## 5. Explicit nonvacuity conditions and what they imply

For any intended absolute risk tolerance epsilon, define

```
Delta_rec=kappa sqrt(Kp epsilon/2)-B-epsilon_D sqrt(Kp),
Delta_geom=Kp+g(p)-J/k,
C_star=min(Delta_rec^2,
           Delta_geom/[1/V+a_K(p)/(kappa^2 e^2)]).
```

A meaningful strict certificate needs `Delta_rec>0`, `Delta_geom>0`, (G1), and `C<C_star`; the finite-sample replacements must leave positive margin uniformly over I_p. The signs must be checked before squaring Delta_rec. Epsilon=.02 is a useful scale diagnostic here, not a redefinition of C2b.

The finite-sample theorem is nonvacuous in the mathematical sense: for any fixed model with a positive population margin and finite Vpsi, its confidence radii vanish as n grows. If `Delta=C_star-C>0` and the KL/p margins have separately been reserved, the conservative sufficient condition

```
l > 32 Vpsi/Delta^2
```

makes the risk upper bound pass on the stated confidence event, since then `C_U<=C+2e_C<C_star`. The KL radius and p interval must also fit their own margins; they are not free.

This is **not** a claim that n=4096 or 8192 is sufficient. The envelope Vpsi can be extremely loose for high-slope gates. The geometry certificate additionally requires a clean-risk scale much smaller than a naive C2-scale coefficient tolerance when e is small. No current-case values of Vpsi, C_star, B, kappa or s have been computed, and this review does not assert practical nonvacuity.

The local-flip result nevertheless removes a real zero-risk exclusion: its margin is positive at all three original confirmation p values .0625/.125/.25, whereas the old zero-pattern theorem excluded the last two. It improves the structural route before any variance-estimation or computation question. It is a sufficient-condition improvement, not a theorem that these models achieve those conditions.

## 6. Smallest decisive follow-up

1. Obtain independent line-by-line review of the local-flip proof, particularly the conditional baseline probability delta/z, whole-row activation sum, rank step and p interval quantifiers. This is the central new result.
2. Freeze one proposed absolute-error scale and a **bank-wide** parameter/norm contract from the actual square-family source freeze. State which parameters are deterministic model quantities and which require fresh-input confidence bounds. Do not pick e, eta, V or a favorable model based on teacher recovery labels.
3. Before sampling, establish a sharper variance envelope or a bound on the relevant confidence radius, and compare its required n with the available audit budget using the displayed nonvacuity inequalities. The generic envelope is a valid fallback but may already rule out an affordable audit. Failure at this stage is an honest result; do not replace it with an empirical SURE point estimate.
4. Only if an affordable margin survives, freeze a small Colab CPU audit of the existing finite bank with fresh non-confirmation input rows and the simultaneous bound. No new training, seed search, gamma search, toy pair or confirmation-world access is needed for that audit.

If the required sample size is prohibitive, the next analytic task is a sharper class-level geometry/recovery bridge or a local-variance confidence theorem, not more of the already completed amplitude mechanism. An X-assisted likelihood/moment method would remain a separately declared route with matched-information controls.

## Boundaries, sources and skills

Read local management AGENTS.md/MEMORY.md, current remote AGENTS.md and project memory, the 2026-10-13 matched-amplitude protocol/report/next plan, the 2026-10-03 next plan, original average-risk bridge, density-identification theorem/options, the current and imported C2 plan, and native model source. No applicable subordinate AGENTS.md was found in the inspected run/import paths. Existing remote modifications were preserved. This report is the only file written; no commit or compute was performed by this reviewer.

The original identification theorem's no-FP inequality and mask-only nonidentification remain in force and are not repackaged as this result. The result here is a new sufficient geometry argument and explicit confidence construction. BR5 remains measurement_unresolved, W2 gated and C2a/C2b unentered.

Applied project-local ARIS formula-derivation, scientific-critical-thinking, physicist-analyst and physics-guide-solver: preserve one risk object, separate exact identities from assumptions, track the matched dimensionless parameter beta*sigma^2, distinguish population risk from selector contrast, and test theorem applicability at the actual K/p roster.

Scientific Agent Skills attribution: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv / DOI](https://doi.org/10.48550/arXiv.2609.00065). Current arXiv metadata was checked on 2026-10-13 (record revised 2026-09-02, v2); no journal reference was listed. The cited skill contributes procedure, not independent evidence for the mathematical claims.

## Addendum: nonempty native class at the actual square-family noise scale

The sufficient class is not only an asymptotic C->0 thought experiment. The following **uniform feasibility witness**, covering the whole interval `p in [1/16,1/4]`, establishes nonemptiness at K=d=16 and sigma=1/20 without a numerical run. It is not a new pairwise counterexample, trained-model result, or method proposal.

Use D=D_*, b=0, constant native amplitude a=1, and the finite affine Bayes sigmoid gates

```
m_i(X)=sigmoid(400 (d_*i^T X-1/2)-log((1-p)/p)).
```

Take objective gamma=12, already in the original native control grid, so `r=sigmoid(-12)`. The gate encoder is free to differ from the objective prior; no equality r=p is assumed in the geometry theorem. This witness supplies p and D only to prove the comparison class is nonempty, not to an estimator.

For each coordinate, conditional independence of the gate draw and S gives `C=2K E[m(1-m)]`. If a and b denote the two weighted Gaussian densities, `ab/(a+b)<=sqrt(ab)/2`. Their Gaussian affinity is `exp(-1/(8 sigma^2))=exp(-50)`. Therefore

```
C <= K sqrt(p(1-p)) exp(-50) <= 8 exp(-50)
  < 2^(-47) < 10^(-14).
```

The last coarse rational comparison uses e>2 and `2^47>10^14`. It intentionally avoids numerical integration or an evaluated certificate. Since E m=p,

```
J=K[p log(1/r)+(1-p)log(1/(1-r))]-K E H_2(m)
 <= M w+(K-M)w0,
w=log(1+exp(12))<12.001,
w0=log(1+exp(-12))<1/4096<.00025.
```

Choose the analytic constants

```
e=eta=1/10000, V=10^(-12), t=3/10000,
u=10000/10001, nu=1-u=1/10001.
```

The elementary entropy bound `H_2(nu)<=nu(log(1/nu)+1)<.0011` and `12nu<.0012` give

```
k=w-12nu-H_2(nu)>11.99,
J/k < M+.005                 (M in [1,4]),
C/V < .01,
a_K(p) C/e^2 < 496*10^(-6)=.000496,
g(p)=p(1-p)>=15/256>.05.
```

Here `log(10001)<10` follows, for example, from `e>8/3`. Thus (G2) holds uniformly over the entire interval. Condition (G1) holds because `sqrt(240)t<16t<1`. The derived dictionary error bound is `.036`; with B=0 and kappa=1, (R) is below `.003`, since `sqrt(C/M)+.036<.037` and `2(.037)^2<.003`.

This proves a finite-parameter native, fixed matched-precision comparison class with a nontrivial absolute C2-scale bound is nonempty. It does **not** show that an optimizer reaches it, that its objective dominates other classes, that a finite bank contains it, or that the proposed confidence envelope can certify it at an affordable n. The latter sample-complexity limitation remains the decisive next obstruction. In particular, the required small-C certificate can demand far more than the plan's 4096/8192 audit rows even though the actual hard coefficient error of this analytical witness is tiny.
