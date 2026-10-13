# Astra generic-certificate feasibility audit — 2026-10-13

## Decision and status

**The smallest decisive audit is already an exact analytic proof. Do not allocate a Colab runtime for it.**

For the fixed native gamma12 witness, the generic moment-envelope/MOM certificate in `ASTRA_INITIAL_REVIEW.md` cannot satisfy its geometry premise for **any input sample** when n<=8192. This statement needs no confidence event, simulation, actual SURE values, fitting, seed or numerical evaluation. It holds even if p and J are supplied exactly and even under the optimistic choice of one MOM block.

This is a limitation of this explicit certificate recipe and its global variance envelope. It is not a lower bound on the actual SURE variance, a failure of the local-flip population theorem, a failure of VG-SAE, or impossibility for other label-free certificates.

The parent supplied the shortest proof; I checked its inequalities and incorporated it below. The exact norm specialization and optional rational receipt design are additional independent derivations. Only this document was written. No numerical execution, Colab session, test, training or data access was performed.

## Frozen information contract

The complete roster, if a machine-readable receipt is later requested, is

```
p in {1/16,1/8,1/4}; K=d=16; sigma=1/20; beta=400;
gamma=12; r=sigmoid(-12);
e=eta=1/10000; V=1/10^12; n=8192;
absolute NMSE diagnostic tolerance epsilon=1/50.
```

There is no search or replacement case. The principal impossibility proof actually holds for every p in [1/16,1/4], so these three rows are only a fixed reporting roster.

Let D=D_* be any orthogonal 16-by-16 matrix, b=0, and use the exact finite native witness

```
W_a=0, c_a=log(exp(1)-1) * 1_16, a(X)=1_16,
W_m=400 D_*^T,
c_m=[-200-log((1-p)/p)] * 1_16,
m_i(X)=sigmoid(400(d_*i^T X-1/2)-log((1-p)/p)).
```

The encoder bias depends on the analytical witness p; its objective prior remains gamma12. Supplying p and D proves nonemptiness of the comparison class, not an estimator result. The certificate obstruction below gives it every advantage by also treating p and J as known: estimating them cannot repair the demonstrated risk-bound failure.

## 1. Shortest pathwise obstruction

For this constant-amplitude orthogonal witness,

```
div f(x)=400 sum_i m_i(x)(1-m_i(x)) >= 0.
```

The observed squared residual and full stochastic-code variance are also nonnegative. Hence the actual SURE summand obeys, for every real x,

```
psi(x)>=-d sigma^2=-1/25.
```

Every block average and its median obey the same lower bound. In the generic envelope, `c2=1` and `Q4>=K^2=256`, so the **chosen upper-bound constant** satisfies

```
Vpsi=3(c0^2+c1^2 Q2+c2^2 Q4)>=768.
```

This is not a lower bound on Var(psi). For any legal number of blocks b>=1 and l=floor(n/b), n<=8192 gives l<=8192, and therefore

```
e_C^2=8 Vpsi/l >= 8*768/8192=3/4 >16/25,
e_C>4/5,
C_U=max(0,med_psi+e_C)>-1/25+4/5=19/25.             (A)
```

On the other hand, passing G2 necessarily requires, since J_U and all other terms are nonnegative,

```
C_U/V < Kp+g(p) <=4+1/4<5,
C_U <5/10^12.                                       (B)
```

Equations (A) and (B) are incompatible. Thus this certificate fails for every possible finite array at n<=8192, not merely with high probability. An unusually negative empirical SURE average cannot defeat this argument because the witness has the deterministic lower support -1/25.

No arithmetic beyond the displayed rational comparisons is needed. This settles the intended feasibility question before a Colab job is justified.

## 2. Exact specialization of the generic envelope

This stronger specialization is useful for documenting where the conservatism enters; it is unnecessary for the main decision. Orthogonality and constant amplitudes give

```
s=kappa=1, B=0,
A0=4, L=0, Gm=1600, La=0,
F0=4, F1=0, D0=1600, D1=0,
c0=701/25, c1=8, c2=1,
Q2=401/25,
Q4=1287209/5000,
Vpsi=31053651/5000.
```

For example, `Q4=256+36/25+9/5000`. The derivation uses only exact squared norms and rational constants; no orthogonal matrix must be generated, no encoder must be instantiated, and no logarithm, exponential or square root must be evaluated numerically.

If one retains the previous confidence allocation with H=3 reporting cases and alpha=1/20, a valid fixed odd block count is b=19, giving l=431. Its confidence admissibility can be checked without logs by

```
sum_{j=0}^{12} (171/32)^j / j! > 180,
```

which implies `exp(9b/32)>3H/alpha` and therefore the original block-count condition. This is a proposed exact-rational receipt check, not an executed check. The analytic obstruction in Section 1 does not depend on b=19 or on verifying that auxiliary inequality.

At b=19, `Vpsi>6000` and l=431 imply `e_C^2>48000/431>100`. Thus the specialized certificate has `C_U>249/25` on every sample. The weaker universal result (A) is already sufficient.

## 3. Conservative C_star bounds

Retain the population quantities and gamma12 witness of the preceding report. Its analytical bounds give

```
0<=C<Cbar=1/10^14,
J/k < M+1/200,
M=16p,
epsilon_D=9/250,
Delta_rec^2=64 M/15625,
D_p=1/V+a_K(p)/e^2=10^12+10^8 a_K(p),
C_star=min(Delta_rec^2,(M+g(p)-J/k)/D_p).
```

The sign conditions `Delta_rec>0` and `Delta_geom>0` are satisfied by the existing witness proof. Since J>=0,

```
L_p:=(g(p)-1/200)/D_p < C_star <= U_p:=(M+g(p))/D_p.  (C)
```

Here U_p<5/10^12, much smaller than the reconstruction branch `Delta_rec^2>=64/15625`, so the geometry branch is the active one. The exact fixed rows are:

| p | M | g(p) | a_K(p) | D_p | L_p numerator / D_p | U_p numerator / D_p |
|---|---:|---:|---:|---:|---:|---:|
| 1/16 | 1 | 15/256 | 496 | 1049600000000 | (343/6400)/D_p | (271/256)/D_p |
| 1/8 | 2 | 7/64 | 184 | 1018400000000 | (167/1600)/D_p | (135/64)/D_p |
| 1/4 | 4 | 3/16 | 76 | 1007600000000 | (73/400)/D_p | (67/16)/D_p |

All three L_p exceed 5/10^14. Thus the conservative population margin obeys

```
C_star-C > L_p-Cbar > 4/10^14.
```

The population certificate is therefore nonempty and has a positive margin. Its observable confidence upper bound is what fails at n=8192. The table is an analytic rational specification, not a report of an executed numerical audit.

## 4. Necessary versus sufficient sample counts

These must be kept separate.

### 4.1 A genuine necessary count for this particular confidence recipe to pass on any data

Because `C_U>=e_C-1/25`, passing `C_U<C_star<=U_p` requires

```
l > 8 Vpsi/(1/25+U_p)^2,
n >= b [floor(8 Vpsi/(1/25+U_p)^2)+1].                (N-path)
```

This is a pathwise lower bound on n for **this envelope and interval formula**. It is not a lower bound for any possible estimator or confidence interval. It is also not a guarantee that the displayed n will work.

Using only `Vpsi>6000` and `U_p<1/1000`, the right-hand ratio exceeds

```
48000/(41/1000)^2 > 28000000.
```

Thus more than 28 million rows per block are necessary for this specialized envelope to have even a chance of passing. At b=19 this gives the conservative necessary bound `n>532000000`. Even b=1 would require more than 28 million. The shorter Section 1 proof needs none of these large-count calculations.

### 4.2 A sufficient count for the previous worst-case confidence-event margin proof

With known p and the analytical J bound, use the deterministic conservative threshold L_p. Let

```
Delta_p=L_p-Cbar>0,
l_suf(p)=floor(32 Vpsi/Delta_p^2)+1,
n_suf(p)=b l_suf(p).                                  (N-suf)
```

On the MOM concentration event this suffices for `C_U<=C+2e_C<L_p<=C_star`. This is a sufficient planning budget for that proof, not a minimum required number of observations.

A uniform, simpler sufficient schedule uses Delta0=4/10^14 in place of Delta_p:

```
n_suf_uniform = 19 * [floor(32*(31053651/5000)/(4/10^14)^2)+1].
```

It is enormously larger than the pathwise lower bound. Both numbers diagnose the global envelope's conservatism; neither estimates the actual variance. For unknown p/J, their confidence margins must additionally be allocated. The simplified schedules above explicitly give the audit the favorable known-p/analytical-J information and do not silently certify the full unknown-nuisance procedure.

## 5. Logic corrections and cautions for the previous report

No new algebraic error was found in the stated local-flip proof or its gamma12 feasibility witness in this follow-up. Three interpretation issues need to be made explicit:

1. The earlier `l>32 Vpsi/Delta^2` was stated correctly as sufficient. Its failure must never be inverted into a lower bound on the sample size needed by every method. Section 1 supplies a separate deterministic impossibility argument for the actual proposed recipe; (N-path) is the corresponding genuine necessary bound.
2. A large valid **upper bound** Vpsi is not evidence that the true variance is large. The witness may admit a much tighter specialized variance bound or a different confidence construction. The present result closes only the proposed global-envelope/MOM route at the available sample size.
3. Finite-sample mathematical nonvacuity as n tends to infinity does not imply useful n=4096/8192 guarantees. Conversely, failure of this confidence recipe does not undo the population nonemptiness theorem. The .02 scale remains an absolute-risk diagnostic here, not the original paired C2b selector contrast.

The lower support of psi is witness-specific. For an arbitrary native model, divergence can have either sign, so `psi>=-d sigma^2` must not be generalized without a new argument.

## 6. Optional exact-rational Colab receipt, only if later requested

A computation is unnecessary for the current decision. If a subsequent explicit request requires a reproducible machine receipt, the entire proposed task is one deterministic Python-standard-library script using `fractions.Fraction`, integer arithmetic, and `math.factorial` only. No Torch, NumPy, Arb, sampling, data download or model construction is needed.

Frozen outputs would be:

- source/plan hashes and the three exact roster rows;
- exact envelope coefficients, Q2, Q4 and Vpsi fractions;
- the rational comparisons proving (A), (B), (C), positive margins, and the optional b=19 condition;
- exact rational quantities in (N-path), (N-suf), with their integer floors and sample counts;
- explicit status `generic_envelope_mom_pathwise_infeasible_at_n8192` for every case;
- `actual_variance_not_computed`, `population_recovery_not_refuted`, `not_a_C2b_test`, `no_samples`, and `no_training`.

If any frozen rational assertion failed, keep that failure and stop; no replacement p, block count, gamma or model is permitted. A proposed process budget is 10 CPU / 30 wall seconds, one thread, CPU-only, through the existing deepstat-managed Colab CLI with runtime storage and immediate fetch. It would be source arithmetic only; Mac and deepstat would remain file/transport managers. This document does not start or request that job because the elementary proof already settles the question.

The next substantive research target is a sharper justified confidence construction or a more tolerant population recovery bridge. Extra inputs under the same global envelope at n=8192, repeated seeds and further pairwise witnesses cannot change the pathwise conclusion above.

## Independent review of the parent's obstruction note

Read `FINITE_SAMPLE_BUDGET_OBSTRUCTION.md` after completing this design. **Approved as stated:** its witness-specific divergence sign, pathwise support bound, generic-envelope lower bound, n<=8192 radius comparison and necessary G2 upper bound are correct. The sharper constant `67/(16*10^12)` is valid because `g(p)=p(1-p)<=3/16` throughout the declared interval. No blocking correction is required. Its no-Colab/no-fresh-audit decision is justified; the scope remains this particular confidence recipe. This approval is Astra-only and is not an Opus/Gemini endorsement.
