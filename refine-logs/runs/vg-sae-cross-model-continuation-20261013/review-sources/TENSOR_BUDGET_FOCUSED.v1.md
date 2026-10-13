# Focused tensor budget algebra review

Audit the unbiased triple estimator, the mixed-coordinate variance lower bound V2, and the Chebyshev radius budget obstruction. The full exact variance V1 and MOM construction are outside this focused review and remain provisional. No numerical execution has occurred.

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


For distinct a,b,c, h_abc=sum_rst c_rst Z_ra Z_sb Z_tc. Coefficients are 1/3 for all row indices equal, -1/6 for exactly two equal, and 1/3 for all distinct. Independent zero-mean coordinates give Var(h_abc)=v^3 sum c_rst^2=(3/2)v^3, v=p(1-p)+sigma^2. Counting ordered entries implies V_H=E||h-T||_F^2 >= (3/2)K(K-1)(K-2)v^3.

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

For n<=8192, N=floor(n/3)<=2730. From (V2),

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

