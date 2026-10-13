# Focused local-flip review source

Exact extracts from ASTRA_INITIAL_REVIEW.md; no new experiment. Review the population geometry lemma and hard coefficient implication only. Finite-sample confidence and resource budget are outside this packet.

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

