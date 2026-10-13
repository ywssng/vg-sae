# Deterministic budget obstruction for the proposed generic confidence envelope

2026-10-13. Analytic result; no numerical execution, audit samples, training, or Colab allocation.

## Fixed scope

Consider only the gamma=12 feasibility witness in ASTRA_INITIAL_REVIEW.md: K=d=16, sigma=1/20, D=D_*, b=0, a=1, coordinatewise increasing Bayes gates, p in [1/16,1/4]. Use exactly the generic native-envelope median-of-means bound in that report and its fixed geometric constants e=eta=1/10000 and V=10^(-12). This note does not rule out tighter confidence constructions, alternate geometric constants, other recovery proofs, or learning recovery.

## Explicit definitions used from the parent theorem

For this unit-column witness, v_i=m_i(1-m_i), f=D_*m, and

    psi(x)=||x-f(x)||^2+sum_i v_i(x)-d sigma^2+2 sigma^2 div f(x),
    Q4=K^2+2 sigma^2(d+2)K+sigma^4 d(d+2),
    g(p)=min(1-2p,p(1-p)),
    a_K(p)=(K-1)/p+1/p^2,
    G2: J_U/k+C_U/V+a_K(p) C_U/(kappa^2 e^2)<Kp+g(p).

Here Q4 is an analytical upper-envelope constant, not a gate moment or an actual lower bound on E||X||^4. All extra terms in Q4 are positive, hence Q4>=K^2. The KL summand is nonnegative and its one-sided Hoeffding increment is nonnegative, so J_U>=0. The witness has kappa=1, V>0, e>0, k>0 and a_K(p)>0. Finally, for p<=1/4, 1-2p>=1/2 while p(1-p)<=3/16; therefore g(p)=p(1-p)<=3/16. N_blocks denotes the MOM block count, distinct from the zero decoder offset b.

## Every possible audit array fails this certificate when n <= 8192

For the witness, div f=sum_i 400 m_i(1-m_i)>=0. Squared residual and stochastic-code variance are nonnegative. Thus the Stein scalar obeys psi(x)>=-d sigma^2=-1/25 for every x. Every block average and its median therefore obey the same lower bound.

The generic moment envelope has L=0 and c2=1. Its Q4>=K^2=256, so Vpsi=3(c0^2+c1^2 Q2+c2^2 Q4)>=768. For a valid certificate n>=N_blocks>=1, so 1<=l=floor(n/N_blocks)<=8192. With these conditions, its confidence radius obeys

    e_C=sqrt(8 Vpsi/l)>=sqrt(3/4)>4/5.

Consequently, for every possible data array,

    C_U=max(0,med_psi+e_C)>4/5-1/25=19/25.

But (G2), since J_U/k and its remaining C_U coefficient are nonnegative, requires

    C_U/V < Kp+g(p) <= 4+3/16=67/16,
    C_U < 67/(16*10^12).

These necessary inequalities contradict C_U>19/25. Even an H=1 bank, known p, and one block cannot rescue the bound. Increasing the simultaneous-bank block count cannot increase l, hence cannot reduce e_C. Requiring G2 uniformly over an interval containing the true p cannot remove its necessary condition at that p. This note retains n<=8192 as the actual planned-budget boundary; it does not claim that 8192 is the largest impossible budget.

## Consequence

Do not allocate a fresh 4096/8192-row audit or run a training sweep to evaluate this particular certificate. The population witness establishes mathematical nonemptiness but not practical certification; the original generic-envelope implementation is deterministically incapable of passing at the proposed audit budget. No simulated rows or decimal evaluation are necessary for this conclusion.

The next viable analytic step is a sharper confidence envelope (exploiting localized sigmoid derivatives and cancellation in psi), a materially less stringent geometry bridge, or a separately declared estimator contract. A proof must first show that its finite-sample bound can fit the available budget. This result does not establish any C2b comparator contrast, hard-between density identification, or failure of the VG estimator.

Review status: Astra xhigh approved the original note. Opus 5.5 xhigh confirmed the contradiction conditionally and requested inline definitions; these are now supplied. Its suggestion to use the population median of individual psi values is not adopted: the statistic here is a median of block MEANS, whose growing-block target is E psi=C. The generic-envelope theorem can be asymptotically valid despite its finite-budget failure. The completed OPUS_FINAL_THEOREM_REVIEW.md approves the amended contradiction as scoped. Its nonblocking minimum-block-size clarification is included. Gemini 3.1 Pro high remains unavailable at inference (429); this is not three-model consensus.
