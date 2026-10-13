# Focused final review packet

Check two standalone algebraic claims only, no computation or new research task. Both are limited sufficient-certificate statements, not C2b performance claims. Please deliver a final verdict with blocking corrections only. The first note includes definitions requested in the completed earlier Opus review. The earlier Opus suggestion to substitute the population median of individual psi into median-of-block-means asymptotics was rejected: block means converge to the expectation, not the individual-variable median. The second is a subsequent primary/Astra proposal. No affordable tensor confidence radius is claimed.

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

The generic moment envelope has L=0 and c2=1. Its Q4>=K^2=256, so Vpsi=3(c0^2+c1^2 Q2+c2^2 Q4)>=768. With N_blocks>=1 and l=floor(n/N_blocks)<=8192, its confidence radius obeys

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

Review status: Astra xhigh approved the original note. Opus 5.5 xhigh confirmed the contradiction conditionally and requested inline definitions; these are now supplied. Its suggestion to use the population median of individual psi values is not adopted: the statistic here is a median of block MEANS, whose growing-block target is E psi=C. The generic-envelope theorem can be asymptotically valid despite its finite-budget failure. Final amended-note Opus review and Gemini 3.1 Pro high review are tracked separately; this is not three-model consensus.


# Third-cumulant geometry candidate

2026-10-13. Primary-agent proposal with Astra xhigh algebra review. Opus/Gemini review and finite-sample affordability remain pending. No numerical execution or new estimator implementation.

## Contract and population identity

Let S_j be iid Bernoulli(p), 0<p<1/2, D_* square orthogonal, and X=offset+D_*S+independent Gaussian noise. The third central cumulant tensor is

    T_*=E[(X-E X)^{tensor 3}]=kappa_3 sum_j d_*j^{tensor 3},
    kappa_3=p(1-p)(1-2p)>0.

Gaussian noise and a constant offset do not change this cumulant. The geometry identity below therefore does not itself require known noise variance or known offset; subsequent clean-risk and bias bridges retain their own explicit assumptions.

## Global positive-permutation bound

For any square orthogonal Q, let T_Q=sum_i q_i^{tensor 3}, A=Q^T D_*, and E=||T_*-kappa_3 T_Q||_F. Orthogonality gives

    E^2/kappa_3^2=2K-2 sum_ij A_ij^3.

The matrix W with W_ij=A_ij^2 is doubly stochastic. By its decomposition as a convex combination of permutation matrices,

    max_pi sum_i A_i,pi(i) >= sum_ij W_ij A_ij=sum_ij A_ij^3.

Hence a permutation P, with no sign flips, satisfies

    ||QP-D_*||_F <= E/kappa_3.

No small-error condition is needed for this inequality. Its quantitative usefulness still depends on E/kappa_3 being small. The positive sign is fixed by p<1/2. For any full-rank square D with orthogonal polar factor Q,

    ||DP-D_*||_F <= ||D-Q||_F+E/kappa_3.

Q is only an analysis object: computing a certificate from it does not replace the trained decoder or modify the VG objective.

## Conditional finite-sample conversion

On an independently justified tensor confidence event ||T_hat-T_*||_F<=epsilon_T, set

    kappa_hat=||T_hat||_F/sqrt(K),
    kappa_L=(||T_hat||_F-epsilon_T)/sqrt(K).

Since ||T_*||_F=sqrt(K) kappa_3, if kappa_L>0 then

    ||DP-D_*||_F <= ||D-Q||_F
      +[||T_hat-kappa_hat T_Q||_F+2 epsilon_T]/kappa_L.

The same tensor-norm event applies to every Q, including data-dependent Q. This geometry step requires no bank-wise union correction once that event is valid. A separate clean-risk confidence bound for a selected encoder still needs its own independence/selection contract.

Do not silently treat a sample-centered third moment as unbiased or its empirical variance as a population confidence bound. The confidence proof must cover unknown mean estimation, finite-sample centering bias or an appropriate unbiased construction, Gaussian tails, tensor norm, and desired failure probability. No such affordable n=8192 radius is established here.

## Relation to the current obstruction

This offers a noncircular geometry route that does not require the tiny clean risk imposed by the local-flip counting proof. It does not by itself establish small coefficient risk, acceptable bias, optimizer convergence, a positive objective gap, generating-p estimation, or C2b advantage. In particular, kappa_3 is not injective as a function of p on (0,1/2); its magnitude must not be inverted into a unique density without more information.

The assumptions are homogeneous unit-amplitude square orthogonal coding. No claim for overcomplete d=8,K=16 or heterogeneous densities is made. Classical orthogonal tensor/moment recovery has the same information and must be retained as a comparator. This is a candidate sufficient certificate, with no novelty claim and no practical sample-budget claim.

Next: obtain the requested independent reviews, then prove an explicit tensor confidence radius and a compatible clean-risk interval at a frozen affordable budget before any Colab experiment.
