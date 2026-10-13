# Third-cumulant geometry candidate

2026-10-13. Primary-agent proposal with Astra xhigh algebra review. Opus 5.5 xhigh completed review found no blocking defect in this conditional certificate. Gemini 3.1 Pro high review and finite-sample affordability remain pending. No numerical execution or new estimator implementation.

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

For P_{i,pi(i)}=1, ||QP-D_*||_F^2=2K-2 sum_i A_i,pi(i). Hence a permutation P, with no sign flips, satisfies

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

A necessary condition for this displayed certificate to report a geometry radius at most rho is

    2 epsilon_T/kappa_L <= rho,
    epsilon_T <= rho ||T_hat||_F/(2 sqrt(K)+rho).

This follows from nonnegativity of the polar discrepancy and residual norm. It is only a floor of this particular bound, not an information-theoretic lower bound for all methods. Test this floor against an independently justified tensor radius before sampling. Any coefficient recovery assessment must align the permitted permutation consistently.

Do not silently treat a sample-centered third moment as unbiased or its empirical variance as a population confidence bound. The confidence proof must cover unknown mean estimation, finite-sample centering bias or an appropriate unbiased construction, Gaussian tails, tensor norm, and desired failure probability. No such affordable n=8192 radius is established here.

## Relation to the current obstruction

This offers a noncircular geometry route that does not require the tiny clean risk imposed by the local-flip counting proof. It does not by itself establish small coefficient risk, acceptable bias, optimizer convergence, a positive objective gap, generating-p estimation, or C2b advantage. In particular, kappa_3 is not injective as a function of p on (0,1/2); its magnitude must not be inverted into a unique density without more information.

The assumptions are homogeneous unit-amplitude square orthogonal coding. No claim for overcomplete d=8,K=16 or heterogeneous densities is made. Classical orthogonal tensor/moment recovery has the same information and must be retained as a comparator. This is a candidate sufficient certificate, with no novelty claim and no practical sample-budget claim.

Next: obtain the requested independent reviews, then prove an explicit tensor confidence radius and a compatible clean-risk interval at a frozen affordable budget before any Colab experiment.
