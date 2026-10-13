# Focused algebra review

Scope: audit only the population identities and inequalities (A)-(C) below. Archived numerical certificates and implementation correctness need a separate evidence review; do not claim to have checked them. Historical pending-review text is a source snapshot, not current connection status.

# Matched-precision native amplitude check, v1

Date: 2026-10-13. Frozen before numerical execution. This is a two-case
analytic certificate and native implementation check, not a training campaign.

## Target and status

Test one finite native direction in which the observed-data VG objective
decreases and hard coefficient NMSE increases, with the dictionary and hard
supports unchanged, even at the true noise precision and prior. This does not
compare against the best recovering class, disprove a dictionary-neighborhood
gap, or decide the original C2 selector endpoints. The analytical derivation is
coherent as stated; the two numerical certificates are pending at this freeze.

## Frozen model and observables

Independent coordinates: S_i ~ Bernoulli(q), X_i=S_i+sigma Z_i, Z_i~N(0,1),
K=d=2, sigma=1/20, beta=400, q in {1/10,3/10}. The two cases are the entire
roster. gamma=log((1-q)/q) is supplied, not estimated. D=I and b=0. Gates
m_i(X)=sigmoid(beta*(X_i-1/2)-gamma) are the exact posterior for this supplied
orthogonal binary generator. They are a finite affine sigmoid native encoder.

Let u0=log(exp(1)-1). Compare a_0=1 against
a_t(X_i)=softplus(u0+t*(X_i-1)), with fixed t=1/2. Both are finite native
affine softplus encoders. Their weights are respectively 0 and t*I; biases
are u0 and u0-t. All other parameters are identical. Precision is held fixed
in the normalized learned-beta branch; standard gate KL and full stochastic
variance are retained. There is no optimizer update or changed model default.

G=beta/2 E||X-B*a(X)||^2 + E sum KL(Bern(m)||Bern(q)); the identical Gaussian
normalizer cancels in the difference. Conditional B draws are independent of
S given X. Hard codes are H*a, H=1{m>1/2}. Population hard coefficient NMSE is
E||H*a-S||^2/(K*q). It is a ratio of population expectations.

## Exact identities and inequalities

For all real x != 1 and 0<t<=1, a_t(x) lies strictly between x and 1, because
softplus is increasing and its derivative lies in (0,1). Thus
g_t(x)=(x-1)^2-(x-a_t(x))^2>0. The pointwise stochastic risk difference is
m(x)*g_t(x), not the squared error of the mean reconstruction alone.

The posterior identity E[m(X)g(X)]=q E[g(1+sigma Z)] gives
G_0-G_t=(beta*K*q/2) E[g_t(1+sigma Z)]>0 and
dG_t/dt|_0=-K*q*(1-exp(-1)) at beta*sigma^2=1.

Set c=t*sigmoid(u0-t*sigma), and
mu2=E[Z^2 1{|Z|<=1}]=erf(1/sqrt(2))-sqrt(2/pi)*exp(-1/2).
On |Z|<=1, c <= (a_t(1+sigma Z)-1)/(sigma Z) < 1, with the value at zero
interpreted by continuity. Since v*(2-v) increases for 0<v<1,

    G_0-G_t >= (K*q/2)*c*(2-c)*mu2.                 (A)

Let h=1/2+sigma^2*gamma, alpha=h/sigma, delta=(1-h)/sigma,
phi(z)=exp(-z^2/2)/sqrt(2*pi), Q(z)=Pr(Z>z). Verify 0<h<1-sigma.
On true-active coordinates, the baseline hard error is zero whenever H=1;
the new amplitude adds a nonnegative squared error. On |Z|<=1, H=1 and the
additional squared error is at least c^2*sigma^2*Z^2. On inactive coordinates,
the decrease in error is at most 1{X>h}. Therefore

    NMSE_t-NMSE_0 >= c^2*sigma^2*mu2 - ((1-q)/q)*Q(alpha)
                 >= c^2*sigma^2*mu2 - ((1-q)/q)*phi(alpha)/alpha.       (B)

For an absolute scale bound, the active error is at most
t^2*sigma^2*Z^2 + 1{X<=h}. For inactive selected coordinates, x>h>0 and
a_t(x)^2 <= max(1,x)^2 <= 1+x^2. The Gaussian tail second moment is
E[Z^2 1{Z>alpha}]=alpha*phi(alpha)+Q(alpha). Mills' inequality Q(z)<=phi(z)/z
for z>0 yields

    NMSE_t <= t^2*sigma^2 + phi(delta)/delta
             + ((1-q)/q)*phi(alpha)*[(1+sigma^2)/alpha+sigma^2*alpha]. (C)

These are population bounds, not quadrature or fitted estimates. A positive
lower bound in (B) demonstrates coefficient-error degradation for this pair.
It supplies no global optimum or actual trained-model conclusion.

## Scope relative to C2

C2a concerns generating-count accuracy, reporting and abstention. C2b concerns
a paired selector improvement in hard coefficient NMSE of at least .02, with
confidence and decoder guards; .02 is not an absolute NMSE-success threshold.
We report whether (C)<.02 only to show the scale of this explicit construction.
Neither (A), (B), nor (C) supplies the hard-between selection law or clears BR5.
The prior, geometry and amplitudes in this pair are supplied for analysis;
no truth-free estimator or unknown-dictionary identification is claimed.

## Execution and falsification

Use Colab CLI CPU only, official python-flint 0.8.0 Arb, 512 bits, one thread.
Certificate process cap: 60 CPU / 120 wall seconds, using the already verified
resource guard. Two fixed cases; no samples, quadrature, fitting, seeds,
search or training. Compare exact rational outward interval endpoints with 0
and 1/50. Any failed bound stays inconclusive. No replacement case is permitted.

Before the certificate, run only the new focused tests: arbitrary fixed input
points check the finite native mapping, unchanged gates/support/decoder, full
stochastic loss identity, and descent; exact arithmetic checks retain both
cases and require the declared premises. These float64 checks are separate
from the real-parameter certificate. They do not transfer rigorous enclosures
to rounded Torch parameters. Record source/plan hashes, dependency versions,
all results, test receipts and whole-process resources; fetch before release.

## Independent review and limitations

Astra xhigh is reviewing this derivation independently. The requested Opus 5.5
xhigh and Gemini 3.1 Pro high reviews are pending connection: neither is in
the collaboration roster, Claude's saved credentials contain no usable token,
and Google's saved OAuth grant was rejected. No substitute-model endorsement
is claimed. The run's review ledger must retain these missing reviews.

Applied workflows: project-local ARIS formula-derivation; Scientific Agent
Skills scientific-critical-thinking; physicist-analyst and physics-guide-solver
(assumptions, scale, dimensionless beta*sigma^2, and observable separation).
Software attribution: [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
Bibliographic metadata was checked in the preceding continuation; no new
literature novelty claim is made here.
