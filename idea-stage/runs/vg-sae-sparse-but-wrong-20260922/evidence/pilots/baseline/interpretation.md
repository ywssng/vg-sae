# P1 interpretation after frozen test evaluation

All225 models completed at3000 updates. All675 calibration checkpoints were
retained;254 unique checkpoints were evaluated on test, including the entire225
final grid. The scientific runner took2796.93 seconds (0.777 hours), including
in-run setup, calibration, checkpoint/JSON I/O and test. Engineering timings and
scope exclusions are in timing_accounting.json. No scientific runs failed or
were retried. Five selection/rescaling tests passed; all artifact validation
checks passed, including675 checkpoint hashes, frozen selection/calibration/config
hashes, shared batches, independent splits, fixed dictionaries and exact L1 gains.

## Main finding and unresolved comparisons

The preregistered primary targetL0=1.8 has **zero VG coverage in all9 worlds**.
The comparison cannot establish either improvement or failure at that target.
The broad wrong-sparsity criterion at1.8 and2.2 is consequently unresolved.
Do not reinterpret missing coverage as evidence of robustness or as a negative
result about the VG research direction.

At targetL0=2, VG has final-step coverage in7/9 worlds. Its selected covered-world
signed cosine and support F1 are approximately1. This confirms that current VG
can learn the positive dictionary and support in this short narrow-amplitude toy.
L1 and Gated also have high-quality covered conditions. No comparison satisfies
the preregistered joint improvement in at least2/3 worlds for either correlation
sign: the tested regime does not establish a VG-specific identity advantage.
BatchTopK comparisons meet the joint threshold in only1 world per correlation
at target2. Cross-method group means may use different covered worlds; use
summary.json paired_worlds rather than comparing those group means directly.

Anthropic-style JumpReLU fails native-L0 coverage at all three targets. Across
the three-world mean curves its smallest penalty.03 gives achievedL0 about.67
to1.00; larger penalties are sparser, including collapsed codes. Its fixed pilot
recipe therefore did not bracket the relevant region. **This is an unresolved
baseline calibration/convergence comparison, not evidence that VG beats a
properly tuned strong JumpReLU.** No grid was changed after outcomes. The
bandwidth/threshold/tanh/pre-act variant,1000-step warmup and lack of input
rescaling remain explicitly recorded deviations from the original full recipe.

## Useful method-development observations

- VG gamma2, fixed a priori, yields mean test nativeL0=5.00,5.00,4.798 for
  rho=-.4,0,+.4 despite hardEV above.999 in all three aggregate curves. It is a
  concrete example of excellent reconstruction with poor support identity.
  Gamma4 reaches the true operating region in several worlds, but success is
  seed/condition dependent. Fixed gamma does not discover the correct support
  count automatically, and parameter-grid fractions are not comparable robustness
  measures across methods.
- On all5 L0-matched target2 VG-versus-L1 paired worlds, calibration-only L1 gains
  reduce coefficient NMSE and test MSE while leaving dictionary and support
  metrics exactly unchanged. For example, rho=-.4/world211 coefficient NMSE falls
  from.06972 to.01328, and MSE from.003895 to.001530. This supports using the
  amplitude-bias control; it does not establish a unique VG anti-mixing effect.
- The preregistered truth-free minimum-c_dec selector is fallible. In rho=+.4,
  world212 it chooses VG gamma2 with signed cosine.99971 but nativeL0=4.39465 and
  supportF1=.62633. In rho=0,world211 it chooses nativeL0=3.20477, signed
  cosine.79815 and F1=.76760. A nearly correct decoder alone does not certify
  correct support decisions or a useful operating point.

## Metric and implementation scope

`mixing_energy` must be read as **signed-assignment leakage**, which can include
sign or matching mismatch. Pure negative atoms can produce large values without
multi-feature mixtures. Signed and absolute cosine are both reported;225 full
final decoder-to-truth cosine matrices and the original signed assignments are
in decoder_geometry.json. This post-run interpretation export changes neither
the registered metric values nor selection/acceptance criteria.

No saved VG calibration checkpoint is within10 times the configured profiled
risk floor (0/135). This is a checkpoint observation, not an all-update boundary
rate. Complete training-loss convergence is not established. The pilot uses
equal widths, controls, checkpoints and training examples, but trainable parameter
counts differ (VG330, L1225, Gated235, Jump230, BatchTopK225), and official baseline
decoder training differs from unit-normalized VG. Full norm metadata, source
hashes, exact recipes and deviations are retained. Three paired worlds and this
narrow orthogonal toy support screening and implementation feasibility only.

The next validation should first preregister a calibration-only range-finding
stage that brackets nativeL0 targets for every method, especially JumpReLU and
the undersparse VG target, then freeze new main-world grids before fresh test
access. This is a future protocol recommendation, not an extra run performed here.
