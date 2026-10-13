# Native population objective versus clean stochastic risk

All twelve frozen cases are retained. Exact rational inputs and 512-bit Arb
outward arithmetic evaluate analytic bounds; no observations, quadrature,
fitting, optimizer updates or GPU computation are used.

- Obstruction certified: 6
- Sufficient bound inconclusive: 6
- Budget-unresolved: 0
- Implementation errors: 0

A certificate separates one explicit finite native witness from the entire
class with clean stochastic squared risk at most epsilon. It does not
exclude accurate dictionaries, prove practical optimization, or establish
unknown-D consistency or generating-density estimation. Inconclusive
matched-precision cases remain in the denominator.

Certificates concern the real-valued architecture at exact rational
weights and stated analytic log/softplus parameters. They do not certify
a rounded float64 implementation. Full bounds, exact endpoints, model
parameters and source hashes are in native_noise_obstruction_results.json.
