# Native finite-beta witness: frozen population calculation

All 16 prescribed cases are retained. The protocol has four exact support states per case,
no observations sampled, no seeds, and no optimizer updates. Every decoder is the
identity teacher dictionary. This is not a dictionary-gap counterexample.

- Resolved float64 decreases: 10
- Unresolved float64 differences: 6
- Budget-unresolved cases: 0
- Implementation failures: 0

The analytic lower bound applies to the ideal finite construction. Saved float64
tensors are independently evaluated at 240-digit Decimal precision. The float64
guard is an engineering resolution rule, not an outward-rounding proof. Tiny
differences remain unresolved; no alternative point is substituted.

Exact parameters, four-state forwards, population weights, parity discrepancies,
closure defects and source hashes are in native_bias_witness_results.json.
